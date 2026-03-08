from __future__ import annotations

import os
import uuid
import time
from pathlib import Path
from typing import Any

import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .citations import attach_citations
from .text_utils import clean_answer, iter_final_text
from .memory import build_memory_block
from .consent import (
    consent_denied_response,
    consent_granted_response,
    consent_request,
    detect_consent,
)
from .engine import RAGEngine
from .dify_client import chat_once as dify_chat_once, stop_generation as dify_stop_generation
from .rag_backends import build_backend_status
from .settings import load_settings
from .state import StateStore
from .router import route_non_rag
from .voice_adapters import build_llm_status, build_voice_statuses, synthesize_audio, transcribe_audio
from .voice_session import VoiceSession

settings = load_settings()
state = StateStore(Path(__file__).resolve().parents[1] / ".state")
engine = RAGEngine(settings, Path(__file__).resolve().parents[1] / ".state")
voice_sessions: dict[str, VoiceSession] = {}

app = FastAPI(title="Micro Leasing RAG Demo")
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    rebuild: bool = False


class RetrieveRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    stream: bool = False
    fast: bool = False
    mode: str | None = None
    backend: str | None = None


class VoiceChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    stream: bool = True
    backend: str | None = None


def _append_turn(session: Any, message: str, answer: str, max_turns: int) -> None:
    if max_turns <= 0:
        return
    session.transcript.append({"role": "user", "text": message})
    session.transcript.append({"role": "assistant", "text": answer})
    limit = max_turns * 2
    if limit > 0 and len(session.transcript) > limit:
        session.transcript = session.transcript[-limit:]


def _selected_backend(requested: str | None) -> str:
    name = (requested or "our_rag").strip() or "our_rag"
    if name not in {"our_rag", "dify_rag"}:
        return "our_rag"
    return name


def _launch_mode() -> str:
    return os.getenv("RAG_LAUNCH_MODE", "direct")


def _rag_statuses() -> dict[str, dict[str, Any]]:
    dify_base_url = os.getenv("DIFY_API_BASE_URL", "")
    dify_key = os.getenv("DIFY_API_KEY", "")
    return {
        "our_rag": {
            "name": "our_rag",
            "available": True,
            "healthy": settings.app.kb_markdown_path.exists(),
            "reason": "ok" if settings.app.kb_markdown_path.exists() else "kb_missing",
        },
        "dify_rag": {
            "name": "dify_rag",
            "available": bool(dify_base_url and dify_key),
            "healthy": bool(dify_base_url and dify_key),
            "reason": "ok" if dify_base_url and dify_key else "not_configured",
        },
    }


def _used_knowledge_from_chunks(final_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": c["chunk_id"],
            "heading_path": c.get("heading_path", []),
            "snippet": c.get("text", "")[:280],
            "source": c.get("source", ""),
            "doc_name": c.get("doc_name", ""),
        }
        for c in final_chunks
    ]


def _dify_inputs(*, fast: bool, voice_fast: bool, mode: str, session_id: str) -> dict[str, Any]:
    return {
        "fast": fast,
        "voice_fast": voice_fast,
        "mode": mode,
        "session_id": session_id,
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "kb": str(settings.app.kb_markdown_path)}


@app.get("/api/backends")
async def backends() -> dict[str, Any]:
    return build_backend_status(
        launch_mode=_launch_mode(),
        rag_statuses=_rag_statuses(),
        voice_statuses=build_voice_statuses(),
        llm_status=build_llm_status(settings.llm.base_url, settings.llm.model),
    )


@app.post("/api/index")
async def index_kb(payload: IndexRequest) -> dict[str, Any]:
    result = engine.index()
    if not result.get("ok"):
        return result
    state.log({"event": "index", "chunks": result.get("chunks")})
    return result


@app.post("/api/retrieve")
async def retrieve(payload: RetrieveRequest) -> dict[str, Any]:
    query = payload.query.strip()
    if not query:
        return {"ok": False, "error": "empty query"}
    result = engine.retrieve(query)
    return result


@app.post("/api/chat")
async def chat(payload: ChatRequest, stream: bool = False) -> Any:
    message = payload.message.strip()
    if not message:
        return {"ok": False, "error": "empty message"}
    stream = bool(payload.stream) or stream
    mode = (payload.mode or "").strip().lower()
    voice_fast = mode == "voice_fast"
    fast = bool(payload.fast) or voice_fast
    backend_name = _selected_backend(payload.backend)

    session_id = payload.session_id or str(uuid.uuid4())
    session = state.get(session_id) or state.create(session_id)
    timings: dict[str, Any] = {}

    decision = detect_consent(message)
    if session.consent_denied:
        response = {
            "ok": True,
            "session_id": session_id,
            "backend": backend_name,
            "answer": consent_denied_response(),
            "consent": "denied",
            "chunks": [],
            "citations": [],
        }
        return _stream_or_json(response, stream)

    if not session.consent_given:
        if decision == "denied":
            session.consent_denied = True
            state.update(session)
            response = {
                "ok": True,
                "session_id": session_id,
                "backend": backend_name,
                "answer": consent_denied_response(),
                "consent": "denied",
                "chunks": [],
                "citations": [],
            }
            return _stream_or_json(response, stream)
        if decision == "granted":
            session.consent_given = True
            state.update(session)
            response = {
                "ok": True,
                "session_id": session_id,
                "backend": backend_name,
                "answer": consent_granted_response(),
                "consent": "granted",
                "chunks": [],
                "citations": [],
            }
            return _stream_or_json(response, stream)
        response = {
            "ok": True,
            "session_id": session_id,
            "backend": backend_name,
            "answer": consent_request(),
            "consent": "needed",
            "chunks": [],
            "citations": [],
        }
        return _stream_or_json(response, stream)

    t_route = time.perf_counter()
    routed = route_non_rag(message, settings.llm.base_url, settings.llm.model)
    timings["route_ms"] = (time.perf_counter() - t_route) * 1000
    if routed:
        state.log({"event": "router", "kind": routed.kind, "session_id": session_id})
        response = {
            "ok": True,
            "session_id": session_id,
            "backend": backend_name,
            "answer": routed.response,
            "consent": "granted",
            "chunks": [],
            "citations": [],
            "timings": timings,
            "conversation_ref": {},
            "can_barge_in": True,
        }
        _append_turn(session, message, routed.response, settings.app.memory_turns)
        state.update(session)
        return _stream_or_json(response, stream)

    if backend_name == "dify_rag":
        dify_base_url = os.getenv("DIFY_API_BASE_URL", "")
        dify_api_key = os.getenv("DIFY_API_KEY", "")
        if not dify_base_url or not dify_api_key:
            response = {
                "ok": True,
                "session_id": session_id,
                "backend": backend_name,
                "answer": "Dify не настроен. Проверьте DIFY_API_BASE_URL и DIFY_API_KEY.",
                "consent": "granted",
                "chunks": [],
                "used_knowledge": [],
                "citations": [],
                "timings": timings,
                "conversation_ref": {},
                "can_barge_in": True,
            }
            return _stream_or_json(response, stream)
        t_dify = time.perf_counter()
        provider_resp = dify_chat_once(
            base_url=dify_base_url,
            api_key=dify_api_key,
            query=message,
            user=session_id,
            inputs=_dify_inputs(fast=fast, voice_fast=voice_fast, mode=mode, session_id=session_id),
            conversation_id=(session.metadata.get("dify") or {}).get("conversation_id"),
            timeout_sec=settings.llm.timeout_sec,
        )
        timings["dify_total_ms"] = (time.perf_counter() - t_dify) * 1000
        session.metadata["dify"] = provider_resp.conversation_ref
        _append_turn(session, message, provider_resp.answer, settings.app.memory_turns)
        state.update(session)
        state.log(
            {
                "event": "chat",
                "backend": "dify_rag",
                "session_id": session_id,
                "question": message,
                "chunks": [c.get("chunk_id") for c in provider_resp.used_knowledge],
                "timings": timings,
            }
        )
        response = {
            "ok": True,
            "session_id": session_id,
            "backend": provider_resp.backend,
            "answer": provider_resp.answer or settings.app.strict_refusal_text,
            "consent": "granted",
            "chunks": [],
            "used_knowledge": provider_resp.used_knowledge,
            "citations": provider_resp.citations,
            "timings": {**provider_resp.timings, **timings},
            "conversation_ref": provider_resp.conversation_ref,
            "can_barge_in": provider_resp.can_barge_in,
            "incomplete": provider_resp.incomplete,
        }
        return _stream_or_json(response, stream)

    retrieval = engine.retrieve(message, fast=fast, voice_fast=voice_fast, session_id=session_id)
    if retrieval.get("timings"):
        timings.update(retrieval.get("timings") or {})
    if not retrieval.get("ok"):
        response = {
            "ok": True,
            "session_id": session_id,
            "backend": backend_name,
            "answer": settings.app.strict_refusal_text,
            "consent": "granted",
            "chunks": [],
            "citations": [],
            "timings": timings,
            "conversation_ref": {},
            "can_barge_in": True,
        }
        _append_turn(session, message, response["answer"], settings.app.memory_turns)
        state.update(session)
        return _stream_or_json(response, stream)

    final_chunks = retrieval.get("final") or []
    weak_context = bool(retrieval.get("weak"))
    if not final_chunks:
        state.log({"event": "no_context", "query": message, "session_id": session_id})
        response = {
            "ok": True,
            "session_id": session_id,
            "backend": backend_name,
            "answer": settings.app.strict_refusal_text,
            "consent": "granted",
            "chunks": [],
            "citations": [],
            "timings": timings,
            "conversation_ref": {},
            "can_barge_in": True,
        }
        _append_turn(session, message, response["answer"], settings.app.memory_turns)
        state.update(session)
        return _stream_or_json(response, stream)

    system_prompt = settings.app.system_prompt_path.read_text(encoding="utf-8")
    system_prompt = system_prompt + "\n\nСогласие на обработку данных уже получено, не запрашивай его."
    memory_block = build_memory_block(session.transcript, settings.app.memory_turns)
    context_block = "\n\n".join(
        [f"[Fragment {i+1}]\n{c['text']}" for i, c in enumerate(final_chunks)]
    )

    expanded = any(trigger in message.lower() for trigger in settings.llm.expand_triggers)
    length_hint = (
        f"Ответ должен быть {settings.llm.concise_sentences_min}–{settings.llm.concise_sentences_max} коротких предложений."
        if not expanded
        else "Можно ответить подробнее, но только на основе контекста."
    )

    weak_hint = ""
    if weak_context:
        weak_hint = (
            "Контекст может быть неполным. Дай ближайшую релевантную информацию из фрагментов, "
            "скажи, что точных данных может не хватать, и задай уточняющий вопрос.\n\n"
        )
    user_prompt = (
        "Ответь строго на основе следующих фрагментов базы знаний. "
        "Если ответа нет — верни точный отказ.\n\n"
        f"{memory_block}{length_hint}\n\n"
        f"{weak_hint}{context_block}\n\nВопрос клиента: {message}"
    )

    from .llm import call_openai_compatible, iter_openai_compatible_stream_events

    model = settings.llm.fast_model if fast and settings.llm.fast_model else settings.llm.model
    base_url = settings.llm.fast_base_url if fast and settings.llm.fast_base_url else settings.llm.base_url
    max_tokens = settings.llm.fast_max_tokens if fast else settings.llm.max_tokens

    if stream:
        def gen() -> Any:
            streamed_parts: list[str] = []
            had_final = False
            finish_reason: str | None = None
            llm_start = time.perf_counter()
            first_token_at: float | None = None
            try:
                def delta_texts() -> Any:
                    nonlocal finish_reason
                    stream_iter = iter_openai_compatible_stream_events(
                    base_url=base_url,
                    model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=settings.llm.temperature,
                        max_tokens=max_tokens,
                        timeout_sec=settings.llm.timeout_sec,
                    )
                    for event in stream_iter:
                        choice = (event.get("choices") or [{}])[0]
                        if choice.get("finish_reason"):
                            finish_reason = choice.get("finish_reason")
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        if text:
                            yield text

                stream_iter = delta_texts()
                for chunk in iter_final_text(stream_iter):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    had_final = True
                    streamed_parts.append(chunk)
                    yield f"data: {json.dumps({'type': 'delta', 'text': chunk}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                state.log({"event": "llm_error", "error": str(exc), "session_id": session_id})
                error_payload = {
                    "type": "final",
                    "ok": True,
                    "session_id": session_id,
                    "backend": backend_name,
                    "answer": (
                        "LLM не настроен или недоступен. "
                        "Проверьте RAG_LLM_BASE_URL и RAG_LLM_MODEL, затем перезапустите backend."
                    ),
                    "consent": "granted",
                    "chunks": [],
                    "used_knowledge": [],
                    "citations": [],
                    "conversation_ref": {},
                    "can_barge_in": True,
                }
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                return

            if had_final:
                answer_text = clean_answer("".join(streamed_parts))
            else:
                answer_text = settings.app.strict_refusal_text

            llm_end = time.perf_counter()
            llm_total = llm_end - llm_start
            ttfb_ms = ((first_token_at - llm_start) * 1000) if first_token_at else None
            token_est = max(1, len(answer_text.split()))
            timings["llm_ttfb_ms"] = ttfb_ms
            timings["llm_total_ms"] = llm_total * 1000
            timings["llm_tokens_per_sec"] = token_est / llm_total if llm_total > 0 else None
            timings["llm_finish_reason"] = finish_reason

            _append_turn(session, message, answer_text, settings.app.memory_turns)
            state.update(session)

            citations = attach_citations(answer_text, final_chunks) if had_final else []
            used_knowledge = [
                {
                    "chunk_id": c["chunk_id"],
                    "heading_path": c.get("heading_path", []),
                    "snippet": c.get("text", "")[:280],
                    "source": c.get("source", ""),
                    "doc_name": c.get("doc_name", ""),
                }
                for c in final_chunks
            ] if had_final else []

            state.log({
                "event": "chat",
                "session_id": session_id,
                "question": message,
                "normalized_query": retrieval.get("normalized_query"),
                "rewritten_query": retrieval.get("rewritten_query"),
                "top_rerank_score": retrieval.get("top_rerank_score"),
                "chunks": [c.get("chunk_id") for c in final_chunks],
                "timings": timings,
            })

            final_payload = {
                "type": "final",
                "ok": True,
                "session_id": session_id,
                "backend": backend_name,
                "answer": answer_text,
                "consent": "granted",
                "chunks": final_chunks if had_final else [],
                "used_knowledge": used_knowledge,
                "citations": citations,
                "timings": timings,
                "incomplete": bool(finish_reason == "length"),
                "conversation_ref": {},
                "can_barge_in": True,
            }
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        llm_start = time.perf_counter()
        llm_resp = call_openai_compatible(
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=settings.llm.temperature,
            max_tokens=max_tokens,
            timeout_sec=settings.llm.timeout_sec,
        )
        answer = clean_answer(llm_resp.text) or settings.app.strict_refusal_text
        llm_total = time.perf_counter() - llm_start
        token_est = max(1, len(answer.split()))
        timings["llm_ttfb_ms"] = None
        timings["llm_total_ms"] = llm_total * 1000
        timings["llm_tokens_per_sec"] = token_est / llm_total if llm_total > 0 else None
    except Exception as exc:
        state.log({"event": "llm_error", "error": str(exc), "session_id": session_id})
        response = {
            "ok": True,
            "session_id": session_id,
            "backend": backend_name,
            "answer": (
                "LLM не настроен или недоступен. "
                "Проверьте RAG_LLM_BASE_URL и RAG_LLM_MODEL, затем перезапустите backend."
            ),
            "consent": "granted",
            "chunks": [],
            "citations": [],
            "timings": timings,
            "conversation_ref": {},
            "can_barge_in": True,
        }
        return _stream_or_json(response, stream)
    citations = attach_citations(answer, final_chunks)
    used_knowledge = [
        {
            "chunk_id": c["chunk_id"],
            "heading_path": c.get("heading_path", []),
            "snippet": c.get("text", "")[:280],
            "source": c.get("source", ""),
            "doc_name": c.get("doc_name", ""),
        }
        for c in final_chunks
    ]

    state.log({
        "event": "chat",
        "session_id": session_id,
        "question": message,
        "normalized_query": retrieval.get("normalized_query"),
        "rewritten_query": retrieval.get("rewritten_query"),
        "top_rerank_score": retrieval.get("top_rerank_score"),
        "chunks": [c.get("chunk_id") for c in final_chunks],
        "timings": timings,
    })

    response = {
        "ok": True,
        "session_id": session_id,
        "backend": backend_name,
        "answer": answer,
        "consent": "granted",
        "chunks": final_chunks,
        "used_knowledge": used_knowledge,
        "citations": citations,
        "timings": timings,
        "conversation_ref": {},
        "can_barge_in": True,
        "retrieval": {
            "normalized_query": retrieval.get("normalized_query"),
            "candidates": retrieval.get("candidates"),
        },
    }
    _append_turn(session, message, answer, settings.app.memory_turns)
    state.update(session)
    return _stream_or_json(response, stream)


@app.post("/api/voice/chat")
async def voice_chat(payload: VoiceChatRequest) -> Any:
    chat_payload = ChatRequest(
        message=payload.message,
        session_id=payload.session_id,
        stream=payload.stream,
        fast=True,
        mode="voice_fast",
        backend=payload.backend,
    )
    return await chat(chat_payload, stream=payload.stream)


@app.get("/api/logs")
async def logs(limit: int = 200) -> dict[str, Any]:
    return {"ok": True, "items": state.tail_logs(limit=limit)}


@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = VoiceSession(session_id=session_id, backend="our_rag")
    voice_sessions[session_id] = session
    audio_chunks: list[str] = []
    await websocket.send_json(
        {
            "type": "session.ready",
            "session_id": session_id,
            "backend": session.backend,
        }
    )
    try:
        while True:
            event = await websocket.receive_json()
            event_type = event.get("type")
            if event_type == "session.update":
                session.backend = _selected_backend(event.get("backend"))
                await websocket.send_json(
                    {
                        "type": "session.updated",
                        "session_id": session_id,
                        "backend": session.backend,
                    }
                )
            elif event_type == "input_audio_buffer.append":
                audio = event.get("audio") or ""
                if audio:
                    audio_chunks.append(audio)
                    for action in session.on_audio_chunk(audio):
                        await websocket.send_json(action)
                        if action.get("type") == "interrupt" and action.get("task_id") and session.backend == "dify_rag":
                            dify_base_url = os.getenv("DIFY_API_BASE_URL", "")
                            dify_api_key = os.getenv("DIFY_API_KEY", "")
                            if dify_base_url and dify_api_key:
                                try:
                                    dify_stop_generation(
                                        base_url=dify_base_url,
                                        api_key=dify_api_key,
                                        task_id=action["task_id"],
                                        user=session_id,
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    await websocket.send_json(
                                        {
                                            "type": "warning",
                                            "session_id": session_id,
                                            "message": f"interrupt stop failed: {exc}",
                                        }
                                    )
            elif event_type == "input_audio_buffer.commit":
                audio_b64 = "".join(audio_chunks)
                audio_chunks.clear()
                try:
                    transcript = transcribe_audio(audio_b64, session_id=session_id)
                except Exception as exc:  # noqa: BLE001
                    await websocket.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "error": f"stt_failed: {exc}",
                        }
                    )
                    continue
                text = (transcript.get("text") or "").strip()
                await websocket.send_json(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "session_id": session_id,
                        "provider": transcript.get("provider"),
                        "transcription": text,
                    }
                )
                if not text:
                    continue
                for action in session.on_transcript_final(text):
                    await websocket.send_json(action)
                response = await chat(
                    ChatRequest(
                        message=text,
                        session_id=session_id,
                        stream=False,
                        fast=True,
                        mode="voice_fast",
                        backend=session.backend,
                    ),
                    stream=False,
                )
                if isinstance(response, dict):
                    for action in session.on_provider_response(response):
                        await websocket.send_json(action)
                    answer_text = response.get("answer", "")
                    if answer_text:
                        await websocket.send_json(
                            {
                                "type": "response.output_text.delta",
                                "session_id": session_id,
                                "delta": answer_text,
                            }
                        )
                    try:
                        audio_response = synthesize_audio(answer_text, session_id=session_id)
                        audio_b64 = audio_response.get("audio_b64") or ""
                        if audio_b64:
                            await websocket.send_json(
                                {
                                    "type": "response.output_audio.delta",
                                    "session_id": session_id,
                                    "delta": audio_b64,
                                    "sample_rate_hz": audio_response.get("sample_rate_hz"),
                                }
                            )
                    except Exception as exc:  # noqa: BLE001
                        await websocket.send_json(
                            {
                                "type": "warning",
                                "session_id": session_id,
                                "message": f"tts_failed: {exc}",
                            }
                        )
                    await websocket.send_json(
                        {
                            "type": "response.done",
                            "session_id": session_id,
                            "backend": response.get("backend"),
                            "used_knowledge": response.get("used_knowledge", []),
                            "citations": response.get("citations", []),
                            "timings": response.get("timings", {}),
                        }
                    )
            elif event_type == "response.cancel":
                session.assistant_speaking = False
                session.interrupted = True
                await websocket.send_json({"type": "response.cancelled", "session_id": session_id})
    except WebSocketDisconnect:
        voice_sessions.pop(session_id, None)


@app.post("/api/voice/start")
async def voice_start() -> JSONResponse:
    return JSONResponse(status_code=501, content={"ok": False, "error": "Voice not implemented"})


@app.post("/api/voice/stop")
async def voice_stop() -> JSONResponse:
    return JSONResponse(status_code=501, content={"ok": False, "error": "Voice not implemented"})


@app.get("/api/voice/status")
async def voice_status() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "launch_mode": _launch_mode(),
            "services": build_voice_statuses(),
        },
    )


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def _stream_or_json(payload: dict[str, Any], stream: bool) -> Any:
    if not stream:
        return payload
    if "type" not in payload:
        payload = {"type": "final", **payload}

    def gen() -> Any:
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
