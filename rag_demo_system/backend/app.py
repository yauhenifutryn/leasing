from __future__ import annotations

import asyncio
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
from .rag_backends import build_backend_status
from .settings import load_settings
from .tools import get_tool_schemas, get_tool, init_tools, get_all_tools
from .tools.filler import get_filler
from .llm_stream import parse_tool_calls_from_events
from .state import StateStore
from .router import route_non_rag
from .voice_adapters import build_llm_status, build_voice_statuses, synthesize_audio, transcribe_audio
from .voice_session import VoiceSession
from .vad import SileroVAD
from .audio_input import WebSocketAudioAdapter
from .rtc_audio import RTCAudioHandler

settings = load_settings()
init_tools(settings)
state = StateStore(Path(__file__).resolve().parents[1] / ".state")
engine = RAGEngine(settings, Path(__file__).resolve().parents[1] / ".state")
voice_sessions: dict[str, VoiceSession] = {}

# SIP monitor: connected WebSocket clients for live event streaming
_sip_monitor_clients: set[WebSocket] = set()


async def broadcast_sip_event(event: dict[str, Any]) -> None:
    """Fire-and-forget broadcast to all connected SIP monitor pages."""
    for ws in list(_sip_monitor_clients):
        try:
            await ws.send_json(event)
        except Exception:  # noqa: BLE001
            _sip_monitor_clients.discard(ws)


app = FastAPI(title="Micro Leasing RAG Demo")
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


_shared_vad: SileroVAD | None = None
_sip_server: asyncio.Server | None = None


@app.on_event("startup")
async def _warmup() -> None:
    """Pre-load models and optionally start SIP server on startup."""
    global _shared_vad, _sip_server
    try:
        engine.retrieve("warmup", fast=True, voice_fast=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        silence_ms = int(os.getenv("VAD_SILENCE_MS", "500"))
        _shared_vad = SileroVAD(sample_rate=24000, silence_ms=silence_ms)
    except Exception:  # noqa: BLE001
        pass

    # Start AudioSocket TCP server if SIP is enabled
    if settings.sip.enabled:
        _sip_server = await asyncio.start_server(
            sip_call_handler,
            host=settings.sip.audiosocket_host,
            port=settings.sip.audiosocket_port,
        )
        print(
            f"[SIP] AudioSocket server listening on "
            f"{settings.sip.audiosocket_host}:{settings.sip.audiosocket_port}",
            flush=True,
        )


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
    return "our_rag"


def _launch_mode() -> str:
    return os.getenv("RAG_LAUNCH_MODE", "direct")


def _rag_statuses() -> dict[str, dict[str, Any]]:
    return {
        "our_rag": {
            "name": "our_rag",
            "available": True,
            "healthy": settings.app.kb_markdown_path.exists(),
            "reason": "ok" if settings.app.kb_markdown_path.exists() else "kb_missing",
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

    # Consent is granted implicitly via UI disclaimer banner.
    # The interactive consent flow (detect_consent / consent_request) is disabled.
    # To re-enable, uncomment the block in git history (commit before this change).

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
        "Ответь строго на основе следующих фрагментов (ЕДИНСТВЕННЫЙ источник фактов, "
        "адреса и числа бери ТОЛЬКО отсюда). Если ответ ЕСТЬ во фрагментах, ты ОБЯЗАНА его использовать. "
        "НЕ говори 'нет данных' если фрагменты содержат ответ. Если ответа действительно нет — скажи что нужно уточнить.\n\n"
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


async def _stream_voice_response(
    *,
    websocket: Any,
    session: Any,
    session_id: str,
    message: str,
    t_speech_stopped: float,
    t_stt_done: float,
    question_id: str,
    rtc_handler: Any | None = None,
) -> None:
    """Sentence-level LLM->TTS streaming for low-latency voice responses.

    As the LLM generates tokens, sentences are detected and sent to TTS
    immediately. Audio chunks are streamed to the browser via WebSocket.
    Supports barge-in: if session.interrupted is set, both LLM and TTS stop.
    """
    from .llm import iter_openai_compatible_stream_events
    from .sentence_detector import SentenceDetector

    backend = session.backend
    brain_model = session.brain_model

    # --- Enrich RAG query with conversation context ---
    # Always include recent context so RAG retrieves topic-relevant fragments,
    # especially critical for short follow-ups like "да", "давай", "Минск"
    rag_query = message
    chat_sess_for_rag = state.get(session_id)
    if chat_sess_for_rag and chat_sess_for_rag.transcript and len(message.split()) <= 5:
        # For short messages, prepend recent USER messages only (less noise than assistant responses)
        user_msgs = [t.get("text", "") for t in chat_sess_for_rag.transcript[-6:] if t.get("role") == "user"]
        if user_msgs:
            rag_query = " ".join(m[:60] for m in user_msgs[-2:]) + " " + message

    # --- RAG retrieval ---
    retrieval = await asyncio.to_thread(
        engine.retrieve, rag_query, True, True, session_id,
    )
    t_retrieval_done = time.time()
    timings: dict[str, Any] = dict(retrieval.get("timings") or {})
    final_chunks = retrieval.get("final") or []

    if not retrieval.get("ok") or not final_chunks:
        answer = settings.app.strict_refusal_text
        await websocket.send_json({
            "type": "response.output_text.delta",
            "session_id": session_id,
            "delta": answer,
        })
        try:
            audio_resp = await asyncio.to_thread(
                synthesize_audio, answer, session_id,
            )
            if audio_resp.get("audio_b64"):
                await websocket.send_json({
                    "type": "response.output_audio.delta",
                    "session_id": session_id,
                    "delta": audio_resp["audio_b64"],
                    "sample_rate_hz": audio_resp.get("sample_rate_hz"),
                })
        except Exception:  # noqa: BLE001
            pass
        t_now = time.time()
        state.log({
            "event": "voice_turn", "question_id": question_id,
            "stack_id": session.stack_id, "session_id": session_id,
            "backend": backend, "brain_model": brain_model,
            "stt_provider": session.stt_provider, "tts_provider": session.tts_provider,
            "transcript": message, "speech_stopped": t_speech_stopped,
            "stt_done": t_stt_done, "retrieval_done": t_retrieval_done,
            "llm_first_token": t_retrieval_done, "tts_first_chunk": t_now,
            "playback_started": t_now,
            "primary_kpi_ms": (t_now - t_speech_stopped) * 1000,
        })
        await websocket.send_json({
            "type": "response.done", "session_id": session_id,
            "backend": backend, "used_knowledge": [],
            "citations": [], "timings": timings,
        })
        return

    # --- Build prompt ---
    system_prompt = settings.app.system_prompt_path.read_text(encoding="utf-8")
    chat_session = state.get(session_id) or state.create(session_id)
    memory_block = build_memory_block(chat_session.transcript, settings.app.memory_turns)
    context_block = "\n\n".join(
        [f"[Fragment {i+1}]\n{c['text']}" for i, c in enumerate(final_chunks)]
    )
    expanded = any(trigger in message.lower() for trigger in settings.llm.expand_triggers)
    length_hint = (
        "Это голосовой разговор. Ответ: 1-2 коротких предложения. Самое важное. Не заканчивай каждый ответ вопросом. Задавай вопрос только если тебе реально нужна информация для продолжения."
        if not expanded
        else "Ответь подробнее, но кратко. Максимум три-четыре предложения."
    )
    weak_context = bool(retrieval.get("weak"))
    weak_hint = (
        "Контекст может быть неполным. Дай ближайшую релевантную информацию из фрагментов, "
        "скажи, что точных данных может не хватать, и задай уточняющий вопрос.\n\n"
    ) if weak_context else ""
    # Get tool schemas early
    tool_schemas = []
    for _tool in get_all_tools().values():
        try:
            tool_schemas.append(_tool.schema(session_phone=session.client_phone))
        except TypeError:
            tool_schemas.append(_tool.schema())

    effective_model = brain_model or settings.llm.fast_model or settings.llm.model
    effective_base_url = settings.llm.fast_base_url or settings.llm.base_url

    # --- Two-path message building ---
    # Qwen3.5 suppresses tool calling when ANY reference/KB text is present.
    # Path 1 (tool-eligible): clean system prompt + clean user message, no RAG.
    # --- Intent classification: tool vs RAG ---
    # Qwen3.5 cannot handle RAG context and tool calling in the same prompt.
    # Solution: fast LLM call to classify intent, then route to the right path.
    # This adds ~200ms latency but guarantees correct behavior.
    from .llm import call_openai_compatible

    sms_triggers = ["отправ", "смс", "sms", "пришли"]
    has_sms_intent = any(t in message.lower() for t in sms_triggers)
    tools_used_in_session = bool(session.tool_calls_this_turn)

    # Fast intent classification (only when tools are available)
    needs_tool = False
    if tool_schemas:
        # If tools were already used, use LLM to classify whether this
        # new message needs a tool or is a general KB question.
        # If tools were never used, also classify to catch first calc request.
        try:
            classify_resp = await asyncio.to_thread(
                call_openai_compatible,
                base_url=effective_base_url,
                model=effective_model,
                system_prompt=(
                    "Классифицируй сообщение клиента. Ответь ОДНИМ словом:\n"
                    "TOOL - если клиент просит расчёт, пересчёт, калькуляцию, отправку СМС, "
                    "или подтверждает отправку/расчёт (да, давай, отправь, пересчитай, хорошо после предложения расчёта)\n"
                    "RAG - если клиент задаёт вопрос о компании, условиях, документах, адресах, "
                    "или любой другой информационный вопрос\n"
                    "Ответь ТОЛЬКО одно слово: TOOL или RAG"
                ),
                user_prompt=f"Контекст диалога: {memory_block[-200:] if memory_block else 'начало разговора'}\nСообщение: {message}",
                temperature=0.0,
                max_tokens=5,
                timeout_sec=5,
            )
            intent = classify_resp.text.strip().upper()
            needs_tool = "TOOL" in intent
        except Exception:
            # Classification failed, fall back to keyword heuristic
            needs_tool = has_sms_intent or any(
                t in message.lower() for t in
                ["рассчит", "расчет", "расчёт", "посчит", "пересчит", "калькул"]
            )
        # Also treat SMS intent as tool
        if has_sms_intent:
            needs_tool = True

    # SMS context for send_sms
    sms_context = ""
    if has_sms_intent and session.tool_calls_this_turn:
        last_calc = next(
            (tc for tc in reversed(session.tool_calls_this_turn)
             if tc.get("tool") == "calculator"), None)
        if last_calc and last_calc.get("result", {}).get("ok"):
            calc_tool = get_tool("calculator")
            sms_context = f"Текст для СМС:\n{calc_tool.format_sms_body(last_calc['result'])}\n\n"

    if needs_tool:
        # TOOL path: clean message, no RAG (tool calling works reliably)
        # Include previous calc params so model can recalculate with changes
        prev_calc_context = ""
        if session.tool_calls_this_turn:
            last_calc = next(
                (tc for tc in reversed(session.tool_calls_this_turn)
                 if tc.get("tool") == "calculator"), None)
            if last_calc:
                import json as _json
                prev_params = last_calc.get("params", {})
                prev_calc_context = (
                    f"Предыдущий расчёт: {_json.dumps(prev_params, ensure_ascii=False)}. "
                    f"Клиент хочет изменить параметры. Вызови calculator с обновлёнными значениями.\n\n"
                )
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{sms_context}{prev_calc_context}{message}"},
        ]
    else:
        # RAG path: full context for KB questions
        user_prompt = (
            f"{memory_block}"
            f"Текущий вопрос клиента: {message}\n\n"
            f"{length_hint}\n\n"
            "Фрагменты из базы знаний (ЕДИНСТВЕННЫЙ источник фактов. "
            "Адреса, числа, ставки бери ТОЛЬКО отсюда):\n\n"
            f"{weak_hint}{context_block}"
        )
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    voice_max_tokens = 200

    # --- Sentence queue: LLM produces sentences, TTS consumes them ---
    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
    t_llm_first_token: float | None = None
    t_tts_first_chunk: float | None = None
    t_playback_started: float | None = None

    async def llm_producer() -> None:
        nonlocal t_llm_first_token
        max_tool_iterations = 3
        # Debug: log tool calling context
        if tool_schemas:
            print(f"[TOOL DEBUG] calc_intent=True, tools={len(tool_schemas)}, "
                  f"msg_count={len(llm_messages)}, user_msg_len={len(llm_messages[-1]['content'])}", flush=True)

        for iteration in range(max_tool_iterations + 1):
            detector = SentenceDetector()
            collected_events: list[dict] = []
            has_content = False

            try:
                # Lower temperature for tool-intent turns (more deterministic)
                # Normal temp for regular turns and for post-tool response
                temp = settings.llm.temperature
                tools_to_send = tool_schemas if iteration < max_tool_iterations else None
                if tool_schemas and iteration == 0:
                    print(f"[TOOL DEBUG] LLM call: temp={temp}, max_tokens={voice_max_tokens}, "
                          f"tools={len(tools_to_send) if tools_to_send else 0}, "
                          f"user_content_first100={llm_messages[-1]['content'][:100]}", flush=True)
                stream = iter_openai_compatible_stream_events(
                    base_url=effective_base_url,
                    model=effective_model,
                    messages=llm_messages,
                    temperature=temp,
                    max_tokens=voice_max_tokens if iteration == 0 else 220,
                    timeout_sec=settings.llm.timeout_sec,
                    tools=tools_to_send,
                )
                _sentinel = object()
                while True:
                    if session.interrupted:
                        break
                    event = await asyncio.to_thread(next, stream, _sentinel)
                    if event is _sentinel:
                        break
                    collected_events.append(event)
                    choice = (event.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    finish_reason = choice.get("finish_reason")

                    # Debug: log first few events and tool call deltas
                    if tool_schemas and len(collected_events) <= 3:
                        tc_delta = delta.get("tool_calls")
                        content_val = delta.get("content")
                        print(f"[TOOL DEBUG] event#{len(collected_events)}: content={repr(content_val)}, "
                              f"tool_calls={'YES' if tc_delta else 'no'}, finish={finish_reason}", flush=True)

                    # Regular content token
                    token = delta.get("content") or ""
                    if token:
                        has_content = True
                        if t_llm_first_token is None:
                            t_llm_first_token = time.time()
                        for sent in detector.feed(token):
                            cleaned = clean_answer(sent)
                            if cleaned:
                                await sentence_queue.put(cleaned)

            except Exception as exc:  # noqa: BLE001
                state.log({"event": "llm_error", "error": str(exc), "session_id": session_id})
                break

            # Flush remaining content
            remaining = detector.flush()
            if remaining and not session.interrupted:
                cleaned = clean_answer(remaining)
                if cleaned:
                    await sentence_queue.put(cleaned)

            # If we got regular content, we are done (no tool call)
            if has_content or session.interrupted:
                break

            # Check for tool calls in the collected events
            tool_calls = parse_tool_calls_from_events(collected_events)
            if not tool_calls:
                break

            # Process each tool call
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                # Notify frontend about tool call
                try:
                    await websocket.send_json({
                        "type": "tool_call.start",
                        "session_id": session_id,
                        "tool": func_name,
                        "params": func_args,
                    })
                except (RuntimeError, WebSocketDisconnect):
                    pass

                # Send filler phrase to TTS immediately
                filler = get_filler(func_name)
                await sentence_queue.put(filler)

                # Execute tool in a thread (synchronous httpx)
                tool_ok = False
                try:
                    tool = get_tool(func_name)
                    filled_params, defaulted = tool.fill_defaults(func_args)
                    result = await asyncio.to_thread(
                        tool.execute, filled_params, {"session_id": session_id, "client_phone": session.client_phone}
                    )
                    result["defaulted"] = defaulted
                    tool_ok = result.get("ok", False)
                    session.tool_calls_this_turn.append({
                        "tool": func_name, "params": filled_params, "result": result,
                    })
                    summary = tool.format_voice_summary(result)
                except KeyError:
                    summary = f"Инструмент '{func_name}' не найден."
                    state.log({"event": "tool_error", "session_id": session_id, "tool": func_name, "error": "tool_not_found"})
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    tb = traceback.format_exc()
                    summary = f"Ошибка выполнения инструмента: {exc}"
                    state.log({"event": "tool_error", "session_id": session_id, "tool": func_name, "error": str(exc), "traceback": tb})
                    print(f"[TOOL ERROR] {func_name}: {exc}\n{tb}", flush=True)

                # Append tool call + result to messages for next LLM iteration
                llm_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc.get("id", f"call_{func_name}"),
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(func_args, ensure_ascii=False),
                        },
                    }],
                })
                llm_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{func_name}"),
                    "content": summary,
                })

                # Notify frontend tool call finished
                try:
                    await websocket.send_json({
                        "type": "tool_call.done",
                        "session_id": session_id,
                        "tool": func_name,
                        "ok": tool_ok,
                    })
                except (RuntimeError, WebSocketDisconnect):
                    pass

                state.log({
                    "event": "tool_call",
                    "session_id": session_id,
                    "tool": func_name,
                    "params": filled_params,
                    "ok": result.get("ok", False) if isinstance(result, dict) else False,
                })

        await sentence_queue.put(None)

    async def tts_consumer() -> None:
        nonlocal t_tts_first_chunk, t_playback_started
        while True:
            if session.interrupted:
                break
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            try:
                await websocket.send_json({
                    "type": "response.output_text.delta",
                    "session_id": session_id,
                    "delta": sentence + " ",
                })
            except (RuntimeError, WebSocketDisconnect):
                session.interrupted = True
                break
            try:
                audio_resp = await asyncio.to_thread(
                    synthesize_audio, sentence, session_id,
                )
                audio_b64 = audio_resp.get("audio_b64") or ""
                if audio_b64:
                    if t_tts_first_chunk is None:
                        t_tts_first_chunk = time.time()
                    if rtc_handler is not None:
                        import base64 as _b64
                        pcm16 = _b64.b64decode(audio_b64)
                        rtc_handler.tts_track.push_audio(pcm16)
                    else:
                        await websocket.send_json({
                            "type": "response.output_audio.delta",
                            "session_id": session_id,
                            "delta": audio_b64,
                            "sample_rate_hz": audio_resp.get("sample_rate_hz"),
                        })
                    t_playback_started = time.time()
            except (RuntimeError, WebSocketDisconnect):
                session.interrupted = True
                break
            except Exception as exc:  # noqa: BLE001
                try:
                    await websocket.send_json({
                        "type": "warning", "session_id": session_id,
                        "message": f"tts_failed: {exc}",
                    })
                except (RuntimeError, WebSocketDisconnect):
                    session.interrupted = True
                    break

    all_sentences: list[str] = []
    _orig_put = sentence_queue.put

    _chunk_texts = [c.get("text", "") for c in final_chunks]

    async def _tracking_put(item: str | None) -> None:
        if item is not None:
            # Strip client name from ALL sentences (unless it's a "name turn")
            if session.client_name:
                from .text_utils import strip_name_from_response
                item = strip_name_from_response(item, session.client_name, session.turn_count)
            # Validate addresses against retrieved context (catch hallucinations)
            from .text_utils import validate_addresses
            item = validate_addresses(item, _chunk_texts)
            all_sentences.append(item)
        await _orig_put(item)

    sentence_queue.put = _tracking_put  # type: ignore[assignment]

    session.assistant_speaking = True
    session.interrupted = False
    producer_task = asyncio.create_task(llm_producer())
    consumer_task = asyncio.create_task(tts_consumer())
    await asyncio.gather(producer_task, consumer_task)
    session.assistant_speaking = False
    if rtc_handler is not None:
        rtc_handler.tts_track.flush()

    full_answer = " ".join(all_sentences)
    if session.interrupted and full_answer:
        full_answer += " [прервано клиентом]"
    if full_answer:
        session.turn_count += 1
        chat_session = state.get(session_id) or state.create(session_id)
        _append_turn(chat_session, message, full_answer, settings.app.memory_turns)
        state.update(chat_session)

    t_now = time.time()
    _llm_ft = t_llm_first_token or t_retrieval_done
    _tts_fc = t_tts_first_chunk or t_now
    _pb_st = t_playback_started or t_now
    voice_timings = {
        "stt_ms": round((t_stt_done - t_speech_stopped) * 1000),
        "rag_ms": round((t_retrieval_done - t_stt_done) * 1000),
        "llm_first_ms": round((_llm_ft - t_retrieval_done) * 1000),
        "tts_first_ms": round((_tts_fc - _llm_ft) * 1000),
        "total_ms": round((_pb_st - t_speech_stopped) * 1000),
    }
    used_knowledge = [
        {"text": c["text"], "chunk_id": c.get("chunk_id")} for c in final_chunks
    ]
    state.log({
        "event": "voice_turn", "question_id": question_id,
        "stack_id": session.stack_id, "session_id": session_id,
        "backend": backend, "brain_model": brain_model,
        "stt_provider": session.stt_provider, "tts_provider": session.tts_provider,
        "transcript": message, "speech_stopped": t_speech_stopped,
        "stt_done": t_stt_done, "retrieval_done": t_retrieval_done,
        "llm_first_token": _llm_ft,
        "tts_first_chunk": _tts_fc,
        "playback_started": _pb_st,
        "primary_kpi_ms": voice_timings["total_ms"],
        **voice_timings,
    })
    await websocket.send_json({
        "type": "response.done", "session_id": session_id,
        "backend": backend, "used_knowledge": used_knowledge,
        "citations": [], "timings": timings,
        "voice_timings": voice_timings,
    })


# ---------------------------------------------------------------------------
# SIP telephony: WebSocket shim + call handler
# ---------------------------------------------------------------------------


class _SIPWebSocketShim:
    """Adapts SIP AudioSocket to the WebSocket interface expected by _stream_voice_response.

    Translates send_json calls: text events go to monitor, audio events go
    to the AudioSocket adapter (resampled from 24kHz to 8kHz).
    """

    def __init__(self, adapter: Any, session_id: str) -> None:
        self._adapter = adapter
        self._session_id = session_id

    async def send_json(self, data: dict[str, Any]) -> None:
        event_type = data.get("type", "")

        if event_type == "response.output_audio.delta":
            import base64 as _b64
            audio_b64 = data.get("delta", "")
            if audio_b64:
                pcm_raw = _b64.b64decode(audio_b64)
                await self._adapter.write_audio(pcm_raw)
            return

        if event_type == "response.output_text.delta":
            await broadcast_sip_event({
                "type": "sip.llm.sentence",
                "call_id": self._session_id,
                "text": data.get("delta", ""),
            })
            return

        if event_type == "tool_call.start":
            await broadcast_sip_event({
                "type": "sip.tool.start",
                "call_id": self._session_id,
                "tool": data.get("tool", ""),
                "params": data.get("params", {}),
            })
            return

        if event_type == "tool_call.done":
            await broadcast_sip_event({
                "type": "sip.tool.result",
                "call_id": self._session_id,
                "tool": data.get("tool", ""),
                "ok": data.get("ok", False),
            })
            return

        if event_type == "response.done":
            await broadcast_sip_event({
                "type": "sip.response.done",
                "call_id": self._session_id,
            })
            return

        # All other events: broadcast to monitor as-is
        await broadcast_sip_event({
            **data,
            "call_id": self._session_id,
        })


async def _sip_send_tts(
    adapter: Any,
    session: VoiceSession,
    session_id: str,
    text: str,
) -> None:
    """Synthesize text and send audio back through AudioSocket."""
    try:
        print(f"[SIP:{session_id[:8]}] TTS synthesizing: {text[:40]}...", flush=True)
        audio_resp = await asyncio.to_thread(synthesize_audio, text, session_id)
        audio_b64 = audio_resp.get("audio_b64") or ""
        sample_rate = audio_resp.get("sample_rate_hz", 24000)
        print(f"[SIP:{session_id[:8]}] TTS got {len(audio_b64)} chars b64, sample_rate={sample_rate}", flush=True)
        if audio_b64:
            import base64 as _b64
            pcm_raw = _b64.b64decode(audio_b64)
            print(f"[SIP:{session_id[:8]}] TTS decoded {len(pcm_raw)} bytes PCM, writing to AudioSocket...", flush=True)
            await adapter.write_audio(pcm_raw)
            print(f"[SIP:{session_id[:8]}] TTS audio sent OK", flush=True)
        else:
            print(f"[SIP:{session_id[:8]}] TTS returned empty audio", flush=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[SIP:{session_id[:8]}] TTS error: {exc}\n{traceback.format_exc()}", flush=True)

    session.assistant_speaking = False

    await broadcast_sip_event({
        "type": "sip.tts.start",
        "call_id": session_id,
        "text": text,
    })


async def _sip_process_utterance(
    adapter: Any,
    session: VoiceSession,
    session_id: str,
    speech_audio: bytes,
) -> None:
    """Transcribe speech audio and stream LLM response back via SIP."""
    import base64 as _b64mod

    audio_b64 = _b64mod.b64encode(speech_audio).decode()
    question_id = str(uuid.uuid4())
    t_speech_stopped = time.time()

    try:
        transcript = transcribe_audio(audio_b64, session_id=session_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[SIP:{session_id[:8]}] STT error: {exc}", flush=True)
        return

    t_stt_done = time.time()
    text = (transcript.get("text") or "").strip()
    if not text:
        return

    print(f"[SIP:{session_id[:8]}] STT: {text}", flush=True)
    await broadcast_sip_event({
        "type": "sip.stt.result",
        "call_id": session_id,
        "text": text,
    })

    session.on_transcript_final(text)

    sip_ws = _SIPWebSocketShim(adapter, session_id)
    await _stream_voice_response(
        websocket=sip_ws,
        session=session,
        session_id=session_id,
        message=text,
        t_speech_stopped=t_speech_stopped,
        t_stt_done=t_stt_done,
        question_id=question_id,
        rtc_handler=None,
    )


async def sip_call_handler(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Handle one SIP call via AudioSocket.

    Lifecycle: read UUID -> query AMI for caller -> create session ->
    run intro -> audio loop with VAD -> pipeline -> TTS back.
    """
    from .sip_audio import SIPAudioAdapter, query_caller_id_ami

    adapter = SIPAudioAdapter(reader, writer)
    session_id = ""
    session: VoiceSession | None = None

    try:
        # 1. Read UUID frame
        first = await adapter.read_next()
        if first is None or first.get("type") != "uuid":
            print("[SIP] No UUID frame received, closing", flush=True)
            await adapter.close()
            return
        session_id = first["uuid"]
        print(f"[SIP:{session_id[:8]}] Connected", flush=True)

        # 2. Query AMI for caller phone
        caller_phone = await query_caller_id_ami(
            channel_id=session_id,
            ami_host=settings.sip.ami_host,
            ami_port=settings.sip.ami_port,
            ami_username=settings.sip.ami_username,
            ami_secret=settings.sip.ami_secret,
        )
        print(f"[SIP:{session_id[:8]}] Caller: {caller_phone or 'unknown'}", flush=True)

        await broadcast_sip_event({
            "type": "sip.call.start",
            "call_id": session_id,
            "phone": caller_phone or "unknown",
        })

        # 3. Create voice session
        session = VoiceSession(
            session_id=session_id,
            backend="our_rag",
            transport="sip",
            client_phone=caller_phone,
            call_id=session_id,
        )
        voice_sessions[session_id] = session
        adapter.caller_phone = caller_phone

        # 4. Create VAD for this call (16kHz, since we resample inbound to 16kHz)
        silence_ms = int(os.getenv("VAD_SILENCE_MS", "500"))
        vad = SileroVAD(sample_rate=16000, silence_ms=silence_ms)

        # 5. Send intro TTS as background task (do not block the read loop)
        intro_text = (
            "Здравствуйте! Меня зовут Ксения, я голосовая помощница компании Микро Лизинг. "
            "Как я могу к вам обращаться?"
        )
        session.assistant_speaking = True
        asyncio.create_task(_sip_send_tts(adapter, session, session_id, intro_text))

        # 6. Audio loop: read frames, feed VAD, dispatch on speech end
        # This runs concurrently with TTS playback (full-duplex AudioSocket).
        _frame_count = 0
        print(f"[SIP:{session_id[:8]}] Entering audio read loop...", flush=True)
        while True:
            frame = await adapter.read_next()
            if _frame_count == 0 and frame is not None:
                print(f"[SIP:{session_id[:8]}] First frame from read_next: type={frame.get('type')}", flush=True)
            if frame is None or frame["type"] == "hangup":
                print(f"[SIP:{session_id[:8]}] Hangup (frames received: {_frame_count})", flush=True)
                await broadcast_sip_event({
                    "type": "sip.call.end",
                    "call_id": session_id,
                })
                break

            if frame["type"] == "dtmf":
                digit = frame["digit"]
                print(f"[SIP:{session_id[:8]}] DTMF: {digit}", flush=True)
                await broadcast_sip_event({
                    "type": "sip.dtmf",
                    "call_id": session_id,
                    "digit": digit,
                })
                continue

            if frame["type"] == "error":
                print(f"[SIP:{session_id[:8]}] Error: {frame.get('message')}", flush=True)
                break

            if frame["type"] != "audio":
                continue

            _frame_count += 1
            if _frame_count == 1:
                print(f"[SIP:{session_id[:8]}] First audio frame received from caller", flush=True)
            if _frame_count % 500 == 0:
                print(f"[SIP:{session_id[:8]}] Audio frames: {_frame_count}, VAD speaking: {vad.is_speaking}, asst: {session.assistant_speaking}", flush=True)

            pcm_16k = frame["pcm16"]

            # VAD processing
            was_speaking = vad.is_speaking
            speech_audio = vad.feed(pcm_16k)

            if not was_speaking and vad.is_speaking:
                print(f"[SIP:{session_id[:8]}] VAD: speech_start", flush=True)
                await broadcast_sip_event({
                    "type": "sip.vad.speech",
                    "call_id": session_id,
                    "event": "start",
                })

            # Barge-in: user started speaking while bot is responding
            if not was_speaking and vad.is_speaking and session.assistant_speaking:
                session.interrupted = True
                session.assistant_speaking = False
                print(f"[SIP:{session_id[:8]}] BARGE-IN", flush=True)
                await broadcast_sip_event({
                    "type": "sip.barge_in",
                    "call_id": session_id,
                })

            # Speech ended: VAD returned accumulated audio
            if speech_audio is not None:
                print(f"[SIP:{session_id[:8]}] VAD: speech_end ({len(speech_audio)} bytes)", flush=True)
                await broadcast_sip_event({
                    "type": "sip.vad.speech",
                    "call_id": session_id,
                    "event": "end",
                })
                # Fire-and-forget: process utterance in background so read loop
                # continues (enables barge-in detection during response)
                session.assistant_speaking = True
                session.interrupted = False
                asyncio.create_task(_sip_process_utterance(
                    adapter, session, session_id, speech_audio,
                ))

    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[SIP:{session_id[:8]}] Error: {exc}\n{traceback.format_exc()}", flush=True)
    finally:
        if session is not None:
            voice_sessions.pop(session_id, None)
        await adapter.close()
        print(f"[SIP:{session_id[:8]}] Cleaned up", flush=True)


# ---------------------------------------------------------------------------
# WebSocket voice endpoint (browser PTT + WebRTC)
# ---------------------------------------------------------------------------


@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = VoiceSession(session_id=session_id, backend="our_rag")
    voice_sessions[session_id] = session
    audio_chunks: list[str] = []

    # -- VAD state --
    vad_enabled = False
    vad: SileroVAD | None = None

    # ------------------------------------------------------------------
    # Helper: STT then sentence-level streaming response.
    # Shared by both push-to-talk commit and VAD speech-end paths.
    # ------------------------------------------------------------------
    async def _process_voice_utterance(audio_b64: str) -> None:
        """Transcribe audio, then stream response with sentence-level TTS."""
        question_id = str(uuid.uuid4())
        t_speech_stopped = time.time()
        try:
            transcript = transcribe_audio(audio_b64, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            await websocket.send_json({
                "type": "error",
                "session_id": session_id,
                "error": f"stt_failed: {exc}",
            })
            return
        t_stt_done = time.time()
        text = (transcript.get("text") or "").strip()
        await websocket.send_json({
            "type": "conversation.item.input_audio_transcription.completed",
            "session_id": session_id,
            "provider": transcript.get("provider"),
            "transcription": text,
        })
        if not text:
            return
        for action in session.on_transcript_final(text):
            await websocket.send_json(action)

        response_coro = _stream_voice_response(
            websocket=websocket,
            session=session,
            session_id=session_id,
            message=text,
            t_speech_stopped=t_speech_stopped,
            t_stt_done=t_stt_done,
            question_id=question_id,
            rtc_handler=rtc_handler,
        )

        if not (vad_enabled and vad is not None):
            # PTT mode: original blocking behavior
            try:
                await response_coro
            except (RuntimeError, WebSocketDisconnect):
                pass
            return

        # ---- Fire-and-forget response ----
        # The response runs as a background task. The main loop continues
        # reading WebSocket events (including response.cancel for barge-in).
        # No separate listener needed; the main loop handles everything.
        session.assistant_speaking = True
        session.interrupted = False
        asyncio.create_task(response_coro)

    # ------------------------------------------------------------------
    # Audio adapter callback: receives raw PCM16 for every chunk.
    # Handles both push-to-talk accumulation and VAD processing.
    # ------------------------------------------------------------------
    import base64 as _b64mod

    async def _on_audio_chunk(raw: bytes) -> None:
        nonlocal vad_enabled, vad, audio_chunks

        # Always accumulate base64 for push-to-talk commit path
        audio_chunks.append(_b64mod.b64encode(raw).decode())

        if not vad_enabled or vad is None:
            return

        # -- VAD processing --
        was_speaking = vad.is_speaking
        speech_audio = vad.feed(raw)
        if not was_speaking and vad.is_speaking:
            print(f"[VAD] speech_start", flush=True)
        if was_speaking and not vad.is_speaking and speech_audio is not None:
            print(f"[VAD] speech_end ({len(speech_audio)} bytes)", flush=True)

        # Barge-in: user started speaking while assistant is streaming TTS
        if not was_speaking and vad.is_speaking and session.assistant_speaking:
            session.interrupted = True
            session.assistant_speaking = False
            await websocket.send_json(
                {
                    "type": "interrupt",
                    "session_id": session_id,
                }
            )

        # Speech ended: VAD returned accumulated audio
        if speech_audio is not None:
            vad_audio_b64 = _b64mod.b64encode(speech_audio).decode()
            audio_chunks.clear()
            await _process_voice_utterance(vad_audio_b64)

    audio_adapter = WebSocketAudioAdapter(on_chunk=_on_audio_chunk)
    rtc_handler: RTCAudioHandler | None = None

    # ------------------------------------------------------------------
    # Helper: send text + TTS audio to client (for hardcoded messages)
    # ------------------------------------------------------------------
    async def _send_tts_message(text: str) -> None:
        await websocket.send_json({
            "type": "response.output_text.delta",
            "session_id": session_id,
            "delta": text,
        })
        try:
            audio_resp = await asyncio.to_thread(synthesize_audio, text, session_id)
            audio_b64 = audio_resp.get("audio_b64") or ""
            if audio_b64:
                if rtc_handler is not None:
                    # Route TTS through RTC track so Chrome AEC gets precise reference
                    pcm16 = _b64mod.b64decode(audio_b64)
                    rtc_handler.tts_track.push_audio(pcm16)
                    rtc_handler.tts_track.flush()
                else:
                    # No RTC: send via WebSocket (PTT mode)
                    await websocket.send_json({
                        "type": "response.output_audio.delta",
                        "session_id": session_id,
                        "delta": audio_b64,
                        "sample_rate_hz": audio_resp.get("sample_rate_hz"),
                    })
        except Exception:  # noqa: BLE001
            pass
        await websocket.send_json({
            "type": "response.done",
            "session_id": session_id,
            "backend": session.backend,
            "used_knowledge": [],
            "citations": [],
            "timings": {},
        })

    # ------------------------------------------------------------------
    # Shared transcript queue: both push-to-talk commit and VAD speech-end
    # push transcribed text here. _wait_for_speech reads from it.
    # ------------------------------------------------------------------
    _speech_queue: asyncio.Queue[str] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Helper: transcribe audio and push to speech queue
    # ------------------------------------------------------------------
    async def _transcribe_and_enqueue(audio_b64: str) -> None:
        try:
            transcript = transcribe_audio(audio_b64, session_id=session_id)
        except Exception:  # noqa: BLE001
            return
        text = (transcript.get("text") or "").strip()
        if text:
            await websocket.send_json({
                "type": "conversation.item.input_audio_transcription.completed",
                "session_id": session_id,
                "provider": transcript.get("provider"),
                "transcription": text,
            })
            await _speech_queue.put(text)

    # ------------------------------------------------------------------
    # Helper: wait for next voice utterance (works with both modes)
    # ------------------------------------------------------------------
    async def _wait_for_speech() -> str:
        nonlocal vad_enabled, vad, session_id, rtc_handler
        while True:
            # Check if RTC VAD already pushed something
            try:
                return _speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

            # When RTC is active, WebSocket events are sparse (no audio).
            # Use a timeout so we periodically check the speech queue.
            try:
                event = await asyncio.wait_for(websocket.receive_json(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            event_type = event.get("type")
            if event_type != "input_audio_buffer.append":
                print(f"[wait_speech] event={event_type} vad={vad_enabled} chunks={len(audio_chunks)}", flush=True)
            if event_type == "session.init":
                if event.get("session_id"):
                    old_id = session_id
                    session_id = str(event["session_id"])
                    session.session_id = session_id
                    voice_sessions.pop(old_id, None)
                    voice_sessions[session_id] = session
                continue
            elif event_type == "session.update":
                session.backend = _selected_backend(event.get("backend"))
                if "vad_mode" in event:
                    vad_enabled = bool(event["vad_mode"])
                    if vad_enabled and vad is None and _shared_vad is not None:
                        vad = _shared_vad
                        vad.reset()
                    if not vad_enabled and vad is not None:
                        vad.reset()
                await websocket.send_json({
                    "type": "session.updated",
                    "session_id": session_id,
                    "backend": session.backend,
                    "vad_mode": vad_enabled,
                })
                continue
            elif event_type == "input_audio_buffer.append":
                audio = event.get("audio") or ""
                if audio:
                    raw = _b64mod.b64decode(audio)
                    # Always accumulate for push-to-talk
                    audio_chunks.append(_b64mod.b64encode(raw).decode())
                    # VAD processing
                    if vad_enabled and vad is not None:
                        was_speaking = vad.is_speaking
                        speech_audio = vad.feed(raw)
                        if speech_audio is not None:
                            vad_b64 = _b64mod.b64encode(speech_audio).decode()
                            audio_chunks.clear()
                            await _transcribe_and_enqueue(vad_b64)
            elif event_type == "input_audio_buffer.commit":
                raw_audio = b"".join(_b64mod.b64decode(c) for c in audio_chunks)
                audio_chunks.clear()
                if not raw_audio:
                    continue
                audio_b64 = _b64mod.b64encode(raw_audio).decode()
                await _transcribe_and_enqueue(audio_b64)
            elif event_type == "rtc.offer":
                # Handle RTC offer during intro (RTC before intro pattern)
                if rtc_handler is not None:
                    await rtc_handler.close()

                _intro_cb_count = 0

                async def _rtc_on_audio_intro(pcm16: bytes) -> None:
                    """No-op during intro. WebSocket audio handles VAD + STT."""
                    pass

                rtc_handler = RTCAudioHandler(
                    on_audio=_rtc_on_audio_intro,
                    sample_rate=24000,
                )
                sdp_offer = event.get("sdp", "")
                try:
                    sdp_answer = await rtc_handler.handle_offer(sdp_offer)
                    await websocket.send_json({"type": "rtc.answer", "sdp": sdp_answer})
                    print("[RTC] peer connection established (intro phase)", flush=True)
                except Exception as exc:
                    print(f"[RTC] offer handling failed: {exc}", flush=True)
                    rtc_handler = None
            elif event_type == "rtc.ice":
                pass  # ICE candidates bundled in SDP
            elif event_type == "response.cancel":
                continue  # ignore during intro

            # Check queue after processing event
            try:
                return _speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                continue

    await websocket.send_json({
        "type": "session.ready",
        "session_id": session_id,
        "backend": session.backend,
    })

    # Consent is granted implicitly via UI disclaimer banner.
    # Introduce and ask name in one step (no consent question).
    await _send_tts_message(
        "Здравствуйте! Меня зовут Ксения, я голосовая помощница компании Микро Лизинг. "
        "Как я могу к вам обращаться?"
    )
    client_name_raw = await _wait_for_speech()

    # Fast check: if input looks like a question, skip name extraction
    _raw_lower = client_name_raw.strip().lower()
    # Topic words that indicate a business question, not a name introduction.
    # "привет" and "здравствуйте" are NOT here because "Привет, я Евгений" is a name.
    _QUESTION_TOPICS = {"какой", "какая", "какие", "где", "когда", "сколько",
                        "можно", "расскажи", "подскажи", "объясни",
                        "адрес", "документы", "условия", "ставка", "аванс",
                        "офис", "процент", "ставку", "погода", "директор"}
    _words_in_input = set(_raw_lower.replace(",", " ").replace(".", " ").replace("?", " ").split())
    # It's a question if it contains topic words AND doesn't look like a name intro
    _has_name_intro = any(w in _raw_lower for w in ("меня зовут", "я ", "мне имя", "зовите"))
    _is_question = "?" in client_name_raw or (bool(_words_in_input & _QUESTION_TOPICS) and not _has_name_intro)

    from .llm import call_openai_compatible
    if _is_question:
        client_name = "друг"
    else:
        try:
            name_resp = await asyncio.to_thread(
                call_openai_compatible,
                base_url=settings.llm.fast_base_url or settings.llm.base_url,
                model=settings.llm.fast_model or settings.llm.model,
                system_prompt="Извлеки имя человека из текста. Верни ТОЛЬКО имя, одно слово, без пояснений. Если имя не найдено, верни слово 'друг'.",
                user_prompt=client_name_raw,
                temperature=0.0,
                max_tokens=10,
                timeout_sec=5,
            )
            client_name = name_resp.text.strip().strip('"').strip("'").strip(".").title()
        except Exception:  # noqa: BLE001
            client_name = client_name_raw.strip().split()[0].title()
        if not client_name or len(client_name) > 20 or len(client_name) < 2:
            client_name = "друг"
        else:
            # Verify it's actually a name using pymorphy3, normalize to nominative
            try:
                import pymorphy3
                _morph_name = pymorphy3.MorphAnalyzer()
                parsed = _morph_name.parse(client_name)[0]
                if "Name" in parsed.tag:
                    client_name = parsed.normal_form.title()
                else:
                    client_name = "друг"
            except Exception:  # noqa: BLE001
                pass

    # If no name was found, the first message is likely a question, not a name.
    # Skip the "Очень приятно" greeting and process it as a real question.
    # If name WAS found but message also contains a question, greet AND answer.
    first_question = None
    _also_has_question = _has_name_intro and bool(_words_in_input & _QUESTION_TOPICS)
    if client_name == "друг":
        first_question = client_name_raw  # save the original message, answer it directly
        # No greeting -- go straight to answering the question
    else:
        if _also_has_question:
            await _send_tts_message(f"Очень приятно, {client_name}!")
            first_question = client_name_raw  # name + question combo, answer the question too
        else:
            await _send_tts_message(f"Очень приятно, {client_name}! Чем могу помочь?")

    # Save intro to session transcript so model knows the name
    session.client_name = client_name
    chat_session = state.get(session_id) or state.create(session_id)
    if client_name != "друг":
        _append_turn(chat_session, f"Меня зовут {client_name}", f"Очень приятно, {client_name}! Чем могу помочь?", settings.app.memory_turns)
    state.update(chat_session)

    # If the first message was a question (not a name), answer it immediately
    if first_question and first_question.strip():
        try:
            question_id = str(uuid.uuid4())
            t_now = time.time()
            await websocket.send_json({
                "type": "conversation.item.input_audio_transcription.completed",
                "session_id": session_id,
                "provider": "intro",
                "transcription": first_question,
            })
            await _stream_voice_response(
                websocket=websocket, session=session, session_id=session_id,
                message=first_question,
                t_speech_stopped=t_now, t_stt_done=t_now, question_id=question_id,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            state.log({"event": "first_question_error", "error": str(exc), "traceback": traceback.format_exc(), "session_id": session_id, "question": first_question})
            print(f"[first_question_error] {exc}\n{traceback.format_exc()}", flush=True)
            await _send_tts_message("Одну секунду, я готова вас слушать.")

    # From here, all messages are normal conversation.
    # ------------------------------------------------------------------

    # RTC was established during the intro. Swap the callback from the
    # intro version (feeds _speech_queue) to the main loop version
    # (calls _process_voice_utterance directly with barge-in detection).
    if rtc_handler is not None:
        # Separate VAD instance for RTC barge-in detection.
        # Main loop's VAD handles WebSocket audio for STT.
        # RTC VAD only detects speech for interruption, never dispatches STT.
        _rtc_barge_vad: SileroVAD | None = None
        if _shared_vad is not None:
            silence_ms = int(os.getenv("VAD_SILENCE_MS", "500"))
            _rtc_barge_vad = SileroVAD(sample_rate=24000, silence_ms=silence_ms)

        _barge_cb_count = 0

        async def _rtc_on_audio_main(pcm16: bytes) -> None:
            """RTC audio: barge-in detection ONLY. STT uses WebSocket audio."""
            nonlocal _barge_cb_count
            _barge_cb_count += 1
            if _rtc_barge_vad is None or not session.assistant_speaking:
                if _barge_cb_count % 500 == 0:
                    print(f"[RTC-BARGE] cb={_barge_cb_count} skipped (asst={session.assistant_speaking} vad={'ok' if _rtc_barge_vad else 'None'})", flush=True)
                return
            if _barge_cb_count % 50 == 0:
                print(f"[RTC-BARGE] cb={_barge_cb_count} ACTIVE vad={_rtc_barge_vad.is_speaking}", flush=True)
            was_speaking = _rtc_barge_vad.is_speaking
            _rtc_barge_vad.feed(pcm16)
            if not was_speaking and _rtc_barge_vad.is_speaking and not session.interrupted:
                session.interrupted = True
                if rtc_handler is not None:
                    rtc_handler.tts_track.clear()
                print("[BARGE-IN-RTC] speech during response", flush=True)
                try:
                    await websocket.send_json({
                        "type": "interrupt",
                        "session_id": session_id,
                    })
                except (RuntimeError, WebSocketDisconnect):
                    pass

        rtc_handler._on_audio = _rtc_on_audio_main
        print("[RTC] switched to main conversation callback", flush=True)

    try:
        while True:
            event = await websocket.receive_json()
            event_type = event.get("type")
            if event_type == "session.init":
                # Client reconnecting with stored session_id
                if event.get("session_id"):
                    old_id = session_id
                    session_id = str(event["session_id"])
                    session.session_id = session_id
                    voice_sessions.pop(old_id, None)
                    voice_sessions[session_id] = session
                continue
            elif event_type == "session.update":
                session.backend = _selected_backend(event.get("backend"))
                # VAD mode toggle
                if "vad_mode" in event:
                    vad_enabled = bool(event["vad_mode"])
                    if vad_enabled and vad is None and _shared_vad is not None:
                        vad = _shared_vad
                        vad.reset()
                    if not vad_enabled and vad is not None:
                        vad.reset()
                await websocket.send_json(
                    {
                        "type": "session.updated",
                        "session_id": session_id,
                        "backend": session.backend,
                        "vad_mode": vad_enabled,
                    }
                )
            elif event_type == "input_audio_buffer.append":
                # Hybrid: WebSocket audio always processed for STT
                # (RTC handles barge-in detection in parallel)
                audio = event.get("audio") or ""
                if audio:
                    raw = _b64mod.b64decode(audio)
                    await audio_adapter.handle_audio_message(raw)
            elif event_type == "input_audio_buffer.commit":
                # Each chunk was independently base64-encoded; decode all,
                # concatenate raw bytes, then re-encode as single base64 string.
                raw_audio = b"".join(_b64mod.b64decode(c) for c in audio_chunks)
                audio_chunks.clear()
                if raw_audio:
                    audio_b64 = _b64mod.b64encode(raw_audio).decode()
                    await _process_voice_utterance(audio_b64)
            elif event_type == "response.cancel":
                session.assistant_speaking = False
                session.interrupted = True
                if rtc_handler is not None:
                    rtc_handler.tts_track.clear()
                print("[BARGE-IN] response.cancel received in main loop", flush=True)
                try:
                    await websocket.send_json({"type": "response.cancelled", "session_id": session_id})
                except (RuntimeError, WebSocketDisconnect):
                    pass

            elif event_type == "rtc.offer":
                # Client sent WebRTC offer for streaming mode
                if rtc_handler is not None:
                    await rtc_handler.close()

                async def _rtc_on_audio(pcm16: bytes) -> None:
                    """RTC inbound audio -> VAD pipeline.

                    CRITICAL: this callback must NEVER block. It runs for
                    every 20ms frame. _process_voice_utterance is fired as
                    a task to avoid blocking barge-in detection.
                    """
                    nonlocal vad, vad_enabled
                    if not vad_enabled or vad is None:
                        return
                    was_speaking = vad.is_speaking
                    speech_audio = vad.feed(pcm16)
                    if not was_speaking and vad.is_speaking:
                        print("[VAD-RTC] speech_start", flush=True)
                    if was_speaking and not vad.is_speaking and speech_audio is not None:
                        print(f"[VAD-RTC] speech_end ({len(speech_audio)} bytes)", flush=True)
                    # Barge-in: user speaking while assistant responds
                    if vad.is_speaking and session.assistant_speaking and not session.interrupted:
                        session.interrupted = True
                        if rtc_handler is not None:
                            rtc_handler.tts_track.clear()
                        print("[BARGE-IN-RTC] speech during response", flush=True)
                        try:
                            await websocket.send_json({
                                "type": "interrupt",
                                "session_id": session_id,
                            })
                        except (RuntimeError, WebSocketDisconnect):
                            pass
                    # Speech ended: fire-and-forget response (don't block audio processing)
                    _MIN_SPEECH_BYTES = 12000  # 0.6s at 24kHz; VAD adds 0.5s silence tail, filters Opus noise bursts
                    if speech_audio is not None:
                        if len(speech_audio) < _MIN_SPEECH_BYTES:
                            print(f"[VAD-RTC] speech_end SKIPPED (too short: {len(speech_audio)} bytes)", flush=True)
                        elif session.assistant_speaking:
                            print(f"[VAD-RTC] speech_end IGNORED (assistant speaking)", flush=True)
                        else:
                            vad_audio_b64 = _b64mod.b64encode(speech_audio).decode()
                            print(f"[VAD-RTC] dispatching ({len(speech_audio)} bytes)", flush=True)
                            asyncio.create_task(_process_voice_utterance(vad_audio_b64))

                rtc_handler = RTCAudioHandler(
                    on_audio=_rtc_on_audio,
                    sample_rate=24000,
                )
                sdp_offer = event.get("sdp", "")
                try:
                    sdp_answer = await rtc_handler.handle_offer(sdp_offer)
                    await websocket.send_json({
                        "type": "rtc.answer",
                        "sdp": sdp_answer,
                    })
                    print("[RTC] peer connection established", flush=True)
                except Exception as exc:
                    print(f"[RTC] offer handling failed: {exc}", flush=True)
                    await websocket.send_json({
                        "type": "error",
                        "error": f"rtc_failed: {exc}",
                    })
                    rtc_handler = None

            elif event_type == "rtc.ice":
                pass  # ICE candidates bundled in SDP; no trickle ICE needed
    except WebSocketDisconnect:
        pass
    finally:
        # Post-session quality analysis (async, non-blocking)
        try:
            chat_session = state.get(session_id)
            if chat_session and len(chat_session.transcript) >= 4:
                from .session_analyzer import analyze_session, save_report
                from .llm import call_openai_compatible
                report = await asyncio.to_thread(
                    analyze_session,
                    chat_session.transcript,
                    call_openai_compatible,
                    settings.llm.base_url,
                    settings.llm.model,
                )
                report["session_id"] = session_id
                save_report(report, Path(__file__).resolve().parents[1] / ".state")
                state.log({"event": "session_analysis", "session_id": session_id, "overall_score": report.get("overall_score")})
        except Exception:  # noqa: BLE001
            pass  # Analysis failure should never break the session cleanup
        if rtc_handler is not None:
            await rtc_handler.close()
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


@app.websocket("/ws/sip-monitor")
async def sip_monitor_ws(websocket: WebSocket) -> None:
    """Read-only WebSocket for SIP call monitoring."""
    await websocket.accept()
    _sip_monitor_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keepalive
    except WebSocketDisconnect:
        pass
    finally:
        _sip_monitor_clients.discard(websocket)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def _stream_or_json(payload: dict[str, Any], stream: bool) -> Any:
    if not stream:
        return payload
    if "type" not in payload:
        payload = {"type": "final", **payload}

    def gen() -> Any:
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
