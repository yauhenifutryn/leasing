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
from .text_utils import clean_answer, clean_voice_output, iter_final_text
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

# Jambonz: store caller phone from control WS for audio WS to read
_jambonz_last_caller_phone: str = ""

# SIP monitor: connected WebSocket clients for live event streaming
_sip_monitor_clients: set[WebSocket] = set()


async def broadcast_sip_event(event: dict[str, Any]) -> None:
    """Fire-and-forget broadcast to all connected SIP monitor pages."""
    for ws in list(_sip_monitor_clients):
        try:
            await ws.send_json(event)
        except Exception:  # noqa: BLE001
            _sip_monitor_clients.discard(ws)


class _JambonzWebSocketShim:
    """Adapts Jambonz audio WebSocket to the interface expected by _stream_voice_response.

    Intercepts audio events: decodes base64 PCM 24kHz and sends as binary
    WebSocket frames (no resampling needed). Other events broadcast to SIP monitor.
    """

    def __init__(self, ws: WebSocket, session_id: str) -> None:
        self._ws = ws
        self._session_id = session_id
        self.audio_bytes_sent = 0

    async def send_bytes(self, data: bytes) -> None:
        """Forward raw PCM bytes to the underlying Jambonz audio WebSocket."""
        await self._ws.send_bytes(data)

    async def send_json(self, data: dict[str, Any]) -> None:
        event_type = data.get("type", "")

        if event_type == "response.output_audio.delta":
            import base64 as _b64
            audio_b64 = data.get("delta", "")
            if audio_b64:
                pcm_raw = _b64.b64decode(audio_b64)
                # Send in chunks to prevent overwhelming mod_audio_fork
                _chunk = 1920
                for _i in range(0, len(pcm_raw), _chunk):
                    await self._ws.send_bytes(pcm_raw[_i : _i + _chunk])
                self.audio_bytes_sent += len(pcm_raw)
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

        await broadcast_sip_event({
            **data,
            "call_id": self._session_id,
        })


app = FastAPI(title="Micro Leasing RAG Demo")
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


_shared_vad: SileroVAD | None = None


@app.on_event("startup")
async def _warmup() -> None:
    """Pre-load models and optionally start Jambonz listener on startup."""
    global _shared_vad
    try:
        engine.retrieve("warmup", fast=True, voice_fast=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        silence_ms = int(os.getenv("VAD_SILENCE_MS", "500"))
        _shared_vad = SileroVAD(sample_rate=24000, silence_ms=silence_ms)
    except Exception:  # noqa: BLE001
        pass

    if settings.jambonz.enabled:
        print("[Jambonz] Enabled. Waiting for calls on /ws/jambonz", flush=True)


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

    # --- Build prompt (needed before parallel tasks) ---
    system_prompt = settings.app.system_prompt_path.read_text(encoding="utf-8")
    chat_session = state.get(session_id) or state.create(session_id)
    memory_block = build_memory_block(chat_session.transcript, settings.app.memory_turns)

    # --- Start RAG retrieval in background (runs while classifier executes) ---
    _t_rag_start = time.time()
    _rag_task = asyncio.create_task(asyncio.to_thread(
        engine.retrieve, rag_query, True, True, session_id,
    ))
    # RAG result awaited after classifier (see _rag_await below).
    # context_block, weak_hint built after await.
    expanded = any(trigger in message.lower() for trigger in settings.llm.expand_triggers)
    length_hint = (
        "Это голосовой разговор. Ответ: 1-2 коротких предложения. Самое важное. Не заканчивай каждый ответ вопросом. Задавай вопрос только если тебе реально нужна информация для продолжения."
        if not expanded
        else "Ответь подробнее, но кратко. Максимум три-четыре предложения."
    )
    # Get tool schemas early (only calculator; SMS is handled deterministically from code)
    tool_schemas = []
    _calc_tool = get_all_tools().get("calculator")
    if _calc_tool:
        try:
            tool_schemas.append(_calc_tool.schema(session_phone=session.client_phone))
        except TypeError:
            tool_schemas.append(_calc_tool.schema())

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

    # Smart intent classifier: sees last 7 turns of conversation, extracts
    # structured data (subject, cost, currency) for immediate tool calling.
    needs_tool = False
    _extracted_hints: dict[str, Any] = {}
    # Fast skip: obvious non-tool messages bypass the classifier entirely (~300ms saved)
    _msg_stripped = message.strip().lower().rstrip(".!,?")
    _SKIP_CLASSIFIER = {
        "спасибо", "спасибо большое", "понял", "понятно", "ясно", "ок",
        "хорошо", "ладно", "пока", "до свидания", "всего доброго",
        "привет", "здравствуйте", "добрый день", "нет", "не надо",
        "всем пока", "это всё", "больше ничего",
    }
    _skip = _msg_stripped in _SKIP_CLASSIFIER and not session.tool_calls_this_turn
    print(f"[Classifier] tools={len(tool_schemas)} msg='{message[:50]}' session={session_id[:8]}{' SKIP(non-tool)' if _skip else ''}", flush=True)
    if tool_schemas and not _skip:
        # Build conversation context: last 7 turns (not just 400 chars)
        _recent_turns = chat_session.transcript[-14:] if chat_session.transcript else []  # 7 pairs
        _conv_lines = []
        for _turn in _recent_turns:
            _role = "Клиент" if _turn.get("role") == "user" else "Бот"
            _conv_lines.append(f"{_role}: {_turn.get('text', '')}")
        _conv_context = "\n".join(_conv_lines) if _conv_lines else "начало разговора"

        _tool_history = ""
        if session.tool_calls_this_turn:
            _last_tools = []
            for tc in session.tool_calls_this_turn[-3:]:
                _tc_params = tc.get("params", {})
                _tc_brief = f"{tc.get('tool', '')}(ok={tc.get('ok', '?')}"
                if _tc_params.get("client_type"):
                    _tc_brief += f", client_type={_tc_params['client_type']}"
                _tc_brief += ")"
                _last_tools.append(_tc_brief)
            _tool_history = f"Инструменты в этом разговоре: {', '.join(_last_tools)}"

        _t_classify_start = time.time()
        try:
            classify_resp = await asyncio.to_thread(
                call_openai_compatible,
                base_url=effective_base_url,
                model=effective_model,
                system_prompt=(
                    "Ты классификатор сообщений голосового бота лизинговой компании. "
                    "Проанализируй НОВОЕ сообщение клиента в контексте диалога.\n\n"
                    "Верни строго JSON:\n"
                    '{"intent": "TOOL" или "RAG", "subject": "тип предмета или null", '
                    '"cost": число или null, "currency": "BYN/USD/EUR или null", '
                    '"client_type": "Физическое лицо/ИП/Юридическое лицо или null", '
                    '"prepaid": число (процент аванса) или null, "term": число (месяцев) или null, '
                    '"action": "calculate/recalculate/sms/change_param/confirm/clarify_client_type или null"}\n\n'
                    "intent=TOOL если клиент:\n"
                    "- хочет рассчитать, взять в лизинг, узнать платежи (даже без слова 'рассчитай')\n"
                    "- называет предмет + стоимость (это запрос на расчёт)\n"
                    "- просит изменить параметры предыдущего расчёта\n"
                    "- подтверждает действие (да, давай, хорошо) после предложения бота\n"
                    "- просит отправить СМС или график\n\n"
                    "intent=RAG если клиент задаёт информационный вопрос (адрес, документы, условия, часы работы)\n\n"
                    "subject: определи тип из контекста. Легковой автомобиль, Грузовой автомобиль, Спецтехника, Оборудование, Недвижимость, Прочий транспорт.\n"
                    "Если клиент назвал марку, определи тип (легковой/грузовой).\n"
                    "cost: извлеки число СТРОГО из НОВОГО сообщения клиента. "
                    "ЗАПРЕЩЕНО брать стоимость из предыдущих сообщений, ответов бота или результатов расчётов. "
                    "Если в НОВОМ сообщении клиент не назвал число, ставь null.\n"
                    "currency: ТОЛЬКО если клиент ЯВНО сказал 'долларов', 'евро', 'USD', 'EUR' в НОВОМ сообщении. "
                    "Если не сказал, ставь null. НЕ угадывай валюту.\n"
                    "client_type: извлеки ТОЛЬКО если клиент ЯВНО указал тип в НОВОМ сообщении. "
                    "'физлицо/физическое лицо' = 'Физическое лицо', "
                    "'ИП/предприниматель/индивидуальный' = 'ИП', "
                    "'юрлицо/юридическое/ООО/компания/фирма' = 'Юридическое лицо'. "
                    "Если не указал, ставь null.\n\n"
                    "БИЗНЕС-ПРАВИЛА:\n"
                    "Грузовые, спецтехника, оборудование, недвижимость: ТОЛЬКО для ИП и юрлиц.\n"
                    "USD/EUR: ТОЛЬКО для ИП и юрлиц. Физлица только BYN.\n"
                    "Если клиент хочет такой предмет или валюту И его client_type НЕИЗВЕСТЕН из всего диалога, "
                    "ставь action='clarify_client_type'.\n"
                    "ВАЖНО: если client_type уже ЯСЕН из диалога (клиент сказал 'физлицо', 'ИП', 'юрлицо' "
                    "ранее, ИЛИ калькулятор уже вызывался с конкретным client_type), "
                    "НЕ ставь clarify_client_type. Используй известный тип.\n\n"
                    "ЛИМИТЫ КАЛЬКУЛЯТОРА:\n"
                    "prepaid: минимум 10%. Если клиент просит 0% или 5%, ставь action='invalid_param'.\n"
                    "Если клиент просит аванс ниже 10%, ставь action='invalid_param'.\n\n"
                    "Никаких пояснений, только JSON."
                ),
                user_prompt=f"{_tool_history}\n\nДиалог:\n{_conv_context}\n\nНОВОЕ сообщение: {message}",
                temperature=0.0,
                max_tokens=80,
                timeout_sec=3,
            )
            _raw = classify_resp.text.strip()
            _js_start = _raw.find("{")
            _js_end = _raw.rfind("}") + 1
            if _js_start >= 0 and _js_end > _js_start:
                import json as _json_classify
                _parsed = _json_classify.loads(_raw[_js_start:_js_end])
                needs_tool = str(_parsed.get("intent", "")).upper() == "TOOL"
                # Extract hints for tool path
                if _parsed.get("subject"):
                    _extracted_hints["subject"] = _parsed["subject"]
                if _parsed.get("cost"):
                    _extracted_hints["cost"] = _parsed["cost"]
                if _parsed.get("currency"):
                    _extracted_hints["currency"] = _parsed["currency"]
                if _parsed.get("client_type"):
                    _extracted_hints["client_type"] = _parsed["client_type"]
                if _parsed.get("prepaid") is not None:
                    _extracted_hints["prepaid"] = _parsed["prepaid"]
                if _parsed.get("term") is not None:
                    _extracted_hints["term"] = _parsed["term"]
                if _parsed.get("action"):
                    _extracted_hints["action"] = _parsed["action"]
            else:
                needs_tool = "TOOL" in _raw.upper()
        except Exception as _classify_exc:
            print(f"[Classifier] ERROR: {_classify_exc}", flush=True)
            # Fallback to keyword heuristic
            needs_tool = has_sms_intent or any(
                t in message.lower() for t in
                ["рассчит", "расчет", "расчёт", "посчит", "пересчит", "калькул",
                 "аванс", "срок", "измени", "помен", "увелич", "уменьш",
                 "лизинг", "взять", "стоимость", "тысяч"]
            )
            if not needs_tool and tools_used_in_session:
                confirm_words = ["да", "давай", "хорошо", "ладно", "согласен", "отправь", "ок"]
                if message.strip().lower().rstrip(".!,") in confirm_words:
                    needs_tool = True
        if has_sms_intent:
            needs_tool = True
        # Override: if classifier extracted a tool action, force TOOL regardless of intent field
        if _extracted_hints.get("action") in ("calculate", "recalculate", "change_param", "sms", "confirm", "clarify_client_type", "invalid_param"):
            needs_tool = True
        _t_classify_ms = (time.time() - _t_classify_start) * 1000
        print(f"[Classifier] result: intent={'TOOL' if needs_tool else 'RAG'} hints={_extracted_hints} ({_t_classify_ms:.0f}ms)", flush=True)

    # --- Await RAG retrieval (started before classifier, should be done by now) ---
    retrieval = await _rag_task
    t_retrieval_done = time.time()
    print(f"[Latency:{session_id[:8]}] RAG: {(t_retrieval_done - _t_rag_start)*1000:.0f}ms (parallel)", flush=True)
    timings: dict[str, Any] = dict(retrieval.get("timings") or {})
    final_chunks = retrieval.get("final") or []
    context_block = "\n\n".join(
        [f"[Fragment {i+1}]\n{c['text']}" for i, c in enumerate(final_chunks)]
    )
    weak_context = bool(retrieval.get("weak"))
    weak_hint = (
        "Контекст может быть неполным. Дай ближайшую релевантную информацию из фрагментов, "
        "скажи, что точных данных может не хватать, и задай уточняющий вопрос.\n\n"
    ) if weak_context else ""

    # Refusal: no RAG chunks found and not a tool request
    if (not retrieval.get("ok") or not final_chunks) and not needs_tool:
        answer = settings.app.strict_refusal_text
        await websocket.send_json({
            "type": "response.output_text.delta",
            "session_id": session_id,
            "delta": answer,
        })
        try:
            audio_resp = await asyncio.to_thread(synthesize_audio, answer, session_id)
            if audio_resp.get("audio_b64"):
                await websocket.send_json({
                    "type": "response.output_audio.delta",
                    "session_id": session_id,
                    "delta": audio_resp["audio_b64"],
                    "sample_rate_hz": audio_resp.get("sample_rate_hz"),
                })
        except Exception:  # noqa: BLE001
            pass
        await websocket.send_json({
            "type": "response.done", "session_id": session_id,
            "backend": backend, "used_knowledge": [], "citations": [], "timings": timings,
        })
        return

    _t_total_parallel = (t_retrieval_done - _t_rag_start) * 1000
    print(f"[Latency:{session_id[:8]}] Total RAG+Classifier parallel: {_t_total_parallel:.0f}ms", flush=True)

    # SMS: direct execution (bypass LLM) when we have calculator data + phone
    # Trigger on: explicit SMS keywords OR classifier detected sms action
    _sms_from_classifier = _extracted_hints.get("action") == "sms" and session.tool_calls_this_turn
    sms_context = ""
    if (has_sms_intent or _sms_from_classifier) and session.tool_calls_this_turn and session.client_phone:
        last_calc = next(
            (tc for tc in reversed(session.tool_calls_this_turn)
             if tc.get("tool") == "calculator"), None)
        if last_calc and last_calc.get("result", {}).get("ok"):
            calc_tool = get_tool("calculator")
            sms_body = calc_tool.format_sms_body(last_calc["result"])
            sms_tool = get_tool("send_sms")
            if sms_tool and sms_body:
                sms_params = {"phone": session.client_phone, "message": sms_body}
                await websocket.send_json({"type": "tool_call.start", "tool": "send_sms", "params": sms_params})
                try:
                    sms_result = await asyncio.to_thread(sms_tool.execute, sms_params, {})
                    sms_ok = sms_result.get("ok", False)
                    session.tool_calls_this_turn.append({"tool": "send_sms", "params": sms_params, "result": sms_result})
                    await websocket.send_json({"type": "tool_call.done", "tool": "send_sms", "ok": sms_ok})
                    print(f"[Jambonz:{session_id[:8]}] SMS sent directly to {session.client_phone} ok={sms_ok}", flush=True)
                except Exception as _sms_exc:
                    print(f"[Jambonz:{session_id[:8]}] SMS direct send error: {_sms_exc}", flush=True)
                    await websocket.send_json({"type": "tool_call.done", "tool": "send_sms", "ok": False})
                # TTS confirmation (separate from SMS so TTS failure doesn't mask SMS success)
                try:
                    _confirm = f"Отправила график платежей по СМС на номер {session.client_phone}."
                    audio_resp = await asyncio.to_thread(synthesize_audio, _confirm, session_id)
                    _ab64 = audio_resp.get("audio_b64", "")
                    if _ab64:
                        import base64 as _sms_b64
                        _pcm = _sms_b64.b64decode(_ab64)
                        _chunk = 1920
                        for _i in range(0, len(_pcm), _chunk):
                            await websocket.send_bytes(_pcm[_i : _i + _chunk])
                        _dur = len(_pcm) / 48000.0
                        _ws = asyncio.get_event_loop().time()
                        while asyncio.get_event_loop().time() - _ws < _dur:
                            if session.interrupted:
                                break
                            await asyncio.sleep(0.1)
                except Exception as _tts_exc:
                    print(f"[Jambonz:{session_id[:8]}] SMS TTS confirmation error: {_tts_exc}", flush=True)
                session.assistant_speaking = False
                return
            sms_context = f"Текст для СМС:\n{sms_body}\n\n"
    elif has_sms_intent and session.tool_calls_this_turn:
        last_calc = next(
            (tc for tc in reversed(session.tool_calls_this_turn)
             if tc.get("tool") == "calculator"), None)
        if last_calc and last_calc.get("result", {}).get("ok"):
            calc_tool = get_tool("calculator")
            sms_context = f"Текст для СМС:\n{calc_tool.format_sms_body(last_calc['result'])}\n\n"

    # ── Deterministic Tool Orchestration ──
    # When classifier extracts enough data, call tools from code directly.
    # LLM only presents the result. This bypasses unreliable LLM tool calling.
    _direct_tool_result = None

    # Parameter change path: user wants to modify previous calculation
    # (e.g., "change advance to 20%", "make it 48 months", "switch to юрлицо").
    # Does NOT require subject+cost; uses previous calc params as base.
    _is_param_change = (
        needs_tool
        and _extracted_hints.get("action") in ("change_param", "recalculate")
        and session.tool_calls_this_turn
    )
    _param_change_params: dict[str, Any] | None = None
    _change_no_new_value = False  # True when user asks about changing but gives no value
    if _is_param_change:
        _prev = next((tc for tc in reversed(session.tool_calls_this_turn)
                      if tc.get("tool") == "calculator"), None)
        if _prev and _prev.get("params"):
            import json as _json_change
            # Check if classifier actually extracted any NEW changeable field
            _changeable = ("prepaid", "term", "currency", "client_type", "condition_new", "type_schedule")
            _has_new_value = any(_extracted_hints.get(k) is not None for k in _changeable)
            if not _has_new_value:
                # User asked about changing but didn't specify a value.
                # Present current params, ask what to change. Don't re-call the API.
                _change_no_new_value = True
                _direct_tool_result = _prev.get("result")
                print(f"[DirectTool] change_param: no new values, presenting current params", flush=True)
            else:
                _param_change_params = dict(_prev["params"])
                # Override only what the classifier extracted in the new message
                for _k, _v in _extracted_hints.items():
                    if _k in ("subject", "cost", "currency", "prepaid", "term",
                              "client_type", "condition_new", "type_schedule") and _v is not None:
                        # Convert prepaid amount to percentage if it looks like an amount
                        if _k == "prepaid" and _v > 100:
                            _prev_cost = _param_change_params.get("cost", 0)
                            if _prev_cost > 0:
                                _v = round((_v / _prev_cost) * 100, 1)
                                print(f"[DirectTool] prepaid amount {_extracted_hints['prepaid']} -> {_v}%", flush=True)
                        _param_change_params[_k] = _v
            print(f"[DirectTool] change_param: {_json_change.dumps({k: v for k, v in _extracted_hints.items() if k != 'action'}, ensure_ascii=False)}", flush=True)

    # If classifier detected tool intent but no cost and not a param change,
    # check if asking about previous result
    if not _is_param_change and needs_tool and _extracted_hints.get("subject") and not _extracted_hints.get("cost"):
        _prev_calc = next((tc for tc in reversed(getattr(session, 'tool_calls_this_turn', []))
                          if tc.get("tool") == "calculator" and tc.get("ok")), None)
        if _prev_calc and _prev_calc.get("result"):
            # Re-present the previous result
            _direct_tool_result = _prev_calc["result"]
            print(f"[DirectTool] re-presenting previous result", flush=True)

    _can_direct_call = (
        _param_change_params is not None
        or (needs_tool and _extracted_hints.get("subject") and _extracted_hints.get("cost"))
    )
    if _can_direct_call:
        _action = _extracted_hints.get("action", "calculate")
        calc_tool = get_tool("calculator")
        import json as _json_direct

        if _param_change_params is not None:
            # Use pre-built params from change_param path
            _direct_params = _param_change_params
        else:
            # Build params: start with extracted hints
            _direct_params: dict[str, Any] = {
                "subject": _extracted_hints["subject"],
                "cost": _extracted_hints["cost"],
            }
            if _extracted_hints.get("currency"):
                _direct_params["currency"] = _extracted_hints["currency"]
            if _extracted_hints.get("client_type"):
                _direct_params["client_type"] = _extracted_hints["client_type"]

        # Check subject restrictions for individuals before calling API.
        # Only block when client_type is EXPLICITLY individual (not defaulted).
        # If client_type is unknown, the classifier should have set clarify_client_type.
        # This is a lightweight fallback in case the classifier missed it.
        _subj_lower = _direct_params.get("subject", "").lower()
        _client = _direct_params.get("client_type")  # None if not set
        _individual_subjects = {"легковой автомобиль", "прочий транспорт"}
        if _client and "Физическое" in str(_client) and _subj_lower not in _individual_subjects and _subj_lower:
            _direct_tool_result = {
                "ok": False,
                "error": f"Для физических лиц доступен лизинг только легковых автомобилей и прочего транспорта. "
                         f"{_direct_params['subject']} доступен для юридических лиц и ИП.",
                "params": _direct_params,
                "defaulted": [],
            }
            print(f"[DirectTool] BLOCKED: {_direct_params['subject']} not available for individuals", flush=True)
        else:
            pass  # proceed to calculator call below

        if _direct_tool_result is None:  # not blocked by restriction check
            print(f"[DirectTool] calculator({_json_direct.dumps(_direct_params, ensure_ascii=False)})", flush=True)
            await broadcast_sip_event({
                "type": "sip.tool.start",
                "call_id": session_id,
                "tool": "calculator",
                "params": _direct_params,
            })
            try:
                _direct_tool_result = await asyncio.to_thread(calc_tool.execute, _direct_params, {})
                _tool_ok = _direct_tool_result.get("ok", False)
                print(f"[DirectTool] result: ok={_tool_ok}", flush=True)
                await broadcast_sip_event({
                    "type": "sip.tool.result",
                    "call_id": session_id,
                    "tool": "calculator",
                    "ok": _tool_ok,
                })
                session.tool_calls_this_turn = getattr(session, 'tool_calls_this_turn', [])
                session.tool_calls_this_turn.append({
                    "tool": "calculator",
                    "params": _direct_tool_result.get("params", _direct_params),
                    "result": _direct_tool_result,
                    "ok": _tool_ok,
                })
            except Exception as _texc:
                print(f"[DirectTool] ERROR: {_texc}", flush=True)
                _direct_tool_result = None

    # ── Clarify client type: classifier detected restricted subject/currency ──
    _clarify_needed = needs_tool and _extracted_hints.get("action") == "clarify_client_type"
    if _clarify_needed:
        _subj = _extracted_hints.get("subject", "предмет лизинга")
        _cur = _extracted_hints.get("currency")
        _reason_parts = []
        if _subj.lower() not in {"легковой автомобиль", "прочий транспорт"}:
            _reason_parts.append(f"{_subj} доступен для юридических лиц и ИП")
        if _cur and _cur != "BYN":
            _reason_parts.append(f"расчёт в {_cur} доступен для юридических лиц и ИП")
        _reason = "; ".join(_reason_parts) if _reason_parts else "для данного расчёта важен тип клиента"
        print(f"[DirectTool] clarify_client_type: {_reason}", flush=True)
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Клиент хочет рассчитать лизинг. {_reason}. "
                "Нужно уточнить тип клиента. Спроси КРАТКО (1 предложение): "
                "они оформляют как физическое лицо, ИП или юридическое лицо? "
                f"Сообщение клиента: {message}"
            )},
        ]
        tool_schemas = []
    elif needs_tool and _extracted_hints.get("action") == "invalid_param":
        # Classifier detected a parameter that will fail the calculator
        _invalid_prepaid = _extracted_hints.get("prepaid")
        if _invalid_prepaid is not None and _invalid_prepaid < 10:
            _invalid_reason = f"Минимальный аванс для лизинга составляет 10%. Клиент просит {_invalid_prepaid}%."
        else:
            _invalid_reason = "Указанные параметры выходят за лимиты калькулятора."
        print(f"[DirectTool] invalid_param: {_invalid_reason}", flush=True)
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"{_invalid_reason}\n"
                f"Сообщение клиента: {message}\n\n"
                "Объясни кратко (1 предложение), что минимальный аванс 10%. "
                "Предложи рассчитать с 10%."
            )},
        ]
        tool_schemas = []
    elif _change_no_new_value and _direct_tool_result and _direct_tool_result.get("ok"):
        # User asked about changing params but didn't specify a new value.
        # Present current params and ask what they want to change.
        _p = _direct_tool_result.get("params", {})
        _cur = _p.get("currency", "BYN")
        _param_summary = (
            f"Текущие параметры: аванс {_p.get('prepaid', 30)}%, "
            f"срок {_p.get('term', 36)} мес., "
            f"валюта {_cur}, "
            f"тип графика {'аннуитетный' if _p.get('type_schedule') == '0' else 'убывающий'}."
        )
        print(f"[DirectTool] presenting current params for change request", flush=True)
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"{_param_summary}\n"
                f"Сообщение клиента: {message}\n\n"
                "Клиент хочет изменить параметры. Назови текущие значения (аванс, срок) "
                "и спроси, какой именно параметр и на какое значение хочет изменить. "
                "Минимальный аванс: 10%."
            )},
        ]
        tool_schemas = []
    elif _direct_tool_result and _direct_tool_result.get("ok"):
        # Tool executed successfully: LLM only presents the result
        _p = _direct_tool_result.get("params", {})
        _cur = _p.get("currency", "BYN")
        _defaulted = set(_direct_tool_result.get("defaulted", []))
        _defaults_note = ""
        if _defaulted:
            _def_parts = []
            if "prepaid" in _defaulted:
                _def_parts.append(f"аванс {_p.get('prepaid', 30)}% (по умолчанию)")
            if "term" in _defaulted:
                _def_parts.append(f"срок {_p.get('term', 36)} мес. (по умолчанию)")
            if "client_type" in _defaulted:
                _def_parts.append(f"тип клиента: {_p.get('client_type', '?')} (по умолчанию)")
            if "type_schedule" in _defaulted:
                _def_parts.append("аннуитетный график (по умолчанию)")
            if _def_parts:
                _defaults_note = f" Параметры по умолчанию: {', '.join(_def_parts)}."
        _result_summary = (
            f"Аванс {_p.get('prepaid', 30)}%: {_direct_tool_result.get('advance_sum', '?')} {_cur}. "
            f"Ежемесячный платёж: {_direct_tool_result.get('payment_min', '?')} {_cur}. "
            f"Выкупной: {_direct_tool_result.get('buyout_sum', '?')} {_cur}. "
            f"Общая сумма: {_direct_tool_result.get('total', '?')} {_cur}. "
            f"Удорожание: {_direct_tool_result.get('increase_percent', '?')}%. "
            f"Срок: {_direct_tool_result.get('num_payments', '?')} мес."
            f"{_defaults_note}"
        )
        print(f"[DirectTool] presenting: {_result_summary[:100]}", flush=True)
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Результат расчёта калькулятора:\n{_result_summary}\n\n"
                f"Сообщение клиента: {message}\n\n"
                "Назови аванс, ежемесячный платёж и общую сумму. "
                "Спроси: хотите изменить аванс, срок или тип платежей, или отправить график по СМС?"
            )},
        ]
        tool_schemas = []  # No function calling needed, just present result
    elif _direct_tool_result and not _direct_tool_result.get("ok"):
        # Tool failed (fallback): explain the API error naturally.
        # This catches cases the classifier didn't flag via clarify_client_type.
        _err_params = _direct_tool_result.get("params", {})
        _err_explanation = _direct_tool_result.get("error", "По заданным параметрам условия не найдены.")
        print(f"[DirectTool] FALLBACK error handler: {_err_explanation[:80]}", flush=True)
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Калькулятор вернул ошибку: {_err_explanation}\n"
                f"Сообщение клиента: {message}\n\n"
                "Объясни кратко (1-2 предложения). Предложи альтернативу или уточни параметры."
            )},
        ]
        tool_schemas = []
    elif needs_tool:
        # TOOL path: classifier detected tool intent but insufficient data for direct execution.
        # Fall back to LLM function calling.
        prev_calc_context = ""
        if session.tool_calls_this_turn:
            last_calc = next(
                (tc for tc in reversed(session.tool_calls_this_turn)
                 if tc.get("tool") == "calculator"), None)
            if last_calc:
                import json as _json_prev
                prev_params = last_calc.get("params", {})
                prev_calc_context = (
                    f"Предыдущий расчёт: {_json_prev.dumps(prev_params, ensure_ascii=False)}. "
                    f"Клиент хочет изменить параметры. Вызови calculator с обновлёнными значениями.\n\n"
                )
        _tool_instruction = (
            "ОБЯЗАТЕЛЬНО вызови инструмент. НЕ отвечай текстом о расчёте без вызова calculator.\n\n"
        )
        _hints_text = ""
        if _extracted_hints:
            import json as _json_hints
            _hints_text = (
                f"Извлечённые данные: {_json_hints.dumps(_extracted_hints, ensure_ascii=False)}.\n\n"
            )
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{_tool_instruction}{_hints_text}{sms_context}{prev_calc_context}{message}"},
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
    session._tts_start_time = 0  # reset for new TTS warmup
    producer_task = asyncio.create_task(llm_producer())
    consumer_task = asyncio.create_task(tts_consumer())
    await asyncio.gather(producer_task, consumer_task)
    # Wait for FreeSWITCH to finish playing buffered audio
    # Jambonz shim tracks bytes sent; 24kHz * 2 = 48000 bytes/sec
    if hasattr(websocket, 'audio_bytes_sent') and websocket.audio_bytes_sent > 0 and not session.interrupted:
        _play_dur = websocket.audio_bytes_sent / 48000.0
        _wait_start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - _wait_start < _play_dur:
            if session.interrupted:
                break
            await asyncio.sleep(0.1)
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
        text = clean_voice_output(text)
        if not text:
            return
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

    # LLM-based first utterance classifier: handles names in any grammatical case,
    # questions, and combined name+question inputs without hardcoded patterns.
    from .llm import call_openai_compatible
    client_name = "друг"
    first_question = None
    try:
        _classify_resp = await asyncio.to_thread(
            call_openai_compatible,
            base_url=settings.llm.fast_base_url or settings.llm.base_url,
            model=settings.llm.fast_model or settings.llm.model,
            system_prompt=(
                "Проанализируй первую реплику клиента в разговоре с голосовым помощником. "
                "Верни строго JSON без пояснений:\n"
                '{"type": "name"|"question"|"both", "name": "имя в именительном падеже или null", "question": "текст вопроса или null"}\n\n'
                "type=name: клиент представился (Привет я Сергей / Меня зовут Илья / Можно Илье обращаться / Евгений)\n"
                "type=question: клиент задал вопрос без имени (Какие условия лизинга? / Сколько стоит?)\n"
                "type=both: клиент назвал имя И задал вопрос (Привет, я Никита, какие условия лизинга?)\n\n"
                "Имя ВСЕГДА в именительном падеже: Илье→Илья, Сергею→Сергей, Никитой→Никита.\n"
                "Если не уверен что это имя, верни null."
            ),
            user_prompt=client_name_raw,
            temperature=0.0,
            max_tokens=80,
            timeout_sec=5,
        )
        _parsed = None
        _text = _classify_resp.text.strip()
        _js_start = _text.find("{")
        _js_end = _text.rfind("}") + 1
        if _js_start >= 0 and _js_end > _js_start:
            import json as _json_mod
            _parsed = _json_mod.loads(_text[_js_start:_js_end])

        if _parsed:
            _utype = _parsed.get("type", "question")
            _uname = _parsed.get("name") or None
            _uquestion = _parsed.get("question") or None

            if _uname and 2 <= len(_uname) <= 20:
                client_name = _uname.strip().title()
            if _utype in ("question", "both") and _uquestion:
                first_question = _uquestion
            elif _utype == "question":
                first_question = client_name_raw
    except Exception:  # noqa: BLE001
        first_question = client_name_raw

    if client_name == "друг":
        if not first_question:
            first_question = client_name_raw
    else:
        if first_question:
            await _send_tts_message(f"Очень приятно, {client_name}!")
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
        # Post-session: save transcript + quality analysis
        _state_dir = Path(__file__).resolve().parents[1] / ".state"
        try:
            chat_session = state.get(session_id)
            if chat_session and chat_session.transcript:
                from .session_analyzer import save_transcript
                save_transcript(session_id, chat_session.transcript, _state_dir, transport="browser")
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
                report["transport"] = "browser"
                save_report(report, _state_dir)
                state.log({"event": "session_analysis", "session_id": session_id, "overall_score": report.get("overall_score")})
        except Exception:  # noqa: BLE001
            pass
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


# ---------------------------------------------------------------------------
# Jambonz telephony: WebSocket handlers
# ---------------------------------------------------------------------------


async def _jambonz_send_tts(
    ws: WebSocket,
    session: VoiceSession,
    session_id: str,
    text: str,
) -> None:
    """Synthesize text and send audio back as binary WebSocket frames."""
    try:
        text = clean_voice_output(text)
        if not text:
            return
        print(f"[Jambonz:{session_id[:8]}] TTS synthesizing: {text[:40]}...", flush=True)
        audio_resp = await asyncio.to_thread(synthesize_audio, text, session_id)
        audio_b64 = audio_resp.get("audio_b64") or ""
        if audio_b64:
            import base64 as _b64
            pcm_24k = _b64.b64decode(audio_b64)
            # Send in 20ms chunks (960 samples * 2 bytes = 1920 bytes at 24kHz)
            # Sending all at once overwhelms mod_audio_fork and disconnects
            chunk_size = 1920
            for i in range(0, len(pcm_24k), chunk_size):
                chunk = pcm_24k[i : i + chunk_size]
                await ws.send_bytes(chunk)
            # Wait for FreeSWITCH to finish playing (audio duration)
            # 24kHz * 2 bytes = 48000 bytes/sec
            audio_duration = len(pcm_24k) / 48000.0
            print(f"[Jambonz:{session_id[:8]}] TTS sent {len(pcm_24k)} bytes PCM 24kHz ({len(pcm_24k) // chunk_size} chunks, {audio_duration:.1f}s)", flush=True)
            # Wait for playback, but stop early if interrupted
            _wait_start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - _wait_start < audio_duration:
                if session.interrupted:
                    break
                await asyncio.sleep(0.1)
        else:
            print(f"[Jambonz:{session_id[:8]}] TTS returned empty audio", flush=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[Jambonz:{session_id[:8]}] TTS error: {exc}\n{traceback.format_exc()}", flush=True)

    session.assistant_speaking = False
    session._tts_finished_at = asyncio.get_event_loop().time()  # type: ignore[attr-defined]
    await broadcast_sip_event({
        "type": "sip.tts.start",
        "call_id": session_id,
        "text": text,
    })


async def _jambonz_process_utterance(
    ws: WebSocket,
    session: VoiceSession,
    session_id: str,
    speech_audio: bytes,
) -> None:
    """Transcribe speech audio and stream LLM response back via Jambonz."""
    import base64 as _b64mod

    audio_b64 = _b64mod.b64encode(speech_audio).decode()
    question_id = str(uuid.uuid4())
    t_speech_stopped = time.time()

    try:
        transcript = transcribe_audio(audio_b64, session_id=session_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[Jambonz:{session_id[:8]}] STT error: {exc}", flush=True)
        session.assistant_speaking = False
        return

    t_stt_done = time.time()
    _stt_ms = (t_stt_done - t_speech_stopped) * 1000
    text = (transcript.get("text") or "").strip()
    if not text:
        print(f"[Jambonz:{session_id[:8]}] STT: empty transcription", flush=True)
        session.assistant_speaking = False
        return

    # Filter known Whisper hallucinations (YouTube training data artifacts)
    _WHISPER_HALLUCINATIONS = [
        "субтитры", "подписывайтесь", "канал", "спасибо за просмотр",
        "dimator", "продолжение следует", "продолжаем", "редактор субтитров",
        "корректор", "семкин", "егорова",
    ]
    _text_lower = text.lower()
    if any(h in _text_lower for h in _WHISPER_HALLUCINATIONS):
        print(f"[Jambonz:{session_id[:8]}] STT: hallucination filtered: {text}", flush=True)
        session.assistant_speaking = False
        return

    # Echo detection: if STT output matches recent bot speech, discard.
    # Catches speaker-mode echo where phone mic picks up bot's own TTS.
    # Uses word-set overlap (not just substring) to catch Whisper paraphrasing.
    _chat_sess = state.get(session_id)
    if _chat_sess and _chat_sess.transcript:
        _recent_bot = " ".join(
            t.get("text", "") for t in _chat_sess.transcript[-4:]
            if t.get("role") == "assistant"
        ).upper()
        _text_up = text.upper().strip()
        if len(_text_up) >= 3 and _recent_bot:
            # Method 1: exact substring (original, catches literal fragments)
            _is_echo = _text_up in _recent_bot
            # Method 2: word-set overlap (catches Whisper paraphrasing of echo)
            if not _is_echo:
                _user_words = set(_text_up.split())
                _bot_words = set(_recent_bot.split())
                if len(_user_words) >= 2:
                    _overlap = len(_user_words & _bot_words) / len(_user_words)
                    _is_echo = _overlap >= 0.6
            if _is_echo:
                print(f"[Jambonz:{session_id[:8]}] Echo filtered: '{text}' (matches bot speech)", flush=True)
                session.assistant_speaking = False
                return

    print(f"[Jambonz:{session_id[:8]}] STT({_stt_ms:.0f}ms): {text}", flush=True)
    await broadcast_sip_event({
        "type": "sip.stt.result",
        "call_id": session_id,
        "text": text,
    })

    session.on_transcript_final(text)

    jambonz_ws = _JambonzWebSocketShim(ws, session_id)
    await _stream_voice_response(
        websocket=jambonz_ws,
        session=session,
        session_id=session_id,
        message=text,
        t_speech_stopped=t_speech_stopped,
        t_stt_done=t_stt_done,
        question_id=question_id,
        rtc_handler=None,
    )


@app.websocket("/ws/jambonz")
async def jambonz_control_ws(websocket: WebSocket) -> None:
    """Jambonz call control WebSocket (subprotocol: ws.jambonz.org)."""
    await websocket.accept(subprotocol="ws.jambonz.org")
    call_sid = ""
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "session:new":
                msgid = msg.get("msgid", "")
                call_sid = msg.get("call_sid", "") or msg.get("callSid", "")
                # Call data is nested under "data"
                _data = msg.get("data", {}) if isinstance(msg.get("data"), dict) else {}
                caller_phone = (
                    _data.get("from", "")
                    or _data.get("callingNumber", "")
                    or msg.get("from", "")
                    or ""
                )
                caller_name = (
                    _data.get("callerName", "")
                    or _data.get("caller_name", "")
                    or msg.get("callerName", "")
                    or ""
                )
                # SIP username (e.g. "sergey", "ilya") for monitor filtering
                sip_user = caller_phone.split("@")[0] if "@" in caller_phone else caller_phone
                # Extract phone from callerName if caller_phone is a SIP username
                import re as _re
                if not caller_phone or not _re.search(r'\d{7,}', caller_phone):
                    _match = _re.search(r'\+?(\d{7,15})', caller_name or "")
                    if _match:
                        caller_phone = _match.group(0)
                # Log full data for debugging
                print(
                    f"[Jambonz:{call_sid[:8]}] session:new from={caller_phone} name={caller_name} data_keys={list(_data.keys()) if _data else 'none'}",
                    flush=True,
                )

                # Broadcast call start to monitor (from control WS for reliability)
                await broadcast_sip_event({
                    "type": "sip.call.start",
                    "call_id": call_sid,
                    "phone": caller_phone or "unknown",
                    "sip_user": sip_user,
                })

                # Pass phone in the audio WS URL (guaranteed delivery, no shared state needed)
                import urllib.parse as _urlparse
                _phone_param = _urlparse.quote(caller_phone) if caller_phone else ""
                audio_ws_url = f"ws://host.docker.internal:8000/ws/jambonz-audio?phone={_phone_param}"
                ack = {
                    "type": "ack",
                    "msgid": msgid,
                    "data": [
                        {
                            "verb": "listen",
                            "url": audio_ws_url,
                            "sampleRate": 16000,
                            "mixType": "mono",
                            "passDtmf": True,
                            "bidirectionalAudio": {
                                "enabled": True,
                                "streaming": True,
                                "sampleRate": 24000,
                            },
                            "metadata": {
                                "from": caller_phone,
                                "callSid": call_sid,
                            },
                        }
                    ],
                }
                await websocket.send_text(json.dumps(ack))

            elif msg_type == "call:status":
                status = msg.get("callStatus", "")
                print(f"[Jambonz:{call_sid[:8]}] call:status={status}", flush=True)
                if status in ("completed", "failed", "no-answer", "busy"):
                    await broadcast_sip_event({
                        "type": "sip.call.end",
                        "call_id": call_sid,
                        "reason": status,
                    })
                    break

            elif msg_type == "verb:hook":
                print(f"[Jambonz:{call_sid[:8]}] verb:hook reason={msg.get('reason', '')}", flush=True)

    except WebSocketDisconnect:
        print(f"[Jambonz:{call_sid[:8]}] Control WS disconnected", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[Jambonz:{call_sid[:8]}] Control WS error: {exc}", flush=True)


@app.websocket("/ws/jambonz-audio")
async def jambonz_audio_ws(websocket: WebSocket) -> None:
    """Jambonz bidirectional audio WebSocket."""
    await websocket.accept()
    session_id = ""
    session: VoiceSession | None = None
    _frame_count = 0

    try:
        # 1. First frame: JSON metadata (or binary if mod_audio_fork skips metadata)
        print("[Jambonz] Audio WS accepted, waiting for first frame...", flush=True)
        first_msg = await websocket.receive()
        if "text" in first_msg and first_msg["text"]:
            raw_meta = first_msg["text"]
        elif "bytes" in first_msg and first_msg["bytes"]:
            # mod_audio_fork might send binary first without metadata
            print(f"[Jambonz] Audio WS got binary first frame ({len(first_msg['bytes'])} bytes), using defaults", flush=True)
            raw_meta = json.dumps({"callSid": "", "sampleRate": 16000, "metadata": {}})
        else:
            print(f"[Jambonz] Audio WS unexpected first frame: {first_msg}", flush=True)
            return
        meta = json.loads(raw_meta)
        call_sid = meta.get("callSid", "")
        sample_rate = meta.get("sampleRate", 16000)
        caller_meta = meta.get("metadata", {})
        caller_phone = caller_meta.get("from", "")
        # Fallback: read phone from URL query parameter
        if not caller_phone:
            import urllib.parse as _urlparse
            _qs = _urlparse.parse_qs(_urlparse.urlparse(str(websocket.url)).query)
            caller_phone = _qs.get("phone", [""])[0]
        session_id = call_sid or str(uuid.uuid4())

        print(
            f"[Jambonz:{session_id[:8]}] Audio WS connected, "
            f"sampleRate={sample_rate}, caller={caller_phone}",
            flush=True,
        )

        await broadcast_sip_event({
            "type": "sip.call.start",
            "call_id": session_id,
            "phone": caller_phone or "unknown",
        })

        # 2. Create voice session
        session = VoiceSession(
            session_id=session_id,
            backend="our_rag",
            transport="jambonz",
            client_phone=caller_phone or None,
            call_id=call_sid,
        )
        voice_sessions[session_id] = session

        # 3. Create VAD (16kHz native)
        silence_ms = int(os.getenv("VAD_SILENCE_MS", "500"))
        vad = SileroVAD(sample_rate=16000, silence_ms=silence_ms)

        # 4. Consent collection via DTMF (press 1 to accept, 2 to decline)
        _dtmf_event: asyncio.Event = asyncio.Event()
        _dtmf_digit: list[str] = []  # mutable container for the collected digit

        def _on_dtmf(digit: str) -> None:
            if digit in ("1", "2"):
                _dtmf_digit.append(digit)
                _dtmf_event.set()

        session._consent_dtmf_callback = _on_dtmf  # type: ignore[attr-defined]

        consent_text = (
            "Здравствуйте! Вас приветствует компания Микро Лизинг. "
            "Для продолжения разговора нам необходимо ваше согласие "
            "на обработку персональных данных. "
            "Нажмите 1 для согласия или 2 для отказа."
        )
        session.assistant_speaking = True
        # Play TTS as background task so DTMF can interrupt it
        asyncio.create_task(_jambonz_send_tts(websocket, session, session_id, consent_text))

        # Listen for DTMF while TTS plays (barge-in with keypad)
        consent_granted = False
        for _attempt in range(2):  # allow one repeat
            _dtmf_event.clear()
            _dtmf_digit.clear()
            try:
                _consent_deadline = asyncio.get_event_loop().time() + 20.0
                while not _dtmf_event.is_set():
                    _remaining = _consent_deadline - asyncio.get_event_loop().time()
                    if _remaining <= 0:
                        break
                    try:
                        _cmsg = await asyncio.wait_for(websocket.receive(), timeout=min(_remaining, 1.0))
                    except asyncio.TimeoutError:
                        continue
                    if "text" in _cmsg and _cmsg["text"]:
                        _ctrl = json.loads(_cmsg["text"])
                        if _ctrl.get("event", _ctrl.get("type", "")) == "dtmf":
                            _d = _ctrl.get("dtmf", "")
                            print(f"[Jambonz:{session_id[:8]}] Consent DTMF: {_d}", flush=True)
                            _on_dtmf(_d)
                    # Ignore audio frames during consent
            except Exception:
                break

            if _dtmf_digit and _dtmf_digit[0] == "1":
                consent_granted = True
                # Stop consent TTS immediately
                await websocket.send_text(json.dumps({"type": "killAudio"}))
                session.assistant_speaking = False
                print(f"[Jambonz:{session_id[:8]}] Consent GRANTED via DTMF", flush=True)
                break
            elif _dtmf_digit and _dtmf_digit[0] == "2":
                await websocket.send_text(json.dumps({"type": "killAudio"}))
                session.assistant_speaking = False
                print(f"[Jambonz:{session_id[:8]}] Consent DENIED via DTMF", flush=True)
                break
            else:
                session.assistant_speaking = False
                if _attempt == 0:
                    repeat_text = "Нажмите 1 для согласия или 2 для отказа."
                    session.assistant_speaking = True
                    asyncio.create_task(_jambonz_send_tts(websocket, session, session_id, repeat_text))
                    # Continue loop to wait for DTMF again

        session._consent_dtmf_callback = None  # type: ignore[attr-defined]

        if not consent_granted:
            denied_text = consent_denied_response()
            session.assistant_speaking = True
            await _jambonz_send_tts(websocket, session, session_id, denied_text)
            await asyncio.sleep(2)
            await websocket.send_text(json.dumps({"type": "disconnect"}))
            return

        # 5. Send welcome TTS (consent passed)
        session.assistant_speaking = True
        intro_text = (
            "Спасибо за согласие! Меня зовут Ксения, я голосовая помощница компании Микро Лизинг. "
            "Как я могу к вам обращаться?"
        )
        asyncio.create_task(_jambonz_send_tts(websocket, session, session_id, intro_text))

        # Pre-roll buffer: keeps last 500ms of audio during TTS playback.
        # On barge-in, this audio is prepended to STT so the speech onset isn't lost.
        # 500ms at 16kHz, 16-bit mono = 16000 bytes
        _PREROLL_BYTES = 16000
        _preroll_buf = bytearray()
        # Post-TTS cooldown: skip VAD for 100ms after natural TTS end.
        # Was 200ms but ate the start of user speech. 100ms is enough for echo tail.
        _COOLDOWN_SEC = 0.1

        # 5. Audio loop
        while True:
            msg = await websocket.receive()

            # Binary frame: caller audio (L16 PCM 16kHz)
            if "bytes" in msg and msg["bytes"]:
                pcm_16k = msg["bytes"]
                _frame_count += 1
                if _frame_count == 1:
                    print(f"[Jambonz:{session_id[:8]}] First audio frame ({len(pcm_16k)} bytes)", flush=True)
                if _frame_count % 250 == 0:
                    import struct as _st
                    import math as _math
                    _n = len(pcm_16k) // 2
                    if _n > 0:
                        _samps = _st.unpack(f"<{_n}h", pcm_16k[:_n*2])
                        _rms = _math.sqrt(sum(s*s for s in _samps) / _n)
                    else:
                        _rms = 0
                    print(f"[Jambonz:{session_id[:8]}] frames={_frame_count} speaking={session.assistant_speaking} vad={vad.is_speaking} rms={_rms:.0f} bytes={len(pcm_16k)}", flush=True)

                # Barge-in on clean caller audio (mono mode, separated tracks)
                if session.assistant_speaking:
                    # Skip first 0.5s of TTS (VAD model warmup).
                    # 0.5s = ~15 frames, enough for Silero VAD to stabilize.
                    if not hasattr(session, '_tts_start_time') or session._tts_start_time == 0:
                        session._tts_start_time = asyncio.get_event_loop().time()
                        session._barge_vad_count = 0
                    _tts_elapsed = asyncio.get_event_loop().time() - session._tts_start_time
                    if _tts_elapsed < 0.5:
                        continue

                    # Maintain pre-roll buffer (rolling 300ms of audio during TTS)
                    _preroll_buf.extend(pcm_16k)
                    if len(_preroll_buf) > _PREROLL_BYTES:
                        _preroll_buf = _preroll_buf[-_PREROLL_BYTES:]

                    # Feed audio to VAD
                    vad.feed(pcm_16k)

                    # Barge-in: require BOTH VAD probability AND audio energy (RMS).
                    # Silero VAD has state residue: after real speech, prob stays 0.99+
                    # for many frames even on silence. RMS check catches this:
                    # echo/silence RMS = 0-110, real speech RMS = 2000+.
                    # RMS floor 300 cleanly separates them.
                    import struct as _st_bi
                    import math as _math_bi
                    _n_bi = len(pcm_16k) // 2
                    _frame_rms = 0.0
                    if _n_bi > 0:
                        _samps_bi = _st_bi.unpack(f"<{_n_bi}h", pcm_16k[:_n_bi * 2])
                        _frame_rms = _math_bi.sqrt(sum(s * s for s in _samps_bi) / _n_bi)

                    _prob = vad.last_probability
                    if _prob >= 0.40 and _frame_rms >= 300:
                        if not hasattr(session, '_barge_vad_count'):
                            session._barge_vad_count = 0
                        session._barge_vad_count += 1
                    else:
                        session._barge_vad_count = max(0, getattr(session, '_barge_vad_count', 0) - 1)

                    # 4 consecutive VAD detections (~128ms) = confirmed speech
                    if getattr(session, '_barge_vad_count', 0) >= 4:
                        session.interrupted = True
                        session.assistant_speaking = False
                        session._tts_start_time = 0
                        session._barge_vad_count = 0
                        # Do NOT set _tts_finished_at here. The post-TTS cooldown
                        # is for natural TTS end (prevent echo). After barge-in,
                        # the user is already speaking; cooldown would eat their speech.
                        session._was_barge_in = True  # type: ignore[attr-defined]
                        # Pre-roll: save buffered audio to prepend to next speech segment.
                        # This captures the speech onset that triggered barge-in.
                        vad.reset()
                        if _preroll_buf:
                            # Inject pre-roll directly into VAD speech buffer so it
                            # becomes part of the next captured utterance
                            vad._speech_buffer = bytes(_preroll_buf)
                            vad._is_speaking = True
                            print(f"[Jambonz:{session_id[:8]}] Pre-roll: {len(_preroll_buf)} bytes injected into speech buffer", flush=True)
                        _preroll_buf.clear()
                        await websocket.send_text(json.dumps({"type": "killAudio"}))
                        print(f"[Jambonz:{session_id[:8]}] BARGE-IN (vad_prob={_prob:.2f} rms={_frame_rms:.0f})", flush=True)
                        await broadcast_sip_event({
                            "type": "sip.barge_in",
                            "call_id": session_id,
                        })
                    continue

                # Post-TTS cooldown: skip VAD for 200ms after TTS ends
                # Prevents echo/reverb from speaker mode triggering false speech
                _tts_fin = getattr(session, '_tts_finished_at', 0.0)
                if _tts_fin > 0:
                    if asyncio.get_event_loop().time() - _tts_fin < _COOLDOWN_SEC:
                        continue
                    session._tts_finished_at = 0.0  # type: ignore[attr-defined]

                # Normal listening: feed VAD
                was_speaking = vad.is_speaking
                speech_audio = vad.feed(pcm_16k)

                if not was_speaking and vad.is_speaking:
                    print(f"[Jambonz:{session_id[:8]}] VAD: speech_start", flush=True)
                    await broadcast_sip_event({
                        "type": "sip.vad.speech",
                        "call_id": session_id,
                        "event": "start",
                    })

                if speech_audio is not None:
                    # Minimum speech guard: prevents Whisper hallucinations on noise bursts.
                    # After barge-in, accept shorter speech (0.4s) to catch "Da", "Net", "OK".
                    # Normal listening uses 0.8s guard since short bursts are usually noise.
                    _was_bi = getattr(session, '_was_barge_in', False)
                    _min_bytes = 12800 if _was_bi else 25600  # 0.4s vs 0.8s at 16kHz
                    if len(speech_audio) < _min_bytes:
                        print(
                            f"[Jambonz:{session_id[:8]}] VAD: speech_end SKIPPED "
                            f"(too short: {len(speech_audio)} bytes, min={_min_bytes}, barge_in={_was_bi})",
                            flush=True,
                        )
                        continue
                    session._was_barge_in = False  # type: ignore[attr-defined]

                    from scipy.signal import resample_poly as _resample_poly
                    import numpy as _np
                    _samples = _np.frombuffer(speech_audio, dtype=_np.int16)
                    _resampled = _resample_poly(_samples, up=3, down=2).astype(_np.int16)
                    speech_24k = _resampled.tobytes()

                    print(
                        f"[Jambonz:{session_id[:8]}] VAD: speech_end "
                        f"({len(speech_audio)} bytes 16kHz -> {len(speech_24k)} bytes 24kHz)",
                        flush=True,
                    )
                    await broadcast_sip_event({
                        "type": "sip.vad.speech",
                        "call_id": session_id,
                        "event": "end",
                    })

                    session.assistant_speaking = True
                    session.interrupted = False
                    asyncio.create_task(_jambonz_process_utterance(
                        websocket, session, session_id, speech_24k,
                    ))

            # Text frame: JSON control messages
            elif "text" in msg and msg["text"]:
                ctrl = json.loads(msg["text"])
                event_type = ctrl.get("event", ctrl.get("type", ""))

                if event_type == "dtmf":
                    digit = ctrl.get("dtmf", "")
                    print(f"[Jambonz:{session_id[:8]}] DTMF: {digit}", flush=True)
                    # Forward to consent callback if active
                    _cb = getattr(session, '_consent_dtmf_callback', None) if session else None
                    if _cb:
                        _cb(digit)
                    await broadcast_sip_event({
                        "type": "sip.dtmf",
                        "call_id": session_id,
                        "digit": digit,
                    })

                elif event_type == "disconnect":
                    print(f"[Jambonz:{session_id[:8]}] Remote disconnect", flush=True)
                    break

    except WebSocketDisconnect:
        print(f"[Jambonz:{session_id[:8]}] Audio WS disconnected (frames: {_frame_count})", flush=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[Jambonz:{session_id[:8]}] Audio WS error: {exc}\n{traceback.format_exc()}", flush=True)
    finally:
        # Post-session: save transcript + quality analysis
        if session is not None:
            _state_dir = Path(__file__).resolve().parents[1] / ".state"
            try:
                chat_session = state.get(session_id)
                if chat_session and chat_session.transcript:
                    from .session_analyzer import save_transcript
                    save_transcript(session_id, chat_session.transcript, _state_dir,
                                    transport="jambonz", phone=session.client_phone or "")
                    print(f"[Jambonz:{session_id[:8]}] Transcript saved ({len(chat_session.transcript)} turns)", flush=True)
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
                    report["transport"] = "jambonz"
                    report["phone"] = session.client_phone or "unknown"
                    save_report(report, _state_dir)
                    print(f"[Jambonz:{session_id[:8]}] Session analysis: score={report.get('overall_score', '?')}", flush=True)
            except Exception:
                pass
            voice_sessions.pop(session_id, None)
        await broadcast_sip_event({
            "type": "sip.call.end",
            "call_id": session_id,
        })
        print(f"[Jambonz:{session_id[:8]}] Cleaned up", flush=True)


@app.post("/api/jambonz/call-status")
async def jambonz_call_status(request: Any = None) -> JSONResponse:
    """Receive call status updates from Jambonz feature-server."""
    return JSONResponse(status_code=200, content={"ok": True})


@app.get("/api/jambonz/credentials")
async def jambonz_credentials() -> JSONResponse:
    """Return SIP credentials for Zoiper setup (shown on monitor page)."""
    if not settings.jambonz.enabled:
        return JSONResponse(status_code=200, content={"ok": False, "reason": "jambonz not enabled"})

    server = settings.jambonz.sip_realm or f"voice.{os.getenv('PUBLIC_IP', 'localhost')}.nip.io"
    accounts = settings.jambonz.sip_accounts
    if not accounts:
        accounts = {settings.jambonz.sip_user: settings.jambonz.sip_password}
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "server": server,
            "transport": "UDP",
            "accounts": [{"username": u, "password": p} for u, p in accounts.items()],
            # Backward compat
            "username": settings.jambonz.sip_user,
            "password": settings.jambonz.sip_password,
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
