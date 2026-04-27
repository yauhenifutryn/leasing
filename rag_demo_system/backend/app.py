from __future__ import annotations

import asyncio
import os
import re
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
from .grounding_validator import replace_ungrounded
from .text_utils import clean_answer, clean_voice_output, contains_stop_word, iter_final_text, split_for_tts_streaming
from .memory import build_memory_block
from .profile_hygiene import filter_patches
from .classifier_schema import ClassifierOutput, parse_classifier_output
from .rag_skip import should_skip_rag
from .profile_prompts import (
    build_change_confirm_text,
    build_clarification_prompt,
    build_readback_text,
    render_calc_result,
)
from .session import ProfileState
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

# Info-seeking question markers — used by RAG-guard to prevent stale
# calculator retriggers on non-tool questions after profile is CONFIRMED.
_INFO_QUESTION_RE = re.compile(
    r"\b(кто|что|где|когда|почему|зачем|какой|какая|какое|какие|сколько|чей|чья|чьё|чьи)\b",
    re.IGNORECASE,
)


async def broadcast_sip_event(event: dict[str, Any]) -> None:
    """Fire-and-forget broadcast to all connected SIP monitor pages."""
    for ws in list(_sip_monitor_clients):
        try:
            await ws.send_json(event)
        except Exception:  # noqa: BLE001
            _sip_monitor_clients.discard(ws)


async def _speak_tts(
    text: str,
    ws: Any,
    session_id: str,
    session: Any,
    *,
    transport: str = "json",
) -> bool:
    """Unified interruptible TTS helper (Fix 36).

    Single implementation of the TTS pattern for every path that emits
    server-side speech: orchestrator read-back / clarification / change-
    confirm (via _emit_plain_assistant_response), Jambonz consent + intro
    + repeats (via _jambonz_send_tts). Bakes in all the barge-in lessons
    from Fix 25 / 33 / 35:

      1. Token ownership (Fix 33): each call stamps a fresh token on the
         session. Only the current token-holder is allowed to reset
         `assistant_speaking=False` at the end — stops an older TTS's
         late-finish from wiping a newer TTS's flag.

      2. Preemptive killAudio (Fix 33): on entry, send killAudio so any
         lingering PCM from an overlapping older TTS is flushed before we
         start pushing our own chunks.

      3. Sentence chunking (Fix 25): text is split on . ! ? … , : ; via
         `split_for_tts_streaming`. Silero synth runs per phrase (~200-
         500 ms each), so barge-in bounds to "next phrase boundary" worst
         case instead of "end of a 15 s monolithic synth".

      4. Per-chunk interrupt check: 1920-byte chunks (40 ms @ 24 kHz
         16-bit mono), `session.interrupted` re-read before each push.

      5. Playback wait (Fix 35): after all chunks are pushed, await the
         accumulated audio duration while polling `session.interrupted`.
         The downstream (Jambonz mod_audio_fork / FreeSWITCH / browser
         audio element) has 10-15 s of buffered PCM — releasing the
         speaking flag as soon as the last `send_json` returns means
         the barge-in gate flips off while the caller is STILL hearing
         the tail of the TTS. That's the bug Fix 35 exists to kill.

      6. killAudio on interrupt: flushes downstream buffer when the
         user barges in, so the caller doesn't hear a 300-500 ms tail.

    `transport` picks the wire format:
      - "json"  → `ws.send_json({type: response.output_audio.delta, ...})`
                  for the orchestrator (browser or `_JambonzWebSocketShim`).
      - "bytes" → `ws.send_bytes(chunk)` direct to a raw Jambonz audio WS.
        Used by consent / intro / repeat that run BEFORE the shim exists.

    Returns True if the call was interrupted by the user.
    """
    import base64 as _b64

    if not text:
        return False
    _my_token = uuid.uuid4().hex
    if session is not None:
        session.tts_speaker_token = _my_token
        try:
            await ws.send_text(json.dumps({"type": "killAudio"}))
        except Exception:  # noqa: BLE001
            pass
        session.assistant_speaking = True
        session.interrupted = False
        session._tts_start_time = 0

    phrases = split_for_tts_streaming(text) or [text]
    was_interrupted = False
    _push_start = asyncio.get_event_loop().time()
    _total_audio_duration = 0.0

    for phrase in phrases:
        if session is not None and session.interrupted:
            was_interrupted = True
            break
        try:
            audio_resp = await asyncio.to_thread(synthesize_audio, phrase, session_id)
        except Exception:  # noqa: BLE001
            continue
        audio_b64 = audio_resp.get("audio_b64") or ""
        if not audio_b64:
            continue
        if session is not None and session.interrupted:
            was_interrupted = True
            break
        pcm = _b64.b64decode(audio_b64)
        sample_rate = audio_resp.get("sample_rate_hz") or 24000
        chunk_size = 1920  # 40 ms @ 24 kHz 16-bit mono
        _phrase_pushed = 0
        _phrase_interrupted = False
        for i in range(0, len(pcm), chunk_size):
            if session is not None and session.interrupted:
                _phrase_interrupted = True
                break
            _end = min(i + chunk_size, len(pcm))
            _chunk = pcm[i:_end]
            try:
                if transport == "bytes":
                    await ws.send_bytes(_chunk)
                else:
                    await ws.send_json({
                        "type": "response.output_audio.delta",
                        "session_id": session_id,
                        "delta": _b64.b64encode(_chunk).decode(),
                        "sample_rate_hz": sample_rate,
                    })
                _phrase_pushed += (_end - i)
            except Exception:  # noqa: BLE001
                if session is not None:
                    session.interrupted = True
                _phrase_interrupted = True
                break
        _total_audio_duration += _phrase_pushed / float(sample_rate * 2)
        if _phrase_interrupted:
            was_interrupted = True
            break

    # Playback wait: downstream buffer plays for real-time seconds even
    # though we pushed the bytes in milliseconds. Keep assistant_speaking
    # True until the actual audio duration has elapsed (or user barges in).
    if not was_interrupted and _total_audio_duration > 0:
        _deadline = _push_start + _total_audio_duration
        while True:
            _now = asyncio.get_event_loop().time()
            if _now >= _deadline:
                break
            if session is not None and session.interrupted:
                was_interrupted = True
                break
            await asyncio.sleep(0.05)

    if was_interrupted:
        try:
            await ws.send_text(json.dumps({"type": "killAudio"}))
        except Exception:  # noqa: BLE001
            pass

    if session is not None:
        if getattr(session, "tts_speaker_token", None) == _my_token:
            session.assistant_speaking = False
            try:
                session._tts_finished_at = asyncio.get_event_loop().time()
            except Exception:  # noqa: BLE001
                pass

    return was_interrupted


async def _emit_plain_assistant_response(
    text: str,
    websocket: Any,
    session_id: str,
    *,
    backend: str = "",
    session: Any = None,
) -> None:
    """Send a plain assistant text + TTS audio response through the websocket.

    Thin wrapper over `_speak_tts` that also emits the text-delta
    (monitor / browser UI) and response.done envelope events expected by
    orchestrator callers. Barge-in behavior is identical to main LLM TTS
    and Jambonz consent TTS — all three paths share `_speak_tts`.
    """
    if not text:
        return
    await websocket.send_json({
        "type": "response.output_text.delta",
        "session_id": session_id,
        "delta": text,
    })
    await _speak_tts(
        text, websocket, session_id, session,
        transport="json",
    )
    if session is not None:
        try:
            chat_sess = state.get(session_id)
            if chat_sess is not None:
                chat_sess.transcript.append({"role": "assistant", "text": text})
        except Exception:  # noqa: BLE001
            pass
    await websocket.send_json({
        "type": "response.done",
        "session_id": session_id,
        "backend": backend,
        "used_knowledge": [],
        "citations": [],
        "timings": {},
    })


class _JambonzWebSocketShim:
    """Adapts Jambonz audio WebSocket to the interface expected by _stream_voice_response.

    Intercepts audio events: decodes base64 PCM 24kHz and sends as binary
    WebSocket frames (no resampling needed). Other events broadcast to SIP monitor.
    """

    def __init__(
        self,
        ws: WebSocket,
        session_id: str,
        control_ws: WebSocket | None = None,
        session: Any | None = None,
    ) -> None:
        self._ws = ws
        self._control_ws = control_ws
        self._session_id = session_id
        self._session = session  # VoiceSession - for interrupt checks in TTS chunk loop
        self.audio_bytes_sent = 0

    async def send_bytes(self, data: bytes) -> None:
        """Forward raw PCM bytes to the underlying Jambonz audio WebSocket."""
        await self._ws.send_bytes(data)

    async def send_text(self, text: str) -> None:
        """Forward control-channel messages (killAudio, disconnect) to Jambonz.

        Jambonz mod_audio_fork accepts JSON control frames on the audio
        websocket (see call sites for killAudio in jambonz_audio_ws,
        e.g. barge-in and consent paths). When a separate control_ws is
        supplied, prefer it (future-proofing for a dedicated control plane);
        otherwise route to the audio ws, which is the established pattern
        in this codebase and known to work for killAudio/disconnect.
        """
        target = self._control_ws if self._control_ws is not None else self._ws
        await target.send_text(text)

    async def send_json(self, data: dict[str, Any]) -> None:
        event_type = data.get("type", "")

        if event_type == "response.output_audio.delta":
            import base64 as _b64
            import json as _json
            audio_b64 = data.get("delta", "")
            if audio_b64:
                pcm_raw = _b64.b64decode(audio_b64)
                _chunk = 1920  # 40ms @ 24kHz 16-bit mono
                _sent = 0
                _interrupted = False
                for _i in range(0, len(pcm_raw), _chunk):
                    if self._session is not None and getattr(self._session, "interrupted", False):
                        _interrupted = True
                        break
                    await self._ws.send_bytes(pcm_raw[_i : _i + _chunk])
                    _sent += len(pcm_raw[_i : _i + _chunk])
                self.audio_bytes_sent += _sent
                if _interrupted:
                    # Tell mod_audio_fork to drop any buffered PCM downstream.
                    try:
                        await self.send_text(_json.dumps({"type": "killAudio"}))
                    except Exception:  # noqa: BLE001
                        pass
            return

        if event_type == "response.output_text.delta":
            # Apply clean_voice_output so the monitor displays what the caller
            # actually hears (e.g. "улица" instead of "ул."). Upstream producers
            # emit whole sentences/phrases here (not per-token), so abbreviation
            # expansion works correctly on each delta.
            _monitor_text = clean_voice_output(data.get("delta", "") or "")
            # Strip trailing TTS-pause artifacts that look ugly in written form:
            # stray commas/ellipsis at the end (from LLM pauses or interrupt-
            # truncated sentences). Spoken audio is unaffected.
            _monitor_text = re.sub(r"[,…\s]+$", "", _monitor_text)
            await broadcast_sip_event({
                "type": "sip.llm.sentence",
                "call_id": self._session_id,
                "text": _monitor_text,
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
        silence_ms = int(os.getenv("VAD_SILENCE_MS", "900"))
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

    # Per-turn reset: prevents stuck-in-calculator loops caused by
    # tool_calls_this_turn persisting across turns (see voice_session.py).
    session.reset_turn_state()

    # Monotonic stamp on each VAD-finalized utterance. apply_turn dispatch
    # rejects out-of-order classifier results via §7.2 invariant #6.
    session.latest_finalized_turn_id = (
        getattr(session, "latest_finalized_turn_id", 0) or 0
    ) + 1
    turn_id = session.latest_finalized_turn_id

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

    # SessionAgent classifier: parses the user's utterance into a
    # ClassifierOutput consumed by apply_turn. Fast-path / skip / exception
    # branches synthesise an output so the dispatch always has a non-None
    # input. `_sa_is_stop` is the only semantic flag read outside apply_turn
    # (powers the literal+classifier stop-request gate below).
    _sa_output: ClassifierOutput | None = None
    _sa_parsed: dict[str, Any] = {}
    _sa_is_stop = False
    # Fast skip: obvious non-tool messages bypass the classifier entirely (~300ms saved)
    _msg_stripped = message.strip().lower().rstrip(".!,?")

    # ── Fast-path: deterministic confirmation detection in readback states ──
    # When we're waiting for the user to confirm a readback or change-confirm
    # (state=READBACK_PENDING or CHANGE_PENDING), the classifier's job is trivial:
    # decide whether the utterance is a "yes" confirmation. A literal match on
    # common confirm words handles this in microseconds, saving the ~900ms
    # classifier round-trip on what is an extremely common turn type.
    _CONFIRM_WORDS = frozenset({
        "да", "верно", "правильно", "подтверждаю", "согласен", "согласна",
        "ок", "хорошо", "конечно", "точно", "именно", "всё верно",
        "все верно", "давайте", "давай",
    })
    _fast_confirm = False
    _current_state = None
    try:
        _current_state = session.client_profile.state
        if _current_state in (ProfileState.READBACK_PENDING, ProfileState.CHANGE_PENDING):
            # Accept multi-word phrases exactly ("всё верно"), plus single-word
            # matches among the confirm set. Cap length to avoid matching longer
            # utterances that merely contain "да" as filler.
            if len(message.split()) <= 3 and _msg_stripped in _CONFIRM_WORDS:
                _fast_confirm = True
    except Exception:  # noqa: BLE001
        pass

    if _fast_confirm:
        # Synthesise the ClassifierOutput consumed by apply_turn — its
        # confirmation logic matches on intent + is_confirmation only.
        _sa_output = ClassifierOutput.model_validate(
            {"intent": "CONVERSATION", "is_confirmation": True},
            context={"utterance": message or ""},
        )
        print(
            f"[Classifier] FAST-PATH: confirm in state={_current_state.value if _current_state else '?'} msg='{_msg_stripped}' session={session_id[:8]}",
            flush=True,
        )

    _SKIP_CLASSIFIER = {
        "спасибо", "спасибо большое", "понял", "понятно", "ясно", "ок",
        "хорошо", "ладно", "пока", "до свидания", "всего доброго",
        "привет", "здравствуйте", "добрый день", "нет", "не надо",
        "всем пока", "это всё", "больше ничего",
    }
    _skip = (
        _fast_confirm  # also cover in the existing skip flag
        or (
            _msg_stripped in _SKIP_CLASSIFIER
            and not (session.tool_calls_history or session.tool_calls_this_turn)
        )
    )
    # Skip turns bypass the classifier call; synthesise a neutral
    # ClassifierOutput so apply_turn handles them via FireLLMFallback.
    if _skip and _sa_output is None:
        _sa_output = ClassifierOutput.model_validate(
            {"intent": "CONVERSATION"},
            context={"utterance": message or ""},
        )
    print(f"[Classifier] tools={len(tool_schemas)} msg='{message[:50]}' session={session_id[:8]}{' SKIP(non-tool)' if _skip else ''}", flush=True)
    if tool_schemas and not _skip:
        # Build conversation context: last 3 turn pairs. Empirically, classifier
        # only needs the last couple of exchanges to detect intent and extract
        # hints; the full 7 pairs caused 1500-2300ms latency in production.
        _recent_turns = chat_session.transcript[-6:] if chat_session.transcript else []  # 3 pairs
        _conv_lines = []
        for _turn in _recent_turns:
            _role = "Клиент" if _turn.get("role") == "user" else "Бот"
            _text = str(_turn.get('text', '') or '')
            # Truncate long bot responses (KB answers can be 500+ chars) to keep
            # classifier input short. User turns are typically short, truncation
            # rarely hits them but keeps inputs bounded.
            if len(_text) > 200:
                _text = _text[:200].rsplit(' ', 1)[0] + '…'
            _conv_lines.append(f"{_role}: {_text}")
        _conv_context = "\n".join(_conv_lines) if _conv_lines else "начало разговора"

        _tool_history = ""
        # Whole-conversation view: prefer history (persists across turns) and
        # also include any tool calls already appended this turn.
        _all_tool_calls = session.tool_calls_history + session.tool_calls_this_turn
        if _all_tool_calls:
            _last_tools = []
            for tc in _all_tool_calls[-3:]:
                _tc_params = tc.get("params", {})
                _tc_brief = f"{tc.get('tool', '')}(ok={tc.get('ok', '?')}"
                if _tc_params.get("client_type"):
                    _tc_brief += f", client_type={_tc_params['client_type']}"
                _tc_brief += ")"
                _last_tools.append(_tc_brief)
            _tool_history = f"Инструменты в этом разговоре: {', '.join(_last_tools)}"

        _t_classify_start = time.time()
        # SessionAgent uses a dedicated small-model vLLM instance (Qwen3-4B on :8788)
        # to avoid scheduler contention with the main 35B model. Falls back to
        # effective_* when the env var is empty (single-instance mode).
        _sa_base_url = settings.llm.session_agent_base_url or effective_base_url
        _sa_model = settings.llm.session_agent_model or effective_model
        try:
            classify_resp = await asyncio.to_thread(
                call_openai_compatible,
                base_url=_sa_base_url,
                model=_sa_model,
                system_prompt=(
                    "Ты SessionAgent голосового бота лизинговой компании. "
                    "Анализируешь НОВОЕ сообщение клиента в контексте диалога.\n\n"
                    "Возвращаешь строго JSON:\n"
                    '{"intent": "TOOL"|"RAG"|"CONVERSATION",\n'
                    ' "subject": "Легковой автомобиль/Грузовой автомобиль/Спецтехника/Оборудование/Недвижимость/Прочий транспорт или null",\n'
                    ' "cost": число или null,\n'
                    ' "currency": "BYN"|"USD"|"EUR"|"RUB" или null,\n'
                    ' "client_type": "Физическое лицо"|"Юридическое лицо" или null,\n'
                    ' "condition_new": 1|0 или null,\n'
                    ' "age_years": число или null,\n'
                    ' "prepaid_pct": число (0-40) или null,\n'
                    ' "prepaid_amount": число в валюте стоимости или null,\n'
                    ' "term_months": число (12-84) или null,\n'
                    ' "type_schedule": "0"|"1" или null,\n'
                    ' "name": "имя клиента или null",\n'
                    ' "is_confirmation": true|false,\n'
                    ' "is_stop_request": true|false,\n'
                    ' "wants_readback": true|false,\n'
                    ' "change_field": "имя поля или null",\n'
                    ' "change_value": значение или null,\n'
                    ' "action": "calculate"|"recalculate"|"change_param"|"sms"|"clarify"|"clarify_client_type"|"confirm"|"invalid_param" или null}\n\n'
                    "intent=TOOL: клиент хочет посчитать / изменить расчёт / отправить СМС / подтвердить расчёт.\n"
                    "intent=RAG: информационный вопрос (офис, документы, условия общие).\n"
                    "intent=CONVERSATION: короткая реакция, подтверждение, остановка, знакомство, шутка.\n\n"
                    "ПРИОРИТЕТ intent — ОЧЕНЬ ВАЖНО:\n"
                    "Если клиент задаёт ВОПРОС о компании или общую информацию "
                    "(кто владелец/директор/учредитель/собственник, какой адрес/телефон/"
                    "график работы, когда вы открыты, чем занимаетесь, какие документы "
                    "нужны, какие условия в целом, история компании, отзывы, контакты), "
                    "это ВСЕГДА intent=RAG — даже если в предыдущих сообщениях вы "
                    "собирали параметры расчёта. Параметры в профиле сохраняются, "
                    "бот вернётся к сбору после ответа на вопрос.\n"
                    "Примеры (mid-collection drift):\n"
                    "  Бот спрашивал параметры. Клиент: 'А кто владелец компании?' -> intent=RAG\n"
                    "  Бот спрашивал параметры. Клиент: 'А кто у вас директор?' -> intent=RAG\n"
                    "  Бот спрашивал параметры. Клиент: 'А какой ваш адрес в Бресте?' -> intent=RAG\n"
                    "  Бот спрашивал параметры. Клиент: 'Когда вы работаете?' -> intent=RAG\n"
                    "intent=TOOL только когда клиент реально прогрессирует расчёт: "
                    "называет параметр (стоимость, валюта, срок, аванс, график, тип/возраст), "
                    "подтверждает, меняет, явно просит посчитать или отправить СМС. "
                    "Сам факт, что разговор шёл о расчёте, НЕ делает следующий вопрос "
                    "TOOL'ом.\n\n"
                    "ПРАВИЛА ИЗВЛЕЧЕНИЯ (только из НОВОГО сообщения, не из истории):\n"
                    "- subject: ТОЛЬКО если клиент явно назвал вид предмета словами. "
                    "НЕ угадывай между легковым/грузовым по контексту.\n"
                    "  'легковой/седан/машин\\w*/авто' + марки (BMW/Toyota/Kia/...) -> 'Легковой автомобиль'.\n"
                    "  'грузовой/грузовик/фура/тягач/самосвал/микроавтобус' -> 'Грузовой автомобиль'.\n"
                    "  'спецтехника/погрузчик/экскаватор/бульдозер/кран/трактор' -> 'Спецтехника'.\n"
                    "  'оборудование/станок/линия/установка' -> 'Оборудование'.\n"
                    "  'недвижимость/квартира/дом/здание/склад/офис' -> 'Недвижимость'.\n"
                    "  'автобус/прицеп/мотоцикл/скутер' -> 'Прочий транспорт'.\n"
                    "  Если вида нет — null. Не выбирай 'Легковой автомобиль' как default.\n"
                    "- cost: число. НЕ берёшь из истории. Если клиент не назвал число, null.\n"
                    "- currency: ВАЖНО:\n"
                    "    * 'рубли'/'рублях'/'рублей'/'BYN'/'белорусские рубли' -> 'BYN' (в Беларуси по умолчанию)\n"
                    "    * 'российские рубли'/'российский рубль'/'RUB' -> 'RUB' (приложение отклонит)\n"
                    "    * 'доллары'/'долларах'/'USD' -> 'USD' (физлицо -> конвертация в BYN 3:1)\n"
                    "    * 'евро'/'EUR' -> 'EUR' (приложение отклонит)\n"
                    "    * Без слова currency -> null\n"
                    "    Бытовое 'в рублях' без уточнения страны ВСЕГДА BYN в беларусском контексте.\n"
                    "- client_type: ТОЛЬКО если клиент ЯВНО назвал свой тип словами. "
                    "НЕ угадывай из контекста (например, 'хочу машину' НЕ значит 'Физическое лицо'). "
                    "Если явного слова нет — ставь null.\n"
                    "  Явные триггеры: 'физлицо/физическое' -> 'Физическое лицо'; "
                    "все остальные юридические формы — ИП / индивидуальный предприниматель / "
                    "самозанятый / юрлицо / юридическое / ООО / ОАО / ЗАО / организация / "
                    "компания / от компании / на компанию / бизнес / бизнесмен / микробизнес / "
                    "предприниматель -> 'Юридическое лицо'. "
                    "Калькулятор принимает только два типа: 'Физическое лицо' и 'Юридическое лицо'.\n"
                    "- condition_new: 'новый/новая/новое' -> 1; "
                    "'б/у / бу / бэу / б-у / подержанный / с пробегом / "
                    "не новый / старый / бывший в употреблении' -> 0.\n"
                    "- age_years: при 'б/у' + число лет (например, '2018 года' -> сколько лет сейчас).\n"
                    "- prepaid_pct: если клиент назвал процент (например, '20 процентов', 'двадцать процентов').\n"
                    "- prepaid_amount: если клиент назвал сумму в валюте стоимости (например, '14 тысяч рублей').\n"
                    "- Либо prepaid_pct либо prepaid_amount, не оба.\n"
                    "- term_months: срок в месяцах. "
                    "Принимай ЛЮБОЙ порядок слов: 'на 5 лет' / 'лет на 5' / "
                    "'5 лет' / 'пять лет' -> 60. 'на 7 лет' / 'лет на 7' -> 84. "
                    "'на 36 месяцев' / '36 месяцев' -> 36.\n"
                    "- type_schedule: '0' (аннуитет) или '1' (линейный). "
                    "Прямые слова: 'аннуитет/аннуитетный' -> '0'; "
                    "'линейный/убывающий/дифференцированный' -> '1'. "
                    "Семантические описания (когда клиент описывает поведение платежей "
                    "вместо названия графика): "
                    "'равные / одинаковые / фиксированные / стабильные / постоянные платежи' / "
                    "'чтобы платежи были равными' / 'чтобы платил одинаково каждый месяц' -> '0' (аннуитет). "
                    "'уменьшающиеся / убывающие / падающие платежи' / "
                    "'первый платёж больше' / 'платежи по убывающей' / "
                    "'тело долга гасится быстрее' -> '1' (линейный). "
                    "Это работает при первичном выборе и при смене (change_field='type_schedule').\n"
                    "- name: имя клиента (когда он представляется: 'меня зовут Сергей' -> 'Сергей').\n\n"
                    "СЕМАНТИЧЕСКИЕ ФЛАГИ:\n"
                    "- is_confirmation: клиент подтверждает запрос бота ('да', 'всё верно', 'правильно', 'давай', 'согласен').\n"
                    "- is_stop_request: true ТОЛЬКО если клиент явно просит ассистента прекратить говорить или молчать. "
                    "Примеры true: 'стоп', 'замолчи', 'помолчи, я думаю', 'подожди секунду, не говори', 'тихо, я сам скажу'. "
                    "Примеры false: 'а подожди, ладно, неважно, а можно машину...' (подожди как маркер речи), "
                    "'ну стоп, давай разберёмся' (стоп как эмоция, не команда), 'нажмите стоп-кран' (о другом), "
                    "'ну и что?', 'алло', 'не слышала?', 'в нашем разговоре уже'. "
                    "Если сомневаешься — false.\n"
                    "- wants_readback: клиент просит повторить параметры ('повтори', 'какие параметры', 'что у нас').\n"
                    "- change_field/change_value: клиент меняет параметр. "
                    "'поменяй срок на 48' -> change_field='term_months', change_value=48. "
                    "'давай без аванса' -> change_field='prepaid_pct', change_value=0. "
                    "'в долларах' -> change_field='currency', change_value='USD'.\n\n"
                    "РАЗРЕШЕНИЕ ССЫЛОК НА ТВОЙ ПРЕДЫДУЩИЙ ОТВЕТ:\n"
                    "Если клиент использует ссылку на твоё предыдущее сообщение в Диалоге "
                    "('тот', 'та', 'тот же', 'такой же', 'что вы сказали', 'что дешевле/"
                    "выгоднее/лучше/проще/удобнее/быстрее', 'первый/второй вариант'), "
                    "посмотри внимательно в Диалог: ЧТО именно ты только что сказал об этом "
                    "параметре, и извлеки соответствующее значение в change_field/change_value "
                    "(или в обычное поле, если параметр ещё не был зафиксирован). "
                    "Это семантическое разрешение ссылок: ты сам сравнил варианты и описал "
                    "их свойства в предыдущей реплике — теперь используй это для понимания "
                    "выбора клиента. Работает для ЛЮБОГО параметра, не только графика.\n"
                    "Пример: ты сказал 'линейный график обычно дает меньшую переплату, "
                    "так как тело долга гасится быстрее. Аннуитет проще для планирования "
                    "бюджета.' Клиент говорит: 'Давай тот, что дешевле.' -> "
                    "change_field='type_schedule', change_value='1' (линейный).\n"
                    "Пример: ты сказал 'обычно для физлиц лучше 30%, для бизнеса 20%'. "
                    "Клиент: 'Давай как для бизнеса.' -> change_field='prepaid_pct', "
                    "change_value=20.\n"
                    "Если в Диалоге ты НЕ давал такого сравнения — change_value=null, "
                    "пусть код спросит уточнение.\n\n"
                    "БИЗНЕС-ПРАВИЛА:\n"
                    "- Физлица: только легковой автомобиль и прочий транспорт. Грузовые, спецтехника, "
                    "оборудование, недвижимость только для ИП/юрлиц.\n"
                    "- Рекомендуемые диапазоны: prepaid_pct 0-40; term_months 12-84.\n"
                    "- КРИТИЧЕСКИ ВАЖНО: term_months, prepaid_pct, prepaid_amount, cost, age_years "
                    "всегда извлекай ТОЧНО как назвал клиент в НОВОМ сообщении, даже если значение "
                    "вне рекомендуемого диапазона. НЕ подменяй, НЕ округляй, НЕ кламповай к границам. "
                    "Если клиент сказал '5 месяцев' -> term_months=5, НЕ 12 и НЕ 60. "
                    "Если сказал '110 процентов' -> prepaid_pct=110, НЕ 40. "
                    "Если сказал '200 месяцев' -> term_months=200, НЕ 84. "
                    "Если сказал 'минус сто тысяч' -> cost=-100000. "
                    "Валидация диапазонов и сообщение клиенту — задача кода, не твоя.\n"
                    "- Если клиент хочет расчёт, но client_type неизвестен из всего диалога, "
                    "ставь action='clarify'.\n\n"
                    "Только JSON, никаких пояснений."
                ),
                user_prompt=f"{_tool_history}\n\nДиалог:\n{_conv_context}\n\nНОВОЕ сообщение: {message}",
                temperature=0.0,
                max_tokens=160,
                timeout_sec=4,
            )
            _raw = classify_resp.text.strip()
            # CP-2.2: route raw classifier text through ClassifierOutput.
            # parse_classifier_output never raises; returns empty model on
            # JSON / validation failure. Utterance is passed as context so
            # the post-validator nulls enum fields lacking a cue in the user
            # utterance (E1/E2 fixes). Legacy call sites below keep reading
            # _sa_parsed.get(...) — the dict is now the Pydantic-validated,
            # utterance-grounded shape.
            _sa_output = parse_classifier_output(_raw, message or "")
            _sa_parsed = _sa_output.model_dump(exclude_none=True)
            try:
                import json as _json
                _sa_summary = {k: v for k, v in _sa_parsed.items() if v not in (None, "", [], {})}
                print(f"[SessionAgent] raw={_json.dumps(_sa_summary, ensure_ascii=False)[:250]} session={session_id[:8]}", flush=True)
            except Exception:  # noqa: BLE001
                pass
            # Stop-request flag (sole semantic field consumed outside apply_turn).
            _sa_is_stop = bool(_sa_parsed.get("is_stop_request"))
        except Exception as _classify_exc:
            print(f"[Classifier] ERROR: {_classify_exc}", flush=True)
            # apply_turn requires a non-None ClassifierOutput; synthesize a
            # neutral CONVERSATION fallback so the dispatch reaches
            # FireLLMFallback rather than silently dropping the turn.
            _sa_output = ClassifierOutput.model_validate(
                {"intent": "CONVERSATION"},
                context={"utterance": message or ""},
            )
        _t_classify_ms = (time.time() - _t_classify_start) * 1000
        print(f"[Classifier] result: ({_t_classify_ms:.0f}ms)", flush=True)

    # --- Handle semantic stop-request: hybrid gate (literal + classifier) ---
    if _sa_is_stop and contains_stop_word(message or ""):
        print(f"[SessionAgent] is_stop_request AND literal match -> listen_mode", flush=True)
        _listen_timeout = settings.turn_taking.listen_mode_timeout_sec
        session.listen_mode = True
        session.listen_mode_until = time.time() + _listen_timeout
        session.interrupted = True
        session.assistant_speaking = False
        try:
            await websocket.send_text(json.dumps({"type": "killAudio"}))
        except Exception as _killexc:
            print(f"[listen_mode] killAudio failed: {_killexc}", flush=True)
        # Cancel any previous auto-exit task to prevent orphaned coroutines (re-entry safety).
        if session.listen_mode_task and not session.listen_mode_task.done():
            session.listen_mode_task.cancel()
        # Spawn the auto-exit task (Task 5 module)
        from .listen_mode import spawn_auto_exit_task
        session.listen_mode_task = spawn_auto_exit_task(session, websocket, session_id)
        return  # no LLM response; bot goes silent
    elif _sa_is_stop and not contains_stop_word(message or ""):
        print(f"[SessionAgent] is_stop_request TRUE but no literal stop-word in '{(message or '')[:60]}' -> ignored", flush=True)

    # --- apply_turn / execute_action dispatch ---
    # Sole orchestration path. apply_turn (§7.2) computes the next TurnAction
    # from the parsed ClassifierOutput + current ProfileState; execute_action
    # consumes the action and drives TTS / tools. _sa_output is guaranteed
    # non-None at this point (classifier success, fast-path/skip synthesis,
    # or exception fallback above).
    #
    # §7.2 invariant #6 — stale-result guard. Another utterance finalised
    # ahead of this one can bump `latest_finalized_turn_id` beyond our
    # stamped `turn_id` during any pending await above; drop in that case
    # so a stale dispatch cannot clobber fresh state.
    if turn_id < session.latest_finalized_turn_id:
        print(f"[apply_turn] stale turn_id={turn_id} < latest={session.latest_finalized_turn_id}; drop", flush=True)
        return

    from .execute_adapters import (
        LLMStreamBackend, TtsSink, CalcAdapter, RagFuture,
    )
    from .turn_dispatcher import apply_turn, execute_action

    _llm_backend = LLMStreamBackend(
        base_url=effective_base_url,
        model=effective_model,
        temperature=settings.llm.temperature,
        max_tokens=200,
        timeout_sec=settings.llm.timeout_sec,
        system_prompt=system_prompt,
    )
    _tts_sink = TtsSink(
        websocket=websocket,
        session_id=session_id,
        session=session,
        rtc_handler=rtc_handler,
    )
    _calc_adapter = CalcAdapter(
        session_id=session_id,
        client_phone=session.client_phone,
    )
    _rag_future_adapter = RagFuture(_rag_task)

    # Codex CP-3.6 P2: stamp the recent-dialogue memory block on the
    # session so execute_action's FireLLMFallback handler can prepend
    # it to the LLM prompt — keeps prior-turn context for follow-up
    # RAG / conversation turns. Mirrors the legacy RAG path's prompt
    # construction.
    session.memory_block = memory_block

    _action = apply_turn(
        session.client_profile, _sa_output, message or "",
        turn_id=turn_id,
    )
    print(f"[apply_turn] turn_id={turn_id} action={type(_action).__name__}", flush=True)

    # Bug K (live call 69941ab4 2026-04-26) — broadcast snapshot IMMEDIATELY
    # after apply_turn returns, before execute_action plays TTS audio.
    # apply_turn has already done all state-machine work (step 1 apply-change,
    # step 5/5a captures, step 6 USD→BYN preflight). Without this earlier
    # broadcast, the monitor UI only updates after `await_playback_drain`
    # — i.e. lagged by a full TTS duration on every turn.
    try:
        _p_imm = session.client_profile
        _conv_cost_imm = None
        _conv_cur_imm = None
        if (
            _p_imm.cost is not None
            and (_p_imm.currency or "").upper() == "USD"
            and "Физическое" in str(_p_imm.client_type or "")
        ):
            try:
                from .profile_prompts import _get_usd_byn_rate
                _r_imm = _get_usd_byn_rate()
                _conv_cost_imm = round(float(_p_imm.cost) * float(_r_imm), 2)
                _conv_cur_imm = "BYN"
            except Exception:
                pass
        await broadcast_sip_event({
            "type": "sip.profile.snapshot",
            "call_id": session_id,
            "state": _p_imm.state.value,
            "fields": {
                "name": _p_imm.name,
                "subject": _p_imm.subject,
                "cost": _p_imm.cost,
                "currency": _p_imm.currency,
                "client_type": _p_imm.client_type,
                "condition_new": _p_imm.condition_new,
                "age_years": getattr(_p_imm, "age_years", None),
                "term_months": _p_imm.term_months,
                "prepaid_pct": _p_imm.prepaid_pct,
                "prepaid_amount": _p_imm.prepaid_amount,
                "type_schedule": _p_imm.type_schedule,
            },
            "original_cost": getattr(_p_imm, "original_cost", None),
            "original_currency": getattr(_p_imm, "original_currency", None),
            "converted_cost": _conv_cost_imm,
            "converted_currency": _conv_cur_imm,
            "missing": sorted(_p_imm.missing_fields()),
        })
    except Exception:  # noqa: BLE001
        pass

    session.assistant_speaking = True
    session.interrupted = False
    _chunks: list[str] = []
    try:
        async for _chunk in execute_action(
            _action,
            ws=websocket,
            session=session,
            backend=_llm_backend,
            tts=_tts_sink,
            calc=_calc_adapter,
            rag_future=_rag_future_adapter,
        ):
            _chunks.append(_chunk)
        # Bug 3 fix: hold assistant_speaking=True through the actual audio
        # playback duration. mod_audio_fork still has buffered PCM after
        # the last yield; without this drain the VAD barge-in gate (which
        # keys on assistant_speaking) flips off while the caller is still
        # hearing the tail.
        await _tts_sink.await_playback_drain()
    finally:
        session.assistant_speaking = False

    _full_answer = " ".join(_chunks).strip()
    if session.interrupted and _full_answer:
        _full_answer += " [прервано клиентом]"
    if _full_answer:
        _append_turn(chat_session, message, _full_answer, settings.app.memory_turns)
        state.update(chat_session)
        session.turn_count += 1
        # Bug 55 — FINAL broadcast so the monitor consistently shows the
        # assembled assistant message.
        try:
            await broadcast_sip_event({
                "type": "sip.llm.final",
                "call_id": session_id,
                "text": _full_answer,
                "interrupted": bool(session.interrupted),
            })
        except Exception:  # noqa: BLE001
            pass

    # Issue #4 (live call cdbcf56b 2026-04-26) — post-dispatch snapshot.
    # Re-broadcast after execute_action completes so the UI reflects any
    # state transitions (CHANGE_PENDING after EmitChangeConfirm, CONFIRMED
    # after FireCalc, profile mutations from step 1/5/5a).
    try:
        _p_post = session.client_profile
        _ts_post = _p_post.type_schedule if _p_post.type_schedule is not None else '-'
        _conv_cost_post = None
        _conv_cur_post = None
        if (
            _p_post.cost is not None
            and (_p_post.currency or "").upper() == "USD"
            and "Физическое" in str(_p_post.client_type or "")
        ):
            try:
                from .profile_prompts import _get_usd_byn_rate
                _r = _get_usd_byn_rate()
                _conv_cost_post = round(float(_p_post.cost) * float(_r), 2)
                _conv_cur_post = "BYN"
            except Exception:
                pass
        await broadcast_sip_event({
            "type": "sip.profile.snapshot",
            "call_id": session_id,
            "state": _p_post.state.value,
            "fields": {
                "name": _p_post.name,
                "subject": _p_post.subject,
                "cost": _p_post.cost,
                "currency": _p_post.currency,
                "client_type": _p_post.client_type,
                "condition_new": _p_post.condition_new,
                "age_years": getattr(_p_post, "age_years", None),
                "term_months": _p_post.term_months,
                "prepaid_pct": _p_post.prepaid_pct,
                "prepaid_amount": _p_post.prepaid_amount,
                "type_schedule": _p_post.type_schedule,
            },
            "original_cost": getattr(_p_post, "original_cost", None),
            "original_currency": getattr(_p_post, "original_currency", None),
            "converted_cost": _conv_cost_post,
            "converted_currency": _conv_cur_post,
            "missing": sorted(_p_post.missing_fields()),
        })
    except Exception:  # noqa: BLE001
        pass

    try:
        await websocket.send_json({
            "type": "response.done",
            "session_id": session_id,
            "backend": backend,
            "used_knowledge": [],
            "citations": [],
            "timings": {},
        })
    except (RuntimeError, WebSocketDisconnect):
        pass
    return


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
            silence_ms = int(os.getenv("VAD_SILENCE_MS", "900"))
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
        if session.listen_mode_task and not session.listen_mode_task.done():
            session.listen_mode_task.cancel()
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
    """Synthesize text and send audio to a raw Jambonz audio WebSocket.

    Fix 36: thin wrapper around the unified `_speak_tts` helper. Consent,
    intro, and repeat TTS now get the exact same barge-in behavior as the
    orchestrator readback and the main LLM path — token ownership,
    sentence chunking, playback wait, killAudio on interrupt. The only
    Jambonz-specific bits left here are the `clean_voice_output` call
    (abbreviation expansion), the wire format (`transport='bytes'` so
    PCM goes out via `ws.send_bytes` instead of JSON deltas), and the
    `sip.tts.start` broadcast to the monitor.
    """
    text = clean_voice_output(text)
    if not text:
        return
    print(f"[Jambonz:{session_id[:8]}] TTS synthesizing: {text[:40]}...", flush=True)
    # Issue (live call ec87a8e1 2026-04-26): Whisper hallucinated "Микро
    # Лизинг" on residual audio at 02:52:42, slipped through echo_filter
    # because the consent + intro text spoken via this helper was never
    # appended to chat_session.transcript — echo_filter only sees the
    # last 4 transcript turns, so company name / "Ксения" / "голосовая
    # помощница" said at greeting time were invisible to the filter.
    # Append the spoken text to transcript here so the existing
    # is_echo() substring path catches future hallucinations of these
    # phrases.
    try:
        _chat_sess = state.get(session_id)
        if _chat_sess is not None:
            _chat_sess.transcript.append({"role": "assistant", "text": text})
    except Exception:  # noqa: BLE001
        pass
    try:
        await _speak_tts(text, ws, session_id, session, transport="bytes")
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(
            f"[Jambonz:{session_id[:8]}] TTS error: {exc}\n{traceback.format_exc()}",
            flush=True,
        )
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
    # Issue 8 fix (live 77cfa127 2026-04-25): the previous in-line
    # word-overlap check at threshold 0.6 false-positived legitimate
    # short user replies that echo bot vocabulary (user "новая машина в
    # долларах" answering bot's "новая или б/у... в рублях или
    # долларах?"). New logic in backend/echo_filter.is_echo:
    #   - Substring of bot speech → echo (unchanged).
    #   - Short user utterances (≤ 5 words) skip overlap check entirely.
    #   - Long utterances need ≥ 0.85 overlap (was 0.6).
    _chat_sess = state.get(session_id)
    if _chat_sess and _chat_sess.transcript:
        _recent_bot = " ".join(
            t.get("text", "") for t in _chat_sess.transcript[-4:]
            if t.get("role") == "assistant"
        )
        from .echo_filter import is_echo as _is_echo_fn
        if _is_echo_fn(text, _recent_bot):
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

    jambonz_ws = _JambonzWebSocketShim(ws, session_id, session=session)
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
        silence_ms = int(os.getenv("VAD_SILENCE_MS", "900"))
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
            if session.listen_mode_task and not session.listen_mode_task.done():
                session.listen_mode_task.cancel()
            _state_dir = Path(__file__).resolve().parents[1] / ".state"
            try:
                chat_session = state.get(session_id)
                if chat_session and chat_session.transcript:
                    from .session_analyzer import save_transcript
                    # Include tool_calls (calculator + SMS invocations) so the
                    # saved transcript carries the same history the live SIP
                    # monitor shows. tool_calls_this_turn may still hold the
                    # current turn's call if the WS dropped mid-turn.
                    _tool_calls = (
                        list(getattr(session, "tool_calls_history", []) or [])
                        + list(getattr(session, "tool_calls_this_turn", []) or [])
                    )
                    save_transcript(
                        session_id,
                        chat_session.transcript,
                        _state_dir,
                        transport="jambonz",
                        phone=session.client_phone or "",
                        tool_calls=_tool_calls,
                    )
                    print(
                        f"[Jambonz:{session_id[:8]}] Transcript saved "
                        f"({len(chat_session.transcript)} turns, "
                        f"{len(_tool_calls)} tool calls)",
                        flush=True,
                    )
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
