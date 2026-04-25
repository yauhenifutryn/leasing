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


def _sticky_calc_ready(
    profile: Any,
    sa_is_confirm: bool,
    needs_tool: bool,
    just_confirmed_this_turn: bool = False,
) -> bool:
    """Return True when the sticky-calc direct-call path should fire.

    is_confirmation=true unlocks recalc in two situations:
      1. State is READBACK_PENDING or CHANGE_PENDING (user about to confirm).
      2. State has JUST transitioned to CONFIRMED this turn (gates 3/4 fired).

    On plain CONFIRMED with no pending change and no transition this turn,
    utterances like "Хорошо." / "Ладно, спасибо." get classifier
    is_confirmation=true but are just acknowledgments — they must NOT
    re-fire the calculator (Fix 39 regression guard).
    """
    try:
        state = profile.state
    except AttributeError:
        return False
    confirm_unlocks = bool(sa_is_confirm) and (
        state in (ProfileState.READBACK_PENDING, ProfileState.CHANGE_PENDING)
        or bool(just_confirmed_this_turn)
    )
    return (
        profile.is_complete_for_calc()
        and (profile.confirmed_at is not None or confirm_unlocks)
        and (bool(needs_tool) or confirm_unlocks)
    )


def _format_invalid_params_msg(markers: list[str]) -> str | None:
    """Translate calculator param_out_of_range / param_bad_type markers to
    a single user-facing Russian message.

    Returns None if no OOR / bad_type markers are present (markers from
    IncompleteProfileError may also include plain missing-field names).
    """
    if not markers:
        return None
    parts: list[str] = []
    _FIELD_RU = {
        "cost": "стоимость",
        "term": "срок",
        "prepaid_pct": "аванс",
        "prepaid_amount": "сумма аванса",
        "age": "возраст предмета",
    }
    for m in markers:
        if not isinstance(m, str):
            continue
        if m.startswith("param_out_of_range:"):
            body = m.split(":", 1)[1]
            field_part, *rest = body.split(":")
            field_name = field_part.split("=")[0]
            ru = _FIELD_RU.get(field_name, field_name)
            mins = next((r.split("=", 1)[1] for r in rest if r.startswith("min=")), None)
            maxs = next((r.split("=", 1)[1] for r in rest if r.startswith("max=")), None)
            if field_name == "term":
                parts.append(f"срок должен быть от {mins} до {maxs} месяцев")
            elif field_name == "prepaid_pct":
                parts.append(f"аванс должен быть от {mins} до {maxs} процентов")
            elif field_name == "cost":
                parts.append(f"стоимость должна быть положительной (до {maxs})")
            elif field_name == "prepaid_amount":
                if maxs == "cost" or not maxs.replace(".", "", 1).isdigit():
                    parts.append("сумма аванса должна быть больше нуля и меньше стоимости")
                else:
                    parts.append(f"сумма аванса не может превышать стоимость ({maxs})")
            elif field_name == "age":
                parts.append(f"возраст предмета должен быть от {mins} до {maxs} лет")
            else:
                parts.append(f"{ru} вне допустимого диапазона ({mins}-{maxs})")
        elif m.startswith("param_bad_type:"):
            body = m.split(":", 1)[1]
            field_name = body.split("=", 1)[0]
            ru = _FIELD_RU.get(field_name, field_name)
            parts.append(f"{ru}: не распознано как число")
    if not parts:
        return None
    return "; ".join(parts)


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

    # Phase 3.D: monotonic stamp on each VAD-finalized utterance. Used by
    # the APPLY_TURN_ENABLED path to reject out-of-order classifier
    # results (§7.2 invariant #6). Legacy path ignores it — safe to stamp
    # unconditionally.
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

    # Bug 4 (live call 6dd5880b 2026-04-25) — affirmation-after-calc path.
    # Bare "Да" / "Да, открой" right after the bot's post-calc SMS prompt was
    # not recognised, so the direct-send path skipped and Qwen3.5 hallucinated
    # "график отправлен" without invoking send_sms. detect_sms_intent now
    # gates on recent successful calc + short affirmation. See sms_intent.py.
    from .sms_intent import detect_sms_intent
    # Bug 15 (live call 1cae210d, 2026-04-25) — pass profile_state so
    # detect_sms_intent rejects "Да" while READBACK_PENDING or
    # CHANGE_PENDING is in flight. Without this, confirming a
    # CHANGE_PENDING ("Да" to "Меняю график на аннуитет, всё верно?")
    # fired SMS with stale (pre-change) calculator output instead of
    # re-running the calculator with the new schedule.
    _sms_intent_state = None
    try:
        _sms_intent_state = session.client_profile.state
    except Exception:  # noqa: BLE001
        pass
    has_sms_intent = detect_sms_intent(
        message,
        list(session.tool_calls_history) + list(session.tool_calls_this_turn),
        profile_state=_sms_intent_state,
    )
    # "session-wide" view: switch to cumulative history since tool_calls_this_turn
    # is reset at turn start (see VoiceSession.reset_turn_state).
    tools_used_in_session = bool(session.tool_calls_history) or bool(session.tool_calls_this_turn)

    # Smart intent classifier: sees last 7 turns of conversation, extracts
    # structured data (subject, cost, currency) for immediate tool calling.
    needs_tool = False
    _extracted_hints: dict[str, Any] = {}
    # Initialize at function scope so the skip-RAG gate (and other post-classifier
    # handlers) can safely reference it when fast-skip bypasses classification
    # or classifier parsing fails.
    _profile_patches: dict[str, Any] = {}
    # SessionAgent semantic flags: initialize at function scope so post-classifier
    # handlers can safely reference them even when fast-skip bypasses classification
    # or classifier parsing fails.
    _sa_is_stop = False
    _sa_is_confirm = False
    _sa_wants_readback = False
    _sa_change_field = None
    _sa_change_value = None
    # Phase 3.D: holds the parsed ClassifierOutput once the classifier
    # returns. Pre-declared at function scope so the APPLY_TURN_ENABLED
    # dispatch can read it even if the classifier API call failed (the
    # except block leaves `_sa_output` as None and the apply-turn branch
    # falls through to the legacy keyword heuristic).
    _sa_output = None
    # Fix 29 helper: True when the classifier-output block staged a fresh
    # pending_change this turn (via explicit change_field or implicit delta
    # detection from Fix 30). Read by the always-on state gate to decide
    # whether to emit the change-confirm prompt on a non-confirm turn.
    _change_staged_this_turn = False
    # Fix 42b (retry): track fields patched this turn for the collect-phase
    # clarify gate below. Narrower than the first attempt: only fires for
    # CONVERSATION intent (TOOL intent is handled by Gate 1 downstream).
    _changed_this_turn: dict[str, Any] = {}
    # Fix 42e: initialize classifier-parse result at function scope. When the
    # fast-path confirm fires (state=READBACK_PENDING + "да"/"верно"/etc.),
    # the classifier block is skipped, which previously left `_sa_parsed`
    # undefined. Downstream references (Fix 42b uses it for intent check)
    # then raised NameError, aborting the turn silently — so the "Всё верно"
    # confirmation never reached _sticky_calc_ready and calc didn't fire
    # (session bd150fd3, 2026-04-18 20:15).
    _sa_parsed: dict[str, Any] = {}
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
        needs_tool = False  # intent=CONVERSATION, is_confirmation handles the rest
        _sa_is_confirm = True
        # Synthesise a minimal ClassifierOutput for the APPLY_TURN_ENABLED
        # path (apply_turn's confirmation logic matches on intent + the
        # `is_confirmation` flag — no other fields are needed). Without
        # this, the fast-path skips the classifier call block where
        # `_sa_output` is normally populated, so the flag gate at
        # app.py:1659 sees `_sa_output is None` and falls through to the
        # legacy 5-gate block. Live regression ac0e35d6 turn 12: the
        # "Да" confirmation bypassed apply_turn silently and legacy
        # DirectTool ran the recalc — which hid the regression behind
        # legacy's safety net.
        _sa_output = ClassifierOutput.model_validate(
            {"intent": "CONVERSATION", "is_confirmation": True},
            context={"utterance": message or ""},
        )
        print(
            f"[Classifier] FAST-PATH: confirm in state={_current_state.value if _current_state else '?'} msg='{_msg_stripped}' session={session_id[:8]}",
            flush=True,
        )
        # Downstream: the APPLY_TURN_ENABLED gate handles the state
        # transition via apply_turn step 1/2; legacy path fallback
        # remains via Gate 3 (READBACK_PENDING) / Gate 4 (CHANGE_PENDING).

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
    # Phase 3.D fix: _skip bypasses the classifier call where `_sa_output`
    # is normally populated. The APPLY_TURN_ENABLED gate below requires
    # `_sa_output is not None`, so any skipped turn falls silently to
    # legacy. Synthesise a minimal ClassifierOutput here so apply_turn
    # handles the skipped turn through FireLLMFallback (intent=CONVERSATION,
    # no confirmation, no captured fields). Preserves the fast-path
    # confirmation signal set above when that branch already synthesised.
    if _skip and _sa_output is None:
        _sa_output = ClassifierOutput.model_validate(
            {"intent": "CONVERSATION", "is_confirmation": bool(_sa_is_confirm)},
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
                    "- term_months: срок в месяцах. 'на 5 лет' -> 60. 'на 7 лет' -> 84.\n"
                    "- type_schedule: 'аннуитет/аннуитетный' -> '0'; 'линейный/убывающий/дифференцированный' -> '1'.\n"
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
            _intent = str(_sa_parsed.get("intent", "")).upper()
            needs_tool = _intent == "TOOL"
            # CP-2.2 fix (Codex adversarial 2026-04-20): `_sa_parsed` always
            # contains the three bool defaults (is_confirmation/is_stop_request/
            # wants_readback serialize even on an empty model), so the old
            # `if not _sa_parsed` emptiness check was unreachable. Route via
            # the Pydantic model's intent field — None only on parse/validation
            # failure — so the raw-text TOOL grep fallback fires again under
            # format drift.
            if _sa_output.intent is None:
                needs_tool = "TOOL" in _raw.upper()
            # Legacy hint extraction for the existing DirectTool path
            if _sa_parsed.get("subject"):
                _extracted_hints["subject"] = _sa_parsed["subject"]
            if _sa_parsed.get("cost") is not None:
                _extracted_hints["cost"] = _sa_parsed["cost"]
            if _sa_parsed.get("currency"):
                # Fix 42a: normalize at hint extraction — classifier sometimes
                # emits RUB for bare "рублей" (which means BYN in Belarus
                # context). Without normalization the staging extras path
                # uses the raw RUB hint and silently flips currency.
                from .profile_hygiene import _normalize_currency as _norm_cur
                _raw_cur = _sa_parsed["currency"]
                _norm = _norm_cur(_raw_cur, message or "")
                _extracted_hints["currency"] = _norm if _norm else _raw_cur
            if _sa_parsed.get("client_type"):
                _extracted_hints["client_type"] = _sa_parsed["client_type"]
            # prepaid: prefer pct, fallback to amount
            if _sa_parsed.get("prepaid_pct") is not None:
                _extracted_hints["prepaid"] = _sa_parsed["prepaid_pct"]
                _extracted_hints["prepaid_pct"] = _sa_parsed["prepaid_pct"]
            if _sa_parsed.get("prepaid_amount") is not None:
                _extracted_hints["prepaid_amount"] = _sa_parsed["prepaid_amount"]
            if _sa_parsed.get("term_months") is not None:
                _extracted_hints["term"] = _sa_parsed["term_months"]
                _extracted_hints["term_months"] = _sa_parsed["term_months"]
            if _sa_parsed.get("condition_new") is not None:
                _extracted_hints["condition_new"] = _sa_parsed["condition_new"]
            if _sa_parsed.get("age_years") is not None:
                _extracted_hints["age_years"] = _sa_parsed["age_years"]
            if _sa_parsed.get("type_schedule"):
                _extracted_hints["type_schedule"] = _sa_parsed["type_schedule"]
            if _sa_parsed.get("action"):
                _extracted_hints["action"] = _sa_parsed["action"]
            # Semantic flags (new)
            _sa_is_stop = bool(_sa_parsed.get("is_stop_request"))
            _sa_is_confirm = bool(_sa_parsed.get("is_confirmation"))
            _sa_wants_readback = bool(_sa_parsed.get("wants_readback"))
            _sa_change_field = _sa_parsed.get("change_field")
            _sa_change_value = _sa_parsed.get("change_value")
            _sa_name = _sa_parsed.get("name")
            # ── Apply to ClientProfile (single source of truth) ──
            _profile_patches: dict[str, Any] = {}
            _profile_current = session.client_profile
            _sa_change_field_val = _sa_parsed.get("change_field")

            # ── Sticky-patch protection: both identity + numeric fields ──
            # Once any calculator-relevant field is captured, the classifier
            # must not silently overwrite it mid-session. The classifier
            # sometimes re-emits stale values from its context window (values
            # from an earlier, already-confirmed calculation) on short turns
            # such as "А можно ли физическое лицо?", which would regress the
            # whole profile back to the older values.
            #
            # Accept a new value only when:
            #   (a) current value is unset (first capture), OR
            #   (b) user's explicit change_field matches this field, OR
            #   (c) new value equals current (no-op re-emit).
            #
            # Fix 21 — identity fields (client_type, subject, condition_new):
            # prevents classifier drift between types (Физ↔Юр, Легк↔Груз).
            # Fix 24 — numeric + currency fields: prevents Phase-A numeric
            # values from regressing over Phase-B values on classifier
            # re-emission turns (see session e4eb325c postmortem).
            _STICKY_IDENTITY_FIELDS = ("client_type", "subject", "condition_new")
            _STICKY_NUMERIC_FIELDS = (
                "cost", "age_years", "prepaid_pct", "prepaid_amount",
                "term_months", "type_schedule", "currency",
            )
            from .profile_hygiene import has_field_signal as _has_signal_sticky
            for _field in _STICKY_IDENTITY_FIELDS + _STICKY_NUMERIC_FIELDS:
                _new_val = _sa_parsed.get(_field)
                if _new_val is None or _new_val == "":
                    continue
                _current_val = getattr(_profile_current, _field, None)
                _is_first_capture = _current_val in (None, "")
                _is_explicit_change = (_sa_change_field_val == _field)
                # prepaid_pct and prepaid_amount share semantic slot "prepaid":
                # a user-requested change via change_field="prepaid_pct" should
                # also unlock prepaid_amount emission on the same turn, and
                # vice-versa.
                if not _is_explicit_change and _field in ("prepaid_pct", "prepaid_amount"):
                    if _sa_change_field_val in ("prepaid_pct", "prepaid_amount", "prepaid"):
                        _is_explicit_change = True
                # Fix 40a + 42d: in COLLECTING state, apply patches directly
                # (first-capture, explicit_change, or live signal) so profile
                # builds up incrementally. In CONFIRMED / CHANGE_PENDING state,
                # DO NOT apply directly — even first_capture for a cleared
                # counterpart (e.g. prepaid_pct=None after switching to
                # prepaid_amount) must route through pending_change. Direct-
                # apply in CONFIRMED bypasses change-confirm:
                #   session a685ce41: "Давай сменим всё" fired calc without prompt
                #   session a7b9803f: multi-field change omitted prepaid from confirm
                _allow_direct_apply = _profile_current.state == ProfileState.COLLECTING
                _has_live_signal = False
                if _allow_direct_apply and not _is_first_capture and not _is_explicit_change:
                    try:
                        _has_live_signal = _has_signal_sticky(_field, _new_val, message or "")
                    except Exception:  # noqa: BLE001
                        _has_live_signal = False
                if _allow_direct_apply and (_is_first_capture or _is_explicit_change or _has_live_signal):
                    _profile_patches[_field] = _new_val
                    if _has_live_signal and not _is_explicit_change and not _is_first_capture:
                        print(
                            f"[Profile] multi-field unlock {_field}='{_new_val}' "
                            f"(was '{_current_val}', literal signal in utterance)",
                            flush=True,
                        )
                elif not _allow_direct_apply and (_is_first_capture or _is_explicit_change):
                    # In CONFIRMED/CHANGE_PENDING, don't apply directly — let
                    # the staging block pick this up via _extracted_hints and
                    # route through pending_change.
                    print(
                        f"[Profile] CONFIRMED-state: deferring {_field}='{_new_val}' "
                        f"(was '{_current_val}') to staging",
                        flush=True,
                    )
                elif _new_val != _current_val:
                    print(
                        f"[Profile] stale {_field} patch ignored: "
                        f"'{_new_val}' (already have '{_current_val}', "
                        f"no explicit change_field={_sa_change_field_val!r})",
                        flush=True,
                    )

            # Fix 40c: prepaid slot sharing — when user switches between
            # percentage and absolute amount (or vice versa), the counterpart
            # field must be cleared so direct-call params don't shadow the
            # fresh value with the stale one. Without this, user says
            # "аванс 80 тысяч рублей" → profile now has prepaid_amount=80000
            # but prepaid_pct=20 (old) is still set, and direct-call prefers
            # prepaid_pct, so the calc uses 20% instead of 80000 BYN.
            # apply_patches ignores None values, so null the counterpart
            # directly on the profile object before merging the new value.
            if "prepaid_pct" in _profile_patches and getattr(_profile_current, "prepaid_amount", None) is not None:
                print(f"[Profile] clearing prepaid_amount (prepaid_pct={_profile_patches['prepaid_pct']} takes over)", flush=True)
                _profile_current.prepaid_amount = None
            elif "prepaid_amount" in _profile_patches and getattr(_profile_current, "prepaid_pct", None) is not None:
                print(f"[Profile] clearing prepaid_pct (prepaid_amount={_profile_patches['prepaid_amount']} takes over)", flush=True)
                _profile_current.prepaid_pct = None

            # Stale-name guard: once profile.name is set, ignore further name
            # patches (classifier hallucinates names from nouns in mid-call).
            if _sa_name and not (session.client_profile.name or "").strip():
                _profile_patches["name"] = _sa_name
            elif _sa_name:
                print(f"[Profile] stale name patch ignored: '{_sa_name}' (already have '{session.client_profile.name}')", flush=True)

            # Issue 7 (live call 77cfa127 2026-04-25) — utterance-fallback
            # subject grounding. Qwen3-4B classifier sometimes returns
            # intent=RAG/CONVERSATION on calc-prep utterances ("Я думаю
            # взять себе машину") and skips slot extraction entirely. Run
            # a deterministic regex pass over the raw utterance so an
            # unambiguous category cue ("машина" → Легковой, "грузовик"
            # → Грузовой, etc.) still seeds profile.subject. Conservative:
            # only fires when classifier didn't provide subject AND
            # profile.subject is currently None.
            if (
                "subject" not in _profile_patches
                and getattr(_profile_current, "subject", None) is None
            ):
                from .utterance_grounding import extract_subject_from_utterance
                _fallback_subject = extract_subject_from_utterance(message or "")
                if _fallback_subject:
                    _profile_patches["subject"] = _fallback_subject
                    print(
                        f"[Profile] utterance-fallback subject='{_fallback_subject}' "
                        f"(classifier omitted)",
                        flush=True,
                    )

            # Issue 1 (live call 3d3e17b9, 2026-04-25) — utterance-fallback
            # age_years grounding. Mirror the subject fallback for terse
            # "N лет" replies the small classifier drops. Only fires when
            # the b/u path actually requires age (condition_new==0) AND
            # age is currently empty AND classifier didn't already provide
            # one — so a numeric leak ("60 месяцев") can't poison age.
            if (
                "age_years" not in _profile_patches
                and getattr(_profile_current, "age_years", None) is None
                and getattr(_profile_current, "condition_new", None) == 0
            ):
                from .utterance_grounding import extract_age_years_from_utterance
                _fallback_age = extract_age_years_from_utterance(message or "")
                if _fallback_age is not None:
                    _profile_patches["age_years"] = _fallback_age
                    print(
                        f"[Profile] utterance-fallback age_years={_fallback_age} "
                        f"(classifier omitted)",
                        flush=True,
                    )

            # Universal slot fallbacks (2026-04-25). Same architectural
            # pattern as subject + age above: when the small classifier
            # omits a slot AND profile field is empty, run a deterministic
            # regex pass over the utterance. Belt-and-suspenders alongside
            # the planned classifier upgrade — the fallback never fights
            # a classifier value, only fills silent gaps.
            from .utterance_grounding import (
                extract_client_type_from_utterance,
                extract_condition_new_from_utterance,
                extract_currency_from_utterance,
                extract_prepaid_pct_from_utterance,
                extract_term_months_from_utterance,
                extract_type_schedule_from_utterance,
            )

            # client_type: terse single-word replies like "Физлицо." / "ИП."
            if (
                "client_type" not in _profile_patches
                and getattr(_profile_current, "client_type", None) is None
            ):
                _fb_client = extract_client_type_from_utterance(message or "")
                if _fb_client:
                    _profile_patches["client_type"] = _fb_client
                    print(
                        f"[Profile] utterance-fallback client_type='{_fb_client}'",
                        flush=True,
                    )

            # condition_new: "Новый." / "Поддержанный." single-word replies.
            if (
                "condition_new" not in _profile_patches
                and getattr(_profile_current, "condition_new", None) is None
            ):
                _fb_cond = extract_condition_new_from_utterance(message or "")
                if _fb_cond is not None:
                    _profile_patches["condition_new"] = _fb_cond
                    print(
                        f"[Profile] utterance-fallback condition_new={_fb_cond}",
                        flush=True,
                    )

            # currency: "В рублях" / "В долларах". Skipped when the
            # utterance simultaneously names a cost so we don't second-
            # guess classifier's combined cost+currency capture.
            if (
                "currency" not in _profile_patches
                and "cost" not in _profile_patches
                and getattr(_profile_current, "currency", None) is None
            ):
                _fb_cur = extract_currency_from_utterance(message or "")
                if _fb_cur:
                    _profile_patches["currency"] = _fb_cur
                    print(
                        f"[Profile] utterance-fallback currency='{_fb_cur}'",
                        flush=True,
                    )

            # term_months: "60 месяцев" / "на 5 лет". The years branch
            # requires age to NOT be the active question (condition_new==1
            # OR age_years already captured) so "5 лет" doesn't fight age.
            _need_term_yrs_gate = (
                getattr(_profile_current, "condition_new", None) == 1
                or getattr(_profile_current, "age_years", None) is not None
            )
            if (
                "term_months" not in _profile_patches
                and getattr(_profile_current, "term_months", None) is None
            ):
                _fb_term = extract_term_months_from_utterance(message or "")
                if _fb_term is not None:
                    # Year-form term answer is gated; month-form is always safe.
                    _is_year_form = _fb_term % 12 == 0 and _fb_term <= 84 and (
                        f"{_fb_term // 12}" in (message or "")
                        and "месяц" not in (message or "").lower()
                    )
                    if not _is_year_form or _need_term_yrs_gate:
                        _profile_patches["term_months"] = _fb_term
                        print(
                            f"[Profile] utterance-fallback term_months={_fb_term}",
                            flush=True,
                        )

            # prepaid_pct: "20 процентов" / "Без аванса". Skipped when
            # an absolute amount is already in patches (slot conflict).
            if (
                "prepaid_pct" not in _profile_patches
                and "prepaid_amount" not in _profile_patches
                and getattr(_profile_current, "prepaid_pct", None) is None
                and getattr(_profile_current, "prepaid_amount", None) is None
            ):
                _fb_pct = extract_prepaid_pct_from_utterance(message or "")
                if _fb_pct is not None:
                    _profile_patches["prepaid_pct"] = _fb_pct
                    print(
                        f"[Profile] utterance-fallback prepaid_pct={_fb_pct}",
                        flush=True,
                    )

            # type_schedule: "Аннуитет." / "Линейный."
            if (
                "type_schedule" not in _profile_patches
                and getattr(_profile_current, "type_schedule", None) is None
            ):
                _fb_sched = extract_type_schedule_from_utterance(message or "")
                if _fb_sched is not None:
                    _profile_patches["type_schedule"] = _fb_sched
                    print(
                        f"[Profile] utterance-fallback type_schedule='{_fb_sched}'",
                        flush=True,
                    )

            # Bug D (live call 2809a6f9 2026-04-26) — RAG-turn patch guard.
            # The classifier extracts hints from EVERY utterance. When the
            # caller asks generic info questions ("адрес офиса?", "кто
            # директор?"), the model can hallucinate leasing fields
            # (subject="Недвижимость", client_type="Юридическое лицо" from
            # "компании"). Without this gate, those hallucinations seed
            # the profile before the user has even said they want to
            # lease anything. Principle: profile-filling only happens
            # when the user is actually talking about leasing OR when we
            # are already mid-flow (any core field filled). Name is the
            # one exception — capturing it from a greeting is always OK.
            _has_any_core_field = any(
                getattr(_profile_current, f, None) is not None
                for f in ("subject", "cost", "client_type", "condition_new",
                          "age_years", "term_months", "prepaid_pct",
                          "prepaid_amount", "type_schedule")
            )
            _tool_flow_actions = {
                "calculate", "recalculate", "change_param", "change_field",
                "confirm", "clarify_client_type", "invalid_param", "sms",
            }
            _action_signals_tool = (
                _extracted_hints.get("action") in _tool_flow_actions
                or bool(_sa_change_field)
                or _sa_is_confirm
                or _sa_wants_readback
            )
            _in_tool_flow = needs_tool or _has_any_core_field or _action_signals_tool
            if not _in_tool_flow and _profile_patches:
                _name_only = {k: v for k, v in _profile_patches.items() if k == "name"}
                _dropped_keys = sorted(set(_profile_patches) - set(_name_only))
                if _dropped_keys:
                    print(
                        f"[Profile] RAG-turn-guard: dropped {_dropped_keys} "
                        f"(intent={_intent}, action={_extracted_hints.get('action')!r}, "
                        f"no core field set yet)",
                        flush=True,
                    )
                _profile_patches = _name_only
            # Hygiene filter before merge: drops noise, normalizes enums, validates MVP ranges.
            _had_patches = bool(_profile_patches)
            if _had_patches:
                print(f"[Profile] patches_pre_filter={_profile_patches}", flush=True)
            _profile_patches = filter_patches(_profile_patches, message or "")
            if _had_patches:
                print(f"[Profile] patches_post_filter={_profile_patches}", flush=True)
            _changed = session.client_profile.apply_patches(_profile_patches)
            if _changed:
                print(f"[Profile] patched: {_changed}", flush=True)
                _changed_this_turn = dict(_changed)
            elif _had_patches and not _profile_patches:
                print(f"[Profile] filter_patches: dropped (noise / invalid values)", flush=True)
            try:
                _p = session.client_profile
                _ts = _p.type_schedule if _p.type_schedule is not None else '-'
                # Fix 40f: use explicit `is not None` checks so 0/empty string
                # don't render as '-'. Prior `or '-'` hid term=0 as '-', which
                # masked a real bug for days.
                _fmt = lambda v: v if v is not None else '-'  # noqa: E731
                print(
                    f"[Profile] snapshot: state={_p.state.value} name={_fmt(_p.name)} "
                    f"subj={_fmt(_p.subject)} cost={_fmt(_p.cost)} {_fmt(_p.currency)} "
                    f"client_type={_fmt(_p.client_type)} cond_new={_fmt(_p.condition_new)} "
                    f"term={_fmt(_p.term_months)} prepaid={_fmt(_p.prepaid_pct)}% "
                    f"prepaid_amt={_fmt(_p.prepaid_amount)} "
                    f"graph={_ts} missing={sorted(_p.missing_fields())}",
                    flush=True,
                )
                # Bug C (live call 6a9d359b 2026-04-26) — show the BYN
                # projection when the conversion path applies. On the
                # legacy DirectTool branch (app.py:2642), Физлицо + USD
                # gets converted to BYN at calc-time only; profile keeps
                # the user's literal answer. The operator-facing snapshot
                # was misleading — it showed 100000 USD with no hint that
                # the calculator would actually run on 300000 BYN. Project
                # the same conversion inline here so the UI can render
                # "100000 USD → 300000 BYN" when applicable.
                _converted_cost = None
                _converted_currency = None
                if (
                    _p.cost is not None
                    and (_p.currency or "").upper() == "USD"
                    and "Физическое" in str(_p.client_type or "")
                ):
                    try:
                        from .profile_prompts import _get_usd_byn_rate
                        _rate = _get_usd_byn_rate()
                        _converted_cost = round(float(_p.cost) * float(_rate), 2)
                        _converted_currency = "BYN"
                    except Exception:  # noqa: BLE001
                        pass
                await broadcast_sip_event({
                    "type": "sip.profile.snapshot",
                    "call_id": session_id,
                    "state": _p.state.value,
                    "fields": {
                        "name": _p.name,
                        "subject": _p.subject,
                        "cost": _p.cost,
                        "currency": _p.currency,
                        "client_type": _p.client_type,
                        "condition_new": _p.condition_new,
                        "age_years": getattr(_p, "age_years", None),
                        "term_months": _p.term_months,
                        "prepaid_pct": _p.prepaid_pct,
                        "prepaid_amount": _p.prepaid_amount,
                        "type_schedule": _p.type_schedule,
                    },
                    "original_cost": getattr(_p, "original_cost", None),
                    "original_currency": getattr(_p, "original_currency", None),
                    "converted_cost": _converted_cost,
                    "converted_currency": _converted_currency,
                    "missing": sorted(_p.missing_fields()),
                })
            except Exception:  # noqa: BLE001
                pass
            # Handle change_field post-confirmation OR extra changes mid-pending.
            # Fix 28 (multi-field) + Fix 30 (implicit change detection on
            # CONFIRMED) + Fix 31 (require cue + literal signal for extras).
            from .profile_hygiene import has_field_signal as _has_field_signal
            _profile_now = session.client_profile
            _EXTRA_KEYS_TO_CHANGE = (
                ("subject", "subject"),
                ("cost", "cost"),
                ("currency", "currency"),
                ("client_type", "client_type"),
                ("condition_new", "condition_new"),
                ("term_months", "term_months"),
                ("type_schedule", "type_schedule"),
                ("prepaid_pct", "prepaid_pct"),
                ("prepaid_amount", "prepaid_amount"),
            )
            # Fix 30: detect implicit change on CONFIRMED state when the
            # classifier emitted at least one hint that differs from the
            # profile AND carries a real utterance signal. Without this, an
            # utterance like "всё-таки легковой автомобиль за 80 тысяч рублей"
            # produced no `change_field` from the classifier, and the stale-
            # patch guard correctly rejected the subject/cost patches — so
            # profile stayed stale and the calculator re-ran old params.
            # Bug E (live call f59681d2 2026-04-26) — post-calc COLLECTING.
            # The legacy direct_call path runs the calculator and presents
            # the result without transitioning state out of COLLECTING. From
            # the user's perspective they have heard a readback and are now
            # changing params — the change-staging path MUST fire here.
            # Without this, classifier change_field/change_value gets dropped
            # silently, calc re-runs with stale values, and the next "Да"
            # bypasses the SMS gate (which only blocks READBACK_PENDING /
            # CHANGE_PENDING). Treat COLLECTING + complete + has-calc-OK
            # as equivalent to CONFIRMED for change-staging purposes.
            _has_successful_calc = any(
                tc.get("tool") == "calculator" and (tc.get("result") or {}).get("ok")
                for tc in (session.tool_calls_history + session.tool_calls_this_turn)
            )
            _post_calc_collecting = (
                _profile_now.state == ProfileState.COLLECTING
                and _profile_now.is_complete_for_calc()
                and _has_successful_calc
            )
            _change_eligible_states = (
                ProfileState.CONFIRMED, ProfileState.CHANGE_PENDING,
            )
            _implicit_enter = False
            if not _sa_change_field and (
                _profile_now.state == ProfileState.CONFIRMED
                or _post_calc_collecting
            ):
                for _hint_key, _field_key in _EXTRA_KEYS_TO_CHANGE:
                    _hv = _extracted_hints.get(_hint_key)
                    if _hv in (None, ""):
                        continue
                    _cur = getattr(_profile_now, _field_key, None)
                    if _cur == _hv:
                        continue
                    if not _has_field_signal(_field_key, _hv, message or ""):
                        continue
                    _implicit_enter = True
                    break
            _enter_change = bool(_sa_change_field) and (
                _profile_now.state in _change_eligible_states
                or _post_calc_collecting
            )
            _enter_change = _enter_change or _implicit_enter
            if _enter_change:
                # Start with existing changes dict if state is already CHANGE_PENDING
                _existing = {}
                _pc = _profile_now.pending_change or {}
                if isinstance(_pc.get("changes"), dict):
                    _existing = dict(_pc["changes"])
                elif _pc.get("field"):
                    _existing = {
                        _pc["field"]: {
                            "old": _pc.get("old_value"),
                            "new": _pc.get("new_value"),
                        }
                    }
                # Primary change from classifier (if any). Fix 34 guard:
                # skip staging when classifier emitted change_field with
                # change_value=None/empty.
                # Fix 40e: for numeric fields, also require a literal signal
                # in the utterance. Classifier sometimes emits change_value=0
                # (or other values) on non-numeric turns — without this guard
                # term_months gets silently set to 0, corrupting the profile.
                # CP-2.3: Fix 41b whitelist retired. ClassifierOutput.change_field
                # is a Literal, so "all" / other non-fields are already None by
                # the time we reach this block. Numeric-field signal check below
                # still runs (Fix 40e — catches classifier emitting change_value=0
                # on non-numeric turns).
                _NUMERIC_CHANGE_FIELDS = {
                    "cost", "term_months", "prepaid_pct", "prepaid_amount", "age_years",
                }
                # Codex adversarial pass 4 (2026-04-20): enum change_fields
                # (currency, client_type, subject, condition_new, type_schedule)
                # previously skipped value-grounding entirely. Classifier could
                # emit change_field='currency', change_value='USD' on "в рублях"
                # and post-confirmation state flipped to USD on a hallucination.
                # Route ALL grounded fields through value_grounded().
                _GROUNDED_CHANGE_FIELDS = _NUMERIC_CHANGE_FIELDS | {
                    "subject", "currency", "client_type", "condition_new", "type_schedule",
                }
                from .classifier_schema import value_grounded as _value_grounded
                _primary_value_ok = _sa_change_value not in (None, "")
                if _primary_value_ok and _sa_change_field in _GROUNDED_CHANGE_FIELDS:
                    if not _value_grounded(_sa_change_field, _sa_change_value, message or ""):
                        print(
                            f"[Profile] change_value not grounded — rejecting "
                            f"{_sa_change_field}={_sa_change_value!r} "
                            f"utterance='{(message or '')[:60]}'",
                            flush=True,
                        )
                        _primary_value_ok = False
                if _sa_change_field and _primary_value_ok:
                    _primary_old = getattr(_profile_now, _sa_change_field, None)
                    _existing[_sa_change_field] = {
                        "old": _primary_old,
                        "new": _sa_change_value,
                    }
                # Fix 31: for extras, require a real utterance signal so the
                # change-confirm prompt only lists fields the user actually
                # mentioned. Without this, derived values echoed back by the
                # classifier (e.g. prepaid_amount=16000 from a prior calc)
                # would end up in "Меняю X на A и сумму аванса на 16000".
                for _hint_key, _field_key in _EXTRA_KEYS_TO_CHANGE:
                    if _field_key == _sa_change_field:
                        continue
                    _hint_val = _extracted_hints.get(_hint_key)
                    if _hint_val in (None, ""):
                        continue
                    _cur = getattr(_profile_now, _field_key, None)
                    if _cur == _hint_val:
                        continue
                    if not _has_field_signal(_field_key, _hint_val, message or ""):
                        print(
                            f"[Profile] extras: dropping {_field_key}={_hint_val!r} "
                            f"(no literal signal in utterance)",
                            flush=True,
                        )
                        continue
                    _existing[_field_key] = {
                        "old": _cur,
                        "new": _hint_val,
                    }
                if not _existing:
                    # Defensive: ran into the block with nothing real to stage.
                    pass
                else:
                    session.client_profile.pending_change = {"changes": _existing}
                    session.client_profile.state = ProfileState.CHANGE_PENDING
                    import time as _time
                    session.client_profile.change_emitted_at = _time.time()
                    _primary_marker = _sa_change_field or next(iter(_existing), None)
                    session.client_profile.last_change_pending = _primary_marker
                    _origin = "explicit" if _sa_change_field else "implicit"
                    _change_staged_this_turn = True
                    print(
                        f"[Profile] CHANGE_PENDING ({_origin}): fields={list(_existing.keys())} "
                        f"primary={_primary_marker}",
                        flush=True,
                    )
            # Confirmation → stamp confirmed_at
            if _sa_is_confirm:
                if session.client_profile.last_change_pending:
                    session.client_profile.confirmed_at = time.time()
                    session.client_profile.last_change_pending = None
                elif (session.client_profile.is_complete_for_calc()
                      and not session.client_profile.confirmed_at):
                    session.client_profile.confirmed_at = time.time()
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

    # --- Phase 3.D: APPLY_TURN_ENABLED dispatch ---
    # When the feature flag is set (deploy-time opt-in, default off until
    # CP-3.5 live validation), route through the structural apply_turn /
    # execute_action pipeline instead of the legacy 5-gate block + sentence
    # queue. The legacy path stays resident as the else-branch fallback so
    # operators can flip the flag to "0" without a git revert if the new
    # path regresses (§7.1).
    if os.environ.get("APPLY_TURN_ENABLED", "0") == "1" and _sa_output is not None:
        # §7.2 invariant #6 — stale-result guard. Another utterance
        # finalised ahead of this one can bump `latest_finalized_turn_id`
        # beyond our stamped `turn_id` during any pending await above; in
        # that case this dispatch would clobber fresh state and we drop it.
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

        _action = apply_turn(
            session.client_profile, _sa_output, message or "",
            turn_id=turn_id,
        )
        print(f"[apply_turn] turn_id={turn_id} action={type(_action).__name__}", flush=True)

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
            # Bug 3 fix: hold assistant_speaking=True through the actual
            # audio playback duration. mod_audio_fork still has buffered
            # PCM after the last yield; without this drain the VAD
            # barge-in gate at line 3140 / 3443 / 3546 (which keys on
            # assistant_speaking) flips off while the caller is still
            # hearing the tail. Mirrors legacy _speak_tts:200-212.
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

    # --- Skip RAG for pure name-capture turns (prevents KB hallucinations on names) ---
    # Skip-RAG name-capture path only applies on the VERY FIRST name turn.
    # Once profile.name is set, the line-1453 gate prevents `name` from being
    # added to `_profile_patches` again (stale-name-patch filter), so a later
    # turn that asks "what is my name?" or "what is the address?" reaches
    # this point with `_profile_patches` empty of "name" and routes to RAG.
    #
    # Bug 5 (live call f5c47706 2026-04-25): the previous gate
    # `not _name_already_captured` was wrong — by the time we reach this
    # line, `apply_patches` (line 1464) has already written profile.name,
    # so the flag was True on the first turn and the open-greeting path
    # never fired. New gate: `"name" in _profile_patches` is True only
    # when this turn captured a fresh name (the patch survived line 1453's
    # "name was empty pre-patch" guard).
    _name_just_captured_this_turn = "name" in _profile_patches
    if _name_just_captured_this_turn and should_skip_rag(message or "", _profile_patches, _extracted_hints):
        print(f"[Grounding] skip-RAG: name-capture session={session_id[:8]}", flush=True)
        # Cancel the in-flight RAG task since we won't use its result
        try:
            if not _rag_task.done():
                _rag_task.cancel()
        except Exception:
            pass
        _name = session.client_profile.name or "клиент"
        _greeting = f"Здравствуйте, {_name}! Чем могу помочь по вопросам лизинга?"
        await _emit_plain_assistant_response(
            _greeting, websocket, session_id,
            backend=backend, session=session,
        )
        session.assistant_speaking = False
        return

    # --- Await RAG retrieval (started before classifier, should be done by now) ---
    retrieval = await _rag_task
    t_retrieval_done = time.time()
    _t_rag_ms = (t_retrieval_done - _t_rag_start) * 1000
    print(f"[Latency:{session_id[:8]}] RAG: {_t_rag_ms:.0f}ms (parallel)", flush=True)
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
    # Use cumulative history (+ in-turn appends) so SMS works for a calc done
    # on a prior turn (tool_calls_this_turn is reset at turn start).
    _sms_all_calls = session.tool_calls_history + session.tool_calls_this_turn
    _sms_from_classifier = _extracted_hints.get("action") == "sms" and _sms_all_calls
    # Bug A (live call 6a9d359b 2026-04-26) — change wins over SMS.
    # When the classifier or sticky-patch logic detected a profile change
    # this turn, the user's utterance is a change request, not an SMS
    # confirm — even if it starts with "Да". Prefer the classifier's
    # structured signals over keyword-based intent: _sa_change_field is
    # set whenever the classifier emitted change_field; _change_staged_this_turn
    # tracks implicit change-detection on CONFIRMED state; _changed_this_turn
    # captures any non-name patches that landed via apply_patches. Any of
    # these means SMS direct-send must defer to the change-confirm path.
    _change_signal_this_turn = bool(
        _change_staged_this_turn
        or _sa_change_field
        or any(k for k in _changed_this_turn if k != "name")
    )
    sms_context = ""
    if (has_sms_intent or _sms_from_classifier) and _change_signal_this_turn:
        print(
            f"[Jambonz:{session_id[:8]}] SMS suppressed by change signal: "
            f"staged={_change_staged_this_turn} field={_sa_change_field!r} "
            f"changed={list(_changed_this_turn)}",
            flush=True,
        )
        # Fall through: legacy LLM / change-confirm path handles it.
    elif (has_sms_intent or _sms_from_classifier) and _sms_all_calls and session.client_phone:
        last_calc = next(
            (tc for tc in reversed(_sms_all_calls)
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
    elif has_sms_intent and _sms_all_calls:
        last_calc = next(
            (tc for tc in reversed(_sms_all_calls)
             if tc.get("tool") == "calculator"), None)
        if last_calc and last_calc.get("result", {}).get("ok"):
            calc_tool = get_tool("calculator")
            sms_context = f"Текст для СМС:\n{calc_tool.format_sms_body(last_calc['result'])}\n\n"

    # ── RAG-Guard: prevent stale calc retriggers ──
    # (a) info-question override: any turn where classifier says calculate but
    #     utterance is a pure info question with no calc-param hints -> RAG.
    #     Fires regardless of profile state because the classifier's "sticky
    #     recalculate" bias happens at any stage, not just after CONFIRMED.
    # (b) circuit-breaker: same calc signature failed >=2 times AFTER readback.
    if needs_tool:
        _msg_lower = (message or "").lower()
        _FRESH_CALC_PARAM_KEYS_GUARD = (
            "subject", "cost", "term", "prepaid", "prepaid_pct",
            "currency", "type_schedule", "client_type", "condition_new",
        )
        _no_fresh_calc_hint = not any(
            _extracted_hints.get(k) not in (None, "")
            for k in _FRESH_CALC_PARAM_KEYS_GUARD
        )
        _has_info_q = bool(_INFO_QUESTION_RE.search(_msg_lower))
        _guard_reason = None
        if _has_info_q and _no_fresh_calc_hint:
            print(
                f"[RAG-Guard] info-question override: '{_msg_lower[:60]}' -> RAG",
                flush=True,
            )
            needs_tool = False
            _guard_reason = "info_q_fired"
        elif (
            session.client_profile.state in (ProfileState.CONFIRMED, ProfileState.CHANGE_PENDING)
            and session.consecutive_calc_failures >= 2
        ):
            print(
                f"[RAG-Guard] circuit-breaker: {session.consecutive_calc_failures} "
                f"consecutive calc failures with sig={session.last_calc_signature!r} -> RAG",
                flush=True,
            )
            needs_tool = False
            _guard_reason = "circuit_breaker"
        else:
            _guard_reason = f"pass (info_q={_has_info_q}, no_fresh_hint={_no_fresh_calc_hint}, state={session.client_profile.state.value}, failures={session.consecutive_calc_failures})"
        print(f"[RAG-Guard] decision={_guard_reason}", flush=True)

    # ── Always-on state gates (Fix 29) ──
    # Gates 3 (READBACK_PENDING) and 4 (CHANGE_PENDING) must run on every turn
    # regardless of what the classifier labelled the intent. When the user
    # denies a readback with a correction ("Нет, автомобиль грузовой"), the
    # classifier often emits intent=CONVERSATION, which used to skip the
    # whole gate block and let the LLM improvise a clarification — bypassing
    # the state machine. Now the state gates fire first, and the tool-intent
    # gates only run afterwards if control falls through.
    _state_profile = session.client_profile
    _STATE_DELTA_KEYS = (
        ("subject", "subject"),
        ("cost", "cost"),
        ("currency", "currency"),
        ("client_type", "client_type"),
        ("condition_new", "condition_new"),
        ("term_months", "term_months"),
        ("type_schedule", "type_schedule"),
        ("prepaid_pct", "prepaid_pct"),
        ("prepaid_amount", "prepaid_amount"),
    )
    # Explicit-deny heuristic: user starts message with "Нет", "неправильно",
    # "неверно", or "ошибка". Used to decide whether to re-prompt readback /
    # change-confirm vs fall through (e.g. user asked an info question mid-state).
    _DENY_PREFIX_RE = re.compile(
        r"^\s*(нет|не\s+верно|не\s+правильно|неправильно|неверно|ошибка|ошибочн\w+)\b",
        re.IGNORECASE,
    )
    _is_explicit_deny = bool(message) and bool(_DENY_PREFIX_RE.match(message))

    # Fix 40 hotfix: track state transitions so _sticky_calc_ready can
    # distinguish "state was pending AND user confirmed this turn" (should
    # fire calc) from "state is already CONFIRMED and user just said 'Хорошо'"
    # (should NOT fire calc). Without this, READBACK_PENDING → CONFIRMED
    # transition below bricks the first calc call.
    _just_confirmed_this_turn = False

    if _state_profile.state == ProfileState.READBACK_PENDING:
        if _sa_is_confirm:
            _state_profile.state = ProfileState.CONFIRMED
            if _state_profile.confirmed_at is None:
                _state_profile.confirmed_at = time.time()
            _just_confirmed_this_turn = True
            print(f"[Orchestrator] profile CONFIRMED (via always-on gate)", flush=True)
            # fall through to tool orchestration
        else:
            # Deny-with-correction detection: did the classifier emit any
            # field whose value differs from the current profile?
            # Codex adversarial confirmation pass (2026-04-20, E-Codex-2):
            # require value_grounded() for each delta so a plain "нет" turn
            # with classifier drift (e.g. stale numeric cost/term from
            # carryover context) cannot stage an unspoken correction.
            # value_grounded delegates to has_field_signal for numerics and
            # to per-value cue maps for enums — one call covers both.
            from .classifier_schema import value_grounded as _vg_readback
            _deltas: dict[str, dict[str, Any]] = {}
            for _hint_key, _field_key in _STATE_DELTA_KEYS:
                _hint_val = _extracted_hints.get(_hint_key)
                if _hint_val in (None, ""):
                    continue
                _cur = getattr(_state_profile, _field_key, None)
                if _cur == _hint_val:
                    continue
                if not _vg_readback(_field_key, _hint_val, message or ""):
                    print(
                        f"[Orchestrator] READBACK delta rejected (ungrounded): "
                        f"{_field_key}={_hint_val!r} utterance='{(message or '')[:60]}'",
                        flush=True,
                    )
                    continue
                _deltas[_field_key] = {"old": _cur, "new": _hint_val}
            if _deltas:
                _state_profile.pending_change = {"changes": _deltas}
                _state_profile.state = ProfileState.CHANGE_PENDING
                _state_profile.change_emitted_at = time.time()
                print(
                    f"[Orchestrator] READBACK deny-with-correction -> CHANGE_PENDING "
                    f"fields={list(_deltas.keys())}",
                    flush=True,
                )
                await _emit_plain_assistant_response(
                    build_change_confirm_text(_state_profile.pending_change),
                    websocket, session_id,
                    backend=backend, session=session,
                )
                session.assistant_speaking = False
                return
            if _is_explicit_deny:
                print(f"[Orchestrator] re-prompting readback (explicit deny, no corrections)", flush=True)
                await _emit_plain_assistant_response(
                    build_readback_text(_state_profile),
                    websocket, session_id,
                    backend=backend, session=session,
                )
                session.assistant_speaking = False
                return
            # No confirm, no deltas, no explicit deny — probably an info
            # question ("а какие офисы в Минске?"). Fall through so RAG
            # answers. State stays READBACK_PENDING.
            print(
                f"[Orchestrator] READBACK_PENDING: no confirm / deltas / deny — falling through",
                flush=True,
            )
    elif _state_profile.state == ProfileState.CHANGE_PENDING:
        # Fix 40 hotfix 2: when a change was STAGED this turn (classifier
        # emitted multi-field change like "Давай сменим всё. Грузовик, 100
        # тысяч на 7 лет, аванс 30%"), is_confirmation=true from the classifier
        # is unreliable — "давай" at the start of the utterance triggers it
        # even though the user hasn't heard or confirmed the change-confirm
        # prompt yet. Force a dedicated confirmation turn.
        if _change_staged_this_turn:
            print(
                f"[Orchestrator] CHANGE_PENDING staged this turn — "
                f"emitting change-confirm regardless of is_confirm={_sa_is_confirm}",
                flush=True,
            )
            await _emit_plain_assistant_response(
                build_change_confirm_text(_state_profile.pending_change),
                websocket, session_id,
                backend=backend, session=session,
            )
            session.assistant_speaking = False
            return
        if _sa_is_confirm:
            # Codex adversarial pass 4 (2026-04-20): only advance to CONFIRMED
            # when apply_pending_change actually applied at least one known
            # field. A malformed payload (unknown fields only) now returns
            # False and keeps pending_change intact — the state machine can
            # re-prompt on the next turn instead of silently losing the edit.
            _applied = _state_profile.apply_pending_change()
            if _applied:
                _state_profile.state = ProfileState.CONFIRMED
                _state_profile.change_emitted_at = time.time()
                _just_confirmed_this_turn = True
                print(
                    f"[Orchestrator] change CONFIRMED (via always-on gate), recalculating",
                    flush=True,
                )
                # fall through to calculator recalc
            else:
                print(
                    f"[Orchestrator] change NOT applied (no known fields in "
                    f"pending_change) — re-prompting change-confirm",
                    flush=True,
                )
                await _emit_plain_assistant_response(
                    build_change_confirm_text(_state_profile.pending_change),
                    websocket, session_id,
                    backend=backend, session=session,
                )
                session.assistant_speaking = False
                return
        elif _is_explicit_deny:
            print(f"[Orchestrator] re-prompting change-confirm (explicit deny)", flush=True)
            await _emit_plain_assistant_response(
                build_change_confirm_text(_state_profile.pending_change),
                websocket, session_id,
                backend=backend, session=session,
            )
            session.assistant_speaking = False
            return
        else:
            # No confirm, no explicit deny. If pending_change was updated
            # this turn (explicit change_field from classifier OR implicit
            # change detected by Fix 30), re-emit the change-confirm so the
            # caller hears the proposal. Otherwise it's likely an info
            # question mid-pending — fall through and let RAG answer.
            if _change_staged_this_turn:
                print(f"[Orchestrator] re-prompting change-confirm (fresh staged)", flush=True)
                await _emit_plain_assistant_response(
                    build_change_confirm_text(_state_profile.pending_change),
                    websocket, session_id,
                    backend=backend, session=session,
                )
                session.assistant_speaking = False
                return
            print(
                f"[Orchestrator] CHANGE_PENDING: no confirm / deny / fresh change — falling through",
                flush=True,
            )

    # ── COLLECTING-clarify gate (intent-agnostic) ──
    # When the user answers a clarification question with a slot-fill
    # utterance and at least one field gets captured, emit the
    # next-missing-field prompt deterministically. Without this gate the
    # turn falls through to the LLM which sometimes re-asks fields that
    # are already in the profile (session b9e9fcfb 2026-04-18, live call
    # 12b9826a 2026-04-25 — Bug 16 юр.лицо path lost subject).
    #
    # Bug 16 (2026-04-25) — gate was previously `intent == CONVERSATION`
    # only. Live call showed it failed to fire even when the classifier
    # logged exactly that. The right architectural rule is intent-
    # agnostic and mirrors the force-readback gate below: if state is
    # COLLECTING, profile is incomplete, and a slot just landed, the
    # next bot action MUST be the deterministic clarify — independent
    # of how the classifier labelled the turn. This makes the behavior
    # universal across (физ/юр) × (легк/груз/спец/оборудование).
    #
    # Guards:
    #   - state == COLLECTING (CONFIRMED / CHANGE_PENDING handled upstream).
    #   - _changed_this_turn truthy (something was captured this turn).
    #   - profile incomplete (otherwise force-readback gate below handles it).
    #   - utterance is not a question (info Qs route to RAG).
    #   - utterance is not a confirmation (handled upstream by state gates).
    _intent_val = str(_sa_parsed.get("intent") if isinstance(_sa_parsed, dict) else "").upper()
    _collect_profile_42b = session.client_profile
    _is_question = (
        "?" in (message or "")
        or bool(re.match(r"^\s*(как|что|какой|какие|какая|какое|где|когда|почему|зачем|сколько)\b",
                         message or "", re.IGNORECASE))
    )
    if (
        _collect_profile_42b.state == ProfileState.COLLECTING
        and bool(_changed_this_turn)
        and not _collect_profile_42b.is_complete_for_calc()
        and not _is_question
        and not _sa_is_confirm
    ):
        _missing_42b = _collect_profile_42b.missing_fields()
        print(
            f"[Orchestrator] COLLECTING clarify: patched={list(_changed_this_turn.keys())} "
            f"still_missing={sorted(_missing_42b)} intent={_intent_val or '-'}",
            flush=True,
        )
        try:
            _clar_42b = build_clarification_prompt(_missing_42b, _collect_profile_42b)
            await _emit_plain_assistant_response(
                _clar_42b, websocket, session_id,
                backend=backend, session=session,
            )
            session.assistant_speaking = False
            return
        except Exception as _clar_exc:
            # Defense in depth: if the gate raises for any reason, log and
            # fall through to LLM rather than silently killing TTS.
            print(f"[Orchestrator] COLLECTING clarify FAILED, falling through: {_clar_exc}", flush=True)

    # ── Profile-complete force-readback (intent-agnostic) ──
    # Live call 04f734c8 (2026-04-25): user said "Ладно, давай линейный."
    # → classifier emitted intent=CONVERSATION → the CONVERSATION-incomplete
    # gate above didn't fire (profile WAS complete after this turn),
    # `needs_tool` was False so the readback gate inside that block didn't
    # see this turn either, and control fell through to the LLM. The LLM
    # ignored the captured snapshot and re-asked for every field.
    #
    # Profile is the source of truth. The moment any patch lands that
    # completes is_complete_for_calc(), the next bot action MUST be the
    # deterministic readback — independent of how the classifier labelled
    # the turn. No LLM in the loop on this critical handoff.
    if (
        bool(_changed_this_turn)
        and _collect_profile_42b.state == ProfileState.COLLECTING
        and _collect_profile_42b.is_complete_for_calc()
        and _collect_profile_42b.confirmed_at is None
        and not _sa_is_confirm
        and not _is_question
    ):
        _collect_profile_42b.state = ProfileState.READBACK_PENDING
        _collect_profile_42b.readback_emitted_at = time.time()
        _readback_complete = build_readback_text(_collect_profile_42b)
        print(
            f"[Orchestrator] force-readback after final-slot capture: "
            f"changed={list(_changed_this_turn.keys())} intent={_intent_val}",
            flush=True,
        )
        await _emit_plain_assistant_response(
            _readback_complete, websocket, session_id,
            backend=backend, session=session,
        )
        session.assistant_speaking = False
        return

    # ── Tool-intent gates (Gates 1 & 2) ──
    # Enforces server-side clarify + readback BEFORE any calculator call.
    # Only active when the classifier detected a tool intent; RAG paths skip.
    #
    # Gate 1 (COLLECTING + incomplete): ask for missing fields, skip calculator.
    # Gate 2 (COLLECTING + complete):   emit readback, transition to READBACK_PENDING.
    if needs_tool:
        _gate_profile = session.client_profile
        print(
            f"[Orchestrator] gate entered: state={_gate_profile.state.value} "
            f"missing={sorted(_gate_profile.missing_fields())} "
            f"action={_extracted_hints.get('action', '-')} "
            f"is_confirm={_sa_is_confirm} "
            f"has_history={bool(session.tool_calls_history)}",
            flush=True,
        )
        # Gate 1: profile incomplete -> clarify missing fields.
        # Skip the gate when we're doing a pure param-change on an existing calc
        # (the change path reuses previous params and doesn't need the full profile).
        # Real param-change turns must (a) reference an existing calc AND
        # (b) include at least one fresh calc-param hint on this turn.
        # Without (b), classifier-emitted `recalculate` on unrelated turns
        # (e.g. info questions) would bypass Gate 1/2 and stale-recalc.
        _FRESH_CALC_PARAM_KEYS = (
            "subject", "cost", "term", "prepaid", "prepaid_pct",
            "currency", "type_schedule", "client_type", "condition_new",
        )
        _has_fresh_param_hint = any(
            _extracted_hints.get(k) not in (None, "")
            for k in _FRESH_CALC_PARAM_KEYS
        )
        _is_param_change_for_gate = (
            _extracted_hints.get("action") in ("change_param", "recalculate")
            and bool(session.tool_calls_history)
            and _has_fresh_param_hint
        )
        print(
            f"[Orchestrator] is_param_change={_is_param_change_for_gate} "
            f"has_fresh_hint={_has_fresh_param_hint} "
            f"action={_extracted_hints.get('action', '-')} "
            f"fresh_hints={[k for k in _FRESH_CALC_PARAM_KEYS if _extracted_hints.get(k) not in (None, '')]}",
            flush=True,
        )
        if not _is_param_change_for_gate:
            _missing = _gate_profile.missing_fields()
            if _missing:
                print(
                    f"[Orchestrator] profile incomplete, missing={sorted(_missing)}",
                    flush=True,
                )
                _clar = build_clarification_prompt(_missing, _gate_profile)
                await _emit_plain_assistant_response(
                    _clar, websocket, session_id,
                    backend=backend, session=session,
                )
                session.assistant_speaking = False
                return
        # Gate 2: profile complete, never confirmed -> emit readback.
        if (
            _gate_profile.state == ProfileState.COLLECTING
            and _gate_profile.is_complete_for_calc()
            and _gate_profile.confirmed_at is None
            and not _sa_is_confirm
            and not _is_param_change_for_gate
        ):
            _gate_profile.state = ProfileState.READBACK_PENDING
            _gate_profile.readback_emitted_at = time.time()
            _readback = build_readback_text(_gate_profile)
            print(f"[Orchestrator] readback emitted", flush=True)
            await _emit_plain_assistant_response(
                _readback, websocket, session_id,
                backend=backend, session=session,
            )
            session.assistant_speaking = False
            return
        # Gates 3 (READBACK_PENDING) and 4 (CHANGE_PENDING) now live in the
        # always-on state-gate block above (Fix 29). They return early before
        # control reaches this point, so no duplicate handling is needed here.

    # ── Deterministic Tool Orchestration ──
    # When classifier extracts enough data, call tools from code directly.
    # LLM only presents the result. This bypasses unreliable LLM tool calling.
    _direct_tool_result = None

    # Parameter change path: user wants to modify previous calculation
    # (e.g., "change advance to 20%", "make it 48 months", "switch to юрлицо").
    # Does NOT require subject+cost; uses previous calc params as base.
    # Use cumulative history (+ in-turn appends) so param-change survives across
    # turns after tool_calls_this_turn is reset at turn start.
    _all_calls_for_change = session.tool_calls_history + session.tool_calls_this_turn
    _is_param_change = (
        needs_tool
        and _extracted_hints.get("action") in ("change_param", "recalculate")
        and bool(_all_calls_for_change)
    )
    _param_change_params: dict[str, Any] | None = None
    _change_no_new_value = False  # True when user asks about changing but gives no value
    if _is_param_change:
        _prev = next((tc for tc in reversed(_all_calls_for_change)
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
        _prev_calls = (
            getattr(session, 'tool_calls_history', [])
            + getattr(session, 'tool_calls_this_turn', [])
        )
        _prev_calc = next((tc for tc in reversed(_prev_calls)
                          if tc.get("tool") == "calculator" and tc.get("ok")), None)
        if _prev_calc and _prev_calc.get("result"):
            # Re-present the previous result
            _direct_tool_result = _prev_calc["result"]
            print(f"[DirectTool] re-presenting previous result", flush=True)

    # Direct-call conditions:
    # 1) change_param path has built explicit params, OR
    # 2) ClientProfile is complete AND (confirmed OR client explicitly confirmed this turn)
    #    — gates calc on explicit confirmation to match the no-defaults + read-back rule.
    # 3) Legacy: current-turn hints have enough for a one-shot confirmed recalc.
    _profile = session.client_profile
    # Profile-ready fires calculator automatically. To prevent sticky re-firing
    # on every post-confirmation turn (info questions, off-topic chat, etc.),
    # also require a fresh signal this turn: either the classifier said this is
    # a tool turn (needs_tool), or the user explicitly confirmed (Верно / Да on
    # a change). Without this, once `confirmed_at` is set, every complete-profile
    # turn retriggers the calculator — even turns where the user asked about the
    # director, office address, or just chatted.
    _profile_ready = _sticky_calc_ready(
        _profile,
        _sa_is_confirm,
        needs_tool,
        just_confirmed_this_turn=_just_confirmed_this_turn,
    )
    _legacy_hint_direct = (
        needs_tool
        and _extracted_hints.get("subject")
        and _extracted_hints.get("cost")
        and _profile.confirmed_at is not None  # only if already confirmed once in session
    )
    _can_direct_call = (
        _param_change_params is not None
        or _profile_ready
        or _legacy_hint_direct
    )
    # Fix 42c: diagnostic log to track why calc may or may not fire after
    # a confirmed change. Catch the "Верно → no tool call" regression.
    print(
        f"[Orchestrator] direct_call gate: can_direct={_can_direct_call} "
        f"profile_ready={_profile_ready} param_change={_param_change_params is not None} "
        f"legacy_hint={_legacy_hint_direct} "
        f"(is_complete={_profile.is_complete_for_calc()}, "
        f"confirmed_at={'set' if _profile.confirmed_at else 'None'}, "
        f"just_confirmed={_just_confirmed_this_turn}, "
        f"is_confirm={_sa_is_confirm}, needs_tool={needs_tool}, "
        f"state={_profile.state.value})",
        flush=True,
    )
    if _can_direct_call:
        _action = _extracted_hints.get("action", "calculate")
        calc_tool = get_tool("calculator")
        import json as _json_direct

        if _param_change_params is not None:
            # Use pre-built params from change_param path
            _direct_params = _param_change_params
        else:
            # Build params from ClientProfile (source of truth) merged with new hints.
            # Profile already has patches applied earlier from SessionAgent output.
            _p = session.client_profile
            _direct_params: dict[str, Any] = {
                "subject": _p.subject or _extracted_hints.get("subject"),
                "cost": _p.cost if _p.cost is not None else _extracted_hints.get("cost"),
            }
            _cur = _p.currency or _extracted_hints.get("currency")
            if _cur:
                _direct_params["currency"] = _cur
            _ct = _p.client_type or _extracted_hints.get("client_type")
            if _ct:
                _direct_params["client_type"] = _ct
            if _p.condition_new is not None:
                _direct_params["condition_new"] = _p.condition_new
            if _p.age_years is not None:
                _direct_params["age"] = _p.age_years
                _direct_params["age_years"] = _p.age_years
            if _p.prepaid_pct is not None:
                _direct_params["prepaid"] = _p.prepaid_pct
                _direct_params["prepaid_pct"] = _p.prepaid_pct
            elif _p.prepaid_amount is not None:
                _direct_params["prepaid_amount"] = _p.prepaid_amount
            if _p.term_months is not None:
                _direct_params["term"] = _p.term_months
            if _p.type_schedule is not None:
                _direct_params["type_schedule"] = _p.type_schedule

        # ── MVP currency policy: USD->BYN for Физ лицо; reject EUR/RUB. ──
        _ct_policy = _direct_params.get("client_type")
        _cur_policy = _direct_params.get("currency")
        _currency_conversion = None
        # Fix 1.2 (2026-04-19) — clear any stale USD disclosure from a
        # previous turn before deciding whether conversion fires this turn.
        # Without this, a client who first quoted USD and then switched to
        # BYN would still see "(это N белорусских рублей по курсу X к 1)"
        # in the readback.
        _p.original_cost = None
        _p.original_currency = None
        if _ct_policy == "Физическое лицо" and _cur_policy in ("EUR", "RUB", "RUR", "CNY"):
            # Block the direct call; emit UnsupportedCurrency fallback message via LLM
            _direct_tool_result = {
                "ok": False,
                "error": (
                    f"Для физических лиц сейчас поддерживаются расчёты в белорусских рублях "
                    f"и в долларах. Валюта {_cur_policy} временно не поддерживается. "
                    "Уточните, пожалуйста, стоимость в BYN или USD."
                ),
                "params": _direct_params,
                "defaulted": [],
            }
            print(f"[DirectTool] currency_policy: reject {_cur_policy} for Физ лицо", flush=True)
        elif _ct_policy == "Физическое лицо" and _cur_policy == "USD" and _direct_params.get("cost") is not None:
            _rate = float(settings.tools.usd_byn_rate)
            _old_cost = float(_direct_params["cost"])
            _new_cost = round(_old_cost * _rate, 2)
            _currency_conversion = {
                "from": "USD", "to": "BYN",
                "amount_from": _old_cost, "amount_to": _new_cost,
                "rate": _rate, "rate_source": "MVP hardcoded",
            }
            _direct_params["cost"] = _new_cost
            _direct_params["currency"] = "BYN"
            # Fix 1.2 (2026-04-19) — stash the USD figures on the profile so
            # the readback / calc-result / SMS paths can disclose both
            # amounts to the client. Without this, downstream renders only
            # see the converted BYN cost and the caller loses sight of the
            # USD number they actually quoted.
            _p.original_cost = _old_cost
            _p.original_currency = "USD"
            print(f"[DirectTool] USD->BYN: {_old_cost} -> {_new_cost} @ {_rate}", flush=True)

        # Check subject restrictions for individuals before calling API.
        # Only block when client_type is EXPLICITLY individual (not defaulted).
        # If client_type is unknown, the classifier should have set clarify_client_type.
        # This is a lightweight fallback in case the classifier missed it.
        _subj_lower = (_direct_params.get("subject") or "").lower()
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
                from .tools.calculator import IncompleteProfileError as _IncProf
                try:
                    _direct_tool_result = await asyncio.to_thread(calc_tool.execute, _direct_params, {})
                except _IncProf as _inc_exc:
                    # Profile incomplete OR params out of range / bad type.
                    print(f"[DirectTool] IncompleteProfile: {_inc_exc.missing}", flush=True)
                    _direct_tool_result = None
                    _extracted_hints["_missing_profile_fields"] = _inc_exc.missing
                    _oor_msg = _format_invalid_params_msg(_inc_exc.missing)
                    if _oor_msg:
                        _extracted_hints["_invalid_params_msg"] = _oor_msg
                        print(f"[DirectTool] OOR user msg: {_oor_msg}", flush=True)
                if _direct_tool_result is not None:
                    _tool_ok = _direct_tool_result.get("ok", False)
                    # Attach conversion disclosure to result for presentation layer
                    if _currency_conversion is not None:
                        _direct_tool_result["currency_conversion"] = _currency_conversion
                    print(f"[DirectTool] result: ok={_tool_ok}", flush=True)
                    await broadcast_sip_event({
                        "type": "sip.tool.result",
                        "call_id": session_id,
                        "tool": "calculator",
                        "ok": _tool_ok,
                    })
                    session.tool_calls_this_turn = getattr(session, 'tool_calls_this_turn', [])
                    print(
                        f"[DirectTool] invoking calculator with params={_direct_params} "
                        f"(path=direct, is_param_change={locals().get('_is_param_change_for_gate', False)})",
                        flush=True,
                    )
                    session.tool_calls_this_turn.append({
                        "tool": "calculator",
                        "params": _direct_tool_result.get("params", _direct_params),
                        "result": _direct_tool_result,
                        "ok": _tool_ok,
                    })
                    # ── Circuit-breaker bookkeeping ──
                    _calc_params = _direct_tool_result.get("params", _direct_params)
                    _calc_sig = str(sorted(_calc_params.items())) if isinstance(_calc_params, dict) else ""
                    _calc_ok = bool(_direct_tool_result.get("ok"))
                    if _calc_sig and _calc_sig == session.last_calc_signature and not _calc_ok:
                        session.consecutive_calc_failures += 1
                    elif _calc_ok:
                        session.consecutive_calc_failures = 0
                        session.last_calc_signature = _calc_sig
                    else:
                        session.consecutive_calc_failures = 1
                        session.last_calc_signature = _calc_sig
            except Exception as _texc:
                print(f"[DirectTool] ERROR: {_texc}", flush=True)
                _direct_tool_result = None

    # ── Clarify client type: classifier detected restricted subject/currency ──
    # Bug B (live call 6a9d359b 2026-04-26) — gate on actual profile state.
    # The 4B classifier sometimes hallucinates action=clarify_client_type
    # when conversation history shows client_type was asked recently, even
    # if it was already captured. Profile is the source of truth: when
    # client_type is filled, ignore the action label and let the normal
    # response path handle the turn. Mirrors the same principle the v2
    # baseline applied to the COLLECTING-clarify gate (commit 9579218).
    _clarify_needed = (
        needs_tool
        and _extracted_hints.get("action") == "clarify_client_type"
        and not (session.client_profile.client_type or "").strip()
    )
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
                f"{memory_block}"
                f"Клиент хочет рассчитать лизинг. {_reason}. "
                "Если сообщение клиента действительно связано с расчётом, "
                "спроси КРАТКО тип клиента: физическое или юридическое лицо. "
                "Если сообщение не про расчёт, просто ответь на вопрос клиента.\n\n"
                f"Сообщение клиента: {message}"
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
                f"{memory_block}"
                f"{_param_summary}\n"
                f"Сообщение клиента: {message}\n\n"
                "Клиент хочет изменить параметры. Назови текущие значения (аванс, срок) "
                "и спроси, какой именно параметр и на какое значение хочет изменить."
            )},
        ]
        tool_schemas = []
    elif _direct_tool_result and _direct_tool_result.get("ok"):
        # Tool executed successfully: LLM only presents the result.
        # Fix 1.1 (2026-04-19) — all numeric fields come from the deterministic
        # renderer in profile_prompts; the LLM below paraphrases tone, never
        # synthesises figures. session_analyzer greps [deterministic_readback]
        # to confirm this path drove the voiced output.
        _result_summary = render_calc_result(_direct_tool_result)
        print(
            f"[deterministic_readback] source=render_calc_result "
            f"session={session_id[:8]} chars={len(_result_summary)}",
            flush=True,
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
    elif _extracted_hints.get("_invalid_params_msg"):
        # Fix 39: one or more calculator params out of allowed range. Surface the
        # specific range to the client instead of letting the LLM improvise a
        # number ("60 максимально" hallucination observed 2026-04-18).
        _oor_message = _extracted_hints["_invalid_params_msg"]
        print(f"[DirectTool] OOR branch: {_oor_message[:80]}", flush=True)
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Клиент запросил параметр вне допустимого диапазона: {_oor_message}.\n"
                f"Сообщение клиента: {message}\n\n"
                "Скажи клиенту ТОЛЬКО указанный диапазон (одной короткой фразой) "
                "и спроси, какое значение в этом диапазоне он выбирает. "
                "НЕ называй других чисел, НЕ предлагай 'типичные' значения."
            )},
        ]
        tool_schemas = []
    elif needs_tool:
        # TOOL path: classifier detected tool intent but insufficient data for direct execution.
        # Fall back to LLM function calling.
        prev_calc_context = ""
        _prev_all_calls = session.tool_calls_history + session.tool_calls_this_turn
        if _prev_all_calls:
            last_calc = next(
                (tc for tc in reversed(_prev_all_calls)
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
        # Explicitly clear tool schemas: on non-tool turns the LLM must NOT be
        # able to call calculator autonomously. Without this, the LLM re-invokes
        # calculator with memorized params on info-question turns, producing
        # stale calc results alongside the real RAG answer.
        tool_schemas = []
        print(f"[LLM] RAG path: tools cleared for non-tool turn", flush=True)
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
                    # ── Circuit-breaker bookkeeping (calculator only) ──
                    if func_name == "calculator":
                        _calc_params = filled_params
                        _calc_sig = str(sorted(_calc_params.items())) if isinstance(_calc_params, dict) else ""
                        _calc_ok = bool(result.get("ok"))
                        if _calc_sig and _calc_sig == session.last_calc_signature and not _calc_ok:
                            session.consecutive_calc_failures += 1
                        elif _calc_ok:
                            session.consecutive_calc_failures = 0
                            session.last_calc_signature = _calc_sig
                        else:
                            session.consecutive_calc_failures = 1
                            session.last_calc_signature = _calc_sig
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
            # Ground high-risk facts (addresses/phones/names) to retrieved chunks.
            # Only runs on RAG intent turns - TOOL turns have no chunks to ground against.
            if not needs_tool and _chunk_texts:
                item = replace_ungrounded(item, _chunk_texts)
            all_sentences.append(item)
        await _orig_put(item)

    sentence_queue.put = _tracking_put  # type: ignore[assignment]

    session.assistant_speaking = True
    session.interrupted = False
    session._tts_start_time = 0  # reset for new TTS warmup
    producer_task = asyncio.create_task(llm_producer())
    consumer_task = asyncio.create_task(tts_consumer())
    await asyncio.gather(producer_task, consumer_task)
    # Save transcript IMMEDIATELY after LLM+TTS finishes (before audio loop
    # can spawn the next utterance task). Uses chat_session from line 627
    # to avoid race conditions with concurrent tasks.
    full_answer = " ".join(all_sentences)
    if session.interrupted and full_answer:
        full_answer += " [прервано клиентом]"
    if full_answer:
        _append_turn(chat_session, message, full_answer, settings.app.memory_turns)
        state.update(chat_session)
        session.turn_count += 1
        print(f"[Jambonz:{session_id[:8]}] Transcript saved ({len(all_sentences)} sentences{', interrupted' if session.interrupted else ''})", flush=True)
        # Belt-and-suspenders: emit the assembled final answer so the SIP
        # monitor always shows the complete assistant message even if a
        # streaming delta got dropped or the consumer broke on barge-in
        # before the last sentence was broadcast.
        await broadcast_sip_event({
            "type": "sip.llm.final",
            "call_id": session_id,
            "text": full_answer,
            "interrupted": bool(session.interrupted),
        })

    # ── Per-turn latency breakdown ──
    # Grep-friendly single-line summary for outlier analysis. One line per turn.
    try:
        _t_now = time.time()
        _lat_classifier_ms = int(_t_classify_ms) if '_t_classify_ms' in locals() and _t_classify_ms is not None else -1
        _lat_rag_ms = int(_t_rag_ms) if '_t_rag_ms' in locals() and _t_rag_ms is not None else -1
        _lat_llm_first_ms = (
            int((t_llm_first_token - t_speech_stopped) * 1000)
            if t_llm_first_token is not None else -1
        )
        _lat_tts_first_ms = (
            int((t_tts_first_chunk - t_speech_stopped) * 1000)
            if t_tts_first_chunk is not None else -1
        )
        _lat_total_e2e_ms = int((_t_now - t_speech_stopped) * 1000)
        _lat_user_len = len(llm_messages[-1].get("content", "")) if llm_messages else 0
        _lat_out_tokens = sum(1 for _s in all_sentences for _ in _s.split())  # approximate
        _lat_path = (
            "rag" if (not tool_schemas or not needs_tool) and _direct_tool_result is None
            else ("direct" if _direct_tool_result is not None else "tool")
        )
        print(
            f"[LATENCY:{session_id[:8]}] "
            f"classifier_ms={_lat_classifier_ms} "
            f"rag_ms={_lat_rag_ms} "
            f"llm_first_ms={_lat_llm_first_ms} "
            f"tts_first_ms={_lat_tts_first_ms} "
            f"total_e2e_ms={_lat_total_e2e_ms} "
            f"user_len={_lat_user_len} "
            f"out_tokens={_lat_out_tokens} "
            f"path={_lat_path}",
            flush=True,
        )
    except Exception as _lat_exc:  # noqa: BLE001
        print(f"[LATENCY:{session_id[:8]}] log failed: {_lat_exc}", flush=True)
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

    # Transcript already saved above (right after asyncio.gather).

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
