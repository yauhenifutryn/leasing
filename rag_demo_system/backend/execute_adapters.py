"""Production adapters that bridge the apply_turn + execute_action
contract (backend.stream / tts.say / calc.calculate / rag_future.result)
to the real orchestrator plumbing in `_stream_voice_response`.

Phase 3.D wire-in (Task 21). Each adapter is a thin shim — zero business
logic — so execute_action stays testable with fakes and the production
paths stay testable by swapping only the shim under test.

Mapping:
  - LLMStreamBackend  → iter_openai_compatible_stream_events
  - TtsSink           → synthesize_audio + websocket.send_json (phrase-
                         boundary path that mirrors app.py:2700-2739)
  - CalcAdapter       → get_all_tools()["calculator"].execute
  - RagFuture         → asyncio.Task[retrieval_dict] → string context
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Optional

from .llm import iter_openai_compatible_stream_events
from .tools import get_all_tools
from .voice_adapters import synthesize_audio


class LLMStreamBackend:
    """Wraps the sync OpenAI-compatible event stream in the async-gen
    token contract expected by `execute_action`'s FireLLMFallback
    handler.

    Pulls events off the sync generator via `asyncio.to_thread(next, ...)`
    so the event loop keeps turning (barge-in tasks, RTC audio push)
    while the LLM back-end is generating. This mirrors legacy behaviour
    at app.py:2551-2579 and is the contract the Task 18 latency test
    exercises.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_sec: float,
        system_prompt: str = "",
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec
        self._system_prompt = system_prompt

    async def stream(self, *, messages) -> AsyncGenerator[str, None]:
        full = list(messages)
        if self._system_prompt and (not full or full[0].get("role") != "system"):
            full = [{"role": "system", "content": self._system_prompt}, *full]
        raw_stream = iter_openai_compatible_stream_events(
            base_url=self._base_url,
            model=self._model,
            messages=full,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout_sec=self._timeout_sec,
            tools=None,
        )
        sentinel = object()
        while True:
            event = await asyncio.to_thread(next, raw_stream, sentinel)
            if event is sentinel:
                break
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            token = delta.get("content") or ""
            if token:
                yield token


class TtsSink:
    """Phrase-level, chunk-aware TTS sink with barge-in parity.

    Mirrors the barge-in mechanics of the legacy `_speak_tts` helper at
    `app.py:82-228` so SIP callers can interrupt mid-sentence with the
    same latency they saw before Section 3:

      1. Phrase-level synth via `split_for_tts_streaming` — each phrase
         is ~200–500ms of audio, so new text never waits for a long
         mono-synth to finish before the next barge-in check.
      2. Per-chunk push (1920-byte / 40ms frames at 24kHz 16-bit mono)
         with `session.interrupted` polled before EVERY frame. On
         interrupt the chunk loop exits immediately so no further PCM
         reaches Jambonz mod_audio_fork's play queue.
      3. `killAudio` control frame sent on interrupt to flush the
         downstream buffer — without this the caller hears 300–500ms
         of tail audio before the bot goes quiet.

    Deliberately omits the 50ms playback-wait loop that `_speak_tts`
    adds at app.py:200-212 — for the FireLLMFallback streaming path we
    want the next sentence's synth to overlap with the prior sentence's
    playback (spec §7.2 #2). The effect on barge-in latency is zero
    because the chunk-level interrupt still fires immediately.
    """

    # 1920 bytes = 40ms at 24kHz 16-bit mono — matches Jambonz frame.
    _CHUNK_SIZE = 1920

    def __init__(
        self,
        *,
        websocket: Any,
        session_id: str,
        session: Any = None,
        rtc_handler: Any = None,
    ) -> None:
        self._ws = websocket
        self._session_id = session_id
        self._session = session
        self._rtc = rtc_handler

    async def say(self, text: str) -> None:
        if not text:
            return
        if self._interrupted():
            return

        # Text-delta goes out ONCE per sentence (monitor + browser UI).
        try:
            await self._ws.send_json({
                "type": "response.output_text.delta",
                "session_id": self._session_id,
                "delta": text + " ",
            })
        except Exception:  # noqa: BLE001
            self._flag_interrupted()
            await self._send_kill_audio()
            return

        from .text_utils import split_for_tts_streaming  # lazy — cycle safe

        phrases = split_for_tts_streaming(text) or [text]
        aborted = False
        for phrase in phrases:
            if self._interrupted():
                aborted = True
                break
            try:
                audio_resp = await asyncio.to_thread(
                    synthesize_audio, phrase, self._session_id,
                )
            except Exception:  # noqa: BLE001
                continue
            audio_b64 = audio_resp.get("audio_b64") or ""
            if not audio_b64:
                continue
            if self._interrupted():
                aborted = True
                break
            if await self._push_chunks(audio_b64, audio_resp):
                aborted = True
                break

        if aborted:
            await self._send_kill_audio()

    async def _push_chunks(self, audio_b64: str, audio_resp: dict) -> bool:
        """Push PCM in 40ms frames, checking barge-in before each send.
        Returns True if interrupted mid-stream."""
        import base64 as _b64
        pcm = _b64.b64decode(audio_b64)
        sample_rate = audio_resp.get("sample_rate_hz") or 24000

        for i in range(0, len(pcm), self._CHUNK_SIZE):
            if self._interrupted():
                return True
            chunk = pcm[i:i + self._CHUNK_SIZE]
            try:
                if self._rtc is not None:
                    self._rtc.tts_track.push_audio(chunk)
                else:
                    await self._ws.send_json({
                        "type": "response.output_audio.delta",
                        "session_id": self._session_id,
                        "delta": _b64.b64encode(chunk).decode(),
                        "sample_rate_hz": sample_rate,
                    })
            except Exception:  # noqa: BLE001
                self._flag_interrupted()
                return True
        return False

    async def _send_kill_audio(self) -> None:
        """Control frame to flush mod_audio_fork's play queue on the
        Jambonz side. Legacy parity — app.py:214-218."""
        if self._rtc is not None:
            return  # RTC track has no equivalent buffer
        try:
            await self._ws.send_text(json.dumps({"type": "killAudio"}))
        except Exception:  # noqa: BLE001
            pass

    def _interrupted(self) -> bool:
        if self._session is None:
            return False
        return bool(getattr(self._session, "interrupted", False))

    def _flag_interrupted(self) -> None:
        if self._session is not None:
            try:
                self._session.interrupted = True
            except Exception:  # noqa: BLE001
                pass


class CalcAdapter:
    """Wraps the sync calculator tool in the async contract expected by
    `execute_action`'s FireCalc handler.

    Defaults are filled through the tool's own `fill_defaults` helper so
    the calc params apply_turn built from ProfileSnapshot get the same
    treatment as the legacy DirectTool path (app.py:2627-2635).
    """

    def __init__(self, *, session_id: str, client_phone: Optional[str] = None) -> None:
        self._session_id = session_id
        self._client_phone = client_phone

    async def calculate(self, params: dict) -> dict:
        calc_tool = get_all_tools().get("calculator")
        if calc_tool is None:
            raise RuntimeError("calculator tool unavailable")
        filled_params, defaulted = calc_tool.fill_defaults(params)
        # Broadcast a `sip.tool.start` event to the SIP monitor page so
        # the operator sees the tool call land in the UI. Legacy
        # DirectTool path emits the same event at app.py:2415-2420; this
        # adapter must mirror it or the monitor shows no calculator line
        # even when apply_turn successfully dispatches FireCalc.
        await self._broadcast({
            "type": "sip.tool.start",
            "call_id": self._session_id,
            "tool": "calculator",
            "params": filled_params,
        })
        try:
            result = await asyncio.to_thread(
                calc_tool.execute,
                filled_params,
                {
                    "session_id": self._session_id,
                    "client_phone": self._client_phone,
                },
            )
        except Exception:
            await self._broadcast({
                "type": "sip.tool.result",
                "call_id": self._session_id,
                "tool": "calculator",
                "ok": False,
            })
            raise
        if isinstance(result, dict) and defaulted:
            result.setdefault("defaulted", defaulted)
        await self._broadcast({
            "type": "sip.tool.result",
            "call_id": self._session_id,
            "tool": "calculator",
            "ok": bool(result.get("ok", False)) if isinstance(result, dict) else False,
        })
        return result

    @staticmethod
    async def _broadcast(event: dict) -> None:
        """Lazy import of broadcast_sip_event to avoid a circular import
        at module load (app.py imports this module via execute_adapters
        → and broadcasts live on app.py). Silently swallows failures
        so a dead monitor page never breaks a live call."""
        try:
            from .app import broadcast_sip_event
            await broadcast_sip_event(event)
        except Exception:  # noqa: BLE001
            pass


class RagFuture:
    """Lazy adapter over the speculative RAG `asyncio.Task`. Resolves
    the retrieval dict on first `.result()` call and caches the
    context string so a handler-retry (shouldn't happen, defensive)
    doesn't double-await.

    Invariant #1: FireLLMFallback is the ONLY caller. If apply_turn
    returns a different TurnAction the task is left running; it's CPU-
    bound inside a thread and garbage-collects without blocking the
    event loop.
    """

    def __init__(self, task) -> None:
        self._task = task
        self._resolved: Optional[str] = None

    async def result(self) -> str:
        if self._resolved is not None:
            return self._resolved
        retrieval = await self._task
        final_chunks = (retrieval or {}).get("final") or []
        self._resolved = "\n\n".join(
            f"[Fragment {i + 1}]\n{chunk.get('text', '')}"
            for i, chunk in enumerate(final_chunks)
        )
        return self._resolved
