"""Regression tests for _emit_plain_assistant_response killAudio on interrupt.

Fix 14: when barge-in flips `session.interrupted` mid-playback, the chunk
loop in `_emit_plain_assistant_response` must break early AND emit a
`killAudio` control frame. Without the frame, Jambonz mod_audio_fork keeps
playing the 100-500ms of PCM already buffered in FreeSWITCH downstream of
our last `send_json` delta, so the caller still hears the readback/
change-confirm/clarify tail for a noticeable beat after barging in.
"""

from __future__ import annotations

import asyncio
import base64
import types


class _FakeWS:
    """Minimal fake for FastAPI's WebSocket — records what was sent."""

    def __init__(self) -> None:
        self.json_sent: list[dict] = []
        self.texts_sent: list[str] = []

    async def send_json(self, data: dict) -> None:
        self.json_sent.append(data)

    async def send_text(self, data: str) -> None:
        self.texts_sent.append(data)


def test_emit_plain_sends_killaudio_on_interrupt(monkeypatch):
    from backend import app as app_module

    async def _run():
        # Fake TTS returns 1 sec of silence so we have many chunks
        # (24000 samples * 2 bytes = 48000 bytes = 25 chunks of 1920).
        pcm_bytes = b"\x00" * (24000 * 2)
        audio_b64 = base64.b64encode(pcm_bytes).decode()
        monkeypatch.setattr(
            app_module, "synthesize_audio",
            lambda text, session_id: {"audio_b64": audio_b64, "sample_rate_hz": 24000},
        )
        # Stub out state.get so transcript append path doesn't crash.
        monkeypatch.setattr(app_module.state, "get", lambda sid: None)

        fake_ws = _FakeWS()
        session = types.SimpleNamespace(
            assistant_speaking=False, interrupted=False, client_profile=None,
        )
        # Flip interrupted after a couple of audio chunks have been sent.
        _calls = {"n": 0}
        _orig_send_json = fake_ws.send_json

        async def _patched_send_json(data):
            await _orig_send_json(data)
            if data.get("type") == "response.output_audio.delta":
                _calls["n"] += 1
                if _calls["n"] == 3:
                    session.interrupted = True

        fake_ws.send_json = _patched_send_json

        await app_module._emit_plain_assistant_response(
            "hello", fake_ws, "sess-1", backend="test", session=session,
        )

        audio_deltas = [
            m for m in fake_ws.json_sent
            if m.get("type") == "response.output_audio.delta"
        ]
        # Some audio deltas went through before the interrupt, but far
        # fewer than the full 25 chunks the loop would have produced.
        assert 1 <= len(audio_deltas) < 25
        # killAudio control frame was sent via send_text.
        assert any("killAudio" in t for t in fake_ws.texts_sent)

    asyncio.run(_run())


def test_emit_plain_no_killaudio_without_interrupt(monkeypatch):
    from backend import app as app_module

    async def _run():
        pcm_bytes = b"\x00" * 3840  # 2 full chunks, no interrupt
        audio_b64 = base64.b64encode(pcm_bytes).decode()
        monkeypatch.setattr(
            app_module, "synthesize_audio",
            lambda text, session_id: {"audio_b64": audio_b64, "sample_rate_hz": 24000},
        )
        monkeypatch.setattr(app_module.state, "get", lambda sid: None)

        fake_ws = _FakeWS()
        session = types.SimpleNamespace(
            assistant_speaking=False, interrupted=False, client_profile=None,
        )

        await app_module._emit_plain_assistant_response(
            "hello", fake_ws, "sess-2", backend="test", session=session,
        )

        audio_deltas = [
            m for m in fake_ws.json_sent
            if m.get("type") == "response.output_audio.delta"
        ]
        assert len(audio_deltas) == 2
        # Fix 33: _emit_plain sends a preemptive killAudio at the start to
        # flush any lingering audio from a concurrent older TTS (e.g. intro
        # still pushing chunks when readback begins). Exactly one killAudio
        # is expected even in the happy path. No second killAudio at the
        # end because no interrupt fired.
        _kills = [t for t in fake_ws.texts_sent if "killAudio" in t]
        assert len(_kills) == 1, fake_ws.texts_sent

    asyncio.run(_run())
