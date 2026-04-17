"""Regression tests for interruptible tool-result TTS in the Jambonz shim.

Fix 6: the `_JambonzWebSocketShim.send_json` chunk loop for
`response.output_audio.delta` must check `session.interrupted` between
chunks and emit a `killAudio` control frame when it breaks early.
"""

from __future__ import annotations

import asyncio
import base64
import types


class _FakeWS:
    """Minimal fake for FastAPI's WebSocket — records what was sent."""

    def __init__(self) -> None:
        self.bytes_sent: list[bytes] = []
        self.texts_sent: list[str] = []

    async def send_bytes(self, b: bytes) -> None:
        self.bytes_sent.append(b)

    async def send_text(self, t: str) -> None:
        self.texts_sent.append(t)


def test_jambonz_shim_sends_full_audio_when_not_interrupted():
    from backend.app import _JambonzWebSocketShim

    async def _run():
        fake_ws = _FakeWS()
        session = types.SimpleNamespace(interrupted=False)
        shim = _JambonzWebSocketShim(fake_ws, "sess", session=session)
        # 10 full chunks = 19200 bytes PCM (1920 bytes = 40ms @ 24kHz s16 mono).
        pcm = b"\x00" * 19200
        b64 = base64.b64encode(pcm).decode()
        await shim.send_json({"type": "response.output_audio.delta", "delta": b64})
        assert len(fake_ws.bytes_sent) == 10
        assert sum(len(b) for b in fake_ws.bytes_sent) == 19200
        assert fake_ws.texts_sent == []  # no killAudio when not interrupted

    asyncio.run(_run())


def test_jambonz_shim_breaks_early_on_interrupt():
    from backend.app import _JambonzWebSocketShim

    async def _run():
        fake_ws = _FakeWS()
        session = types.SimpleNamespace(interrupted=False)
        shim = _JambonzWebSocketShim(fake_ws, "sess", session=session)

        # Flip interrupt BEFORE the send: no chunks should be sent,
        # but killAudio IS sent so mod_audio_fork drops buffered audio.
        session.interrupted = True
        pcm = b"\x00" * 19200
        b64 = base64.b64encode(pcm).decode()
        await shim.send_json({"type": "response.output_audio.delta", "delta": b64})
        assert len(fake_ws.bytes_sent) == 0
        assert any("killAudio" in t for t in fake_ws.texts_sent)

    asyncio.run(_run())


def test_jambonz_shim_without_session_preserves_old_behavior():
    from backend.app import _JambonzWebSocketShim

    async def _run():
        fake_ws = _FakeWS()
        shim = _JambonzWebSocketShim(fake_ws, "sess")  # no session kwarg
        pcm = b"\x00" * 3840  # 2 full chunks
        b64 = base64.b64encode(pcm).decode()
        await shim.send_json({"type": "response.output_audio.delta", "delta": b64})
        assert len(fake_ws.bytes_sent) == 2
        assert fake_ws.texts_sent == []

    asyncio.run(_run())
