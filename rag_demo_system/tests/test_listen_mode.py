"""Regression tests for the listen_mode auto-exit flag-reset contract.

Fix 17: when listen_mode is entered (user said "стоп"), the stop-handler sets
``session.interrupted = True`` to kill the then-current TTS playback. When the
auto-exit loop later fires and emits the "Слушаю Вас" prompt, the Jambonz
shim's ``send_json`` chunk loop (Fix 6) honors ``session.interrupted`` and
will drop the audio if it is still True.

So the auto-exit task MUST reset ``session.interrupted = False`` BEFORE it
sends the audio delta, and must leave ``session.assistant_speaking = False``
after ``response.done`` so barge-in on the next turn works.
"""

from __future__ import annotations

import asyncio
import base64
import types
from unittest.mock import patch


class _FakeWS:
    """Minimal fake WebSocket that records everything sent."""

    def __init__(self, session) -> None:
        self.session = session
        self.json_sent: list[dict] = []
        self.texts_sent: list[str] = []
        # Snapshot of session.interrupted at the moment each audio delta
        # was sent. This is what the Jambonz shim would observe.
        self.interrupted_at_audio: list[bool] = []

    async def send_json(self, data: dict) -> None:
        if data.get("type") == "response.output_audio.delta":
            self.interrupted_at_audio.append(bool(self.session.interrupted))
        self.json_sent.append(data)

    async def send_text(self, data: str) -> None:
        self.texts_sent.append(data)


def _make_session(*, interrupted: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        listen_mode=True,
        listen_mode_until=0.0,  # already past -> fire immediately
        interrupted=interrupted,
        assistant_speaking=False,
        backend="test",
    )


def test_listen_mode_auto_exit_resets_interrupted_before_audio(monkeypatch):
    """At the moment the audio delta is sent, session.interrupted must be
    False, otherwise the Jambonz shim would silently drop the 'Слушаю Вас'
    audio and the caller would hear nothing."""
    from backend import listen_mode

    async def _run():
        # Stub the lazy-imported synthesize_audio so we don't load the TTS
        # engine. Return a tiny 1-chunk PCM so there's something to send.
        pcm = b"\x00" * 1920  # 1 full chunk @ 24kHz s16 mono
        audio_b64 = base64.b64encode(pcm).decode()
        import backend.voice_adapters as va
        monkeypatch.setattr(
            va,
            "synthesize_audio",
            lambda text, session_id: {
                "audio_b64": audio_b64,
                "sample_rate_hz": 24000,
            },
        )
        # Patch asyncio.sleep inside listen_mode so we don't wait.
        async def _instant_sleep(_):
            return
        monkeypatch.setattr(listen_mode.asyncio, "sleep", _instant_sleep)

        session = _make_session(interrupted=True)  # stop-handler set this
        ws = _FakeWS(session)

        task = listen_mode.spawn_auto_exit_task(session, ws, "test-sess-17")
        await task

        # CRITICAL: at least one audio delta was sent, and at every audio
        # delta moment session.interrupted was False.
        assert ws.interrupted_at_audio, "no audio delta was sent"
        assert all(v is False for v in ws.interrupted_at_audio), (
            "listen_mode auto-exit sent audio while session.interrupted was True "
            "— Jambonz shim would drop it."
        )

        # Final session state: flags cleaned up so the next turn works.
        assert session.listen_mode is False, "listen_mode flag not cleared"
        assert session.interrupted is False, "interrupted flag not cleared"
        assert session.assistant_speaking is False, (
            "assistant_speaking must be False after response.done"
        )

        # Expected message sequence.
        types_sent = [m.get("type") for m in ws.json_sent]
        assert "response.output_text.delta" in types_sent
        assert "response.output_audio.delta" in types_sent
        assert "response.done" in types_sent

    asyncio.run(_run())


def test_listen_mode_auto_exit_clears_assistant_speaking_after_done(monkeypatch):
    """After the turn ends, assistant_speaking must be False so the next
    barge-in / VAD cycle works correctly."""
    from backend import listen_mode

    async def _run():
        pcm = b"\x00" * 1920
        audio_b64 = base64.b64encode(pcm).decode()
        import backend.voice_adapters as va
        monkeypatch.setattr(
            va,
            "synthesize_audio",
            lambda text, session_id: {
                "audio_b64": audio_b64,
                "sample_rate_hz": 24000,
            },
        )
        async def _instant_sleep(_):
            return
        monkeypatch.setattr(listen_mode.asyncio, "sleep", _instant_sleep)

        session = _make_session(interrupted=False)
        # Pre-condition: assistant not currently speaking.
        session.assistant_speaking = False

        ws = _FakeWS(session)
        task = listen_mode.spawn_auto_exit_task(session, ws, "test-sess-17b")
        await task

        assert session.assistant_speaking is False

    asyncio.run(_run())


def test_listen_mode_auto_exit_skips_when_flag_cleared(monkeypatch):
    """If the client spoke first and cleared listen_mode before the task
    wakes, the task exits silently without sending anything."""
    from backend import listen_mode

    async def _run():
        async def _instant_sleep(_):
            return
        monkeypatch.setattr(listen_mode.asyncio, "sleep", _instant_sleep)

        session = _make_session(interrupted=True)
        session.listen_mode = False  # client already spoke

        ws = _FakeWS(session)
        task = listen_mode.spawn_auto_exit_task(session, ws, "test-sess-17c")
        await task

        # Nothing was sent: neither text, audio, nor response.done.
        assert ws.json_sent == []
        # And we did NOT touch session.interrupted in the skip path
        # (no need to - that's the caller's responsibility).
        assert session.interrupted is True

    asyncio.run(_run())
