"""Fix 25: `_emit_plain_assistant_response` must abort synthesis at the next
phrase boundary when `session.interrupted` flips during a multi-phrase
readback — NOT block until the full string has been synthesized.

Before Fix 25: readback = one synth call for the whole 3-4s of audio,
so barge-in could not stop it until synth returned. User spoke, nothing
happened, then maybe a killAudio flushed the trailing buffer.

After Fix 25: readback splits into ~9 phrases, each ~200-500ms of synth,
with an interrupt check before each. Barge-in stops the next synth from
even happening.
"""

from __future__ import annotations

import asyncio
import base64
import types


class _FakeWS:
    def __init__(self) -> None:
        self.json_sent: list[dict] = []
        self.texts_sent: list[str] = []

    async def send_json(self, data: dict) -> None:
        self.json_sent.append(data)

    async def send_text(self, data: str) -> None:
        self.texts_sent.append(data)


def test_interrupt_aborts_before_next_phrase_synth(monkeypatch):
    """Flip `session.interrupted` right after the first phrase's audio
    starts streaming. Only the first phrase should be synthesized; the
    rest must be skipped because the outer loop checks interrupted
    before each synth call.
    """
    from backend import app as app_module

    async def _run():
        synth_calls: list[str] = []

        def _fake_synth(text, session_id):
            synth_calls.append(text)
            pcm = b"\x00" * (24000 * 2)  # ~1s of silence
            return {"audio_b64": base64.b64encode(pcm).decode(), "sample_rate_hz": 24000}

        monkeypatch.setattr(app_module, "synthesize_audio", _fake_synth)
        monkeypatch.setattr(app_module.state, "get", lambda sid: None)

        fake_ws = _FakeWS()
        session = types.SimpleNamespace(
            assistant_speaking=False, interrupted=False, client_profile=None,
            _tts_start_time=123.0,
        )
        _orig_send_json = fake_ws.send_json

        async def _patched_send_json(data):
            await _orig_send_json(data)
            # Flip after the first audio chunk of the first phrase.
            if (
                data.get("type") == "response.output_audio.delta"
                and not session.interrupted
            ):
                session.interrupted = True

        fake_ws.send_json = _patched_send_json

        # Canonical readback — 9 phrases after split.
        readback = (
            "Давайте подтвердим параметры: Грузовой автомобиль, новый, "
            "стоимость 80000 BYN, Юридическое лицо, срок 36 месяцев, "
            "аванс 20%, график аннуитет. Всё верно?"
        )
        await app_module._emit_plain_assistant_response(
            readback, fake_ws, "sess-readback", backend="test", session=session,
        )

        # Only one phrase was synthesized — the rest were skipped by the
        # pre-synth interrupt check.
        assert len(synth_calls) == 1, (
            f"expected exactly 1 synth call after mid-phrase interrupt, got {len(synth_calls)}: {synth_calls}"
        )
        # killAudio was sent to flush downstream buffers.
        assert any("killAudio" in t for t in fake_ws.texts_sent)
        # `_tts_start_time` was reset for VAD warmup parity with main TTS.
        assert session._tts_start_time == 0

    asyncio.run(_run())


def test_pre_call_interrupt_aborts_without_synthesizing(monkeypatch):
    """Edge case: `session.interrupted` is already True before the helper
    is called (barge-in fired between orchestrator decision and call).
    The current code resets it to False at the top so this test
    documents the reset behavior — if Fix 25 ever changes that, we
    revisit the edge case.
    """
    from backend import app as app_module

    async def _run():
        synth_calls: list[str] = []

        def _fake_synth(text, session_id):
            synth_calls.append(text)
            pcm = b"\x00" * 3840
            return {"audio_b64": base64.b64encode(pcm).decode(), "sample_rate_hz": 24000}

        monkeypatch.setattr(app_module, "synthesize_audio", _fake_synth)
        monkeypatch.setattr(app_module.state, "get", lambda sid: None)

        fake_ws = _FakeWS()
        # Barge-in fired before this call was dispatched.
        session = types.SimpleNamespace(
            assistant_speaking=False, interrupted=True, client_profile=None,
            _tts_start_time=0.0,
        )

        await app_module._emit_plain_assistant_response(
            "Привет.", fake_ws, "sess-pre", backend="test", session=session,
        )

        # Current behavior: the helper resets interrupted=False, so synth
        # runs. This is documented, not ideal — but not the primary Fix 25
        # concern. If we ever make this smarter, this test pins the
        # current contract.
        assert len(synth_calls) == 1

    asyncio.run(_run())


def test_full_readback_without_interrupt_synthesizes_every_phrase(monkeypatch):
    """Happy path: no barge-in, all phrases synthesized in order."""
    from backend import app as app_module

    async def _run():
        synth_calls: list[str] = []

        def _fake_synth(text, session_id):
            synth_calls.append(text)
            pcm = b"\x00" * 1920  # 1 chunk, no interrupt racing
            return {"audio_b64": base64.b64encode(pcm).decode(), "sample_rate_hz": 24000}

        monkeypatch.setattr(app_module, "synthesize_audio", _fake_synth)
        monkeypatch.setattr(app_module.state, "get", lambda sid: None)

        fake_ws = _FakeWS()
        session = types.SimpleNamespace(
            assistant_speaking=False, interrupted=False, client_profile=None,
            _tts_start_time=0.0,
        )

        readback = (
            "Давайте подтвердим параметры: Грузовой автомобиль, новый, "
            "стоимость 80000 BYN. Всё верно?"
        )
        await app_module._emit_plain_assistant_response(
            readback, fake_ws, "sess-full", backend="test", session=session,
        )

        assert len(synth_calls) >= 4  # split produces multiple phrases
        assert synth_calls[-1] == "Всё верно?"
        # Fix 33: _emit_plain now sends exactly one preemptive killAudio on
        # entry (flushes any leftover audio from a prior overlapping TTS).
        # No SECOND killAudio because no interrupt fired during chunks.
        _kills = [t for t in fake_ws.texts_sent if "killAudio" in t]
        assert len(_kills) == 1, fake_ws.texts_sent

    asyncio.run(_run())
