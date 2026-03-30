"""Tests for server-side Silero VAD wrapper."""
import struct
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Provide a fake 'torch' module so that backend.vad can be imported and its
# feed() method can call torch.FloatTensor without the real torch installed.
# ---------------------------------------------------------------------------
_mock_torch = ModuleType("torch")


class _FakeTensor:
    """Minimal stand-in for torch.FloatTensor; division returns self."""

    def __init__(self, data):
        self._data = data

    def __truediv__(self, other):
        return self


_mock_torch.FloatTensor = _FakeTensor  # type: ignore[attr-defined]
_mock_torch.jit = MagicMock()  # type: ignore[attr-defined]

# Only inject if torch is not genuinely installed.
if "torch" not in sys.modules:
    sys.modules["torch"] = _mock_torch


def _make_pcm_chunk(n_samples: int = 512, amplitude: int = 0) -> bytes:
    """Generate a PCM16 chunk of silence (amplitude=0) or noise."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _make_mock_model(prob: float = 0.0):
    """Create a mock that behaves like Silero VAD model callable."""
    model = MagicMock()
    model.return_value.item.return_value = prob
    return model


class TestSileroVAD:
    def test_init_default_params(self):
        from backend.vad import SileroVAD

        vad = SileroVAD(sample_rate=16000, silence_ms=500, model=_make_mock_model())
        assert vad.sample_rate == 16000
        assert vad.silence_ms == 500
        assert vad.is_speaking is False

    def test_feed_silence_no_speech(self):
        from backend.vad import SileroVAD

        vad = SileroVAD(sample_rate=16000, silence_ms=500, model=_make_mock_model(0.0))
        result = vad.feed(_make_pcm_chunk(512, amplitude=0))
        assert result is None

    def test_speech_detected_sets_is_speaking(self):
        from backend.vad import SileroVAD

        vad = SileroVAD(sample_rate=16000, silence_ms=500, model=_make_mock_model(0.9))
        chunk = _make_pcm_chunk(512, amplitude=1000)
        vad.feed(chunk)
        assert vad.is_speaking is True

    def test_speech_end_returns_audio_buffer(self):
        from backend.vad import SileroVAD

        model = _make_mock_model(0.9)
        vad = SileroVAD(sample_rate=16000, silence_ms=100, model=model)

        # Feed speech chunk
        speech_chunk = _make_pcm_chunk(512, amplitude=5000)
        vad.feed(speech_chunk)
        assert vad.is_speaking is True

        # Switch model to return low probability (silence)
        model.return_value.item.return_value = 0.05
        silence_chunk = _make_pcm_chunk(512, amplitude=0)
        result = None
        for _ in range(5):
            result = vad.feed(silence_chunk)
            if result is not None:
                break

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert vad.is_speaking is False

    def test_buffer_contains_speech_and_trailing_silence(self):
        """Returned buffer should contain both speech and trailing silence chunks."""
        from backend.vad import SileroVAD

        model = _make_mock_model(0.9)
        vad = SileroVAD(sample_rate=16000, silence_ms=100, model=model)

        speech_chunk = _make_pcm_chunk(512, amplitude=5000)
        vad.feed(speech_chunk)

        model.return_value.item.return_value = 0.05
        silence_chunk = _make_pcm_chunk(512, amplitude=0)
        result = None
        for _ in range(5):
            result = vad.feed(silence_chunk)
            if result is not None:
                break

        # Buffer must be larger than just the speech chunk (includes trailing silence)
        assert result is not None
        assert len(result) > len(speech_chunk)

    def test_reset_clears_state(self):
        from backend.vad import SileroVAD

        model = _make_mock_model()
        vad = SileroVAD(sample_rate=16000, silence_ms=500, model=model)
        vad._speech_buffer = b"\x00" * 100
        vad._is_speaking = True
        vad.reset()
        assert vad.is_speaking is False
        assert vad._speech_buffer == b""

    def test_reset_calls_model_reset_states(self):
        from backend.vad import SileroVAD

        model = _make_mock_model()
        vad = SileroVAD(sample_rate=16000, silence_ms=500, model=model)
        vad.reset()
        model.reset_states.assert_called_once()

    def test_multiple_utterances(self):
        """After first utterance returned, VAD should detect second utterance."""
        from backend.vad import SileroVAD

        model = _make_mock_model(0.9)
        vad = SileroVAD(sample_rate=16000, silence_ms=100, model=model)

        # First utterance
        speech_chunk = _make_pcm_chunk(512, amplitude=5000)
        vad.feed(speech_chunk)

        model.return_value.item.return_value = 0.05
        silence_chunk = _make_pcm_chunk(512, amplitude=0)
        result1 = None
        for _ in range(5):
            result1 = vad.feed(silence_chunk)
            if result1 is not None:
                break
        assert result1 is not None
        assert vad.is_speaking is False

        # Second utterance
        model.return_value.item.return_value = 0.9
        vad.feed(speech_chunk)
        assert vad.is_speaking is True

        model.return_value.item.return_value = 0.05
        result2 = None
        for _ in range(5):
            result2 = vad.feed(silence_chunk)
            if result2 is not None:
                break
        assert result2 is not None
        assert vad.is_speaking is False
