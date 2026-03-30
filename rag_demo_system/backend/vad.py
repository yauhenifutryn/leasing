"""Server-side Silero Voice Activity Detection.

Wraps silero-vad to process incoming PCM16 audio chunks and detect
speech boundaries. When speech ends (silence exceeds threshold),
returns the accumulated speech audio buffer.

The model file (silero_vad.jit) is downloaded during server provisioning
and loaded from local disk at runtime; no internet required.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path


class SileroVAD:
    """Stateful VAD that accumulates speech and emits on silence.

    Feed PCM16 audio chunks via `feed()`. When speech is detected followed
    by silence longer than `silence_ms`, the accumulated speech bytes are
    returned. Otherwise `feed()` returns None.

    Silence duration is tracked by counting audio samples rather than
    wall-clock time, which makes the behavior deterministic and testable.

    For testing, pass a mock via the `model` parameter to skip torch loading.
    """

    SPEECH_THRESHOLD = 0.5
    """Probability above which a chunk is considered speech."""

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_ms: int = 500,
        model: object | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_ms = silence_ms

        if model is not None:
            self._model = model
        else:
            import torch  # noqa: F811 -- lazy import, heavy dependency

            vad_path = os.environ.get(
                "SILERO_VAD_PATH",
                str(
                    Path(__file__).resolve().parent.parent
                    / "models"
                    / "silero_vad.jit"
                ),
            )
            self._model = torch.jit.load(vad_path)

        self._is_speaking = False
        self._speech_buffer = b""
        # Track silence duration by counting samples, not wall-clock time.
        self._silence_samples = 0
        self._silence_samples_threshold = int(sample_rate * silence_ms / 1000)

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def feed(self, pcm16_bytes: bytes) -> bytes | None:
        """Feed a PCM16 audio chunk.

        Returns the accumulated speech audio when speech ends (silence
        exceeds ``silence_ms``), otherwise returns None.
        """
        import torch  # noqa: F811 -- lazy import

        n_samples = len(pcm16_bytes) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm16_bytes)
        tensor = torch.FloatTensor(samples) / 32768.0

        prob = self._model(tensor, self.sample_rate).item()

        if prob >= self.SPEECH_THRESHOLD:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_buffer = b""
            self._speech_buffer += pcm16_bytes
            self._silence_samples = 0
            return None

        # Below threshold: either trailing silence during speech, or idle silence.
        if self._is_speaking:
            self._speech_buffer += pcm16_bytes
            self._silence_samples += n_samples
            if self._silence_samples >= self._silence_samples_threshold:
                audio = self._speech_buffer
                self._speech_buffer = b""
                self._is_speaking = False
                self._silence_samples = 0
                return audio

        return None

    def reset(self) -> None:
        """Clear all internal state."""
        self._is_speaking = False
        self._speech_buffer = b""
        self._silence_samples = 0
        self._model.reset_states()
