from __future__ import annotations

import base64
import os

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class SpeakRequest(BaseModel):
    text: str
    session_id: str
    language: str = "ru"


class SileroTTSSynthesizer:
    def __init__(
        self, speaker: str, sample_rate: int, model_variant: str = "v4_ru", speaker_pt: str | None = None
    ) -> None:
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=model_variant,
            trust_repo=True,
        )
        self._model.to(torch.device("cpu"))
        self._speaker = speaker
        self._speaker_embedding: torch.Tensor | None = None
        self._sample_rate = sample_rate

        if speaker_pt and os.path.isfile(speaker_pt):
            self._speaker_embedding = torch.load(speaker_pt, map_location="cpu")
            print(f"[silero_tts] Loaded custom voice from {speaker_pt}")

    def synthesize(self, text: str) -> tuple[bytes, int]:
        if self._speaker_embedding is not None:
            audio = self._model.apply_tts(
                text=text,
                speaker=self._speaker_embedding,
                sample_rate=self._sample_rate,
                put_accent=True,
                put_yo=True,
            )
        else:
            audio = self._model.apply_tts(
                text=text,
                speaker=self._speaker,
                sample_rate=self._sample_rate,
                put_accent=True,
                put_yo=True,
            )
        import numpy as np
        from scipy import signal as sp_signal

        audio_np = audio.detach().cpu().numpy().astype(np.float64)

        # 1. Notch filter at 2.5kHz and 3.5kHz to suppress metallic ringing
        #    (phase reconstruction artifact from vocoder)
        for notch_freq in [2500, 3500]:
            b, a = sp_signal.iirnotch(notch_freq, Q=8.0, fs=self._sample_rate)
            audio_np = sp_signal.filtfilt(b, a, audio_np)

        # 2. Gentle high-shelf cut above 4kHz to reduce breathiness
        sos = sp_signal.butter(2, 4000, btype="low", fs=self._sample_rate, output="sos")
        high_cut = sp_signal.sosfilt(sos, audio_np)
        # Mix: 70% original + 30% low-passed to preserve clarity
        audio_np = 0.7 * audio_np + 0.3 * high_cut

        # 3. Noise gate: silence segments below threshold get zeroed
        frame_size = int(self._sample_rate * 0.02)  # 20ms frames
        for i in range(0, len(audio_np) - frame_size, frame_size):
            frame_rms = np.sqrt(np.mean(audio_np[i:i+frame_size] ** 2))
            if frame_rms < 0.008:  # below noise floor
                audio_np[i:i+frame_size] *= 0.05  # near-silent, not hard zero

        # 4. Normalize to 90% peak
        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak * 0.9

        pcm16 = (audio_np * 32767).astype(np.int16).tobytes()
        return pcm16, self._sample_rate


def create_app(synthesizer: SileroTTSSynthesizer) -> FastAPI:
    app = FastAPI(title="Silero TTS Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        voice_source = (
            "custom_pt" if synthesizer._speaker_embedding is not None else "native"
        )
        return {
            "ok": True,
            "provider": "silero_tts",
            "speaker": synthesizer._speaker,
            "voice_source": voice_source,
        }

    @app.post("/speak")
    async def speak(payload: SpeakRequest) -> dict[str, object]:
        audio_bytes, sample_rate_hz = synthesizer.synthesize(payload.text)
        return {
            "ok": True,
            "provider": "silero_tts",
            "session_id": payload.session_id,
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate_hz": sample_rate_hz,
        }

    return app


def create_unavailable_app(reason: str) -> FastAPI:
    app = FastAPI(title="Silero TTS Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": False, "provider": "silero_tts", "reason": reason}

    @app.post("/speak")
    async def speak(_: SpeakRequest) -> dict[str, object]:
        raise HTTPException(status_code=503, detail=reason)

    return app


def _build_default_app() -> FastAPI:
    speaker = (os.getenv("SILERO_TTS_SPEAKER") or "eugene").strip()
    model_variant = (os.getenv("SILERO_TTS_MODEL") or "v4_ru").strip()
    sample_rate = int(os.getenv("SILERO_TTS_SAMPLE_RATE") or "24000")
    speaker_pt = os.getenv("SILERO_TTS_SPEAKER_PT")
    try:
        synthesizer = SileroTTSSynthesizer(speaker, sample_rate, model_variant=model_variant, speaker_pt=speaker_pt)
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"silero_tts_not_ready: {exc}")
    return create_app(synthesizer)


app = _build_default_app()
