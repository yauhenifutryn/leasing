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
        # Clean up v4_ru echo/reverb: high-pass filter + normalization
        import numpy as np
        audio_np = audio.detach().cpu().numpy()
        # High-pass filter at 80Hz to remove low-frequency reverb
        if len(audio_np) > 2:
            alpha = 0.98  # cutoff ~80Hz at 24kHz
            filtered = np.zeros_like(audio_np)
            filtered[0] = audio_np[0]
            for i in range(1, len(audio_np)):
                filtered[i] = alpha * (filtered[i - 1] + audio_np[i] - audio_np[i - 1])
            audio_np = filtered
        # Normalize to prevent clipping
        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak * 0.95
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
