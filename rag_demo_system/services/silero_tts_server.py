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
    def __init__(self, speaker: str, sample_rate: int) -> None:
        from silero import silero_tts

        self._model, _ = silero_tts(language="ru", speaker="v5_4_ru")
        self._model.to(torch.device("cpu"))
        self._speaker = speaker
        self._sample_rate = sample_rate

    def synthesize(self, text: str) -> tuple[bytes, int]:
        audio = self._model.apply_tts(
            text=text,
            speaker=self._speaker,
            sample_rate=self._sample_rate,
            put_accent=True,
            put_yo=True,
        )
        pcm16 = (audio * 32767).to(torch.int16).numpy().tobytes()
        return pcm16, self._sample_rate


def create_app(synthesizer: SileroTTSSynthesizer) -> FastAPI:
    app = FastAPI(title="Silero TTS Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "provider": "silero_tts", "speaker": synthesizer._speaker}

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
    speaker = (os.getenv("SILERO_TTS_SPEAKER") or "xenia").strip()
    sample_rate = int(os.getenv("SILERO_TTS_SAMPLE_RATE") or "24000")
    try:
        synthesizer = SileroTTSSynthesizer(speaker, sample_rate)
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"silero_tts_not_ready: {exc}")
    return create_app(synthesizer)


app = _build_default_app()
