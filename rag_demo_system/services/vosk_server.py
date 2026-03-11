from __future__ import annotations

import base64
import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.audio_utils import resample_pcm16_mono


class TranscribeRequest(BaseModel):
    audio_b64: str
    session_id: str
    language: str = "ru"
    sample_rate_hz: int = 24000


class VoskTranscriber:
    def __init__(self, model_path: str) -> None:
        from vosk import KaldiRecognizer, Model

        self._recognizer_cls = KaldiRecognizer
        self._model = Model(model_path=model_path)

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate_hz: int, language: str) -> str:
        _ = language
        pcm = resample_pcm16_mono(audio_bytes, sample_rate_hz, 16000)
        recognizer = self._recognizer_cls(self._model, 16000)
        recognizer.AcceptWaveform(pcm)
        result = json.loads(recognizer.FinalResult() or "{}")
        return str(result.get("text") or "").strip()


def create_app(transcriber: VoskTranscriber) -> FastAPI:
    app = FastAPI(title="Vosk STT Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "provider": "vosk"}

    @app.post("/transcribe")
    async def transcribe(payload: TranscribeRequest) -> dict[str, object]:
        try:
            audio_bytes = base64.b64decode(payload.audio_b64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid_audio_b64: {exc}") from exc
        return {
            "ok": True,
            "provider": "vosk",
            "session_id": payload.session_id,
            "text": transcriber.transcribe_pcm16(audio_bytes, payload.sample_rate_hz, payload.language),
        }

    return app


def create_unavailable_app(reason: str) -> FastAPI:
    app = FastAPI(title="Vosk STT Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": False, "provider": "vosk", "reason": reason}

    @app.post("/transcribe")
    async def transcribe(_: TranscribeRequest) -> dict[str, object]:
        raise HTTPException(status_code=503, detail=reason)

    return app


def _build_default_app() -> FastAPI:
    model_path = (os.getenv("VOSK_MODEL_PATH") or "").strip()
    if not model_path:
        return create_unavailable_app("vosk_not_ready: VOSK_MODEL_PATH is not configured")
    try:
        transcriber = VoskTranscriber(model_path=model_path)
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"vosk_not_ready: {exc}")
    return create_app(transcriber)


app = _build_default_app()
