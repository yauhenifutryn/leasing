from __future__ import annotations

import base64
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class TranscribeRequest(BaseModel):
    audio_b64: str
    session_id: str
    language: str = "ru"
    sample_rate_hz: int = 24000


@dataclass
class FasterWhisperTranscriber:
    model_size: str
    device: str
    compute_type: str

    def __post_init__(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate_hz: int, language: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = Path(temp_file.name)
        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate_hz)
                wav_file.writeframes(audio_bytes)
            segments, _info = self._model.transcribe(str(wav_path), language=language, vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments if segment.text).strip()
        finally:
            wav_path.unlink(missing_ok=True)


def create_app(transcriber: FasterWhisperTranscriber) -> FastAPI:
    app = FastAPI(title="Whisper Fallback Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "provider": "whisper"}

    @app.post("/transcribe")
    async def transcribe(payload: TranscribeRequest) -> dict[str, object]:
        try:
            audio_bytes = base64.b64decode(payload.audio_b64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid_audio_b64: {exc}") from exc
        text = transcriber.transcribe_pcm16(audio_bytes, payload.sample_rate_hz, payload.language)
        return {
            "ok": True,
            "provider": "whisper",
            "session_id": payload.session_id,
            "text": text,
        }

    return app


def create_unavailable_app(reason: str) -> FastAPI:
    app = FastAPI(title="Whisper Fallback Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": False, "provider": "whisper", "reason": reason}

    @app.post("/transcribe")
    async def transcribe(_: TranscribeRequest) -> dict[str, object]:
        raise HTTPException(status_code=503, detail=reason)

    return app


def _build_default_app() -> FastAPI:
    try:
        transcriber = FasterWhisperTranscriber(
            model_size=os.getenv("WHISPER_MODEL_SIZE", "large-v3"),
            device=os.getenv("WHISPER_DEVICE", "cuda"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
        )
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"whisper_not_ready: {exc}")
    return create_app(transcriber)


app = _build_default_app()
