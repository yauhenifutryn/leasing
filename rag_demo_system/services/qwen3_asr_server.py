from __future__ import annotations

import base64
import os
import tempfile
import wave
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class TranscribeRequest(BaseModel):
    audio_b64: str
    session_id: str
    language: str = "ru"
    sample_rate_hz: int = 24000


class Qwen3ASRTranscriber:
    def __init__(self, model_id: str, device: str) -> None:
        from qwen_asr import Qwen3ASRModel  # deferred import: keeps import errors in _build_default_app

        self._model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device,
        )

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate_hz: int, language: str) -> str:
        # Write raw PCM16 bytes into a temporary WAV file so the Qwen3-ASR model
        # can load it via a path. The model handles 16kHz resampling internally.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = Path(temp_file.name)
        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate_hz)
                wav_file.writeframes(audio_bytes)
            results = self._model.transcribe(audio=str(wav_path), language=language)
            return results[0].text.strip() if results else ""
        finally:
            wav_path.unlink(missing_ok=True)


def create_app(transcriber: Qwen3ASRTranscriber) -> FastAPI:
    app = FastAPI(title="Qwen3 ASR Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "provider": "qwen3_asr"}

    @app.post("/transcribe")
    async def transcribe(payload: TranscribeRequest) -> dict[str, object]:
        try:
            audio_bytes = base64.b64decode(payload.audio_b64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid_audio_b64: {exc}") from exc
        text = transcriber.transcribe_pcm16(audio_bytes, payload.sample_rate_hz, payload.language)
        return {
            "ok": True,
            "provider": "qwen3_asr",
            "session_id": payload.session_id,
            "text": text,
        }

    return app


def create_unavailable_app(reason: str) -> FastAPI:
    app = FastAPI(title="Qwen3 ASR Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": False, "provider": "qwen3_asr", "reason": reason}

    @app.post("/transcribe")
    async def transcribe(_: TranscribeRequest) -> dict[str, object]:
        raise HTTPException(status_code=503, detail=reason)

    return app


def _build_default_app() -> FastAPI:
    model_id = (os.getenv("QWEN3_ASR_MODEL_ID") or "Qwen/Qwen3-ASR-1.7B").strip()
    device = (os.getenv("QWEN3_ASR_DEVICE") or "cuda:0").strip()
    try:
        transcriber = Qwen3ASRTranscriber(model_id, device)
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"qwen3_asr_not_ready: {exc}")
    return create_app(transcriber)


app = _build_default_app()
