from __future__ import annotations

import base64
import io
import os
import wave

import numpy as np
import scipy.signal
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class TranscribeRequest(BaseModel):
    audio_b64: str
    session_id: str
    language: str = "ru"
    sample_rate_hz: int = 24000


class VoxtralTranscriber:
    def __init__(self, model_id: str, device: str) -> None:
        # Deferred import: VoxtralRealtimeForConditionalGeneration requires transformers>=5.2.0
        # which lives in its own isolated venv, separate from the shared backend venv
        # (transformers==4.37.2).
        from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor  # noqa: PLC0415

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.bfloat16,
        )
        # Target sample rate is 16kHz (set by Voxtral's feature extractor).
        # Pitfall 3: audio arriving at 24kHz must be resampled before calling processor().
        self._target_sr: int = self._processor.feature_extractor.sampling_rate  # 16000

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate_hz: int, language: str) -> str:
        # Convert raw PCM16 bytes to float32 numpy array in [-1.0, 1.0].
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Resample from input sample rate to the target 16kHz if necessary.
        # Pitfall 3: processor rejects audio not at 16kHz.
        if sample_rate_hz != self._target_sr:
            num_samples = int(len(audio_float) * self._target_sr / sample_rate_hz)
            audio_float = scipy.signal.resample(audio_float, num_samples)

        # Batch / offline inference API.
        # Pitfall 6: do NOT use the streaming API (input_features_generator / padding_cache).
        # Use processor() -> model.generate() -> batch_decode().
        inputs = self._processor(audio_float, sampling_rate=self._target_sr, return_tensors="pt")
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)
        outputs = self._model.generate(**inputs)
        text: str = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return text.strip()


def create_app(transcriber: VoxtralTranscriber) -> FastAPI:
    app = FastAPI(title="Voxtral STT Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "provider": "voxtral"}

    @app.post("/transcribe")
    async def transcribe(payload: TranscribeRequest) -> dict[str, object]:
        try:
            audio_bytes = base64.b64decode(payload.audio_b64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid_audio_b64: {exc}") from exc
        text = transcriber.transcribe_pcm16(audio_bytes, payload.sample_rate_hz, payload.language)
        return {
            "ok": True,
            "provider": "voxtral",
            "session_id": payload.session_id,
            "text": text,
        }

    return app


def create_unavailable_app(reason: str) -> FastAPI:
    app = FastAPI(title="Voxtral STT Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": False, "provider": "voxtral", "reason": reason}

    @app.post("/transcribe")
    async def transcribe(_: TranscribeRequest) -> dict[str, object]:
        raise HTTPException(status_code=503, detail=reason)

    return app


def _build_default_app() -> FastAPI:
    model_id = (os.getenv("VOXTRAL_MODEL_ID") or "mistralai/Voxtral-Mini-4B-Realtime-2602").strip()
    device = (os.getenv("VOXTRAL_DEVICE") or "cuda:0").strip()
    try:
        transcriber = VoxtralTranscriber(model_id, device)
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"voxtral_not_ready: {exc}")
    return create_app(transcriber)


app = _build_default_app()
