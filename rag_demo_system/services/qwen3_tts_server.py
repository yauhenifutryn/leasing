from __future__ import annotations

import base64
import io
import os

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class SpeakRequest(BaseModel):
    text: str
    session_id: str
    language: str = "ru"


class Qwen3TTSSynthesizer:
    def __init__(self, model_id: str, device: str, speaker: str) -> None:
        from qwen_tts import Qwen3TTSModel  # deferred import: keeps import errors in _build_default_app

        self._model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.bfloat16,
        )
        self._speaker = speaker

    def synthesize(self, text: str, language: str) -> tuple[bytes, int]:
        # Qwen3-TTS CustomVoice uses full language name, not ISO code.
        wavs, sr = self._model.generate_custom_voice(
            text=text, language="Russian", speaker=self._speaker,
        )

        # Convert the numpy waveform to raw PCM16 bytes by writing a WAV in-memory
        # and then discarding the 44-byte header so the caller gets headerless PCM16.
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV", subtype="PCM_16")
        buf.seek(44)  # skip RIFF/WAV header (44 bytes for standard PCM WAV)
        pcm16_bytes = buf.read()
        return pcm16_bytes, sr


def create_app(synthesizer: Qwen3TTSSynthesizer) -> FastAPI:
    app = FastAPI(title="Qwen3 TTS Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "provider": "qwen3_tts"}

    @app.post("/speak")
    async def speak(payload: SpeakRequest) -> dict[str, object]:
        audio_bytes, sample_rate_hz = synthesizer.synthesize(payload.text, payload.language)
        return {
            "ok": True,
            "provider": "qwen3_tts",
            "session_id": payload.session_id,
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate_hz": sample_rate_hz,
        }

    return app


def create_unavailable_app(reason: str) -> FastAPI:
    app = FastAPI(title="Qwen3 TTS Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": False, "provider": "qwen3_tts", "reason": reason}

    @app.post("/speak")
    async def speak(_: SpeakRequest) -> dict[str, object]:
        raise HTTPException(status_code=503, detail=reason)

    return app


def _build_default_app() -> FastAPI:
    model_id = (os.getenv("QWEN3_TTS_MODEL_ID") or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice").strip()
    device = (os.getenv("QWEN3_TTS_DEVICE") or "cuda:0").strip()
    speaker = (os.getenv("QWEN3_TTS_SPEAKER") or "Vivian").strip()
    try:
        synthesizer = Qwen3TTSSynthesizer(model_id, device, speaker)
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"qwen3_tts_not_ready: {exc}")
    return create_app(synthesizer)


app = _build_default_app()
