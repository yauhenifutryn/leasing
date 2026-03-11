from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class SpeakRequest(BaseModel):
    text: str
    session_id: str
    language: str = "ru"


class VoskTtsSynthesizer:
    def __init__(self, model_name: str, sample_rate_hz: int = 22050) -> None:
        from vosk_tts import Model, Synth

        self._model = Model(model_name=model_name)
        self._synth = Synth(self._model)
        self._sample_rate_hz = sample_rate_hz

    def synthesize(self, text: str, language: str) -> tuple[bytes, int]:
        _ = language
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = Path(temp_file.name)
        try:
            self._synth.synth(text, wav_path=str(wav_path))
            audio_bytes = wav_path.read_bytes()
            if audio_bytes[:44].startswith(b"RIFF"):
                audio_bytes = audio_bytes[44:]
            return audio_bytes, self._sample_rate_hz
        finally:
            wav_path.unlink(missing_ok=True)


def create_app(synthesizer: VoskTtsSynthesizer) -> FastAPI:
    app = FastAPI(title="Vosk TTS Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "provider": "vosk_tts"}

    @app.post("/speak")
    async def speak(payload: SpeakRequest) -> dict[str, object]:
        audio_bytes, sample_rate_hz = synthesizer.synthesize(payload.text, payload.language)
        return {
            "ok": True,
            "provider": "vosk_tts",
            "session_id": payload.session_id,
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate_hz": sample_rate_hz,
        }

    return app


def create_unavailable_app(reason: str) -> FastAPI:
    app = FastAPI(title="Vosk TTS Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": False, "provider": "vosk_tts", "reason": reason}

    @app.post("/speak")
    async def speak(_: SpeakRequest) -> dict[str, object]:
        raise HTTPException(status_code=503, detail=reason)

    return app


def _build_default_app() -> FastAPI:
    model_name = (os.getenv("VOSK_TTS_MODEL_NAME") or "vosk-model-tts-ru-0.9-multi").strip()
    try:
        synthesizer = VoskTtsSynthesizer(
            model_name=model_name,
            sample_rate_hz=int(os.getenv("VOSK_TTS_SAMPLE_RATE_HZ", "22050")),
        )
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"vosk_tts_not_ready: {exc}")
    return create_app(synthesizer)


app = _build_default_app()
