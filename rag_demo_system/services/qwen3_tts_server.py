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
    """Supports two modes via env vars:

    1. CustomVoice (default): preset speakers, no ref audio needed.
       QWEN3_TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
       QWEN3_TTS_SPEAKER=Vivian

    2. Voice clone: clones a voice from a reference audio file.
       QWEN3_TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base
       QWEN3_TTS_REF_AUDIO=/path/to/ref_voice_ru.wav
       QWEN3_TTS_REF_TEXT="transcript of the reference audio"
    """

    def __init__(self, model_id: str, device: str, speaker: str,
                 ref_audio: str | None, ref_text: str | None) -> None:
        from qwen_tts import Qwen3TTSModel

        self._model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.bfloat16,
        )
        self._speaker = speaker
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._voice_clone_prompt = None

        # If ref_audio is provided, pre-compute the voice clone prompt once at startup
        if ref_audio and ref_text:
            print(f"[qwen3_tts] Building voice clone prompt from {ref_audio}...")
            self._voice_clone_prompt = self._model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            print("[qwen3_tts] Voice clone prompt ready.")
            self._mode = "clone"
        else:
            self._mode = "custom"
            print(f"[qwen3_tts] Using CustomVoice speaker: {speaker}")

    def synthesize(self, text: str, language: str) -> tuple[bytes, int]:
        if self._mode == "clone":
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language="Russian",
                voice_clone_prompt=self._voice_clone_prompt,
            )
        else:
            wavs, sr = self._model.generate_custom_voice(
                text=text, language="Russian", speaker=self._speaker,
            )

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV", subtype="PCM_16")
        buf.seek(44)
        pcm16_bytes = buf.read()
        return pcm16_bytes, sr


def create_app(synthesizer: Qwen3TTSSynthesizer) -> FastAPI:
    app = FastAPI(title="Qwen3 TTS Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "provider": "qwen3_tts", "mode": synthesizer._mode}

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
    ref_audio = (os.getenv("QWEN3_TTS_REF_AUDIO") or "").strip() or None
    ref_text = (os.getenv("QWEN3_TTS_REF_TEXT") or "").strip() or None
    try:
        synthesizer = Qwen3TTSSynthesizer(model_id, device, speaker, ref_audio, ref_text)
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"qwen3_tts_not_ready: {exc}")
    return create_app(synthesizer)


app = _build_default_app()
