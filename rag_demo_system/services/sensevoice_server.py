from __future__ import annotations

import base64
import io
import os
import wave

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class TranscribeRequest(BaseModel):
    audio_b64: str
    session_id: str
    language: str = "auto"
    sample_rate_hz: int = 24000


class SenseVoiceTranscriber:
    def __init__(self, model_id: str, device: str) -> None:
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self._postprocess = rich_transcription_postprocess
        self._model = AutoModel(
            model=model_id,
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
        )

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate_hz: int, language: str) -> str:
        # Convert PCM16 bytes to float32 numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # SenseVoice expects 16kHz; resample if needed
        if sample_rate_hz != 16000:
            import scipy.signal
            num_samples = int(len(audio_np) * 16000 / sample_rate_hz)
            audio_np = scipy.signal.resample(audio_np, num_samples)

        res = self._model.generate(
            input=audio_np,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        if not res or not res[0].get("text"):
            return ""
        return self._postprocess(res[0]["text"])


def create_app(transcriber: SenseVoiceTranscriber) -> FastAPI:
    app = FastAPI(title="SenseVoice STT Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "provider": "sensevoice"}

    @app.post("/transcribe")
    async def transcribe(payload: TranscribeRequest) -> dict[str, object]:
        try:
            audio_bytes = base64.b64decode(payload.audio_b64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid_audio_b64: {exc}") from exc
        audio_len_bytes = len(audio_bytes)
        audio_duration_s = audio_len_bytes / (2 * payload.sample_rate_hz)
        print(f"[sensevoice] transcribe: {audio_len_bytes} bytes, {audio_duration_s:.1f}s, sr={payload.sample_rate_hz}, lang={payload.language}")
        text = transcriber.transcribe_pcm16(audio_bytes, payload.sample_rate_hz, payload.language)
        print(f"[sensevoice] result: '{text[:100]}'" if text else "[sensevoice] result: (empty)")
        return {
            "ok": True,
            "provider": "sensevoice",
            "session_id": payload.session_id,
            "text": text,
        }

    return app


def create_unavailable_app(reason: str) -> FastAPI:
    app = FastAPI(title="SenseVoice STT Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": False, "provider": "sensevoice", "reason": reason}

    @app.post("/transcribe")
    async def transcribe(_: TranscribeRequest) -> dict[str, object]:
        raise HTTPException(status_code=503, detail=reason)

    return app


def _build_default_app() -> FastAPI:
    model_id = (os.getenv("SENSEVOICE_MODEL_ID") or "FunAudioLLM/SenseVoiceSmall").strip()
    device = (os.getenv("SENSEVOICE_DEVICE") or "cuda:0").strip()
    try:
        transcriber = SenseVoiceTranscriber(model_id, device)
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"sensevoice_not_ready: {exc}")
    return create_app(transcriber)


app = _build_default_app()
