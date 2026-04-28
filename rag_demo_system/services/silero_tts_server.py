from __future__ import annotations

import base64
import os

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class SpeakRequest(BaseModel):
    text: str
    session_id: str
    language: str = "ru"


class SileroTTSSynthesizer:
    def __init__(
        self, speaker: str, sample_rate: int, model_variant: str = "v5_4_ru", speaker_pt: str | None = None
    ) -> None:
        if model_variant.startswith("v5"):
            from silero import silero_tts
            self._model, _ = silero_tts(language="ru", speaker=model_variant)
        else:
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
        rate_pct = int(os.getenv("SILERO_TTS_RATE_PCT") or "100")
        speaker_arg = (
            self._speaker_embedding if self._speaker_embedding is not None else self._speaker
        )
        common_kwargs = {
            "speaker": speaker_arg,
            "sample_rate": self._sample_rate,
            "put_accent": True,
            "put_yo": True,
        }
        # Silero v5_4_ru's SSML <prosody rate> is parsed but silently ignored
        # (verified live 2026-04-28: 100% and 120% produced byte-for-byte
        # identical audio). Naive numpy.interp resample (1cbbee7+9caf11b)
        # speeds up but raises pitch ~2 semitones — caused "weird voice"
        # complaint 2026-04-29. Replace with librosa phase-vocoder
        # time-stretch which preserves pitch.
        audio = self._model.apply_tts(text=text, **common_kwargs)
        audio_np = audio.detach().cpu().numpy().astype(np.float32)
        if rate_pct != 100 and len(audio_np) > 1:
            # Lazy import: librosa is ~50MB and only loads when speedup is
            # enabled. First call adds ~150ms cold-start; subsequent calls
            # add ~10-30ms per phrase for the time-stretch itself.
            import librosa
            rate = rate_pct / 100.0
            audio_np = librosa.effects.time_stretch(audio_np, rate=rate)
        pcm16 = (audio_np * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
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
            "rate_pct": int(os.getenv("SILERO_TTS_RATE_PCT") or "100"),
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
    speaker = (os.getenv("SILERO_TTS_SPEAKER") or "xenia").strip()
    model_variant = (os.getenv("SILERO_TTS_MODEL") or "v5_4_ru").strip()
    sample_rate = int(os.getenv("SILERO_TTS_SAMPLE_RATE") or "24000")
    speaker_pt = os.getenv("SILERO_TTS_SPEAKER_PT")
    try:
        synthesizer = SileroTTSSynthesizer(speaker, sample_rate, model_variant=model_variant, speaker_pt=speaker_pt)
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"silero_tts_not_ready: {exc}")
    return create_app(synthesizer)


app = _build_default_app()
