from __future__ import annotations

import base64
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Domain vocabulary for Whisper: biases transcription toward the bot name,
# Belarusian leasing vocabulary, car brand aliases and graph types.
# Whisper initial_prompt is capped at 224 tokens; high-ROI terms placed at
# the end (guaranteed to survive truncation). See test_whisper_prompt.py.
_DEFAULT_INITIAL_PROMPT = (
    "Помощница Ксения компании Микро Лизинг. "
    "Клиенты: физическое лицо, физлицо, ИП, ипэшник, юридическое лицо, юрлицо. "
    "Предметы: легковой автомобиль, грузовой автомобиль, спецтехника, "
    "оборудование, недвижимость, прочий транспорт. "
    "Марки: BMW бэха, Mercedes мерс, Audi аудюха, BYD. "
    "Термины: аванс, срок лизинга, ежемесячный платёж, выкупной платёж, "
    "нагрузка, переплата, удорожание, лизингополучатель. "
    "Состояние: новый, б/у, бэу, пробег. "
    "Графики: аннуитетный, линейный, дифференцированный. "
    "Ксения, Ксения."
)

# Known Whisper training-data hallucinations that appear on short/silent/noisy
# audio. These are Russian YouTube caption boilerplate that Whisper memorized
# during training and reproduces when the input has no real speech signal.
# When detected, we treat the transcription as empty so the client-facing
# "not understood" fallback handles the turn cleanly.
_HALLUCINATION_BLACKLIST: frozenset[str] = frozenset({
    "продолжение следует",
    "продолжение следует...",
    "продолжаем",
    "продолжаем.",
    "продолжаем...",
    "субтитры создавал dimatorzok",
    "субтитры подогнал «а.с.с.»",
    "субтитры выполнены а.с.с.",
    "субтитры сделал dimatorzok",
    "редактор субтитров",
    "корректор",
    "спасибо за внимание",
    "спасибо за просмотр",
    "спасибо за внимание!",
    "спасибо за просмотр!",
    "подписывайтесь на канал",
    "не забудьте подписаться",
    "ставьте лайки",
    "ставьте лайк",
    "all rights reserved",
    "субтитры",
    "субтитры.",
    "※",
})


def _is_hallucination(text: str) -> bool:
    """Return True if transcription matches a known Whisper training-data leak."""
    if not text:
        return False
    normalized = text.strip().lower().rstrip(".!?…;:")
    normalized_nodots = normalized.replace("...", "").strip()
    return (
        text.strip().lower() in _HALLUCINATION_BLACKLIST
        or normalized in _HALLUCINATION_BLACKLIST
        or normalized_nodots in _HALLUCINATION_BLACKLIST
    )


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
            segments, _info = self._model.transcribe(
                str(wav_path),
                language=language,
                vad_filter=True,
                initial_prompt=os.getenv("WHISPER_INITIAL_PROMPT", _DEFAULT_INITIAL_PROMPT),
            )
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
        audio_len_bytes = len(audio_bytes)
        audio_duration_s = audio_len_bytes / (2 * payload.sample_rate_hz)  # PCM16 = 2 bytes/sample
        print(f"[whisper] transcribe: {audio_len_bytes} bytes, {audio_duration_s:.1f}s, sr={payload.sample_rate_hz}, lang={payload.language}")
        text = transcriber.transcribe_pcm16(audio_bytes, payload.sample_rate_hz, payload.language)
        if _is_hallucination(text):
            print(f"[whisper] hallucination filtered: '{text[:60]}' -> empty", flush=True)
            text = ""
        print(f"[whisper] result: '{text[:100]}'" if text else "[whisper] result: (empty)")
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
