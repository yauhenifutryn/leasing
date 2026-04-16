from __future__ import annotations

import base64
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Domain vocabulary for Whisper: biases transcription towards car brands,
# leasing terms, and financial amounts common in Belarus leasing calls.
_DEFAULT_INITIAL_PROMPT = (
    "Клиент звонит в компанию Микро Лизинг. "
    "Типы клиентов: ИП (индивидуальный предприниматель), физическое лицо, юридическое лицо. "
    "Лизинг автомобилей и грузового транспорта в Беларуси. "
    "Марки: Volkswagen Фольксваген, Toyota Тойота, BMW бэха, "
    "Mercedes-Benz мерс, Audi аудюха, Hyundai Хёндай, Kia Киа, "
    "Skoda Шкода, Renault Рено, Nissan Ниссан, Mazda Мазда, Ford Форд, "
    "Opel Опель, Honda Хонда, Subaru Субару, Mitsubishi Мицубиши, "
    "Chevrolet Шевроле, Lexus Лексус, Peugeot Пежо, Citroen Ситроен, "
    "Volvo Вольво, Land Rover Ленд Ровер, Porsche Порше. "
    "Китайские: Geely Джили, Chery Чери, Haval Хавал, Exeed Эксид, "
    "Changan Чанган, JAC Джак, BYD, Jetour Джетур, Omoda Омода. "
    "Отечественные: Lada Лада ВАЗ, ГАЗ ГАЗель, МАЗ. "
    "Термины: аванс, ежемесячный платёж, выкупной платёж, график платежей, "
    "удорожание, лизингодатель, лизингополучатель, юрлицо, физлицо, ИП, "
    "договор лизинга, VIN, УНП, НДС, КАСКО, ОСАГО, б/у, рассрочка, "
    "рефинансирование, реструктуризация, тягач, полуприцеп, спецтехника. "
    "Суммы: десять тысяч, двадцать тысяч, пятьдесят тысяч, сто тысяч "
    "белорусских рублей, долларов, евро, полтора миллиона."
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
