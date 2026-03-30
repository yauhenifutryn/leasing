from __future__ import annotations

import base64
import os
import re
from typing import Any

import requests

try:
    from num2words import num2words as _num2words

    def _number_to_russian(match: re.Match) -> str:
        """Convert a numeric match to Russian words."""
        num_str = match.group(0).replace(" ", "").replace("\u00a0", "")
        try:
            n = float(num_str) if "." in num_str or "," in num_str else int(num_str)
            return _num2words(n, lang="ru")
        except (ValueError, OverflowError):
            return match.group(0)

    _NUM_RE = re.compile(r"\d[\d\s\u00a0]*[\d.,]*\d|\d+")

    def normalize_numbers_for_tts(text: str) -> str:
        """Replace digits with Russian words for natural TTS pronunciation."""
        return _NUM_RE.sub(_number_to_russian, text)

except ImportError:
    def normalize_numbers_for_tts(text: str) -> str:
        """Passthrough when num2words is not installed."""
        return text


# Russian abbreviation pronunciation map for TTS.
# Silero TTS reads abbreviations as words, not letter-by-letter.
# This map expands them to how they should be spoken aloud.
_ABBREV_TTS: dict[str, str] = {
    "НДФЛ": "эн-дэ-фэ-эл",
    "ИП": "и-пэ",
    "ИНН": "и-эн-эн",
    "УНП": "у-эн-пэ",
    "НДС": "эн-дэ-эс",
    "ВНЖ": "вэ-эн-жэ",
    "ВУ": "вэ-у",
    "BYN": "бэ-уай-эн",
    "USD": "долларов",
    "EUR": "евро",
    "GPS": "джи-пи-эс",
    "КАСКО": "каско",
    "ОСАГО": "осаго",
    "ООО": "о-о-о",
    "ЗАО": "зэ-а-о",
    "ОАО": "о-а-о",
}

# Build regex that matches any abbreviation (longest first to avoid partial matches)
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREV_TTS, key=len, reverse=True)) + r")\b"
)


def normalize_abbreviations_for_tts(text: str) -> str:
    """Expand abbreviations to spoken Russian for natural TTS pronunciation."""
    return _ABBREV_RE.sub(lambda m: _ABBREV_TTS[m.group(0)], text)


def _service_status(name: str, base_url: str | None) -> dict[str, Any]:
    if not base_url:
        return {"name": name, "available": False, "healthy": False, "reason": "not_configured"}
    health_url = base_url.rstrip("/") + "/health"
    try:
        resp = requests.get(health_url, timeout=2)
        return {
            "name": name,
            "available": True,
            "healthy": resp.ok,
            "reason": "ok" if resp.ok else f"http_{resp.status_code}",
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "available": True, "healthy": False, "reason": str(exc)}


def build_voice_statuses() -> dict[str, dict[str, Any]]:
    return {
        "whisper": _service_status("whisper", os.getenv("WHISPER_BASE_URL")),
        "silero_tts": _service_status("silero_tts", os.getenv("SILERO_TTS_BASE_URL")),
    }


def build_llm_status(base_url: str, model: str) -> dict[str, dict[str, Any]]:
    if not base_url or not model:
        return {"qwen": {"name": "qwen", "available": False, "healthy": False, "reason": "not_configured"}}
    url = base_url.rstrip("/") + "/models"
    try:
        resp = requests.get(url, timeout=2)
        return {
            "qwen": {
                "name": "qwen",
                "available": True,
                "healthy": resp.ok,
                "reason": "ok" if resp.ok else f"http_{resp.status_code}",
                "model": model,
            }
        }
    except Exception as exc:  # noqa: BLE001
        return {"qwen": {"name": "qwen", "available": True, "healthy": False, "reason": str(exc), "model": model}}


def transcribe_audio(audio_b64: str, session_id: str) -> dict[str, Any]:
    """Transcribe audio using the Whisper STT service."""
    base_url = os.getenv("WHISPER_BASE_URL")
    if not base_url:
        raise RuntimeError("WHISPER_BASE_URL is not configured")
    resp = requests.post(
        base_url.rstrip("/") + "/transcribe",
        json={"audio_b64": audio_b64, "session_id": session_id, "language": "ru", "sample_rate_hz": 24000},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("text"):
        raise RuntimeError("Whisper returned empty transcription")
    data.setdefault("provider", "whisper")
    return data


def synthesize_audio(text: str, session_id: str) -> dict[str, Any]:
    """Synthesize speech using the Silero TTS service."""
    base_url = os.getenv("SILERO_TTS_BASE_URL")
    if not base_url:
        raise RuntimeError("SILERO_TTS_BASE_URL is not configured")
    tts_text = normalize_abbreviations_for_tts(normalize_numbers_for_tts(text))
    resp = requests.post(
        base_url.rstrip("/") + "/speak",
        json={"text": tts_text, "session_id": session_id, "language": "ru"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    data.setdefault("provider", "silero_tts")
    data.setdefault("session_id", session_id)
    return data
