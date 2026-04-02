from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any

import requests
import yaml

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

    def _pct_to_russian(match: re.Match) -> str:
        """Convert '39%' to 'тридцать девять процентов'."""
        num_str = match.group(1).replace(" ", "").replace(",", "").replace("\u00a0", "")
        try:
            n = float(num_str) if "." in num_str else int(num_str)
            return _num2words(n, lang="ru") + " процентов"
        except (ValueError, OverflowError):
            return match.group(0)

    def _dollar_to_russian(match: re.Match) -> str:
        """Convert '$20,000' to 'двадцать тысяч долларов'."""
        num_str = match.group(1).replace(" ", "").replace(",", "").replace("\u00a0", "")
        try:
            n = float(num_str) if "." in num_str else int(num_str)
            return _num2words(n, lang="ru") + " долларов"
        except (ValueError, OverflowError):
            return match.group(0)

    _RE_PCT = re.compile(r"(\d[\d\s.,]*\d|\d)\s*%")
    _RE_DOLLAR = re.compile(r"\$\s*(\d[\d\s.,]*\d|\d)")
    _RE_SPACED_NUM = re.compile(r"\d{1,3}(?:[\s\u00a0]\d{3})+")
    _RE_PLAIN_NUM = re.compile(r"\d+(?:[.,]\d+)?")

    def normalize_numbers_for_tts(text: str) -> str:
        """Replace digits, percentages, and dollar amounts with Russian words."""
        text = _RE_PCT.sub(_pct_to_russian, text)
        text = _RE_DOLLAR.sub(_dollar_to_russian, text)
        text = _RE_SPACED_NUM.sub(_number_to_russian, text)
        text = _RE_PLAIN_NUM.sub(_number_to_russian, text)
        return text

except ImportError:
    def normalize_numbers_for_tts(text: str) -> str:
        """Passthrough when num2words is not installed."""
        return text


# ---------------------------------------------------------------------------
# Latin-to-Cyrillic transliteration for TTS
# ---------------------------------------------------------------------------

_TRANSLITERATION_DICT: dict[str, str] | None = None
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

_PHONETIC_MAP = [
    ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"),
    ("wh", "в"), ("ck", "к"), ("oo", "у"), ("ee", "и"),
    ("ea", "и"), ("ou", "ау"), ("ow", "ау"), ("igh", "ай"),
    ("tion", "шн"), ("sion", "жн"), ("ous", "ас"),
    ("qu", "кв"), ("x", "кс"), ("w", "в"), ("j", "дж"),
    ("y", "и"), ("a", "а"), ("b", "б"), ("c", "к"),
    ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"),
    ("h", "х"), ("i", "и"), ("k", "к"), ("l", "л"),
    ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"),
    ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"),
    ("v", "в"), ("z", "з"),
]

_RE_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z\-]*[A-Za-z]|[A-Za-z]")


def _load_transliteration_dict() -> dict[str, str]:
    global _TRANSLITERATION_DICT
    if _TRANSLITERATION_DICT is not None:
        return _TRANSLITERATION_DICT
    path = _CONFIG_DIR / "transliteration.yaml"
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _TRANSLITERATION_DICT = {str(k).lower(): str(v) for k, v in raw.items()}
    else:
        _TRANSLITERATION_DICT = {}
    return _TRANSLITERATION_DICT


def _phonetic_transliterate(word: str) -> str:
    result: list[str] = []
    i = 0
    lower = word.lower()
    while i < len(lower):
        matched = False
        for eng, rus in _PHONETIC_MAP:
            if lower[i:].startswith(eng):
                result.append(rus)
                i += len(eng)
                matched = True
                break
        if not matched:
            result.append(lower[i])
            i += 1
    return "".join(result)


def transliterate_latin(text: str) -> str:
    """Replace Latin words with Cyrillic using dictionary + phonetic fallback."""
    dictionary = _load_transliteration_dict()
    # Multi-word dictionary matches first (e.g., "Land Rover")
    for key, value in sorted(dictionary.items(), key=lambda x: -len(x[0])):
        if " " not in key:
            continue
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        text = pattern.sub(value, text)

    def _replace(m: re.Match) -> str:
        word = m.group(0)
        lower = word.lower()
        if lower in dictionary:
            return dictionary[lower]
        return _phonetic_transliterate(word)

    text = _RE_LATIN_WORD.sub(_replace, text)
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
    tts_text = transliterate_latin(normalize_abbreviations_for_tts(normalize_numbers_for_tts(text)))
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
