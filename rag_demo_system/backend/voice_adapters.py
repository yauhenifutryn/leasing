from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any

import requests
import yaml

try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
except ImportError:
    _morph = None

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

    # Time with preposition: "с 9:00" -> "с девяти", "до 18:00" -> "до восемнадцати"
    _RE_TIME_WITH_PREP = re.compile(r"(с|до|в|после|к)\s+(\d{1,2}):(\d{2})\b", re.I)
    _RE_TIME_PLAIN = re.compile(r"\b(\d{1,2}):(\d{2})\b")

    # Nominative -> genitive for hours (after с/до/от/после)
    _HOUR_GENITIVE = {
        "0": "нуля", "1": "часа", "2": "двух", "3": "трёх", "4": "четырёх",
        "5": "пяти", "6": "шести", "7": "семи", "8": "восьми", "9": "девяти",
        "10": "десяти", "11": "одиннадцати", "12": "двенадцати",
        "13": "тринадцати", "14": "четырнадцати", "15": "пятнадцати",
        "16": "шестнадцати", "17": "семнадцати", "18": "восемнадцати",
        "19": "девятнадцати", "20": "двадцати", "21": "двадцати одного",
        "22": "двадцати двух", "23": "двадцати трёх",
    }
    _HOUR_NOMINATIVE = {
        "0": "ноль", "1": "час", "2": "два", "3": "три", "4": "четыре",
        "5": "пять", "6": "шесть", "7": "семь", "8": "восемь", "9": "девять",
        "10": "десять", "11": "одиннадцать", "12": "двенадцать",
        "13": "тринадцать", "14": "четырнадцать", "15": "пятнадцать",
        "16": "шестнадцать", "17": "семнадцать", "18": "восемнадцать",
        "19": "девятнадцать", "20": "двадцать", "21": "двадцать один",
        "22": "двадцать два", "23": "двадцать три",
    }

    def _time_with_prep(match: re.Match) -> str:
        prep, h, m = match.group(1), match.group(2), match.group(3)
        # "с" and "до" require genitive; "в" requires accusative (same as nominative for time)
        use_genitive = prep.lower() in ("с", "до", "после", "от")
        table = _HOUR_GENITIVE if use_genitive else _HOUR_NOMINATIVE
        h_word = table.get(h, _num2words(int(h), lang="ru"))
        if m == "00":
            return f"{prep} {h_word}"
        m_word = _num2words(int(m), lang="ru")
        return f"{prep} {h_word} {m_word}"

    def _time_plain(match: re.Match) -> str:
        h, m = match.group(1), match.group(2)
        h_word = _HOUR_NOMINATIVE.get(h, _num2words(int(h), lang="ru"))
        if m == "00":
            return h_word
        m_word = _num2words(int(m), lang="ru")
        return f"{h_word} {m_word}"

    _GENITIVE_PREPS = {"от", "до", "с", "без", "после", "более", "менее", "свыше"}
    # Words that are part of compound numbers (should stay in genitive zone)
    _NUMBER_WORDS = {
        "ноль", "один", "одна", "одно", "два", "две", "три", "четыре",
        "пять", "шесть", "семь", "восемь", "девять", "десять",
        "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
        "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать",
        "девятнадцать", "двадцать", "тридцать", "сорок", "пятьдесят",
        "шестьдесят", "семьдесят", "восемьдесят", "девяносто", "сто",
        "двести", "триста", "четыреста", "пятьсот", "тысяча", "тысяч",
    }

    def _to_genitive(word: str) -> str:
        """Inflect a Russian word to genitive case using pymorphy3."""
        if _morph is None:
            return word
        parsed = _morph.parse(word)[0]
        inflected = parsed.inflect({"gent"})
        return inflected.word if inflected else word

    def _fix_genitives_after_preps(text: str) -> str:
        """Fix nominative -> genitive after prepositions requiring genitive case."""
        words = text.split()
        result = []
        apply_genitive = False
        for w in words:
            if w.lower() in _GENITIVE_PREPS:
                apply_genitive = True
                result.append(w)
                continue
            if apply_genitive and w.lower() in _NUMBER_WORDS:
                result.append(_to_genitive(w))
            else:
                if w.lower() not in _NUMBER_WORDS:
                    apply_genitive = False
                result.append(w)
        return " ".join(result)

    def normalize_numbers_for_tts(text: str) -> str:
        """Replace digits, percentages, dollar amounts, and times with Russian words."""
        # Times with prepositions first (genitive case)
        text = _RE_TIME_WITH_PREP.sub(_time_with_prep, text)
        text = _RE_TIME_PLAIN.sub(_time_plain, text)
        text = _RE_PCT.sub(_pct_to_russian, text)
        text = _RE_DOLLAR.sub(_dollar_to_russian, text)
        text = _RE_SPACED_NUM.sub(_number_to_russian, text)
        text = _RE_PLAIN_NUM.sub(_number_to_russian, text)
        # Fix genitive after prepositions (от десять -> от десяти, до тридцать девять -> до тридцати девяти)
        text = _fix_genitives_after_preps(text)
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
