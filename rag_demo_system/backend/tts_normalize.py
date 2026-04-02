from __future__ import annotations

import re
from pathlib import Path

import yaml
from num2words import num2words


def _num_to_words(match: re.Match) -> str:
    raw = match.group(0)
    cleaned = raw.replace(" ", "").replace(",", "").replace("\u00a0", "")
    dollar_prefix = False
    if cleaned.startswith("$"):
        dollar_prefix = True
        cleaned = cleaned[1:]
    try:
        if "." in cleaned:
            n = float(cleaned)
        else:
            n = int(cleaned)
    except ValueError:
        return raw
    try:
        words = num2words(n, lang="ru")
    except Exception:
        return raw
    if dollar_prefix:
        words += " долларов"
    return words


def _pct_to_words(match: re.Match) -> str:
    num_part = match.group(1).replace(" ", "").replace(",", "").replace("\u00a0", "")
    try:
        if "." in num_part:
            n = float(num_part)
        else:
            n = int(num_part)
    except ValueError:
        return match.group(0)
    try:
        words = num2words(n, lang="ru")
    except Exception:
        return match.group(0)
    return words + " процентов"


_RE_PCT = re.compile(r"(\d[\d\s.,]*\d|\d)\s*%")
_RE_DOLLAR = re.compile(r"\$\s*(\d[\d\s.,]*\d|\d)")
_RE_SPACED_NUM = re.compile(r"\d{1,3}(?:[\s\u00a0]\d{3})+")
_RE_PLAIN_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_for_tts(text: str) -> str:
    text = _RE_PCT.sub(_pct_to_words, text)
    text = _RE_DOLLAR.sub(lambda m: _num_to_words(m), text)
    text = _RE_SPACED_NUM.sub(lambda m: _num_to_words(m), text)
    text = _RE_PLAIN_NUM.sub(lambda m: _num_to_words(m), text)
    text = transliterate_latin(text)
    return text


# ---------------------------------------------------------------------------
# Latin-to-Cyrillic transliteration
# ---------------------------------------------------------------------------

_TRANSLITERATION_DICT: dict[str, str] | None = None
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# Basic English-to-Russian phonetic rules for unknown words
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
    dictionary = _load_transliteration_dict()

    # First pass: multi-word dictionary matches (e.g., "Land Rover")
    for key, value in sorted(dictionary.items(), key=lambda x: -len(x[0])):
        if " " not in key:
            continue
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        text = pattern.sub(value, text)

    # Second pass: remaining single Latin words
    def _replace(m: re.Match) -> str:
        word = m.group(0)
        lower = word.lower()
        if lower in dictionary:
            return dictionary[lower]
        return _phonetic_transliterate(word)

    text = _RE_LATIN_WORD.sub(_replace, text)
    return text
