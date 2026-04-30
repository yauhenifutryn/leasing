"""Russian number-word parser for cost-grounding.

Handles the subset of Russian number words that appear in leasing
conversations: units (один-девять), teens (десять-девятнадцать),
tens (двадцать-девяносто), hundreds (сто-девятьсот), and scale words
(тысяча, миллион). Returns None for any input that doesn't contain a
parseable number, so callers can fall through to digit-based grounding.

Skip cases:
- percent-form ("двадцать процентов") — caller wants a cost-quantity;
- purely digit-form ("20000 долларов") — caller already does that.
"""
from __future__ import annotations

import re
from typing import Optional

_UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3,
    "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
    "девять": 9,
}
_TEENS = {
    "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
    "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80,
    "девяносто": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800,
    "девятьсот": 900,
}
_SCALES = {
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000,
    "миллион": 1000000, "миллиона": 1000000, "миллионов": 1000000,
}
# Bug 10 (live calls 2026-04-29): users say "20 косарей" / "30 тонн" /
# "15 кусков" / "20 штук" — slang for "thousand". Finite, well-known
# lexicon — stem-prefix match catches every grammatical variant
# (косарь / косаря / косарей / косаре) without listing them all.
_SLANG_THOUSAND_STEMS = ("косар", "тонн", "куск", "кусок", "штук")
_PERCENT_STOPWORDS = {"процент", "процента", "процентов"}

_WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)


def parse_ru_number(text: str) -> Optional[int]:
    """Parse a Russian number-word phrase into an int. Return None when
    no number is present, when the only number is in a percent context,
    or when only digits are present.
    """
    if not text:
        return None
    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    if not tokens:
        return None

    result = 0
    current = 0
    found_any = False
    for tok in tokens:
        if tok in _UNITS:
            current += _UNITS[tok]
            found_any = True
            continue
        if tok in _TEENS:
            current += _TEENS[tok]
            found_any = True
            continue
        if tok in _TENS:
            current += _TENS[tok]
            found_any = True
            continue
        if tok in _HUNDREDS:
            current += _HUNDREDS[tok]
            found_any = True
            continue
        if tok in _SCALES:
            multiplier = _SCALES[tok]
            if current == 0:
                # "тысяча" with no preceding count → 1000.
                current = 1
            result += current * multiplier
            current = 0
            found_any = True
            continue
        if any(tok.startswith(stem) for stem in _SLANG_THOUSAND_STEMS):
            # Bug 10: slang "косарь" / "тонн" / "кусков" / "штук" all
            # mean thousand in Russian colloquial speech. Stem-prefix
            # match covers every grammatical case.
            if current == 0:
                current = 1
            result += current * 1000
            current = 0
            found_any = True
            continue
        if tok in _PERCENT_STOPWORDS:
            # Percent context: drop everything accumulated so far so
            # "двадцать процентов" doesn't surface as 20.
            current = 0
            result = 0
            found_any = False
            continue
        # Unknown word — skip silently so phrases like "оставим двадцать
        # тысяч долларов" still parse.

    if current:
        result += current

    return result if found_any and result > 0 else None
