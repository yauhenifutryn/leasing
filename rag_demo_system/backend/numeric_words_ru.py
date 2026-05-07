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


# ── Inline word→digit rewriter (chat normalization) ──────────────────────
#
# Used by the chat path to normalize messages BEFORE the classifier sees
# them. Voice gets digit-form numbers from Whisper STT; chat gets whatever
# the user typed ("тридцать процентов"). The 4B classifier is unreliable
# on word-form numbers, so we rewrite them to digits in place. Reuses the
# same lookup tables as parse_ru_number above; conservative — only rewrites
# maximal sequences of word-numbers, skipping mixed digit-form tokens
# entirely so "100 тысяч долларов" stays untouched.

_NUMBER_TOKENS = set(_UNITS) | set(_TEENS) | set(_TENS) | set(_HUNDREDS) | set(_SCALES)


def _is_number_word(tok: str) -> bool:
    tl = tok.lower()
    if tl in _NUMBER_TOKENS:
        return True
    if any(tl.startswith(stem) for stem in _SLANG_THOUSAND_STEMS):
        return True
    return False


def _parse_number_run(words: list[str]) -> int | None:
    """Parse a run of pure number-words into an int. Returns None if the
    run cannot be confidently parsed — for example, a standalone scale
    word ("тысяч долларов") or slang stem ("косарей") with no preceding
    user-stated quantity. The caller then emits the original tokens
    unchanged so digit-mixed input ("100 тысяч долларов") survives.
    """
    result = 0
    current = 0
    saw_user_quantity = False  # at least one units/teens/tens/hundreds word
    for tok in words:
        tl = tok.lower()
        if tl in _UNITS:
            current += _UNITS[tl]
            saw_user_quantity = True
            continue
        if tl in _TEENS:
            current += _TEENS[tl]
            saw_user_quantity = True
            continue
        if tl in _TENS:
            current += _TENS[tl]
            saw_user_quantity = True
            continue
        if tl in _HUNDREDS:
            current += _HUNDREDS[tl]
            saw_user_quantity = True
            continue
        if tl in _SCALES:
            # No implicit "1" multiplier when the user didn't say a number.
            # "тысяч" alone is preserved as-is by the caller.
            if not saw_user_quantity and current == 0:
                return None
            base = current if current else 1
            result += base * _SCALES[tl]
            current = 0
            continue
        if any(tl.startswith(stem) for stem in _SLANG_THOUSAND_STEMS):
            if not saw_user_quantity and current == 0:
                return None
            base = current if current else 1
            result += base * 1000
            current = 0
            continue
        # Unknown — caller should not include it in the run.
    result += current
    return result if (result > 0 and saw_user_quantity) else None


def replace_ru_number_words(text: str) -> str:
    """Rewrite Russian word-form numbers in `text` to digit form.

    "тридцать процентов"               → "30 процентов"
    "три года и тридцать процентов"    → "3 года и 30 процентов"
    "сто тысяч долларов"               → "100000 долларов"
    "двадцать пять процентов"          → "25 процентов"
    "100 тысяч"                        → "100 тысяч"  (digit-mixed; untouched)
    "хочу машину"                      → "хочу машину"

    Idempotent on already-digit text. Preserves separators verbatim by
    only collapsing whitespace WITHIN a detected number run.

    Implementation: walk the source text token-by-token (Cyrillic words
    only — digits are not consumed and break runs). Build a list of
    (kind, content) chunks where kind ∈ {"raw", "num"}. "num" chunks are
    collapsed to a single digit; "raw" chunks are emitted verbatim.
    """
    if not text:
        return text

    chunks: list[tuple[str, str]] = []   # ("raw" | "num", content)
    raw_buf: list[str] = []
    num_words: list[str] = []
    last_end = 0

    def _flush_num() -> None:
        if num_words:
            chunks.append(("num", " ".join(num_words)))
            num_words.clear()

    def _flush_raw() -> None:
        if raw_buf:
            chunks.append(("raw", "".join(raw_buf)))
            raw_buf.clear()

    for m in _WORD_RE.finditer(text):
        gap = text[last_end:m.start()]
        word = m.group(0)
        last_end = m.end()
        if _is_number_word(word):
            # Bridge a single-whitespace gap between number-words; longer
            # gaps (newlines / punctuation) break the run.
            if num_words and gap.strip() == "":
                pass  # absorb the space; already collapsed by " ".join
            else:
                if num_words:
                    _flush_num()
                _flush_raw()
                raw_buf.append(gap)
                _flush_raw()
            num_words.append(word)
        else:
            _flush_num()
            raw_buf.append(gap + word)
    raw_buf.append(text[last_end:])
    _flush_num()
    _flush_raw()

    out: list[str] = []
    for kind, content in chunks:
        if kind == "raw":
            out.append(content)
        else:
            parsed = _parse_number_run(content.split())
            out.append(str(parsed) if parsed is not None else content)
    return "".join(out)
