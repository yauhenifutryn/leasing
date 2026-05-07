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

# Bug session-2026-05-08: colloquial fractional Russian numbers. Live
# transcript "за полмиллион рублей" was not understood by the classifier.
# Three patterns to handle:
#   1. Compound single-token: "полмиллион(а|ов)", "полтысячи", "полста"
#   2. "пол" + scale word: "пол миллиона", "пол тысячи" (0.5x scale)
#   3. "полтора" / "полторы" + scale word: "полтора миллиона",
#      "полторы тысячи" (1.5x scale; gendered against the scale)
_HALF_COMPOUND = {
    "полмиллион": 500_000,
    "полмиллиона": 500_000,
    "полмиллионов": 500_000,
    "полмиллиард": 500_000_000,
    "полмиллиарда": 500_000_000,
    "полтысячи": 500,
    "полста": 50,
}
_ONE_AND_HALF = {"полтора", "полторы"}
_HALF_PREFIX = {"пол"}

_WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)


def _consume_compound(tokens: list[str], i: int) -> Optional[tuple[int, int]]:
    """Try to consume a compound numeric pattern starting at tokens[i].

    Returns (value, consumed_count) on success, or None when no compound
    pattern matches. Tokens must already be lowercased.

    Patterns:
      - "полмиллиона" (1 token) → 500_000
      - "полтора миллиона" / "полторы тысячи" (2 tokens) → 1.5 × scale
      - "пол миллиона" / "пол тысячи" (2 tokens) → 0.5 × scale
    Bare "полтора" / "пол" without a scale word returns None so the
    caller can skip the token (e.g., "деревянный пол" stays as text).
    """
    tok = tokens[i]
    if tok in _HALF_COMPOUND:
        return (_HALF_COMPOUND[tok], 1)
    if tok in _ONE_AND_HALF:
        if i + 1 < len(tokens) and tokens[i + 1] in _SCALES:
            return (_SCALES[tokens[i + 1]] * 3 // 2, 2)
        return None
    if tok in _HALF_PREFIX:
        if i + 1 < len(tokens) and tokens[i + 1] in _SCALES:
            return (_SCALES[tokens[i + 1]] // 2, 2)
        return None
    return None


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
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]

        compound = _consume_compound(tokens, i)
        if compound is not None:
            value, consumed = compound
            result += value
            found_any = True
            i += consumed
            continue
        if tok in _ONE_AND_HALF or tok in _HALF_PREFIX:
            # Bare "полтора" / "пол" with no scale follower — skip.
            i += 1
            continue

        if tok in _UNITS:
            current += _UNITS[tok]
            found_any = True
            i += 1
            continue
        if tok in _TEENS:
            current += _TEENS[tok]
            found_any = True
            i += 1
            continue
        if tok in _TENS:
            current += _TENS[tok]
            found_any = True
            i += 1
            continue
        if tok in _HUNDREDS:
            current += _HUNDREDS[tok]
            found_any = True
            i += 1
            continue
        if tok in _SCALES:
            multiplier = _SCALES[tok]
            if current == 0:
                # "тысяча" with no preceding count → 1000.
                current = 1
            result += current * multiplier
            current = 0
            found_any = True
            i += 1
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
            i += 1
            continue
        if tok in _PERCENT_STOPWORDS:
            # Percent context: drop everything accumulated so far so
            # "двадцать процентов" doesn't surface as 20.
            current = 0
            result = 0
            found_any = False
            i += 1
            continue
        # Unknown word — skip silently so phrases like "оставим двадцать
        # тысяч долларов" still parse.
        i += 1

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

_NUMBER_TOKENS = (
    set(_UNITS) | set(_TEENS) | set(_TENS) | set(_HUNDREDS) | set(_SCALES)
    | set(_HALF_COMPOUND) | _ONE_AND_HALF | _HALF_PREFIX
)


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
    user-stated quantity, or a bare "пол" / "полтора" without a scale
    follower. The caller then emits the original tokens unchanged so
    digit-mixed input ("100 тысяч долларов") survives.
    """
    lower = [w.lower() for w in words]
    result = 0
    current = 0
    saw_explicit = False  # any concrete numeric token (units/teens/tens/
                          # hundreds, OR a compound half/one-and-a-half form)
    i = 0
    n = len(lower)
    while i < n:
        tl = lower[i]

        compound = _consume_compound(lower, i)
        if compound is not None:
            value, consumed = compound
            result += value
            saw_explicit = True
            i += consumed
            continue
        if tl in _ONE_AND_HALF or tl in _HALF_PREFIX:
            # Bare "полтора" / "пол" with no scale follower — bail out;
            # caller emits original tokens verbatim ("деревянный пол").
            return None

        if tl in _UNITS:
            current += _UNITS[tl]
            saw_explicit = True
            i += 1
            continue
        if tl in _TEENS:
            current += _TEENS[tl]
            saw_explicit = True
            i += 1
            continue
        if tl in _TENS:
            current += _TENS[tl]
            saw_explicit = True
            i += 1
            continue
        if tl in _HUNDREDS:
            current += _HUNDREDS[tl]
            saw_explicit = True
            i += 1
            continue
        if tl in _SCALES:
            # No implicit "1" multiplier when the user didn't say a number.
            # "тысяч" alone is preserved as-is by the caller.
            if not saw_explicit and current == 0:
                return None
            base = current if current else 1
            result += base * _SCALES[tl]
            current = 0
            i += 1
            continue
        if any(tl.startswith(stem) for stem in _SLANG_THOUSAND_STEMS):
            if not saw_explicit and current == 0:
                return None
            base = current if current else 1
            result += base * 1000
            current = 0
            i += 1
            continue
        # Unknown — caller should not include it in the run.
        i += 1
    result += current
    return result if (result > 0 and saw_explicit) else None


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
