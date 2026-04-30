"""Bug 10 (live calls 2026-04-29): users spelled cost in slang —
"20 косарей", "30 тонн", "15 кусков", "20 штук" — and the dispatcher
extracted cost=20 / 30 / 15 / 20 instead of 20000 / 30000 / 15000 / 20000.
The bot then read back "20 рублей" / "20 долларов" which is wrong.

Slang thousand multipliers are finite and well-known, so a small lexical
addition to the parser is the right scope (not regex post-processing of
the LLM output, not a "guess what the user meant" heuristic).

Two layers:
1. numeric_words_ru.parse_ru_number — accept word-form spelling
   ("двадцать косарей" → 20000).
2. utterance_grounding.extract_cost_from_utterance — accept digit-form
   spelling ("20 косарей" → 20000), the common case.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from backend.numeric_words_ru import parse_ru_number
from backend.utterance_grounding import extract_cost_from_utterance


@pytest.mark.parametrize("text,expected", [
    # косарь / косаря / косарей / косаре
    ("двадцать косарей", 20000),
    ("сто косарей", 100000),
    # тонна / тонн / тонны
    ("тридцать тонн", 30000),
    ("пятьдесят тонн", 50000),
    # кусок / куска / кусков / куски
    ("пятнадцать кусков", 15000),
    # штука / штук / штуки
    ("двадцать штук", 20000),
])
def test_parse_ru_number_accepts_slang_thousand_stems(text, expected):
    assert parse_ru_number(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("20 косарей", 20000),
    ("100 косарей", 100000),
    ("30 тонн", 30000),
    ("50 тонн", 50000),
    ("15 кусков", 15000),
    ("20 штук", 20000),
    # mid-sentence variants from realistic utterances
    ("стоимость 20 косарей долларов", 20000),
    ("где-то 30 тонн", 30000),
])
def test_extract_cost_from_utterance_accepts_digit_slang(text, expected):
    assert extract_cost_from_utterance(text) == expected


def test_slang_below_cost_min_still_rejects():
    """Range gate (10_000 ≤ cost ≤ 100_000_000) must still reject
    out-of-band slang values so unrelated chatter doesn't ground."""
    # 5 косарей = 5000, below COST_MIN — extractor must drop.
    assert extract_cost_from_utterance("5 косарей") is None


def test_slang_does_not_affect_percent_context():
    """Percent context drop in parse_ru_number must continue to work
    even when slang stems are added to scales."""
    # "двадцать процентов" still resolves to None (percent stopwords
    # zero out the accumulator).
    assert parse_ru_number("двадцать процентов") is None
