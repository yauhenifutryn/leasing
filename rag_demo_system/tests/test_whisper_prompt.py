"""Regression tests for Whisper initial_prompt: token budget and key-term inclusion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("faster_whisper")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.whisper_server import _DEFAULT_INITIAL_PROMPT  # noqa: E402


def test_prompt_fits_224_token_budget() -> None:
    """Whisper's initial_prompt is capped at 224 tokens. Anything above silently truncates."""
    from faster_whisper.tokenizer import Tokenizer

    class _StubModel:
        is_multilingual = True
        num_languages = 99

    tokenizer = Tokenizer(
        tokenizer=None,
        multilingual=True,
        task="transcribe",
        language="ru",
    )
    tokens = tokenizer.encode(_DEFAULT_INITIAL_PROMPT)
    assert len(tokens) < 224, f"Prompt is {len(tokens)} tokens, cap is 224"


def test_prompt_contains_critical_vocabulary() -> None:
    """Key terms missing before this change; must all appear at least once now."""
    required = [
        "Ксения",
        "линейный",
        "аннуитет",
        "дифференцированный",
        "нагрузка",
        "переплата",
        "физик",
        "юрик",
        "ипэшник",
        "лизингополучатель",
        "выкупной",
    ]
    for term in required:
        assert term in _DEFAULT_INITIAL_PROMPT, f"Missing required term: {term}"


def test_prompt_places_bot_name_near_end() -> None:
    """Whisper keeps only the last 224 tokens; Ксения must be in the final half to be guaranteed."""
    mid = len(_DEFAULT_INITIAL_PROMPT) // 2
    assert "Ксения" in _DEFAULT_INITIAL_PROMPT[mid:], "Ксения must appear in second half of prompt"
