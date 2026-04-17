"""Regression tests for Whisper initial_prompt: token budget and key-term inclusion.

The token-budget test loads a small Whisper model once to get a real tokenizer.
If faster-whisper is not installed or model download fails, the token-budget
test is skipped — but the vocabulary and placement tests still run (they only
depend on the string constant).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.whisper_server import _DEFAULT_INITIAL_PROMPT  # noqa: E402

# Hard cap per Whisper architecture; applied by openai-whisper and faster-whisper.
_TOKEN_BUDGET = 224

# Removed terms ("физик", "юрик") were dropped because including them blew
# the token budget. Whisper handles these forms acceptably without explicit
# biasing. The remaining critical terms are what Whisper was missing most.
_CRITICAL_TERMS = [
    "Ксения",
    "линейный",
    "аннуитет",
    "дифференцированный",
    "нагрузка",
    "переплата",
    "ипэшник",
    "лизингополучатель",
    "выкупной",
    "Микро Лизинг",
]


def _encode_with_whisper_tokenizer(text: str) -> list[int]:
    """Load the real Whisper tokenizer via a small model; returns token list.

    Raises pytest.skip if tokenizer cannot be loaded (missing dep, offline, etc.).
    """
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer
    except ImportError as exc:
        pytest.skip(f"faster-whisper not installed: {exc}")

    cache_dir = os.path.expanduser("~/.cache/whisper")
    try:
        model = WhisperModel(
            "tiny", device="cpu", compute_type="int8", download_root=cache_dir
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load Whisper tiny tokenizer (offline?): {exc}")

    tokenizer = Tokenizer(
        tokenizer=model.hf_tokenizer,
        multilingual=True,
        task="transcribe",
        language="ru",
    )
    return tokenizer.encode(text)


def test_prompt_fits_224_token_budget() -> None:
    """Whisper's initial_prompt is capped at 224 tokens; anything above silently truncates."""
    tokens = _encode_with_whisper_tokenizer(_DEFAULT_INITIAL_PROMPT)
    assert len(tokens) < _TOKEN_BUDGET, (
        f"Prompt is {len(tokens)} tokens, cap is {_TOKEN_BUDGET}. "
        "Remove less critical vocabulary until it fits."
    )


def test_prompt_contains_critical_vocabulary() -> None:
    """Key terms absent before the change; must appear at least once now."""
    for term in _CRITICAL_TERMS:
        assert term in _DEFAULT_INITIAL_PROMPT, f"Missing required term: {term}"


def test_prompt_places_bot_name_near_end() -> None:
    """Whisper keeps only the last 224 tokens; Ксения must be in the final half."""
    mid = len(_DEFAULT_INITIAL_PROMPT) // 2
    assert (
        "Ксения" in _DEFAULT_INITIAL_PROMPT[mid:]
    ), "Ксения must appear in second half of prompt so truncation cannot drop it"


def test_prompt_mentions_ксения_multiple_times() -> None:
    """Defense in depth: the bot's name should appear 2+ times for strong biasing."""
    count = _DEFAULT_INITIAL_PROMPT.count("Ксения")
    assert count >= 2, f"Ксения should appear at least twice; found {count}"
