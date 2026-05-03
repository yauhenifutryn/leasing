"""Token-budget guard for the classifier system prompt.

Bug 29 (2026-05-03 evening) was caused by silently growing the inline
classifier prompt past vLLM max_model_len=4096. vLLM returned HTTP 400 on
every classifier request and the bot fell to LLM fallback. This guard
asserts the prompt fits within a fraction of max_model_len so the next
regression of this class is caught at startup, not in production.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.classifier_prompt import (  # noqa: E402
    assert_prompt_token_budget,
    build_classifier_system_prompt,
    count_tokens_approx,
)


def test_build_classifier_system_prompt_returns_nonempty_string():
    prompt = build_classifier_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 500
    assert "END_CALL" in prompt  # canary for one of the Bug 29 additions


def test_count_tokens_approx_is_reasonable():
    n = count_tokens_approx(build_classifier_system_prompt())
    assert 500 < n < 6000


def test_assert_prompt_token_budget_passes_at_current_size():
    assert_prompt_token_budget(max_model_len=8192, fraction=0.80)


def test_assert_prompt_token_budget_raises_when_over_budget():
    with pytest.raises(AssertionError, match="exceeds budget"):
        assert_prompt_token_budget(max_model_len=512, fraction=0.80)
