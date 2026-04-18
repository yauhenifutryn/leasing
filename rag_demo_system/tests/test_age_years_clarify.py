"""Fix 1.5 — age_years clarification branch.

Regression from live call on 2026-04-19: a физлицо saying "бу мотоцикл"
reliably got condition_new=0 (Fix 1.3 working), which triggers
`ClientProfile.missing_fields()` to require `age_years`. Before this fix,
`build_clarification_prompt` had no branch for age_years and fell through
to the generic "Уточните параметры расчёта, пожалуйста", which the LLM
cannot act on — the orchestrator looped indefinitely while the client
repeated already-captured term/prepaid/graph values.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_prompts import build_clarification_prompt


def test_age_years_alone_returns_specific_prompt() -> None:
    text = build_clarification_prompt({"age_years"}, None)
    low = text.lower()
    # Must ask about the age, not fall through to the generic prompt.
    assert "лет" in low, f"clarify must ask about years: {text}"
    assert "б/у" in low or "бу " in low or "техник" in low, (
        f"clarify should mention б/у / техника context: {text}"
    )
    assert "уточните параметры расчёта" not in low, (
        f"must NOT be the generic fallback: {text}"
    )


def test_age_years_with_cost_yields_cost_prompt_first() -> None:
    # Priority ordering check: when cost is also missing, ask for cost first
    # (the higher-value ask). age_years comes next turn once cost is filled.
    text = build_clarification_prompt({"age_years", "cost", "currency"}, None)
    assert "стоимость" in text.lower()


def test_age_years_with_term_yields_term_prompt_first() -> None:
    # Same priority principle for the term/prepaid block.
    text = build_clarification_prompt({"age_years", "term_months", "prepaid"}, None)
    assert "срок" in text.lower()
