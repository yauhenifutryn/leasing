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


def test_age_years_with_term_yields_age_prompt_first() -> None:
    # Fix 1.13 (2026-04-19) — swap the priority. age must be asked before
    # term/prepaid. Live call 743c1a0e: client said "Два года" meaning a
    # 2-year lease; classifier assigned it to BOTH term_months=24 AND
    # age_years=2 because they hadn't been separated in the prompt
    # sequence. Asking age on its own turn kills the collision.
    text = build_clarification_prompt({"age_years", "term_months", "prepaid"}, None)
    assert "лет" in text.lower(), f"age must be asked first: {text}"
    assert "срок" not in text.lower(), f"term should be deferred: {text}"


def test_age_years_after_cost_still_deferred() -> None:
    # Priority ordering: when cost/currency are also missing, those still
    # come first (they're upstream of the calculator in the funnel).
    # Age only outranks term/prepaid/graph, not cost/currency.
    text = build_clarification_prompt({"age_years", "cost", "currency"}, None)
    assert "стоимость" in text.lower()
    assert "возраст" not in text.lower() and "сколько лет" not in text.lower()


def test_term_prepaid_schedule_after_client_type_capture() -> None:
    """Bug 16 (live call 12b9826a, 2026-04-25): юр.лицо + грузовик flow.
    User had subject + cost + currency + condition + age + client_type set.
    Only term/prepaid/schedule remained. The clarify prompt must ask for
    those three and NOT re-ask subject (which the LLM was hallucinating
    when the deterministic gate failed to fire)."""
    text = build_clarification_prompt(
        {"prepaid", "term_months", "type_schedule"}, None
    )
    low = text.lower()
    assert "срок" in low, f"must ask term: {text}"
    assert "аванс" in low, f"must ask prepaid: {text}"
    assert "график" in low or "аннуитет" in low, f"must ask schedule: {text}"
    # Subject hallucination guard — must NOT re-ask which kind of subject.
    assert "транспорт" not in low or "лизинг" in low, (
        f"prompt should not re-ask subject category: {text}"
    )


def test_change_confirm_speaks_russian_age_label() -> None:
    # Fix 1.12 — "Меняю age_years на 5" must never be spoken. The change-
    # confirm must render the Russian label. Live call 743c1a0e shipped
    # the bug in the wild (call audio confirmed).
    from backend.profile_prompts import build_change_confirm_text
    text = build_change_confirm_text({"field": "age_years", "new_value": 5})
    assert "age_years" not in text, f"raw field name leaked to TTS: {text}"
    assert "возраст" in text.lower(), f"Russian label missing: {text}"
    assert "5" in text
