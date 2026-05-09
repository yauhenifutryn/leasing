"""B3 — grammatical case in change-readback ("Меняю + accusative").

The change-readback says "Меняю X на Y" where X must be in the accusative
case (винительный падеж). The current renderer emits the nominative
field labels:
    Меняю валюта на белорусские рубли  (wrong — feminine -а should be -у)
    Меняю сумма аванса на 5000        (wrong — same)

Russian rule for inanimate nouns:
    masculine -∅ (срок, аванс, тип, предмет, возраст): nom == acc — OK
    feminine -ь (стоимость): nom == acc — OK
    feminine -а (валюта, сумма): nom -а → acc -у — NEEDS FIX
    neuter (состояние): nom == acc — OK

This test pins the accusative form so future label additions don't
regress.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_prompts import build_change_confirm_text  # noqa: E402


def test_currency_change_uses_accusative():
    pending = {"changes": {"currency": {"old": "USD", "new": "BYN"}}}
    out = build_change_confirm_text(pending)
    assert "валюту" in out, f"expected accusative 'валюту', got: {out!r}"
    assert "Меняю валюта" not in out, f"nominative leaked: {out!r}"


def test_prepaid_amount_change_uses_accusative():
    pending = {"changes": {"prepaid_amount": {"old": 1000, "new": 5000}}}
    out = build_change_confirm_text(pending)
    assert "сумму аванса" in out, f"expected accusative, got: {out!r}"
    assert "Меняю сумма аванса" not in out


def test_cost_change_unaffected_by_accusative_fix():
    """стоимость is feminine ending in soft sign — nom == acc.
    Sanity guard so the fix doesn't accidentally mutate this label."""
    pending = {"changes": {"cost": {"old": 80000, "new": 90000}}}
    out = build_change_confirm_text(pending)
    assert "стоимость на 90000" in out


def test_term_change_unaffected_by_accusative_fix():
    """срок is masculine inanimate — nom == acc."""
    pending = {"changes": {"term_months": {"old": 36, "new": 48}}}
    out = build_change_confirm_text(pending)
    assert "срок на 48" in out


def test_subject_change_unaffected_by_accusative_fix():
    """предмет лизинга is masculine inanimate — nom == acc."""
    pending = {"changes": {"subject": {"old": "Недвижимость", "new": "Легковой автомобиль"}}}
    out = build_change_confirm_text(pending)
    assert "предмет лизинга на" in out


def test_multifield_join_with_accusative_currency():
    """Multi-field join must apply accusative to each label individually."""
    pending = {
        "changes": {
            "currency": {"old": "USD", "new": "BYN"},
            "term_months": {"old": 36, "new": 48},
        },
    }
    out = build_change_confirm_text(pending)
    assert "валюту на белорусские рубли" in out
    assert "срок на 48" in out
