"""B2 — change-readback must not emit empty values.

Live trace shape:
    Меняю предмет лизинга на легковой автомобиль и **возраст на ,**
    остальное оставляю. Всё верно?

The empty " на , " happens when subject flip nulls age_years (because the
new subject category doesn't carry an age question), and the multi-field
renderer doesn't gracefully handle a field whose new value is None.

Fix shape: omit fields with new=None from the visible readback. The
clarify gate will re-ask the missing field on a later turn naturally.
If ONLY None-valued changes are staged (degenerate), fall back to the
clarify prompt.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_prompts import build_change_confirm_text  # noqa: E402


def test_subject_flip_with_age_nulled_omits_age_phrase():
    """Live regression: subject Недвижимость→Легковой автомобиль, age
    nulled. The readback should mention only the subject change."""
    pending = {
        "changes": {
            "subject": {"old": "Недвижимость", "new": "Легковой автомобиль"},
            "age_years": {"old": 5, "new": None},
        },
    }
    out = build_change_confirm_text(pending)
    # No empty " на , " or trailing-empty value.
    assert " на , " not in out, f"empty-value phrase leaked: {out!r}"
    assert "на ," not in out.replace(", остальное", ""), (
        f"empty-value comma leaked: {out!r}"
    )
    # Subject change must still be there (canonical capitalization preserved).
    assert "предмет лизинга на Легковой автомобиль" in out
    # Age phrase should be omitted entirely.
    assert "возраст на" not in out, (
        f"age phrase leaked despite null new value: {out!r}"
    )


def test_all_none_changes_falls_back_to_clarify():
    """Edge: every staged change has new=None. The renderer must not
    emit a degenerate "Меняю , остальное оставляю" — fall back to clarify.
    """
    pending = {
        "changes": {
            "age_years": {"old": 5, "new": None},
            "term_months": {"old": 36, "new": None},
        },
    }
    out = build_change_confirm_text(pending)
    assert "Уточните" in out, f"expected clarify fallback, got: {out!r}"
    # Must not emit a malformed "Меняю".
    assert "Меняю , остальное" not in out
    assert "Меняю  и" not in out


def test_single_field_none_value_falls_back_to_clarify():
    """Single-field shape with new=None — same fallback as the all-None
    multi-field case."""
    pending = {
        "changes": {"age_years": {"old": 5, "new": None}},
    }
    out = build_change_confirm_text(pending)
    assert "Уточните" in out, f"expected clarify fallback, got: {out!r}"


def test_normal_two_field_change_unaffected():
    """Sanity: normal multi-field change (no None values) renders the
    legacy two-field 'и' join."""
    pending = {
        "changes": {
            "term_months": {"old": 36, "new": 60},
            "prepaid_pct": {"old": 20, "new": 30},
        },
    }
    out = build_change_confirm_text(pending)
    assert "Меняю" in out
    assert "срок на 60" in out
    assert "аванс на 30" in out
    assert " и " in out
    assert "Всё верно?" in out


def test_one_real_one_null_renders_only_real():
    """Primary B2 case generalized: a real change + a null-side-effect
    change should produce a single-field readback, not the broken join.
    """
    pending = {
        "changes": {
            "currency": {"old": "USD", "new": "BYN"},
            "prepaid_amount": {"old": 5000, "new": None},
        },
    }
    out = build_change_confirm_text(pending)
    # Single-field shape (no "и" join because only one real change left).
    assert "валюту на белорусские рубли" in out or "валюта на белорусские рубли" in out
    assert "сумма аванса" not in out, (
        f"null prepaid_amount leaked into readback: {out!r}"
    )
