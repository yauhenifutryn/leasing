"""Fix 28: multi-field change-confirm + apply.

Scenario from production 2026-04-18 (session acf1ffb5):
  User: "Ладно, давай всё-таки возьмём легковой автомобиль за 80 тысяч рублей."
  Bot:  "Меняю предмет лизинга на Легковой автомобиль, остальное оставляю. Всё верно?"
  User: "Нет, ещё и стоимость поменять на 80 тысяч рублей."
  Bot:  "Меняю предмет лизинга на Легковой автомобиль, остальное оставляю. Всё верно?"  ← same!
  User: "Да."
  → calc runs with cost=80000 despite bot saying "остальное оставляю"

Root cause: pending_change was single-field, extras got applied via
apply_patches bypassing the change-confirm gate. Fix 28 stores changes
as a `{changes: {field: {old,new}}}` dict so all edits are visible.
"""

from __future__ import annotations

import pytest

from backend.profile_prompts import build_change_confirm_text
from backend.session import ClientProfile


# ----- build_change_confirm_text -----

def test_legacy_single_field_shape_still_works():
    text = build_change_confirm_text({"field": "term_months", "new_value": 48})
    assert "срок" in text
    assert "48" in text
    assert "остальное оставляю" in text


def test_multi_field_shape_lists_both_changes():
    pc = {
        "changes": {
            "subject": {"old": "Грузовой автомобиль", "new": "Легковой автомобиль"},
            "cost": {"old": 150000, "new": 80000},
        }
    }
    text = build_change_confirm_text(pc)
    assert "Легковой автомобиль" in text
    assert "80000" in text
    assert "предмет" in text.lower() or "лизинг" in text.lower()
    assert "стоимость" in text
    assert "остальное оставляю" in text


def test_multi_field_three_changes():
    pc = {
        "changes": {
            "subject": {"old": "Грузовой автомобиль", "new": "Легковой автомобиль"},
            "cost": {"old": 150000, "new": 80000},
            "term_months": {"old": 36, "new": 48},
        }
    }
    text = build_change_confirm_text(pc)
    assert "Легковой" in text
    assert "80000" in text
    assert "48" in text
    # Joined with "и" before the last item
    assert " и " in text


def test_multi_field_single_entry_reads_like_single():
    """If `changes` dict has one key, text should still be grammatical."""
    pc = {"changes": {"term_months": {"old": 36, "new": 48}}}
    text = build_change_confirm_text(pc)
    assert "срок" in text
    assert "48" in text
    assert " и " not in text


def test_empty_changes_returns_fallback():
    pc = {"changes": {}}
    text = build_change_confirm_text(pc)
    assert "Уточните" in text or "что именно" in text.lower()


def test_none_returns_clarification():
    text = build_change_confirm_text(None)
    assert "Уточните" in text


# ----- ClientProfile.apply_pending_change -----

def test_apply_pending_change_single_field_legacy():
    p = ClientProfile()
    p.term_months = 36
    p.pending_change = {"field": "term_months", "old_value": 36, "new_value": 60}
    assert p.apply_pending_change() is True
    assert p.term_months == 60
    assert p.pending_change is None


def test_apply_pending_change_multi_field():
    p = ClientProfile()
    p.subject = "Грузовой автомобиль"
    p.cost = 150000
    p.pending_change = {
        "changes": {
            "subject": {"old": "Грузовой автомобиль", "new": "Легковой автомобиль"},
            "cost": {"old": 150000, "new": 80000},
        }
    }
    assert p.apply_pending_change() is True
    assert p.subject == "Легковой автомобиль"
    assert p.cost == 80000
    assert p.pending_change is None


def test_apply_pending_change_skips_unknown_field():
    p = ClientProfile()
    p.term_months = 36
    p.pending_change = {
        "changes": {
            "term_months": {"old": 36, "new": 60},
            "not_a_field": {"old": "x", "new": "y"},
        }
    }
    assert p.apply_pending_change() is True
    assert p.term_months == 60


def test_apply_pending_change_empty_returns_false():
    p = ClientProfile()
    p.pending_change = None
    assert p.apply_pending_change() is False
