"""Tests for ClientProfile: completeness checks and field merge semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.session import ClientProfile  # noqa: E402


def test_empty_profile_is_not_complete() -> None:
    p = ClientProfile()
    assert p.is_complete_for_calc() is False
    missing = p.missing_fields()
    assert "client_type" in missing
    assert "subject" in missing
    assert "cost" in missing


def test_profile_with_all_fields_is_complete() -> None:
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=70000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=84,
        type_schedule="0",
    )
    assert p.is_complete_for_calc() is True
    assert p.missing_fields() == set()


def test_used_subject_requires_age() -> None:
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=50000.0,
        currency="BYN",
        condition_new=0,  # used
        prepaid_pct=20.0,
        term_months=60,
        type_schedule="0",
    )
    assert p.is_complete_for_calc() is False
    assert "age_years" in p.missing_fields()


def test_prepaid_either_pct_or_amount() -> None:
    base = dict(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=70000.0,
        currency="BYN",
        condition_new=1,
        term_months=60,
        type_schedule="0",
    )
    assert ClientProfile(**base, prepaid_pct=20.0).is_complete_for_calc() is True
    assert ClientProfile(**base, prepaid_amount=14000.0).is_complete_for_calc() is True
    assert ClientProfile(**base).is_complete_for_calc() is False


def test_apply_patches_updates_fields() -> None:
    p = ClientProfile()
    changed = p.apply_patches({"subject": "Легковой автомобиль", "cost": 70000.0})
    assert p.subject == "Легковой автомобиль"
    assert p.cost == 70000.0
    assert "subject" in changed
    assert "cost" in changed


def test_apply_patches_skips_none() -> None:
    p = ClientProfile(subject="Легковой автомобиль")
    p.apply_patches({"subject": None, "cost": 70000.0})
    assert p.subject == "Легковой автомобиль"
    assert p.cost == 70000.0


def test_apply_patches_respects_locked_fields() -> None:
    p = ClientProfile(term_months=84)
    p.locked_fields.add("term_months")
    p.apply_patches({"term_months": 48})
    assert p.term_months == 84


def test_apply_patches_returns_empty_for_empty_input() -> None:
    p = ClientProfile()
    assert p.apply_patches({}) == {}
    assert p.apply_patches(None) == {}


def test_apply_patches_ignores_unknown_keys() -> None:
    p = ClientProfile()
    changed = p.apply_patches({"unknown_field": "value"})
    assert changed == {}


def test_to_dict_roundtrip() -> None:
    p = ClientProfile(
        name="Сергей",
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=70000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=84,
        type_schedule="0",
    )
    d = p.to_dict()
    assert d["name"] == "Сергей"
    assert d["client_type"] == "Физическое лицо"
    assert d["confirmed_at"] is None
    assert d["locked_fields"] == []
