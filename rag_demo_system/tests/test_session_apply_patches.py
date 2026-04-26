"""Tests for ClientProfile.apply_additive_patches: first-time capture semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.session import ClientProfile  # noqa: E402


def test_apply_additive_patches_clears_prepaid_sibling_pct_to_amount():
    p = ClientProfile(prepaid_pct=20.0)
    changed = p.apply_additive_patches({"prepaid_amount": 5000.0})
    assert p.prepaid_amount == 5000.0
    assert p.prepaid_pct is None
    assert changed == {"prepaid_amount": 5000.0}


def test_apply_additive_patches_clears_prepaid_sibling_amount_to_pct():
    p = ClientProfile(prepaid_amount=5000.0)
    p.apply_additive_patches({"prepaid_pct": 20.0})
    assert p.prepaid_pct == 20.0
    assert p.prepaid_amount is None


def test_apply_additive_patches_respects_locked_fields():
    p = ClientProfile(cost=10000.0, locked_fields={"cost"})
    changed = p.apply_additive_patches({"cost": 99999.0})
    assert p.cost == 10000.0
    assert "cost" not in changed


def test_apply_additive_patches_skips_none_values():
    p = ClientProfile(subject="Легковой автомобиль")
    changed = p.apply_additive_patches({"subject": None, "term_months": 24})
    assert p.subject == "Легковой автомобиль"
    assert p.term_months == 24
    assert "subject" not in changed


def test_apply_additive_patches_skips_unknown_attrs():
    p = ClientProfile()
    changed = p.apply_additive_patches({"not_a_field": 42})
    assert "not_a_field" not in changed


def test_apply_additive_patches_prepaid_pct_wins_when_both_provided():
    # When both prepaid_pct and prepaid_amount arrive in the same patch
    # dict, prepaid_pct wins — both are applied in dict order, then the
    # post-loop sibling-clear's `if _applied_prepaid_pct` branch runs
    # before the `elif`, so prepaid_amount ends up None regardless of
    # dict insertion order. This is deterministic and matches
    # apply_pending_change semantics.
    p = ClientProfile()
    changed = p.apply_additive_patches({
        "prepaid_amount": 5000.0,
        "prepaid_pct": 20.0,
    })
    assert p.prepaid_pct == 20.0
    assert p.prepaid_amount is None
    assert "prepaid_pct" in changed
    # Note: prepaid_amount was briefly applied then cleared; the caller
    # sees it in `changed` because it WAS applied from patches, but the
    # final profile state has it as None.
    assert changed.get("prepaid_amount") == 5000.0
