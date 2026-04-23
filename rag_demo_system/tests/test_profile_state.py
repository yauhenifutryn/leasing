"""Unit tests for backend.profile_state pure functions.

Phase 3.A of the apply_turn refactor. These tests lock in behavior of
the extracted helpers before the orchestrator is rewritten against them.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_state import build_snapshot
from backend.session import ClientProfile


def test_build_snapshot_copies_every_field() -> None:
    profile = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="USD",
        original_cost=80000.0,
        original_currency="USD",
        condition_new=1,
        age_years=None,
        prepaid_pct=20.0,
        prepaid_amount=None,
        term_months=36,
        type_schedule="0",
    )
    snap = build_snapshot(profile)
    assert snap.client_type == "Физическое лицо"
    assert snap.subject == "Легковой автомобиль"
    assert snap.cost == 80000.0
    assert snap.currency == "USD"
    assert snap.original_cost == 80000.0
    assert snap.original_currency == "USD"
    assert snap.condition_new == 1
    assert snap.age_years is None
    assert snap.prepaid_pct == 20.0
    assert snap.prepaid_amount is None
    assert snap.term_months == 36
    assert snap.type_schedule == "0"


def test_build_snapshot_is_frozen() -> None:
    profile = ClientProfile()
    snap = build_snapshot(profile)
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.cost = 1.0  # type: ignore[misc]


def test_build_snapshot_from_empty_profile_yields_all_none() -> None:
    snap = build_snapshot(ClientProfile())
    assert snap.client_type is None
    assert snap.subject is None
    assert snap.cost is None
    assert snap.term_months is None
