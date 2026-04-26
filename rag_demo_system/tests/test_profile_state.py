"""Unit tests for backend.profile_state pure functions.

Phase 3.A of the apply_turn refactor. These tests lock in behavior of
the extracted helpers before the orchestrator is rewritten against them.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_state import (
    build_snapshot,
    partition_patches,
    derive_implied_flips,
    build_calc_params,
)
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


# --- partition_patches ---

def test_partition_patches_first_time_only() -> None:
    profile = ClientProfile()
    proposed = {"cost": 80000.0, "currency": "USD", "term_months": 36}
    first_time, delta = partition_patches(profile, proposed)
    assert first_time == {"cost": 80000.0, "currency": "USD", "term_months": 36}
    assert delta == {}


def test_partition_patches_delta_on_captured() -> None:
    profile = ClientProfile(cost=80000.0, term_months=36)
    proposed = {"cost": 80000.0, "term_months": 60}
    first_time, delta = partition_patches(profile, proposed)
    assert first_time == {}
    assert delta == {"term_months": {"old": 36, "new": 60}}


def test_partition_patches_drops_no_op_equal_values() -> None:
    profile = ClientProfile(subject="Легковой автомобиль")
    proposed = {"subject": "Легковой автомобиль"}
    first_time, delta = partition_patches(profile, proposed)
    assert first_time == {}
    assert delta == {}


def test_partition_patches_mixed_first_time_and_delta() -> None:
    profile = ClientProfile(subject="Легковой автомобиль", cost=80000.0)
    proposed = {"subject": "Грузовой автомобиль", "term_months": 36}
    first_time, delta = partition_patches(profile, proposed)
    assert first_time == {"term_months": 36}
    assert delta == {
        "subject": {"old": "Легковой автомобиль", "new": "Грузовой автомобиль"},
    }


def test_partition_patches_empty_proposed_returns_empty() -> None:
    first_time, delta = partition_patches(ClientProfile(cost=80000.0), {})
    assert first_time == {}
    assert delta == {}


def test_partition_patches_none_new_value_on_captured_field_is_delta() -> None:
    # Implied flip clearing age_years (condition_new -> 1) is a legitimate
    # change-confirm trigger when age_years was previously captured.
    profile = ClientProfile(age_years=5)
    proposed = {"age_years": None}
    first_time, delta = partition_patches(profile, proposed)
    assert first_time == {}
    assert delta == {"age_years": {"old": 5, "new": None}}


# --- derive_implied_flips ---

def test_no_implied_flip_truck_for_phys_lico() -> None:
    """Bug M (2026-04-26): the truck→Юр auto-flip caused an infinite loop
    when user re-asserted Физ on a CHANGE_PENDING confirm. The rule was
    removed; the conflict surfaces via _preflight_calc_policy at readback
    time as a FireOORMessage instead.
    """
    profile = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
    )
    flips = derive_implied_flips(profile, {"subject": "Грузовой автомобиль"})
    assert flips == {}


def test_no_implied_flip_spectech_for_phys_lico() -> None:
    profile = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
    )
    flips = derive_implied_flips(profile, {"subject": "Спецтехника"})
    assert flips == {}


def test_no_implied_flip_commercial_transport_for_phys_lico() -> None:
    profile = ClientProfile(client_type="Физическое лицо")
    flips = derive_implied_flips(profile, {"subject": "Коммерческий транспорт"})
    assert flips == {}


def test_no_flip_when_client_type_already_yur() -> None:
    profile = ClientProfile(
        client_type="Юридическое лицо",
        subject="Легковой автомобиль",
    )
    flips = derive_implied_flips(profile, {"subject": "Грузовой автомобиль"})
    assert flips == {}


def test_no_flip_when_classifier_already_proposes_yur() -> None:
    # Classifier already flipped client_type to Юр; no additional implied flip.
    profile = ClientProfile(client_type="Физическое лицо")
    flips = derive_implied_flips(
        profile,
        {"subject": "Грузовой автомобиль", "client_type": "Юридическое лицо"},
    )
    assert flips == {}


def test_no_flip_for_passenger_vehicle() -> None:
    profile = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
    )
    flips = derive_implied_flips(profile, {"subject": "Легковой автомобиль"})
    assert flips == {}


def test_condition_new_clears_age_years() -> None:
    profile = ClientProfile(condition_new=0, age_years=5)
    flips = derive_implied_flips(profile, {"condition_new": 1})
    assert flips == {"age_years": None}


def test_condition_new_stays_0_does_not_clear_age_years() -> None:
    profile = ClientProfile(condition_new=0, age_years=5)
    flips = derive_implied_flips(profile, {"condition_new": 0})
    assert flips == {}


def test_condition_new_1_without_age_years_is_noop() -> None:
    profile = ClientProfile(condition_new=0, age_years=None)
    flips = derive_implied_flips(profile, {"condition_new": 1})
    assert flips == {}


def test_implied_flips_combined_subject_and_condition() -> None:
    # Edge case: user says "Грузовой новый" — only the condition→age_years
    # wipe still fires (subject→client_type flip was removed in Bug M).
    profile = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        condition_new=0,
        age_years=5,
    )
    flips = derive_implied_flips(
        profile,
        {"subject": "Грузовой автомобиль", "condition_new": 1},
    )
    assert flips == {"age_years": None}


# --- build_calc_params ---
#
# These tests mirror the legacy construction at app.py:2211-2234 exactly.
# Any change here implies a change to the calculator API contract.

def test_build_calc_params_full_byn_physical() -> None:
    profile = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=120000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        type_schedule="0",
    )
    params = build_calc_params(profile)
    assert params == {
        "subject": "Легковой автомобиль",
        "cost": 120000.0,
        "currency": "BYN",
        "client_type": "Физическое лицо",
        "condition_new": 1,
        "prepaid": 20.0,
        "prepaid_pct": 20.0,   # legacy alias for compat
        "term": 36,
        "type_schedule": "0",
    }


def test_build_calc_params_with_age_years_sets_both_aliases() -> None:
    profile = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=15000.0,
        currency="USD",
        condition_new=0,
        age_years=3,
        prepaid_pct=30.0,
        term_months=24,
        type_schedule="0",
    )
    params = build_calc_params(profile)
    # Both `age` and `age_years` set for backward compat (app.py:2223-2225).
    assert params["age"] == 3
    assert params["age_years"] == 3


def test_build_calc_params_prepaid_amount_branch() -> None:
    profile = ClientProfile(
        client_type="Юридическое лицо",
        subject="Спецтехника",
        cost=500000.0,
        currency="BYN",
        condition_new=1,
        prepaid_amount=100000.0,
        term_months=48,
        type_schedule="1",
    )
    params = build_calc_params(profile)
    # prepaid_amount path (elif branch): no `prepaid` or `prepaid_pct` key.
    assert params["prepaid_amount"] == 100000.0
    assert "prepaid" not in params
    assert "prepaid_pct" not in params


def test_build_calc_params_omits_none_optional_fields() -> None:
    # Currency, client_type, condition_new, type_schedule, age_years,
    # prepaid_* all conditionally set. A profile with only cost+subject
    # should produce a minimal params dict.
    profile = ClientProfile(cost=50000.0, subject="Легковой автомобиль")
    params = build_calc_params(profile)
    assert params["cost"] == 50000.0
    assert params["subject"] == "Легковой автомобиль"
    assert "currency" not in params
    assert "client_type" not in params
    assert "condition_new" not in params
    assert "term" not in params
    assert "type_schedule" not in params
    assert "age" not in params
    assert "age_years" not in params
    assert "prepaid" not in params
