"""Currency drift for Физ лицо: foreign currencies (EUR/RUB/CNY/etc.)
must drift to BYN at the calc preflight, not get rejected with FireOORMessage.

User intent (2026-05-09): keep the conversation moving instead of
looping the user back to "specify in BYN or USD". For individuals
the calc is always in BYN — convert at the NBRB rate of the source
currency. For Юр лицо the foreign currency is left alone (the
calculator API supports BYN/USD/EUR/RUB for legal entities).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.session import ProfileState  # noqa: E402
from backend.turn_action import FireOORMessage  # noqa: E402
from backend.turn_dispatcher import _preflight_calc_policy  # noqa: E402

from tests.test_apply_turn import make_complete_profile  # noqa: E402


@pytest.fixture
def stubbed_eur_rate():
    """Inject a deterministic EUR/BYN rate via the per-currency cache."""
    import backend.profile_prompts as pp
    import time as _time

    pp._NBRB_RATE_CACHE["EUR"] = (3.55, _time.monotonic())
    yield 3.55
    pp._NBRB_RATE_CACHE.pop("EUR", None)


@pytest.fixture
def stubbed_rub_rate():
    """Inject a deterministic RUB/BYN rate via the per-currency cache."""
    import backend.profile_prompts as pp
    import time as _time

    pp._NBRB_RATE_CACHE["RUB"] = (0.0342, _time.monotonic())
    yield 0.0342
    pp._NBRB_RATE_CACHE.pop("RUB", None)


def test_eur_for_phys_drifts_to_byn_does_not_oor(stubbed_eur_rate):
    profile = make_complete_profile(cost=80000.0, currency="EUR")
    profile.state = ProfileState.READBACK_PENDING
    action = _preflight_calc_policy(profile)
    # Drift, not reject: preflight returns None so caller proceeds with FireCalc.
    assert action is None, f"expected drift (None), got {type(action).__name__}: {action!r}"
    assert profile.currency == "BYN"
    assert profile.cost == round(80000.0 * stubbed_eur_rate, 2)
    assert profile.original_currency == "EUR"
    assert profile.original_cost == 80000.0


def test_rub_for_phys_drifts_to_byn_does_not_oor(stubbed_rub_rate):
    profile = make_complete_profile(cost=5_000_000.0, currency="RUB")
    profile.state = ProfileState.READBACK_PENDING
    action = _preflight_calc_policy(profile)
    assert action is None
    assert profile.currency == "BYN"
    assert profile.cost == round(5_000_000.0 * stubbed_rub_rate, 2)
    assert profile.original_currency == "RUB"


def test_yur_eur_left_untouched_no_drift():
    """Юр лицо asks in EUR — calculator handles it directly, no drift."""
    profile = make_complete_profile(
        client_type="Юридическое лицо",
        subject="Грузовой автомобиль",
        cost=80000.0,
        currency="EUR",
    )
    profile.state = ProfileState.READBACK_PENDING
    action = _preflight_calc_policy(profile)
    assert action is None
    assert profile.currency == "EUR"
    assert profile.cost == 80000.0
    assert profile.original_currency is None


def test_phys_byn_no_op():
    """BYN for Физ лицо is the native case — no conversion."""
    profile = make_complete_profile(cost=240000.0, currency="BYN")
    profile.state = ProfileState.READBACK_PENDING
    action = _preflight_calc_policy(profile)
    assert action is None
    assert profile.currency == "BYN"
    assert profile.cost == 240000.0
    assert profile.original_currency is None


def test_phys_unknown_currency_drifts_to_byn():
    """For currencies NBRB doesn't list (or temporarily unreachable),
    fall back gracefully — drift to BYN at whatever rate the helper
    returns, no OOR. Test isolates by stubbing _get_nbrb_rate directly."""
    import backend.profile_prompts as pp

    profile = make_complete_profile(cost=80000.0, currency="CNY")
    profile.state = ProfileState.READBACK_PENDING
    with patch.object(pp, "_get_nbrb_rate", return_value=0.42):
        action = _preflight_calc_policy(profile)
    assert action is None
    assert profile.currency == "BYN"
    assert profile.original_currency == "CNY"
    assert profile.cost == round(80000.0 * 0.42, 2)


def test_phys_eur_no_old_oor_message():
    """Regression: ensure the old FireOORMessage path is gone for
    EUR/RUB/etc. on Физ лицо (the user explicitly asked for drift)."""
    profile = make_complete_profile(cost=80000.0, currency="EUR")
    profile.state = ProfileState.READBACK_PENDING
    action = _preflight_calc_policy(profile)
    assert not isinstance(action, FireOORMessage)
