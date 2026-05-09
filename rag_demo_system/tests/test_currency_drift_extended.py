"""Currency drift — full pipeline through ClassifierOutput.

Distinct from test_currency_drift.py which exercises only the dispatcher
preflight on a pre-set ClientProfile.currency. This suite exercises the
whole pipeline:

    user utterance  →  classifier output (validated by schema)
                    →  apply_turn (proposes patches, grounding, fallback)
                    →  preflight drift policy
                    →  ClientProfile mutated (cost & currency)

Catches the live regression on call b31925a8 (2026-05-09 evening): user
said "500 тысяч юаней", classifier emitted currency="CNY", pydantic
Literal rejected it → profile.currency stayed None → drift never fired
→ bot fell to LLM-fallback and hallucinated "юани не принимаем".
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.classifier_schema import ClassifierOutput  # noqa: E402
from backend.session import ClientProfile, ProfileState  # noqa: E402
from backend.turn_action import EmitReadback  # noqa: E402
from backend.turn_dispatcher import apply_turn  # noqa: E402


def _phys_partial_profile(**over):
    """Phys profile with everything except currency+cost. The utterance
    will supply those, exercising the cost+currency capture path."""
    base = dict(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        type_schedule="0",
    )
    base.update(over)
    return ClientProfile(**base)


@pytest.fixture
def stubbed_cny_rate():
    """Inject a deterministic CNY/BYN rate."""
    import backend.profile_prompts as pp
    import time as _t
    pp._NBRB_RATE_CACHE["CNY"] = (0.42, _t.monotonic())
    yield 0.42
    pp._NBRB_RATE_CACHE.pop("CNY", None)


def test_phys_cny_via_schema_drifts_to_byn(stubbed_cny_rate):
    """Live regression repro: utterance contains 'юаней', classifier
    emits currency=CNY. Schema MUST accept it. Drift converts to BYN.
    """
    utt = "за 500 тысяч юаней"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "calculate",
            "cost": 500000,
            "currency": "CNY",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    # Schema must accept CNY (the original bug: Literal rejected it).
    assert co.currency == "CNY", (
        f"schema dropped CNY: {co.currency!r}. "
        f"_CURRENCY_VALUES needs to include CNY."
    )

    profile = _phys_partial_profile()
    action = apply_turn(profile, co, utt)
    # Profile should now have cost+currency captured, then drifted to BYN.
    assert profile.currency == "BYN", f"drift didn't fire: {profile.currency}"
    assert profile.original_currency == "CNY"
    assert profile.original_cost == 500000.0
    # Drift target: 500000 * 0.42 = 210000.
    assert profile.cost == 210000.0
    # Result is the readback (profile completed via this turn).
    assert isinstance(action, EmitReadback)


def test_phys_pln_via_schema_drifts_to_byn():
    """PLN (Polish złoty) — common border-trade currency."""
    import backend.profile_prompts as pp
    utt = "за 100 тысяч злотых"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "calculate",
            "cost": 100000,
            "currency": "PLN",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    assert co.currency == "PLN"
    profile = _phys_partial_profile()
    with patch.object(pp, "_get_nbrb_rate", return_value=0.81):
        apply_turn(profile, co, utt)
    assert profile.currency == "BYN"
    assert profile.original_currency == "PLN"


def test_phys_unknown_iso_currency_still_drifts():
    """An ISO-3 code we don't have a regex for (CHF here) should still
    pass the schema; drift policy converts it to BYN."""
    import backend.profile_prompts as pp
    utt = "за 50 тысяч швейцарских франков"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "calculate",
            "cost": 50000,
            "currency": "CHF",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    assert co.currency == "CHF"
    profile = _phys_partial_profile()
    with patch.object(pp, "_get_nbrb_rate", return_value=3.5):
        apply_turn(profile, co, utt)
    assert profile.currency == "BYN"
    assert profile.original_currency == "CHF"


def test_phys_byn_unchanged_through_pipeline():
    """Sanity: BYN stays BYN, no drift, no original_* stash."""
    utt = "за 200 тысяч белорусских рублей"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "calculate",
            "cost": 200000,
            "currency": "BYN",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    profile = _phys_partial_profile()
    apply_turn(profile, co, utt)
    assert profile.currency == "BYN"
    assert profile.cost == 200000.0
    assert profile.original_currency is None


def test_yur_cny_left_for_calc_layer():
    """For Юр лицо the calc API only accepts BYN/USD/EUR/RUB; foreign
    currencies stay un-drifted at preflight (the calc itself OOR-rejects
    if needed). Verify CNY does NOT silently drift for Юр."""
    utt = "за 500 тысяч юаней"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "calculate",
            "cost": 500000,
            "currency": "CNY",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    profile = _phys_partial_profile(
        client_type="Юридическое лицо",
        subject="Грузовой автомобиль",
    )
    apply_turn(profile, co, utt)
    # No drift for Юр — currency stays CNY (downstream calc handles).
    assert profile.currency == "CNY"
    assert profile.original_currency is None
