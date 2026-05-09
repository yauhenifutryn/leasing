"""Latency P2: change_param routing must stay deterministic.

HANDOVER (P2) reports that 5 of 6 final turns on the live transcript fell to
FireLLMFallback, costing ~1-2s of LLM streaming each. Specific case from
turn 34:

    Profile (held):  cost=20000 USD, prepaid_pct=10, term_months=48, ...
    Utterance:       "Нет, я имел в виду, что валюту договора лизинга
                      изменить на белорусский рубль, а стоимость по-прежнему
                      остаётся 20 тысяч долларов"
    Classifier:      intent=TOOL, action=change_param,
                     cost=20000, currency=BYN, prepaid_pct=10, term_months=48,
                     change_field=None, change_value=None

The user clearly wants currency BYN; the other fields are echoes of the
held profile. The dispatcher should stage a single-field
EmitChangeConfirm(currency: USD→BYN) — not fall to LLM fallback.

These tests pin the deterministic path so any drift back into LLM-fallback
shows up as a unit-test failure instead of a +1.5s latency regression in
production.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.classifier_schema import ClassifierOutput  # noqa: E402
from backend.session import ProfileState  # noqa: E402
from backend.turn_action import EmitChangeConfirm, FireLLMFallback  # noqa: E402
from backend.turn_dispatcher import apply_turn  # noqa: E402

from tests.test_apply_turn import make_complete_profile  # noqa: E402


HANDOVER_TURN_34_UTTERANCE = (
    "Нет, я имел в виду, что валюту договора лизинга изменить на "
    "белорусский рубль, а стоимость по-прежнему остаётся 20 тысяч долларов"
)


def _live_classifier(utt: str) -> ClassifierOutput:
    """Reproduce the classifier output observed on live turn 34: full
    profile echo with the new currency, no change_field/change_value
    pair."""
    return ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "change_param",
            "cost": 20000,
            "currency": "BYN",
            "prepaid_pct": 10,
            "term_months": 48,
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )


def test_handover_p2_currency_only_change_routes_to_change_confirm():
    """The headline case: post-calc, user wants only currency to change,
    classifier echoes other fields, no change_field pair. Must stage a
    single-field currency change-confirm.
    """
    profile = make_complete_profile(
        cost=20000.0,
        currency="USD",
        prepaid_pct=10.0,
        term_months=48,
    )
    profile.state = ProfileState.CONFIRMED  # post-calc

    co = _live_classifier(HANDOVER_TURN_34_UTTERANCE)
    action = apply_turn(profile, co, HANDOVER_TURN_34_UTTERANCE)

    assert isinstance(action, EmitChangeConfirm), (
        f"expected EmitChangeConfirm, got {type(action).__name__}: {action!r}"
    )
    assert "currency" in action.changes, (
        f"currency missing from staged delta: {action.changes!r}"
    )
    cur_change = action.changes["currency"]
    assert cur_change["old"] == "USD"
    assert cur_change["new"] == "BYN"
    # Echoed fields (cost/prepaid/term) match the held profile so they
    # must NOT appear as deltas.
    assert "cost" not in action.changes, (
        f"cost was echoed not changed; should not be in delta: {action.changes!r}"
    )
    assert "prepaid_pct" not in action.changes
    assert "term_months" not in action.changes


def test_handover_p2_does_not_fall_to_llm_fallback():
    """Negative-shape: explicitly assert the regression doesn't recur
    (FireLLMFallback was the production behavior the HANDOVER is asking
    us to eliminate).
    """
    profile = make_complete_profile(
        cost=20000.0,
        currency="USD",
        prepaid_pct=10.0,
        term_months=48,
    )
    profile.state = ProfileState.CONFIRMED

    co = _live_classifier(HANDOVER_TURN_34_UTTERANCE)
    action = apply_turn(profile, co, HANDOVER_TURN_34_UTTERANCE)

    assert not isinstance(action, FireLLMFallback), (
        "regression: change_param with single currency delta routed to "
        "FireLLMFallback (~+1.5s latency)"
    )


def test_handover_p2_change_param_currency_post_readback():
    """Same case but state=READBACK_PENDING (user changing mind before
    confirming the readback). Must still route to a deterministic
    change-confirm — falling to LLM fallback here would force the user
    to hear a freewheeled prompt mid-readback.
    """
    profile = make_complete_profile(
        cost=20000.0,
        currency="USD",
        prepaid_pct=10.0,
        term_months=48,
    )
    profile.state = ProfileState.READBACK_PENDING

    co = _live_classifier(HANDOVER_TURN_34_UTTERANCE)
    action = apply_turn(profile, co, HANDOVER_TURN_34_UTTERANCE)

    assert isinstance(action, EmitChangeConfirm)
    assert "currency" in action.changes


def test_handover_p2_currency_only_change_with_compact_utterance():
    """Variant: compact "поменяй валюту на белорусские рубли" with
    no echoed numeric fields. Classifier may still emit just
    currency=BYN (no change_field pair). Must route deterministically.
    """
    utt = "поменяй валюту на белорусские рубли"
    profile = make_complete_profile(
        cost=20000.0,
        currency="USD",
        prepaid_pct=10.0,
        term_months=48,
    )
    profile.state = ProfileState.CONFIRMED

    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "change_param",
            "currency": "BYN",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    action = apply_turn(profile, co, utt)

    assert isinstance(action, EmitChangeConfirm), (
        f"expected EmitChangeConfirm, got {type(action).__name__}"
    )
    assert action.changes.get("currency", {}).get("new") == "BYN"


def test_handover_p2_free_form_rag_mid_calc_still_falls_to_llm():
    """Constraint preservation (per user request): a legitimate
    free-form RAG question mid-calc must STILL route to FireLLMFallback.
    The P2 fix must not eliminate flexibility — only over-routing.

    Example: post-calc the user asks "что такое аннуитет?". No
    change-intent signal, no top-level field deltas, action != change_param.
    Expected: FireLLMFallback (so the LLM can answer the RAG question).
    """
    utt = "что такое аннуитет?"
    profile = make_complete_profile()
    profile.state = ProfileState.CONFIRMED

    co = ClassifierOutput.model_validate(
        {
            "intent": "RAG",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    action = apply_turn(profile, co, utt)

    assert isinstance(action, FireLLMFallback), (
        f"flexibility regression: free-form RAG mid-calc must keep falling "
        f"to FireLLMFallback, got {type(action).__name__}"
    )
