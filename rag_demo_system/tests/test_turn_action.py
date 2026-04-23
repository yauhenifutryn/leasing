"""Shape + immutability tests for the TurnAction ADT.

Phase 3.B of the apply_turn refactor. The 7 variants + ProfileSnapshot
are the contract between `apply_turn` (pure decision) and
`execute_action` (pure IO). Any breaking change here ripples across
both modules, so lock the shape down with explicit tests.
"""
from pathlib import Path
import sys
import dataclasses

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.turn_action import (
    ProfileSnapshot,
    EmitReadback,
    EmitClarify,
    EmitChangeConfirm,
    FireCalc,
    FireLLMFallback,
    FireOORMessage,
    Noop,
    TurnAction,
)


def _sample_snapshot() -> ProfileSnapshot:
    return ProfileSnapshot(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="USD",
        original_cost=None,
        original_currency=None,
        condition_new=1,
        age_years=None,
        prepaid_pct=20.0,
        prepaid_amount=None,
        term_months=36,
        type_schedule="0",
    )


# --- frozen immutability on every variant ---

def test_profile_snapshot_is_frozen() -> None:
    snap = _sample_snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.cost = 1.0  # type: ignore[misc]


def test_emit_readback_frozen() -> None:
    r = EmitReadback(snapshot=_sample_snapshot())
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.snapshot = None  # type: ignore[misc]


def test_emit_clarify_frozen() -> None:
    c = EmitClarify(missing=["cost"], snapshot=_sample_snapshot())
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.missing = []  # type: ignore[misc]


def test_emit_change_confirm_frozen() -> None:
    c = EmitChangeConfirm(
        changes={"term_months": {"old": 36, "new": 60}},
        snapshot=_sample_snapshot(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.changes = {}  # type: ignore[misc]


def test_fire_calc_frozen() -> None:
    f = FireCalc(snapshot=_sample_snapshot(), calc_params={"cost": 80000.0})
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.calc_params = {}  # type: ignore[misc]


def test_fire_llm_fallback_frozen() -> None:
    f = FireLLMFallback(user_utterance="привет")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.user_utterance = ""  # type: ignore[misc]


def test_fire_oor_message_frozen() -> None:
    f = FireOORMessage(message="вне диапазона")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.message = ""  # type: ignore[misc]


def test_noop_frozen() -> None:
    n = Noop(reason="nothing_to_do")
    with pytest.raises(dataclasses.FrozenInstanceError):
        n.reason = ""  # type: ignore[misc]


# --- payload shape ---

def test_emit_clarify_carries_missing_and_snapshot() -> None:
    c = EmitClarify(missing=["cost", "term_months"], snapshot=_sample_snapshot())
    assert c.missing == ["cost", "term_months"]
    assert c.snapshot.cost == 80000.0


def test_emit_change_confirm_carries_changes() -> None:
    c = EmitChangeConfirm(
        changes={"term_months": {"old": 36, "new": 60}},
        snapshot=_sample_snapshot(),
    )
    assert c.changes["term_months"]["old"] == 36
    assert c.changes["term_months"]["new"] == 60


def test_fire_calc_carries_snapshot_and_params() -> None:
    f = FireCalc(
        snapshot=_sample_snapshot(),
        calc_params={"cost": 80000.0, "currency": "USD"},
    )
    assert f.calc_params["cost"] == 80000.0
    assert f.snapshot.currency == "USD"


def test_fire_llm_fallback_defaults() -> None:
    f = FireLLMFallback(user_utterance="сколько стоит доставка?")
    assert f.user_utterance == "сколько стоит доставка?"
    assert f.rag_context is None
    assert f.snapshot is None


def test_fire_llm_fallback_with_full_payload() -> None:
    f = FireLLMFallback(
        user_utterance="расскажи про КАСКО",
        rag_context="КАСКО — добровольное...",
        snapshot=_sample_snapshot(),
    )
    assert f.rag_context.startswith("КАСКО")
    assert f.snapshot.cost == 80000.0


def test_fire_oor_message_carries_text() -> None:
    f = FireOORMessage(message="стоимость вне допустимого диапазона")
    assert "диапазона" in f.message


def test_noop_default_reason() -> None:
    assert Noop().reason == ""


def test_noop_custom_reason() -> None:
    assert Noop(reason="stale_turn").reason == "stale_turn"


# --- Union coverage ---

def test_turn_action_union_covers_all_seven_variants() -> None:
    # Every variant must be assignable to TurnAction.
    variants: tuple[TurnAction, ...] = (
        EmitReadback(snapshot=_sample_snapshot()),
        EmitClarify(missing=[], snapshot=_sample_snapshot()),
        EmitChangeConfirm(changes={}, snapshot=_sample_snapshot()),
        FireCalc(snapshot=_sample_snapshot(), calc_params={}),
        FireLLMFallback(user_utterance=""),
        FireOORMessage(message=""),
        Noop(),
    )
    for v in variants:
        assert isinstance(v, TurnAction.__args__)  # type: ignore[attr-defined]
    assert len(variants) == 7
