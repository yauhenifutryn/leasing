"""Unit + characterization suite for apply_turn.

Every E# acceptance test from spec §6 lives under the matching section
below. Characterization tests (ported from existing integration tests)
are grouped by source file with the original test name preserved as a
comment so the mapping stays visible.

Phase 3.C of the apply_turn refactor. Tests are written RED FIRST per
task; each test is wired to its dispatch-step implementation in the
same commit.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.classifier_schema import ClassifierOutput
from backend.session import ClientProfile, ProfileState
from backend.turn_action import (
    EmitReadback,
    EmitClarify,
    EmitChangeConfirm,
    FireCalc,
    FireLLMFallback,
    FireOORMessage,
    Noop,
)
from backend.turn_dispatcher import apply_turn


# ---------------------------------------------------------------- fixtures


def make_complete_profile(**overrides) -> ClientProfile:
    """Profile with every calc-required field populated (passes
    `is_complete_for_calc()`)."""
    base = dict(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="USD",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        type_schedule="0",
    )
    base.update(overrides)
    return ClientProfile(**base)


def make_partial_profile(**overrides) -> ClientProfile:
    """Profile with no fields set except those explicitly overridden."""
    return ClientProfile(**overrides)


def make_classifier(**overrides) -> ClassifierOutput:
    """ClassifierOutput with safe defaults. `intent=None` is accepted by
    the schema, but most tests pass a concrete one."""
    base = dict(intent="CONVERSATION", is_confirmation=False)
    base.update(overrides)
    return ClassifierOutput(**base)


# ---------------------------------------------------------------- smoke


def test_scaffold_returns_noop_for_empty_utterance() -> None:
    """Smoke: scaffold exists and returns a TurnAction."""
    profile = make_partial_profile()
    classifier = make_classifier()
    action = apply_turn(profile, classifier, utterance="")
    assert isinstance(action, Noop)
