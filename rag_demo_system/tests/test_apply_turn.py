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


def make_classifier(*, utterance: str = "", **overrides) -> ClassifierOutput:
    """ClassifierOutput with safe defaults, mirroring production
    construction via ``model_validate(context={"utterance": ...})``.

    Passing the same utterance here as to ``apply_turn`` is REQUIRED
    whenever test overrides set enum fields (subject, client_type,
    currency, condition_new, type_schedule, change_field) — otherwise
    the Section 2 post-validator nullifies the field at construction
    time and the test input becomes empty.
    """
    base = dict(intent="CONVERSATION", is_confirmation=False)
    base.update(overrides)
    return ClassifierOutput.model_validate(
        base, context={"utterance": utterance}
    )


# ---------------------------------------------------------------- smoke


def test_smoke_empty_profile_tool_intent_emits_clarify() -> None:
    """Smoke: empty profile + TOOL intent → EmitClarify for every
    required field (step 5b). The calc-intent gate landed 2026-04-24
    after the live regression where EmitClarify was over-firing on
    conversational turns. Keep the test intent=TOOL to pin step 5b's
    happy path."""
    profile = make_partial_profile()
    classifier = make_classifier(intent="TOOL")
    action = apply_turn(profile, classifier, utterance="хочу посчитать")
    assert isinstance(action, EmitClarify)
    # Every calc-required field is missing.
    missing = set(action.missing)
    for field in ("client_type", "subject", "cost", "currency",
                  "condition_new", "term_months", "type_schedule"):
        assert field in missing


# ---------------------------------------------------------------- E5
# E5 — Readback emits when profile first completes, regardless of
# classifier intent label. This is the structural fix for the bug
# surfaced on live call cc7fc318 where `intent=CONVERSATION` caused
# the readback gate to skip and the bot freewheeled an LLM clarify.


def test_e5_readback_emits_on_profile_complete_even_when_intent_conversation() -> None:
    profile = make_complete_profile()  # currency="USD", cost=80000.0
    profile.state = ProfileState.COLLECTING
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="Аннуитетный график")
    assert isinstance(action, EmitReadback)
    assert profile.state == ProfileState.READBACK_PENDING
    # Preflight (step 5a) converts USD→BYN before the readback emits;
    # snapshot carries the BYN cost. USD figures stash in original_*.
    assert action.snapshot.currency == "BYN"
    assert action.snapshot.original_cost == 80000.0
    assert action.snapshot.original_currency == "USD"


def test_e5_readback_emits_on_profile_complete_even_when_intent_none() -> None:
    # Belt-and-suspenders: even if classifier omits intent entirely.
    profile = make_complete_profile()
    profile.state = ProfileState.COLLECTING
    classifier = make_classifier(intent=None, is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="ну")
    assert isinstance(action, EmitReadback)


def test_e5_readback_does_not_emit_if_profile_incomplete() -> None:
    profile = make_partial_profile(cost=80000.0, currency="USD")
    profile.state = ProfileState.COLLECTING
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="ну")
    assert not isinstance(action, EmitReadback)


def test_e5_readback_does_not_emit_on_confirmation_turn() -> None:
    # is_confirmation suppresses readback re-emission.
    profile = make_complete_profile()
    profile.state = ProfileState.COLLECTING
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="да")
    assert not isinstance(action, EmitReadback)


# ---------------------------------------------------------------- E6
# E6 — change_field routes to EmitChangeConfirm, NEVER direct calc.
# Kills the live-call cc7fc318 bug where "поменяем срок на 60" fired
# calc within the same turn, with no confirmation beat.


def test_e6_change_field_routes_to_change_confirm_not_direct_calc() -> None:
    profile = make_complete_profile(term_months=36)
    profile.state = ProfileState.CONFIRMED
    utterance = "А давай всё-таки поменяем срок на 60 месяцев"
    classifier = make_classifier(
        utterance=utterance,
        intent="TOOL",
        action="change_param",
        change_field="term_months",
        change_value=60,
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance=utterance)
    assert isinstance(action, EmitChangeConfirm)
    assert action.changes == {"term_months": {"old": 36, "new": 60}}
    assert profile.state == ProfileState.CHANGE_PENDING
    assert profile.term_months == 36   # NOT mutated yet
    # Snapshot carries projected post-change term.
    assert action.snapshot.term_months == 60


def test_e6_ungrounded_change_value_drops_silently() -> None:
    # Qwen hallucinates "change_value=60" but utterance has no numeric cue
    # for term. value_grounded returns False; delta is empty; no
    # EmitChangeConfirm. Profile and state unchanged.
    profile = make_complete_profile(term_months=36)
    profile.state = ProfileState.CONFIRMED
    utterance = "ну хорошо"
    classifier = make_classifier(
        utterance=utterance,
        intent="CONVERSATION",
        change_field="term_months",
        change_value=60,
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance=utterance)
    assert not isinstance(action, EmitChangeConfirm)
    assert profile.term_months == 36
    assert profile.state == ProfileState.CONFIRMED


def test_polish_a_intent_tool_pair_accepted_without_verbatim_grounding() -> None:
    # Polish A (live call eb3d0a3d 2026-04-27): bot just explained the
    # schedule comparison ("линейный обычно дешевле, аннуитет ровнее"),
    # user replied "Давай тот, что дешевле." Classifier reasoned across
    # the prior bot turn and emitted intent=TOOL with a (change_field,
    # change_value) pair resolving to type_schedule="1". The value isn't
    # verbatim in the user utterance — the old grounding gate dropped
    # the pair and the dispatcher fell through to FireLLMFallback.
    # New behavior: intent=TOOL bypasses verbatim grounding for the
    # structurally-paired change_field/change_value signal.
    profile = make_complete_profile(type_schedule="0")
    profile.state = ProfileState.CONFIRMED
    utterance = "Давай тот, что дешевле."
    classifier = make_classifier(
        utterance=utterance,
        intent="TOOL",
        change_field="type_schedule",
        change_value="1",
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance=utterance)
    assert isinstance(action, EmitChangeConfirm)
    assert "type_schedule" in action.changes
    assert action.changes["type_schedule"]["old"] == "0"
    assert action.changes["type_schedule"]["new"] == "1"
    # No mutation yet — change goes through confirm.
    assert profile.type_schedule == "0"


def test_polish_a_intent_conversation_pair_still_requires_grounding() -> None:
    # Belt-and-suspenders: when intent != TOOL, the verbatim grounding
    # gate stays in place so phantom (change_field, change_value) pairs
    # from non-action turns still get dropped. This is the same
    # protection covered by test_e6_ungrounded_change_value_drops_silently
    # but spelled out for type_schedule to confirm the gate is selective
    # on intent rather than on change_field identity.
    profile = make_complete_profile(type_schedule="0")
    profile.state = ProfileState.CONFIRMED
    utterance = "ну хорошо"
    classifier = make_classifier(
        utterance=utterance,
        intent="CONVERSATION",
        change_field="type_schedule",
        change_value="1",
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance=utterance)
    assert not isinstance(action, EmitChangeConfirm)
    assert profile.type_schedule == "0"


def test_e6_top_level_subject_delta_also_routes_to_change_confirm() -> None:
    # Classifier fires top-level `subject` (not change_field pair) with a
    # grounded value that differs from profile.subject. This is the E7b
    # uniformity requirement: ANY captured-field delta → change-confirm,
    # whether it comes from change_field/change_value or from top-level.
    profile = make_complete_profile(
        subject="Легковой автомобиль",
        client_type="Юридическое лицо",  # avoid implied flip complication
    )
    profile.state = ProfileState.CONFIRMED
    utterance = "Хочу грузовой автомобиль"
    classifier = make_classifier(
        utterance=utterance,
        intent="CONVERSATION",
        subject="Грузовой автомобиль",
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance=utterance)
    assert isinstance(action, EmitChangeConfirm)
    assert "subject" in action.changes
    assert action.changes["subject"]["old"] == "Легковой автомобиль"
    assert action.changes["subject"]["new"] == "Грузовой автомобиль"


# ---------------------------------------------------------------- E7
# E7 — EmitReadback / EmitClarify / FireLLMFallback carry the current
# profile as a ProfileSnapshot so downstream renderers can use it as
# an anti-hallucination anchor. Kills the live-call cc7fc318 bug
# where the LLM clarify prompt re-asked for already-captured fields
# after an implicit subject pivot.


def test_e7_emit_readback_snapshot_carries_every_captured_field() -> None:
    profile = make_complete_profile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="USD",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        type_schedule="0",
    )
    profile.state = ProfileState.COLLECTING
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="ну")
    assert isinstance(action, EmitReadback)
    # Every captured field must surface on the snapshot so the
    # deterministic readback renderer has full context.
    assert action.snapshot.client_type == "Физическое лицо"
    assert action.snapshot.subject == "Легковой автомобиль"
    # Preflight (step 5a) converts USD→BYN before the readback emits;
    # snapshot carries BYN cost, original USD values stash in original_*.
    assert action.snapshot.currency == "BYN"
    assert action.snapshot.original_cost == 80000.0
    assert action.snapshot.original_currency == "USD"
    assert action.snapshot.condition_new == 1
    assert action.snapshot.prepaid_pct == 20.0
    assert action.snapshot.term_months == 36
    assert action.snapshot.type_schedule == "0"


def test_e7_snapshot_anchor_invariant_holds_for_any_snapshot_carrying_action() -> None:
    # Partial profile: whatever action fires, if it carries a snapshot
    # the snapshot must surface what was captured. Noop is acceptable.
    profile = make_partial_profile(
        subject="Грузовой автомобиль",
        cost=80000.0,
        currency="USD",
        term_months=36,
        client_type="Юридическое лицо",  # avoids implied flip
    )
    classifier = make_classifier(intent="CONVERSATION")
    action = apply_turn(profile, classifier, utterance="ну")
    assert hasattr(action, "snapshot") or isinstance(action, Noop)
    if hasattr(action, "snapshot") and action.snapshot is not None:
        assert action.snapshot.cost == 80000.0
        assert action.snapshot.currency == "USD"
        assert action.snapshot.term_months == 36


# ---------------------------------------------------------------- E7b
# E7b — Implicit cross-field flip (subject→truck forces client_type→Юр)
# emits a PAIRED EmitChangeConfirm, not two separate confirmations.
# Rule table lives in profile_state.derive_implied_flips; apply_turn
# step 4 folds the flip into the delta set uniformly.


def test_e7b_subject_flip_to_truck_only_changes_subject_not_client_type() -> None:
    """Bug M (2026-04-26): subject→Грузовой no longer auto-flips client_type
    to Юр. The change-confirm only mentions the field the user actually
    changed. The truck+физ conflict is caught at readback/preflight time
    as a FireOORMessage with two actionable branches (см. apply_turn step
    5a + _preflight_calc_policy)."""
    profile = make_complete_profile(
        subject="Легковой автомобиль",
        client_type="Физическое лицо",
    )
    profile.state = ProfileState.CONFIRMED
    utterance = "Да, я всё-таки хочу грузовой автомобиль"
    classifier = make_classifier(
        utterance=utterance,
        intent="CONVERSATION",
        subject="Грузовой автомобиль",
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance=utterance)
    assert isinstance(action, EmitChangeConfirm)
    assert "subject" in action.changes
    assert "client_type" not in action.changes  # no auto-flip
    assert action.changes["subject"]["new"] == "Грузовой автомобиль"
    assert profile.state == ProfileState.CHANGE_PENDING
    # Profile fields NOT mutated yet — confirmation gates the change.
    assert profile.subject == "Легковой автомобиль"
    assert profile.client_type == "Физическое лицо"


def test_e7b_no_implied_flip_when_client_type_already_yur() -> None:
    # Same subject flip but client_type already Юр → only subject is in
    # delta (no client_type entry). Verifies the implied-flip guard.
    profile = make_complete_profile(
        subject="Легковой автомобиль",
        client_type="Юридическое лицо",
    )
    profile.state = ProfileState.CONFIRMED
    utterance = "переключи на грузовой"
    classifier = make_classifier(
        utterance=utterance,
        intent="CONVERSATION",
        subject="Грузовой автомобиль",
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance=utterance)
    assert isinstance(action, EmitChangeConfirm)
    assert "subject" in action.changes
    assert "client_type" not in action.changes


# ---------------------------------------------------------------- E8a
# E8a — apply_turn emits FireCalc with USD-aware snapshot and correctly
# shaped calc_params. The post-calc narration itself is rendered by
# execute_action (E8b integration test lands in Phase 3.D) using
# profile_prompts.render_calc_result(result), bypassing the LLM.


def test_e8a_fire_calc_params_preserve_original_usd_fields() -> None:
    profile = make_complete_profile(
        cost=240000.0, currency="BYN",
        original_cost=80000.0, original_currency="USD",
    )
    # Canonical readback→confirm flow: bot just spoke the readback, user
    # confirmed. Step 2 transitions READBACK_PENDING→CONFIRMED in this
    # turn, step 6 fires calc. Issue 4 (live 2ab41112) gate requires
    # the pre-turn state to be a legitimate pre-calc state.
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(intent="TOOL", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="Да, рассчитывай")
    assert isinstance(action, FireCalc)
    # Snapshot carries both the applied BYN cost AND the original USD
    # fields so render_calc_result can emit the dual-disclosure prefix.
    assert action.snapshot.cost == 240000.0
    assert action.snapshot.currency == "BYN"
    assert action.snapshot.original_cost == 80000.0
    assert action.snapshot.original_currency == "USD"
    # calc_params is what the calculator API consumes (build_calc_params).
    assert action.calc_params["cost"] == 240000.0
    assert action.calc_params["currency"] == "BYN"
    assert action.calc_params["client_type"] == "Физическое лицо"
    assert action.calc_params["term"] == 36


def test_e8a_fire_calc_does_not_emit_without_confirmation() -> None:
    # CONFIRMED state alone is not enough — is_confirmation gates FireCalc.
    profile = make_complete_profile()
    profile.state = ProfileState.CONFIRMED
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="расскажи про страховку")
    assert not isinstance(action, FireCalc)


def test_e8a_fire_calc_does_not_emit_when_profile_incomplete() -> None:
    profile = make_partial_profile(cost=80000.0, currency="USD")
    profile.state = ProfileState.CONFIRMED
    classifier = make_classifier(intent="TOOL", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="рассчитывай")
    assert not isinstance(action, FireCalc)


# ---------------------------------------------------------------- Step 1
# Step 1 — CHANGE_PENDING + is_confirmation → apply the staged change,
# transition to CONFIRMED, and RE-DISPATCH (the now-complete + confirmed
# state may unlock step 6 FireCalc on the second loop iteration).


def test_change_pending_confirm_applies_change_and_fires_calc() -> None:
    profile = make_complete_profile(term_months=36)
    profile.state = ProfileState.CHANGE_PENDING
    profile.pending_change = {"changes": {"term_months": {"old": 36, "new": 60}}}
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="Да, верно")
    # Re-dispatch lands in step 6 FireCalc because profile is complete
    # + CONFIRMED + is_confirmation.
    assert isinstance(action, FireCalc)
    assert profile.term_months == 60
    assert profile.state == ProfileState.CONFIRMED
    assert profile.pending_change is None


def test_change_pending_confirm_applies_multi_field_changes() -> None:
    profile = make_complete_profile(
        subject="Легковой автомобиль",
        client_type="Физическое лицо",
    )
    profile.state = ProfileState.CHANGE_PENDING
    profile.pending_change = {
        "changes": {
            "subject": {"old": "Легковой автомобиль", "new": "Грузовой автомобиль"},
            "client_type": {"old": "Физическое лицо", "new": "Юридическое лицо"},
        }
    }
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="да")
    assert profile.subject == "Грузовой автомобиль"
    assert profile.client_type == "Юридическое лицо"
    assert profile.state == ProfileState.CONFIRMED
    assert profile.pending_change is None
    # Calc fires on the re-dispatch pass (profile complete + CONFIRMED).
    assert isinstance(action, FireCalc)


def test_change_pending_deny_does_not_apply_and_stays_in_change_pending() -> None:
    # is_confirmation=False on CHANGE_PENDING means the user rejected
    # the change; staged data stays, profile stays in CHANGE_PENDING
    # for the next turn's clarification.
    profile = make_complete_profile(term_months=36)
    profile.state = ProfileState.CHANGE_PENDING
    profile.pending_change = {"changes": {"term_months": {"old": 36, "new": 60}}}
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="нет, подожди")
    assert profile.term_months == 36
    assert profile.state == ProfileState.CHANGE_PENDING
    assert profile.pending_change is not None


# ---------------------------------------------------------------- Bug 1
# Bug 1 (live call 6ca0eaca, 2026-04-25): when in CHANGE_PENDING the
# classifier sometimes re-emits the same change_field/change_value pair
# on the next turn (Qwen seeing prior context bleeding through). Without
# a loop guard, step 4 keeps re-staging the same EmitChangeConfirm and
# the bot asks the same confirmation question forever.


def test_change_pending_re_emit_same_delta_does_not_loop() -> None:
    """Identical re-emit should NOT produce another EmitChangeConfirm."""
    profile = make_complete_profile(term_months=36)
    profile.state = ProfileState.CHANGE_PENDING
    profile.pending_change = {"changes": {"term_months": {"old": 36, "new": 48}}}
    classifier = make_classifier(
        utterance="48 месяцев",
        intent="CONVERSATION",
        is_confirmation=False,
        change_field="term_months",
        change_value=48,
    )
    action = apply_turn(profile, classifier, utterance="48 месяцев")
    # The loop bug: action would be EmitChangeConfirm with the same delta.
    # Fix: drop already-staged identical deltas → step 4 doesn't fire.
    assert not isinstance(action, EmitChangeConfirm)
    # Staged change is preserved — user can still confirm it on next turn.
    assert profile.state == ProfileState.CHANGE_PENDING
    assert profile.pending_change == {
        "changes": {"term_months": {"old": 36, "new": 48}}
    }


def test_change_pending_re_emit_different_value_re_stages() -> None:
    """A NEW value while in CHANGE_PENDING should re-stage as a fresh
    EmitChangeConfirm — only identical delta is dropped, not a genuine
    correction ("нет, 60")."""
    profile = make_complete_profile(term_months=36)
    profile.state = ProfileState.CHANGE_PENDING
    profile.pending_change = {"changes": {"term_months": {"old": 36, "new": 48}}}
    classifier = make_classifier(
        utterance="нет, 60 месяцев",
        intent="CONVERSATION",
        is_confirmation=False,
        change_field="term_months",
        change_value=60,
    )
    action = apply_turn(profile, classifier, utterance="нет, 60 месяцев")
    assert isinstance(action, EmitChangeConfirm)
    assert profile.pending_change["changes"]["term_months"]["new"] == 60


# ---------------------------------------------------------------- Step 2
# Step 2 — READBACK_PENDING + is_confirmation → CONFIRMED, and calc
# fires immediately on the same call (step 6 picks it up after the
# state transition).


def test_readback_pending_confirm_transitions_to_confirmed_and_fires_calc() -> None:
    profile = make_complete_profile()
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="Верно, да")
    assert isinstance(action, FireCalc)
    assert profile.state == ProfileState.CONFIRMED


def test_readback_pending_deny_does_not_transition_nor_fire_calc() -> None:
    profile = make_complete_profile()
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="нет, неправильно")
    # Without a grounded correction (step 3, Task 13.3), a plain deny
    # stays in READBACK_PENDING.
    assert profile.state == ProfileState.READBACK_PENDING
    assert not isinstance(action, FireCalc)


# ---------------------------------------------------------------- Step 3
# Step 3 — READBACK_PENDING + is_confirmation=False + grounded correction
# → stage correction via EmitChangeConfirm (re-dispatched step 4).
# Ports behavior from test_readback_deny_grounding.py.


def test_readback_deny_with_grounded_correction_stages_change_confirm() -> None:
    profile = make_complete_profile(cost=80000.0, currency="USD")
    profile.state = ProfileState.READBACK_PENDING
    utterance = "нет, 150 тысяч долларов"
    classifier = make_classifier(
        utterance=utterance,
        intent="CONVERSATION",
        cost=150000.0,
        currency="USD",
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance=utterance)
    # Correction is staged as EmitChangeConfirm; profile.cost not yet
    # mutated. State moved to CHANGE_PENDING for the next turn's
    # confirmation pass.
    assert isinstance(action, EmitChangeConfirm)
    assert "cost" in action.changes
    assert action.changes["cost"]["new"] == 150000.0
    assert profile.cost == 80000.0   # NOT mutated yet
    assert profile.state == ProfileState.CHANGE_PENDING


def test_readback_deny_with_ungrounded_correction_stays_in_readback_pending() -> None:
    # Codex E-Codex-2 guard: plain "нет" + classifier-hallucinated
    # cost=150000 must NOT stage a change. value_grounded drops the
    # ungrounded delta; state stays in READBACK_PENDING.
    profile = make_complete_profile(cost=80000.0, currency="USD")
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(
        utterance="нет",
        intent="CONVERSATION",
        cost=150000.0,     # hallucinated; utterance has no numeric cue
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance="нет")
    assert not isinstance(action, EmitChangeConfirm)
    assert profile.state == ProfileState.READBACK_PENDING
    assert profile.cost == 80000.0


# ---------------------------------------------------------------- Step 5b
# Step 5b — patches applied but profile still incomplete → EmitClarify.
# Carries the current snapshot as E7 anti-hallucination anchor AND the
# sorted list of missing fields so the renderer can ask for exactly the
# right slots in the right order.


def test_step_5b_emits_clarify_with_missing_and_snapshot() -> None:
    profile = make_partial_profile(cost=80000.0, currency="USD")
    utterance = "36 месяцев"
    classifier = make_classifier(
        utterance=utterance,
        intent="TOOL",  # calc-intent gate (2026-04-24): step 5b requires it
        term_months=36,
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance=utterance)
    assert isinstance(action, EmitClarify)
    # term_months WAS applied (first-time capture); snapshot reflects it.
    assert action.snapshot.term_months == 36
    assert action.snapshot.cost == 80000.0
    # Still-missing fields show up in .missing (client_type, subject,
    # condition_new, type_schedule, prepaid — order: sorted).
    missing = set(action.missing)
    assert "client_type" in missing
    assert "subject" in missing
    assert "condition_new" in missing
    assert "type_schedule" in missing
    assert "prepaid" in missing


def test_step_5b_no_patches_applied_still_emits_clarify_when_incomplete() -> None:
    # When classifier emits nothing grounded AND profile is incomplete,
    # step 5b still fires on TOOL intent so the caller knows what to
    # ask for. Calc-intent gate (2026-04-24) requires intent=TOOL.
    profile = make_partial_profile(cost=80000.0)
    classifier = make_classifier(intent="TOOL", is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="ну")
    assert isinstance(action, EmitClarify)
    assert action.snapshot.cost == 80000.0
    assert "subject" in action.missing


# ---------------------------------------------------------------- Step 7
# Step 7 — classifier flags invalid_param (cost bounds violated,
# unsupported currency, etc.) → FireOORMessage with deterministic text.
# Fires BEFORE step 8 (LLM fallback) to avoid wasting an LLM call on
# a structurally-rejected input.


def test_step_7_invalid_param_routes_to_fire_oor_message() -> None:
    profile = make_partial_profile()
    classifier = make_classifier(
        intent="CONVERSATION",
        action="invalid_param",
        is_confirmation=False,
    )
    action = apply_turn(profile, classifier, utterance="миллиард рублей")
    assert isinstance(action, FireOORMessage)
    assert len(action.message) > 0


# ---------------------------------------------------------------- Step 8
# Step 8 — catch-all → FireLLMFallback. Fires when no structural
# dispatch matched and the user's utterance is a freeform question.
# Snapshot carries any captured values as anti-hallucination anchor (E7).


def test_step_8_fires_llm_fallback_for_freeform_question_from_confirmed() -> None:
    # Profile is CONFIRMED (passed through readback earlier); user now
    # asks a freeform question. No structural branch matches; step 8
    # catches it.
    profile = make_complete_profile()
    profile.state = ProfileState.CONFIRMED
    classifier = make_classifier(intent="RAG", is_confirmation=False)
    action = apply_turn(
        profile, classifier,
        utterance="Какие документы нужны для оформления?",
    )
    assert isinstance(action, FireLLMFallback)
    assert action.user_utterance == "Какие документы нужны для оформления?"
    # Snapshot carries the confirmed profile as anchor.
    assert action.snapshot is not None
    assert action.snapshot.cost == 80000.0


def test_step_8_snapshot_is_none_when_profile_is_empty() -> None:
    # Profile has no captures (but in a non-COLLECTING terminal state).
    profile = make_partial_profile()
    profile.state = ProfileState.CONFIRMED  # edge case — forced
    classifier = make_classifier(intent="RAG", is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="что такое КАСКО?")
    assert isinstance(action, FireLLMFallback)
    assert action.snapshot is None


def test_step_8_readback_pending_deny_no_correction_falls_to_llm_fallback() -> None:
    # Plain "нет" on readback with no grounded correction: step 2 skipped
    # (!is_confirmation), pre-compute yields no delta, no other step
    # matches — FireLLMFallback handles the denial naturally (LLM will
    # ask "что хотите изменить?").
    profile = make_complete_profile()
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="нет, неправильно")
    assert isinstance(action, FireLLMFallback)
    # Profile state NOT changed by FireLLMFallback.
    assert profile.state == ProfileState.READBACK_PENDING


# ---------------------------------------------------------------- Coverage
# Legacy `pending_change` single-field payload shape {'field':..,'new_value':..}
# — predates Fix 28's multi-field {'changes': {...}}. Still supported on
# the apply-path so old payloads persisted in-flight survive the refactor.


def test_change_pending_legacy_single_field_shape_applies_on_confirm() -> None:
    profile = make_complete_profile(term_months=36)
    profile.state = ProfileState.CHANGE_PENDING
    profile.pending_change = {"field": "term_months", "new_value": 48}
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="да")
    assert profile.term_months == 48
    assert profile.state == ProfileState.CONFIRMED
    assert isinstance(action, FireCalc)


# ---------------------------------------------------------------- calc-intent gate (step 5b)
# 2026-04-24 live regression: session 5e2a8f73 on 38.80.122.90 showed
# apply_turn returning EmitClarify on EVERY conversational turn (name
# capture, push-back, small talk) because step 5b fired whenever the
# profile was incomplete + COLLECTING, regardless of intent. Legacy
# had an implicit `if needs_tool:` wrapper around the whole gate block.
# These tests pin the fix: step 5b requires TOOL intent (or a calc-path
# action); anything else falls through to FireLLMFallback.


def test_name_capture_on_empty_profile_routes_to_llm_fallback() -> None:
    """User says 'Меня зовут Евгений' on a fresh session. Classifier
    emits intent=CONVERSATION with name='Евгений'. apply_turn MUST NOT
    emit the missing-fields clarify — the user hasn't asked to calculate
    anything. Fall through to FireLLMFallback so the LLM improvises a
    natural greeting."""
    profile = make_partial_profile()
    classifier = make_classifier(intent="CONVERSATION", name="Евгений")
    action = apply_turn(profile, classifier, utterance="Меня зовут Евгений")
    assert isinstance(action, FireLLMFallback)


def test_conversation_pushback_on_empty_profile_routes_to_llm_fallback() -> None:
    """User pushes back on being rushed ('Подожди, я не просил ещё
    этого'). apply_turn must NOT loop on the same clarify question —
    route to LLM so the bot can respond naturally."""
    profile = make_partial_profile()
    classifier = make_classifier(intent="CONVERSATION")
    action = apply_turn(profile, classifier, utterance="Подожди, я не просил ещё этого")
    assert isinstance(action, FireLLMFallback)


def test_small_talk_on_empty_profile_routes_to_llm_fallback() -> None:
    """Anything non-calc on an empty profile in COLLECTING ends up in
    FireLLMFallback. Covers 'алло', 'да о чём вообще', etc."""
    profile = make_partial_profile()
    classifier = make_classifier(intent="CONVERSATION")
    for utterance in ("Алло.", "Да о чём вообще?", "Ну и?"):
        action = apply_turn(profile, classifier, utterance=utterance)
        assert isinstance(action, FireLLMFallback), f"utterance={utterance!r}"


def test_rag_intent_on_empty_profile_routes_to_llm_fallback() -> None:
    """Info question ('где вы находитесь?') — classifier says RAG. Still
    not calc intent, so step 5b must NOT fire."""
    profile = make_partial_profile()
    classifier = make_classifier(intent="RAG")
    action = apply_turn(profile, classifier, utterance="Где вы находитесь?")
    assert isinstance(action, FireLLMFallback)


def test_calc_action_without_tool_intent_still_fires_clarify() -> None:
    """Defensive: some classifier paths emit intent=CONVERSATION but
    action='calculate' (observed in legacy). Calc-intent gate accepts
    either signal so the calc flow isn't stuck behind a label mismatch."""
    profile = make_partial_profile()
    classifier = make_classifier(intent="CONVERSATION", action="calculate")
    action = apply_turn(profile, classifier, utterance="хочу посчитать")
    assert isinstance(action, EmitClarify)


def test_clarify_client_type_action_routes_to_emit_clarify() -> None:
    """Classifier's 'clarify_client_type' action (foreign subject /
    unsupported currency) is part of the calc flow — step 5b should
    still fire so the user gets asked for the missing identity field."""
    profile = make_partial_profile()
    classifier = make_classifier(
        intent="CONVERSATION", action="clarify_client_type",
    )
    action = apply_turn(profile, classifier, utterance="")
    assert isinstance(action, EmitClarify)


def test_recalculate_action_on_partial_profile_fires_clarify() -> None:
    """User says 'пересчитай' with missing fields — classifier emits
    action=recalculate. Calc-intent gate treats this as calc intent."""
    profile = make_partial_profile(cost=80000.0, currency="BYN")
    classifier = make_classifier(intent="TOOL", action="recalculate")
    action = apply_turn(profile, classifier, utterance="пересчитай")
    assert isinstance(action, EmitClarify)


# ---------------------------------------------------------------- Bug 1: step 6 preflight
# 2026-04-24 live regression session ac0e35d6: FireCalc fired with raw
# USD cost, calc API rejected it, `render_calc_result` printed "?"
# placeholders. apply_turn step 6 now applies the legacy DirectTool
# preprocessing: currency policy (EUR/RUB reject), subject restriction
# (non-individual subject reject), USD→BYN conversion before FireCalc.


def test_step6_usd_profile_converts_to_byn_and_stashes_original() -> None:
    """Fresh capture: user said 80k USD, all other fields captured,
    state=CONFIRMED + confirmation. Step 6 converts USD→BYN in-place
    and stashes the USD figures on the profile for the disclosure
    prefix. FireCalc carries the converted cost in both snapshot and
    calc_params."""
    profile = make_complete_profile(cost=80000.0, currency="USD")
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(intent="TOOL", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="Да")
    assert isinstance(action, FireCalc)
    # 80k USD × 3.0 = 240k BYN at the default MVP rate.
    assert action.snapshot.cost == 240000.0
    assert action.snapshot.currency == "BYN"
    assert action.snapshot.original_cost == 80000.0
    assert action.snapshot.original_currency == "USD"
    # calc_params ship BYN to the calculator API.
    assert action.calc_params["cost"] == 240000.0
    assert action.calc_params["currency"] == "BYN"


def test_step6_eur_for_physical_person_rejected_as_oor() -> None:
    """Физ лицо + EUR → FireOORMessage. Calculator API currently
    supports BYN/USD only for individuals."""
    profile = make_complete_profile(cost=80000.0, currency="EUR")
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(intent="TOOL", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="Да")
    assert isinstance(action, FireOORMessage)
    assert "EUR" in action.message or "евро" in action.message.lower() or "валюта" in action.message.lower()


def test_step6_rub_for_physical_person_rejected_as_oor() -> None:
    """Физ лицо + RUB rejected (Belarusian бытие plus MVP scope)."""
    profile = make_complete_profile(cost=5000000.0, currency="RUB")
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(intent="TOOL", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="Да")
    assert isinstance(action, FireOORMessage)


def test_step6_commercial_subject_restriction_is_safety_net() -> None:
    """Физ лицо + spec-tech: the normal flow never reaches step 6 with
    this combination because E7b (derive_implied_flips) flips client_type
    to Юр in step 4. But _preflight_calc_policy still carries the legacy
    safety-net check (app.py:2386-2395) in case a migrated / manually
    staged profile bypasses E7b. Exercise the helper directly."""
    from backend.turn_dispatcher import _preflight_calc_policy
    profile = make_complete_profile(
        subject="Спецтехника", cost=200000.0, currency="BYN",
    )
    action = _preflight_calc_policy(profile)
    assert isinstance(action, FireOORMessage)
    assert "физических" in action.message.lower()


def test_step6_legal_person_usd_does_not_convert() -> None:
    """Юр лицо + USD: conversion does NOT fire — ЮЛ calculator supports
    USD natively. No original_* stash."""
    profile = make_complete_profile(
        client_type="Юридическое лицо", cost=80000.0, currency="USD",
    )
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(intent="TOOL", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="Да")
    assert isinstance(action, FireCalc)
    assert action.snapshot.cost == 80000.0
    assert action.snapshot.currency == "USD"
    assert action.snapshot.original_cost is None
    assert action.snapshot.original_currency is None


def test_step6_byn_native_profile_no_conversion_no_stash() -> None:
    """Physical person who quotes in BYN from the start: no conversion,
    no disclosure."""
    profile = make_complete_profile(cost=240000.0, currency="BYN")
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(intent="TOOL", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="Да")
    assert isinstance(action, FireCalc)
    assert action.snapshot.original_cost is None
    assert action.snapshot.original_currency is None


def test_step6_re_calc_preserves_usd_disclosure_on_same_params() -> None:
    """After a USD→BYN conversion on turn N, turn N+1 re-confirms with
    the same params (user said 'да'). Preflight preserves the stashed
    original_* so the disclosure prefix still narrates. Live ac0e35d6
    turn 12 depends on this — user 'да' after change-confirm still
    gets the 'Стоимость 80000 долларов...' prefix."""
    profile = make_complete_profile(
        cost=240000.0, currency="BYN",
        original_cost=80000.0, original_currency="USD",
    )
    profile.state = ProfileState.READBACK_PENDING
    classifier = make_classifier(intent="TOOL", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="Да")
    assert isinstance(action, FireCalc)
    assert action.snapshot.original_cost == 80000.0
    assert action.snapshot.original_currency == "USD"


def test_step1_currency_change_from_usd_to_byn_clears_original_stash() -> None:
    """User switched USD→BYN mid-session via a change-confirm. Step 1
    apply-patches must clear the stale USD disclosure or the next
    calc narrates "Стоимость N долларов..." even though the user
    explicitly moved off USD."""
    profile = make_complete_profile(
        cost=240000.0, currency="BYN",
        original_cost=80000.0, original_currency="USD",
    )
    # Stage a user-initiated currency change from the prior USD capture.
    profile.state = ProfileState.CHANGE_PENDING
    profile.pending_change = {"changes": {
        "currency": {"old": "USD", "new": "BYN"},
    }}
    classifier = make_classifier(intent="TOOL", is_confirmation=True)
    action = apply_turn(profile, classifier, utterance="Да")
    # After applying, profile.original_* should be cleared.
    assert profile.original_cost is None
    assert profile.original_currency is None
    assert isinstance(action, FireCalc)
    assert action.snapshot.original_cost is None


# ---------------------------------------------------------------- Bug 3: name capture + snapshot
# 2026-04-24 live regression ac0e35d6 turn 14: STT garbled user's
# follow-up as "Боянс патиба", classifier emitted name="Боянс",
# profile correctly rejected stale name (first-time-only), but the
# FireLLMFallback handler's anchor didn't include the captured name
# so the LLM greeted "Здравствуйте, Боянс!" on top of the profile
# already holding name="Евгений".


def test_apply_turn_captures_name_first_time() -> None:
    """Fresh session: user said 'Меня зовут Евгений'. apply_turn mirrors
    legacy app.py:1414-1416 — accept only when profile.name is empty."""
    profile = make_partial_profile()
    classifier = make_classifier(intent="CONVERSATION", name="Евгений")
    apply_turn(profile, classifier, utterance="Меня зовут Евгений")
    assert profile.name == "Евгений"


def test_apply_turn_ignores_stale_name_on_later_turn() -> None:
    """Profile already has name="Евгений"; classifier re-emits
    name="Боянс" on a garbled STT turn. apply_turn must NOT overwrite —
    first-time-only semantics."""
    profile = make_partial_profile()
    profile.name = "Евгений"
    classifier = make_classifier(intent="CONVERSATION", name="Боянс")
    apply_turn(profile, classifier, utterance="Боянс патиба")
    assert profile.name == "Евгений"


def test_fire_llm_fallback_snapshot_carries_name() -> None:
    """FireLLMFallback.snapshot must include profile.name so the LLM's
    anti-hallucination anchor tells it 'the client is Евгений' and it
    doesn't improvise a different greeting."""
    profile = make_partial_profile()
    profile.name = "Евгений"
    classifier = make_classifier(intent="CONVERSATION")
    action = apply_turn(profile, classifier, utterance="Привет ещё раз")
    assert isinstance(action, FireLLMFallback)
    assert action.snapshot is not None
    assert action.snapshot.name == "Евгений"


def test_snapshot_anchor_lines_include_name_field() -> None:
    """Anchor-line renderer must include `name` as the first prompt
    line so the LLM treats it as captured identity."""
    from backend.turn_dispatcher import _snapshot_anchor_lines
    from backend.turn_action import ProfileSnapshot
    snap = ProfileSnapshot(
        client_type=None, subject=None, cost=None, currency=None,
        original_cost=None, original_currency=None,
        condition_new=None, age_years=None,
        prepaid_pct=None, prepaid_amount=None,
        term_months=None, type_schedule=None,
        name="Евгений",
    )
    lines = _snapshot_anchor_lines(snap)
    assert any("name: Евгений" in line for line in lines)


# ---------------------------------------------------------------- Codex adversarial high #1
# Two invariants that the old raw-setattr step 1 loop violated:
#   1. prepaid_pct/prepaid_amount slot invariant: confirming a change to
#      prepaid_amount must null the stale prepaid_pct (and vice versa).
#   2. locked_fields guard: a confirmed change to a locked field must be
#      silently rejected (field stays at its current value).
# Both are now delegated to ClientProfile.apply_pending_change().


def test_step1_apply_pending_change_clears_prepaid_sibling():
    # Regression for Codex high #1: a change-confirm that flips prepaid_pct
    # to prepaid_amount must null the stale prepaid_pct after apply.
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=30000.0,
        currency="BYN",
        condition_new=1,
        term_months=24,
        prepaid_pct=20.0,
        type_schedule="0",
        state=ProfileState.CHANGE_PENDING,
        pending_change={"changes": {"prepaid_amount": {"old": None, "new": 5000.0}}},
    )
    co = ClassifierOutput.model_validate(
        {"intent": "CONVERSATION", "is_confirmation": True},
        context={"utterance": "да"},
    )

    apply_turn(p, co, "да", turn_id=1)

    assert p.prepaid_amount == 5000.0
    assert p.prepaid_pct is None, "Codex high #1: sibling not cleared"
    assert p.state == ProfileState.CONFIRMED


def test_step1_apply_pending_change_respects_locked_fields():
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.turn_action import EmitChangeConfirm
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=10000.0,
        currency="BYN",
        condition_new=1,
        term_months=24,
        prepaid_pct=20.0,
        type_schedule="0",
        state=ProfileState.CHANGE_PENDING,
        pending_change={"changes": {"cost": {"old": 10000.0, "new": 99999.0}}},
        locked_fields={"cost"},
    )
    co = ClassifierOutput.model_validate(
        {"intent": "CONVERSATION", "is_confirmation": True},
        context={"utterance": "да"},
    )

    action = apply_turn(p, co, "да", turn_id=1)

    # Codex CP-3.6 high #1: when apply_pending_change returns False (locked-
    # only payload), apply_turn must re-emit EmitChangeConfirm — NOT advance
    # to CONFIRMED with stale data and NOT fire the calculator.
    assert p.cost == 10000.0, "locked field was mutated"
    assert isinstance(action, EmitChangeConfirm), (
        f"locked-only pending_change must re-prompt change-confirm, got {type(action).__name__}"
    )
    assert p.state == ProfileState.CHANGE_PENDING, (
        f"profile must stay CHANGE_PENDING on failed apply, got {p.state}"
    )
    assert p.pending_change is not None, (
        "pending_change must be preserved for retry on locked-only payload"
    )


def test_step1_apply_pending_change_rejects_all_unknown_fields():
    """Codex CP-3.6 high #1: pending_change containing only unknown field
    names (e.g. 'prepaid' instead of 'prepaid_pct') must NOT advance the
    profile to CONFIRMED or fire the calculator. apply_pending_change()
    returns False in this case and preserves pending_change for retry.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.turn_action import EmitChangeConfirm
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=10000.0,
        currency="BYN",
        condition_new=1,
        term_months=24,
        prepaid_pct=20.0,
        type_schedule="0",
        state=ProfileState.CHANGE_PENDING,
        # 'prepaid' is not a real ClientProfile attribute — only
        # prepaid_pct / prepaid_amount are. Simulates classifier drift
        # that staged a malformed payload via a prior CHANGE_PENDING turn.
        pending_change={"changes": {"prepaid": {"old": 20.0, "new": 30.0}}},
    )
    co = ClassifierOutput.model_validate(
        {"intent": "CONVERSATION", "is_confirmation": True},
        context={"utterance": "да"},
    )

    action = apply_turn(p, co, "да", turn_id=1)

    assert p.prepaid_pct == 20.0, "unrelated profile field must not change"
    assert isinstance(action, EmitChangeConfirm), (
        f"unknown-field pending_change must re-prompt change-confirm, got {type(action).__name__}"
    )
    assert p.state == ProfileState.CHANGE_PENDING
    assert p.pending_change is not None


def test_step5_first_time_prepaid_amount_clears_pct_sibling():
    # Regression for Codex high #1: first-time capture of prepaid_amount
    # when prepaid_pct was already set must null the pct, matching
    # Fix 42d slot-invariant semantics.
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=10000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        state=ProfileState.COLLECTING,
    )
    co = ClassifierOutput.model_validate(
        {"intent": "TOOL", "prepaid_amount": 5000.0, "action": "calculate"},
        context={"utterance": "аванс 5000 рублей"},
    )

    apply_turn(p, co, "аванс 5000 рублей", turn_id=1)

    assert p.prepaid_amount == 5000.0
    assert p.prepaid_pct is None, "Codex high #1: sibling pct survived first-time amount capture"


def test_step5_first_time_respects_locked_fields():
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        # cost is None (first-time path)
        state=ProfileState.COLLECTING,
        locked_fields={"cost"},
    )
    co = ClassifierOutput.model_validate(
        {"intent": "TOOL", "cost": 99999.0, "action": "calculate"},
        context={"utterance": "стоимость 99999 рублей"},
    )

    apply_turn(p, co, "стоимость 99999 рублей", turn_id=1)

    assert p.cost is None, "locked field was set in step 5"


def test_step5a_rub_for_phys_emits_oor_not_readback():
    # Regression for live call f7e5aa1d (2026-04-24): classifier emitted
    # currency=RUB while profile was COLLECTING; apply_turn step 5a
    # transitioned to READBACK_PENDING and EmitReadback spoke
    # "стоимость 10000 RUB" as confirmed parameters. Preflight must fire
    # BEFORE the readback so RUB → FireOORMessage immediately.
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.turn_action import FireOORMessage
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=10000.0,
        currency="RUB",
        condition_new=1,
        term_months=24,
        prepaid_pct=20.0,
        type_schedule="0",
        state=ProfileState.COLLECTING,
    )
    co = ClassifierOutput.model_validate(
        {"intent": "TOOL", "action": "calculate"},
        context={"utterance": "аннуитетный"},
    )

    action = apply_turn(p, co, "аннуитетный", turn_id=1)

    assert isinstance(action, FireOORMessage)
    assert p.state != ProfileState.READBACK_PENDING, "RUB reached readback"


def test_preflight_commercial_subject_for_phys_emits_oor():
    # Unit test for _preflight_calc_policy subject restriction: Физ лицо +
    # Грузовой автомобиль must return FireOORMessage.
    # Note: in the apply_turn dispatch flow, derive_implied_flips intercepts
    # this combination BEFORE step 5a (it proposes client_type →
    # Юридическое лицо as a delta, firing EmitChangeConfirm at step 4).
    # This test exercises the backstop policy directly so the guard is
    # verified independently of the dispatch routing.
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import _preflight_calc_policy
    from backend.turn_action import FireOORMessage

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Грузовой автомобиль",
        cost=10000.0,
        currency="BYN",
        condition_new=1,
        term_months=24,
        prepaid_pct=20.0,
        type_schedule="0",
        state=ProfileState.COLLECTING,
    )

    action = _preflight_calc_policy(p)

    assert isinstance(action, FireOORMessage)


def test_multi_field_change_surfaces_when_companion_grounding_drops():
    # Regression for live call f7e5aa1d turn 11: user said "для юрлица
    # коммерческие автомобили" — classifier emitted client_type=Юр +
    # subject=Грузовой, but subject grounding dropped "коммерческие".
    # Only client_type delta staged silently, calc then ran with
    # Легковой + Юр — wrong outcome. System must instead emit a clarify
    # that asks which subject category the user wants.
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.turn_action import EmitClarify
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=30000.0,
        currency="BYN",
        condition_new=1,
        term_months=24,
        prepaid_pct=20.0,
        type_schedule="0",
        state=ProfileState.CONFIRMED,
    )
    co = ClassifierOutput.model_validate(
        {
            "intent": "CONVERSATION",
            "client_type": "Юридическое лицо",
            "action": "clarify",
        },
        context={"utterance": "для юрлица коммерческие автомобили"},
    )

    action = apply_turn(p, co, "для юрлица коммерческие автомобили", turn_id=1)

    # Must be clarify, NOT a change-confirm that silently drops subject.
    assert isinstance(action, EmitClarify)
    assert "subject" in action.missing


# ---------------------------------------------------------------- Bugs Q + S
# Year-form disambiguation: "X лет/года" carries either age-of-vehicle
# OR term, depending on conversation state. The semantic rule lives in
# turn_dispatcher._dispatch_once (state-aware), not in stateless regex.


def test_year_form_in_age_phase_grounds_age_not_term() -> None:
    """Bug Q (live call 730d3aab): condition_new=0, age_years=None.
    User says 'Два года' answering 'Сколько лет вашему транспорту?'.
    Classifier silent → utterance fallback grounds age_years=2; profile
    must NOT carry term_months=24."""
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="BYN",
        condition_new=0,
        age_years=None,
        state=ProfileState.COLLECTING,
    )
    co = ClassifierOutput.model_validate(
        {"intent": "TOOL", "is_confirmation": False},
        context={"utterance": "два года"},
    )
    apply_turn(p, co, "два года", turn_id=1)
    assert p.age_years == 2
    assert p.term_months is None


def test_year_form_age_phase_drops_misattributed_term() -> None:
    """Bug Q hardening: even when classifier mis-emits term_months=24
    from 'Два года' while we're in age-collection state, dispatcher
    drops it (year-form belongs to age in this phase)."""
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="BYN",
        condition_new=0,
        age_years=None,
        state=ProfileState.COLLECTING,
    )
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "is_confirmation": False,
            "term_months": 24,
        },
        context={"utterance": "два года"},
    )
    apply_turn(p, co, "два года", turn_id=1)
    assert p.age_years == 2
    assert p.term_months is None  # misattribution dropped


def test_year_form_in_term_phase_grounds_term_via_fallback() -> None:
    """Bug S (live call 4e522fb5): age already captured (used vehicle
    with age_years=2), term_months still None. User says 'три года
    срок'. Classifier conservative on this surface → utterance
    fallback in term-phase grounds term_months=36."""
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="BYN",
        condition_new=0,
        age_years=2,
        term_months=None,
        prepaid_pct=38.0,
        state=ProfileState.COLLECTING,
    )
    co = ClassifierOutput.model_validate(
        {"intent": "TOOL", "is_confirmation": False},
        context={"utterance": "три года срок"},
    )
    apply_turn(p, co, "три года срок", turn_id=1)
    assert p.term_months == 36


def test_year_form_in_term_phase_new_vehicle_grounds_term() -> None:
    """condition_new=1 (no age applies); term still missing.
    User says 'на пять лет'. Even with classifier silent, fallback
    grounds term_months=60 because we're past the age-collection state."""
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="BYN",
        condition_new=1,
        term_months=None,
        prepaid_pct=20.0,
        state=ProfileState.COLLECTING,
    )
    co = ClassifierOutput.model_validate(
        {"intent": "TOOL", "is_confirmation": False},
        context={"utterance": "на пять лет"},
    )
    apply_turn(p, co, "на пять лет", turn_id=1)
    assert p.term_months == 60
    assert p.age_years is None  # no age for new vehicles


def test_year_form_term_phase_drops_misattributed_age() -> None:
    """Inverse safeguard: term-phase, classifier mis-emits age_years=5
    from 'на пять лет'. Dispatcher routes the year-form to term and
    drops the misattributed age (age has no role for new vehicles
    or already-captured age)."""
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="BYN",
        condition_new=1,
        term_months=None,
        prepaid_pct=20.0,
        state=ProfileState.COLLECTING,
    )
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "is_confirmation": False,
            "age_years": 5,
        },
        context={"utterance": "на пять лет"},
    )
    apply_turn(p, co, "на пять лет", turn_id=1)
    assert p.term_months == 60
    assert p.age_years is None  # condition_new=1 disqualifies age


# ---------------------------------------------------------------------------
# Codex CP-3.6 P1: utterance fallback for slots the classifier dropped.
# ---------------------------------------------------------------------------

def test_utterance_fallback_subject_when_classifier_omits():
    """Codex CP-3.6 P1: classifier returns intent=RAG with no subject on
    a calc-prep utterance ('Я думаю взять себе машину'); the utterance-
    fallback regex must still seed profile.subject so the next turn's
    capture flow does not re-ask.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(state=ProfileState.COLLECTING)
    co = ClassifierOutput.model_validate(
        {"intent": "RAG", "is_confirmation": False},
        context={"utterance": "Я думаю взять себе машину"},
    )
    apply_turn(p, co, "Я думаю взять себе машину", turn_id=1)
    assert p.subject == "Легковой автомобиль", (
        f"utterance-fallback subject must seed profile, got {p.subject!r}"
    )


def test_utterance_fallback_skipped_when_profile_field_already_set():
    """Sticky guarantee: the fallback never overrides an existing value.
    User says 'хочу машину' but profile.subject is already 'Грузовой' —
    keep the truck.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(subject="Грузовой автомобиль", state=ProfileState.COLLECTING)
    co = ClassifierOutput.model_validate(
        {"intent": "RAG", "is_confirmation": False},
        context={"utterance": "хочу машину"},
    )
    apply_turn(p, co, "хочу машину", turn_id=1)
    assert p.subject == "Грузовой автомобиль"


def test_utterance_fallback_skipped_when_classifier_proposes_field():
    """If the classifier already proposed subject, the fallback must
    not double-write a different value.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(state=ProfileState.COLLECTING)
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "is_confirmation": False,
            "subject": "Спецтехника",
        },
        context={"utterance": "хочу машину под спецтехнику"},
    )
    apply_turn(p, co, "хочу машину под спецтехнику", turn_id=1)
    assert p.subject == "Спецтехника"


# ---------------------------------------------------------------------------
# Codex CP-3.6 P2: memory_block threading into FireLLMFallback prompt.
# ---------------------------------------------------------------------------

def test_build_fallback_messages_includes_memory_block():
    """The LLM fallback prompt must prepend the orchestrator-provided
    memory_block when present so the model sees prior dialogue context.
    """
    from backend.turn_dispatcher import _build_fallback_messages

    msgs = _build_fallback_messages(
        utterance="а где у вас офис?",
        rag_context=None,
        snapshot=None,
        memory_block="Контекст диалога:\nКлиент: привет\nБот: здравствуйте",
    )
    content = msgs[0]["content"]
    assert "Контекст диалога" in content, (
        "memory_block must be present in fallback prompt"
    )
    assert "Сообщение клиента: а где у вас офис?" in content
    assert content.index("Контекст диалога") < content.index("Сообщение клиента"), (
        "memory_block must precede the current utterance"
    )


def test_build_fallback_messages_omits_memory_block_when_absent():
    """memory_block is optional; absence must not introduce empty noise.
    The role-guard block IS unconditional (anti-hallucination contract),
    so we assert structure rather than exact equality."""
    from backend.turn_dispatcher import _build_fallback_messages

    msgs = _build_fallback_messages(
        utterance="привет",
        rag_context=None,
        snapshot=None,
    )
    content = msgs[0]["content"]
    assert "Сообщение клиента: привет" in content
    # No memory or KB blocks injected.
    assert "Контекст диалога" not in content
    assert "Фрагменты из базы знаний" not in content
    assert "НЕ переспрашивай" not in content  # no snapshot anchor


def test_build_fallback_messages_role_guard_blocks_change_confirm_hallucination():
    """Anti-hallucination role guard (live call 8cb0bfaf 2026-04-28):
    the LLM hallucinated 'Меняем стоимость и график. Подтвердите?' when
    classifier dropped a multi-field change. State machine never staged
    the change; user said 'Да' on next turn → silent data loss.

    Architectural truth: change-confirm wording is ALWAYS produced by
    build_change_confirm_text via EmitChangeConfirm. The LLM is NEVER the
    source. The role-guard prompt must explicitly forbid the LLM from
    writing change-confirm phrases so it cannot fabricate one even when
    the classifier silently drops fields."""
    from backend.turn_dispatcher import _build_fallback_messages

    msgs = _build_fallback_messages(
        utterance="давай поменяем стоимость и график",
        rag_context=None,
        snapshot=None,
    )
    content = msgs[0]["content"]
    # Must explicitly forbid change-confirm wording.
    assert "Меняю" in content and "всё верно" in content, (
        "role guard must name the forbidden change-confirm shape"
    )
    assert "Подтверждаю изменение" in content or "Подтвердите параметры" in content
    # Must instruct the model to clarify when uncertain.
    assert "переспроси" in content or "уточни" in content


def test_build_fallback_messages_role_guard_present_with_snapshot_and_kb():
    """Role guard is unconditional — applies whether or not snapshot or
    KB context is present. Verifies ordering: memory → anchor → role
    guard → KB → utterance, so the LLM sees the contract before the
    knowledge it should answer from."""
    from backend.turn_dispatcher import _build_fallback_messages
    from backend.turn_action import ProfileSnapshot

    snap = ProfileSnapshot(
        client_type="Физическое лицо", subject="Легковой автомобиль",
        cost=300000.0, currency="BYN", original_cost=None,
        original_currency=None, condition_new=0, age_years=2,
        prepaid_pct=30.0, prepaid_amount=None, term_months=36,
        type_schedule="0", name="Никита",
    )
    msgs = _build_fallback_messages(
        utterance="а в чём разница графиков?",
        rag_context="[Fragment 1]\nАннуитетный график — равные платежи.",
        snapshot=snap,
        memory_block="Контекст диалога:\nКлиент: подскажи",
    )
    content = msgs[0]["content"]
    # All four blocks present.
    assert "Контекст диалога" in content
    assert "НЕ переспрашивай" in content  # snapshot anchor
    assert "НЕ пиши формулировки" in content  # role guard
    assert "Фрагменты из базы знаний" in content
    # Ordering check.
    pos_memory = content.index("Контекст диалога")
    pos_anchor = content.index("НЕ переспрашивай")
    pos_guard = content.index("НЕ пиши формулировки")
    pos_kb = content.index("Фрагменты из базы знаний")
    pos_utt = content.index("Сообщение клиента")
    assert pos_memory < pos_anchor < pos_guard < pos_kb < pos_utt


# ---------------------------------------------------------------------------
# Section 3 polish (2026-04-26 follow-ups to live test bf7a95a8):
# Bug 1 — subject-grounding flag prevents silent-default subject from
#         passing through to readback.
# Bug 2 — meta-question detector routes "А какой лучше?" away from a
#         re-emitted EmitClarify into FireLLMFallback so the bot can
#         actually answer the user's question.
# ---------------------------------------------------------------------------

def test_subject_grounded_flag_defaults_true_when_constructed_with_subject():
    """ClientProfile fixtures and snapshots that pass `subject=...` at
    construction time treat that as already user-grounded. Without this,
    every existing test that bootstraps a partial profile with a subject
    would unexpectedly fall into the "missing subject" branch.
    """
    from backend.session import ClientProfile
    p = ClientProfile(subject="Легковой автомобиль")
    assert p.subject_user_grounded is True


def test_subject_grounded_flag_defaults_false_for_empty_profile():
    from backend.session import ClientProfile
    p = ClientProfile()
    assert p.subject_user_grounded is False


def test_missing_fields_includes_subject_when_set_silently():
    """The bug 1 invariant: even if profile.subject is set, when the
    user-grounded flag is False the field is treated as missing so the
    clarify gate fires instead of letting the silent default reach the
    readback.
    """
    from backend.session import ClientProfile
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        type_schedule="0",
    )
    # Simulate the silent-leak path: subject is present, but the grounded
    # flag was never flipped True (e.g. the value was set via a path that
    # bypassed the strict utterance-cue gate).
    p.subject_user_grounded = False
    assert "subject" in p.missing_fields()
    assert not p.is_complete_for_calc()


def test_subject_grounded_flips_true_when_utterance_fallback_seeds_it():
    """When the utterance-fallback regex extracts subject from a clear
    user mention, the grounded flag must flip True so the readback fires.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(state=ProfileState.COLLECTING)
    co = ClassifierOutput.model_validate(
        {"intent": "RAG", "is_confirmation": False},
        context={"utterance": "Я думаю взять себе машину"},
    )
    apply_turn(p, co, "Я думаю взять себе машину", turn_id=1)
    assert p.subject == "Легковой автомобиль"
    assert p.subject_user_grounded is True


def test_subject_grounded_flips_true_when_classifier_grounds_it():
    """When the classifier proposes subject and the value is grounded by
    an utterance cue, the flag must flip True.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(state=ProfileState.COLLECTING)
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "is_confirmation": False,
            "subject": "Грузовой автомобиль",
        },
        context={"utterance": "хочу взять грузовик"},
    )
    apply_turn(p, co, "хочу взять грузовик", turn_id=1)
    assert p.subject == "Грузовой автомобиль"
    assert p.subject_user_grounded is True


def test_silent_subject_blocks_readback_and_emits_subject_clarify():
    """Live regression bf7a95a8 (2026-04-26): user says vague calc-prep
    utterance ("Я бы себе хотел что-то купить"), every other slot lands
    in subsequent turns, but subject was set silently. Without this fix
    the readback fires with subject=Легковой автомобиль the user never
    confirmed. With the fix, the clarify gate asks for subject before
    any readback.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    # Simulate the post-leak state: profile is otherwise complete but
    # subject was set via a path that did NOT flip the grounded flag.
    p = ClientProfile(
        client_type="Физическое лицо",
        cost=80000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        type_schedule="0",
        state=ProfileState.COLLECTING,
    )
    p.subject = "Легковой автомобиль"          # silent assignment
    p.subject_user_grounded = False             # never confirmed by utterance

    co = ClassifierOutput.model_validate(
        {"intent": "TOOL", "is_confirmation": False},
        context={"utterance": "ну хорошо"},
    )
    action = apply_turn(p, co, "ну хорошо", turn_id=1)
    assert isinstance(action, EmitClarify), (
        f"silent subject must force a subject-ask clarify, got {type(action).__name__}"
    )
    assert "subject" in action.missing


def test_meta_question_routes_to_llm_fallback_instead_of_reclarify():
    """Bug 2 (live regression bf7a95a8 2026-04-26): user replies to the
    type_schedule clarify with "А какой лучше?". The bot must answer the
    meta-question via LLM/RAG, not re-emit the same clarify prompt.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    # Profile is missing only type_schedule; everything else is captured.
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        state=ProfileState.COLLECTING,
    )
    co = ClassifierOutput.model_validate(
        {"intent": "CONVERSATION", "is_confirmation": False},
        context={"utterance": "А какой лучше?"},
    )
    action = apply_turn(p, co, "А какой лучше?", turn_id=1)
    assert isinstance(action, FireLLMFallback), (
        f"meta-question must drift to LLM, got {type(action).__name__}"
    )
    # Snapshot must be carried so the LLM has the captured-field anchor.
    assert action.snapshot is not None
    assert action.snapshot.subject == "Легковой автомобиль"


def test_meta_question_variations_all_drift_to_llm():
    """The meta-question detector must be generic across phrasings, not
    hard-coded per prompt. Sample a handful of variants on the same
    incomplete profile.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    variants = [
        "А какой лучше тип графика, подскажи, ну какой дешевле?",
        "Что лучше выбрать?",
        "В чём разница между ними?",
        "Не понимаю, что выбрать.",
        "Объясни, пожалуйста, чем отличаются.",
        "Что выгоднее по итогу?",
    ]
    for utt in variants:
        p = ClientProfile(
            client_type="Физическое лицо",
            subject="Легковой автомобиль",
            cost=80000.0,
            currency="BYN",
            condition_new=1,
            prepaid_pct=20.0,
            term_months=36,
            state=ProfileState.COLLECTING,
        )
        co = ClassifierOutput.model_validate(
            {"intent": "CONVERSATION", "is_confirmation": False},
            context={"utterance": utt},
        )
        action = apply_turn(p, co, utt, turn_id=1)
        assert isinstance(action, FireLLMFallback), (
            f"meta-question variant {utt!r} must drift to LLM, "
            f"got {type(action).__name__}"
        )


def test_clarify_still_fires_when_no_meta_question():
    """The new meta-question gate must NOT swallow plain calc-intent
    turns that legitimately need a clarify ask.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(state=ProfileState.COLLECTING)
    co = ClassifierOutput.model_validate(
        {"intent": "TOOL", "is_confirmation": False},
        context={"utterance": "хочу посчитать лизинг"},
    )
    action = apply_turn(p, co, "хочу посчитать лизинг", turn_id=1)
    assert isinstance(action, EmitClarify)


def test_meta_question_with_actual_answer_captures_field_first():
    """If the user phrases a meta-question but ALSO names the answer
    in the same breath (\"А какой лучше? Линейный давай.\"), the
    structured slot capture wins — type_schedule fills, profile becomes
    complete, readback fires. The meta gate is only consulted when
    the clarify branch would otherwise return.
    """
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        state=ProfileState.COLLECTING,
    )
    utt = "А какой лучше? Линейный давай."
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "is_confirmation": False,
            "type_schedule": "1",
        },
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=1)
    assert p.type_schedule == "1"
    assert isinstance(action, EmitReadback), (
        f"answer-with-meta-question still captures the answer; "
        f"expected EmitReadback, got {type(action).__name__}"
    )


# ---------------------------------------------------------------------------
# Live regression 5e6f4c48 (2026-04-26): RAG turns must not enter the step 5b
# clarify branch even when the profile carries captured fields. Bug R's
# _has_any_core_field gate previously fired EmitClarify on every RAG turn
# once any field was captured (often silently from utterances like "вашей
# компании"), looping the bot's subject prompt while the user kept asking
# unrelated questions about the leasing company.
# ---------------------------------------------------------------------------

def test_rag_intent_drifts_to_llm_even_when_core_field_captured():
    """Live repro 5e6f4c48: profile has a captured field, user asks a
    plain RAG question. The clarify gate must be skipped so the LLM
    can answer instead of re-emitting subject-ask."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        client_type="Юридическое лицо",  # captured legitimately on a prior turn
        state=ProfileState.COLLECTING,
    )
    utt = "Расскажи, кто директор?"
    co = ClassifierOutput.model_validate(
        {"intent": "RAG", "is_confirmation": False},
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=1)
    assert isinstance(action, FireLLMFallback), (
        f"RAG turn must drift to LLM even with a captured field, "
        f"got {type(action).__name__}"
    )


def test_rag_intent_drifts_to_llm_with_subject_captured():
    """Mirror case for subject — RAG question after subject was
    captured legitimately must still drift to LLM."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        subject="Легковой автомобиль",
        state=ProfileState.COLLECTING,
    )
    utt = "А какой у вас адрес в Минске?"
    co = ClassifierOutput.model_validate(
        {"intent": "RAG", "is_confirmation": False},
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=1)
    assert isinstance(action, FireLLMFallback)


def test_calc_intent_still_emits_clarify_when_incomplete():
    """Sanity: TOOL intent is unaffected by the RAG-skip — calc-prep
    turns still flow through step 5b to gather missing slots."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(state=ProfileState.COLLECTING)
    utt = "хочу посчитать лизинг"
    co = ClassifierOutput.model_validate(
        {"intent": "TOOL", "is_confirmation": False},
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=1)
    assert isinstance(action, EmitClarify)


def test_bug_r_bare_affirmation_still_emits_clarify():
    """Bug R regression guard: 'Давай' classified as CONVERSATION (NOT
    RAG) with a captured field must still enter step 5b so we ask for
    the missing slot rather than letting the LLM invent a default."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(
        subject="Легковой автомобиль",
        state=ProfileState.COLLECTING,
    )
    utt = "Давай"
    co = ClassifierOutput.model_validate(
        {"intent": "CONVERSATION", "is_confirmation": False},
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=1)
    assert isinstance(action, EmitClarify)


def test_silent_client_type_from_other_reference_no_longer_captured():
    """End-to-end: the live regression utterance no longer captures
    client_type even when the classifier emits it."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = ClientProfile(state=ProfileState.COLLECTING)
    utt = "Расскажи, кто директор вашей компании?"
    co = ClassifierOutput.model_validate(
        {
            "intent": "RAG",
            "is_confirmation": False,
            "client_type": "Юридическое лицо",
        },
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=1)
    assert p.client_type is None, (
        f"silent юр capture from 'вашей компании' must be dropped, "
        f"got {p.client_type!r}"
    )
    assert isinstance(action, FireLLMFallback), (
        f"with no captured fields, RAG turn drifts to LLM — "
        f"got {type(action).__name__}"
    )


# ============================================================ Issues 2+3
# Live call 5fa0bb3d 2026-04-26: post-calc, last_offer="sms" stamped.
# User says "А, хорошо, а можно всё-таки другой график сделать?"
# (action=change_param, change_field=type_schedule, change_value
# could not be grounded — no annuity/linear cue). Two regressions:
#
# Issue 2 — last_offer="sms" leaks into the next turn. User's "Да"
#   matches step 5c (last_offer + is_confirmation + no change_field +
#   action != change_param) → FireSMS instead of recalc.
#
# Issue 3 — falling through to FireLLMFallback hallucinates a
#   change-confirm ("Меняем тип графика на равные платежи, верно?")
#   without staging anything, leading the user to confirm a non-
#   existent change.


def test_change_intent_without_value_clears_last_offer_and_clarifies():
    """Issue 3: bare change-intent emits a deterministic clarify keyed
    by change_field instead of falling through to LLM fallback."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED
    p.last_offer = "sms"

    utt = "А, хорошо, а можно всё-таки другой график сделать?"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "change_param",
            "change_field": "type_schedule",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=2)

    # Issue 3: deterministic clarify, not LLM fallback.
    assert isinstance(action, EmitClarify), (
        f"bare change-intent must clarify, got {type(action).__name__}"
    )
    assert "type_schedule" in action.missing
    # Issue 2: last_offer cleared so next "Да" can't fire SMS via 5c.
    assert p.last_offer is None


def test_change_intent_no_field_still_clears_last_offer():
    """Issue 2 (Fix 2): change_param without change_field still clears
    last_offer so a follow-up bare-Да doesn't trigger 5c→FireSMS."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED
    p.last_offer = "sms"

    utt = "хочу что-то поменять"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "change_param",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    apply_turn(p, co, utt, turn_id=2)
    # last_offer cleared even when no specific field was identified —
    # the user has signaled intent to change, the SMS offer is stale.
    assert p.last_offer is None


def test_change_field_set_without_action_change_param_clears_last_offer():
    """Fix 2: change_field is the broader signal — even without
    action=change_param, presence of change_field clears last_offer."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED
    p.last_offer = "sms"

    utt = "линейный"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "is_confirmation": False,
            "change_field": "type_schedule",
            "change_value": "1",
        },
        context={"utterance": utt},
    )
    apply_turn(p, co, utt, turn_id=2)
    assert p.last_offer is None


def test_bare_da_after_calc_no_change_mention_still_fires_sms():
    """Sanity guard: Fix 2 must NOT break the canonical "Да → SMS"
    path when the user is genuinely confirming the post-calc SMS offer
    (no change-intent signaled)."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_action import FireSMS
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED
    p.last_offer = "sms"

    utt = "Да"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "is_confirmation": True,
        },
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=2)
    assert isinstance(action, FireSMS), (
        f"bare-Да after SMS offer must fire SMS, got {type(action).__name__}"
    )


def test_change_intent_with_grounded_value_routes_to_change_confirm():
    """Counterpart to Fix 3 clarify: when change_value DOES ground,
    step 4 stages EmitChangeConfirm normally (no clarify, no LLM)."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED
    p.last_offer = "sms"

    utt = "Поменяй на линейный"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "change_param",
            "change_field": "type_schedule",
            "change_value": "1",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=2)
    assert isinstance(action, EmitChangeConfirm)
    # Step 4 already clears last_offer; this is the same invariant
    # via a different code path.
    assert p.last_offer is None


# ============================================================ Issue 4
# Live call 2ab41112 2026-04-26: after a successful calc + SMS fire
# (last_offer cleared), state stayed CONFIRMED. User said
# "Спасибо. Последний вопрос. Кто владелец Вашей компании?" — the
# small classifier emitted is_confirmation=True (cued by "Спасибо")
# along with intent=RAG. Step 6's bare CONFIRMED+is_confirmation+
# complete gate matched and re-fired FireCalc with stale params.
# Expected behaviour: a RAG-intent turn after the SMS offer drifts
# to FireLLMFallback so the bot can answer the actual question.
#
# Universal fix: gate step 6 on the original (pre-turn) state being
# READBACK_PENDING (the canonical readback-confirm flow that
# transitions COLLECTING→READBACK_PENDING→CONFIRMED in one turn) OR
# on the classifier explicitly emitting calculate/recalculate. Bare
# is_confirmation in a sticky CONFIRMED state is not enough signal.


def test_post_calc_thanks_plus_rag_question_does_not_refire_calc():
    """Issue 4: 'Спасибо. Кто владелец?' classified as
    is_confirmation=True must NOT re-fire FireCalc when state has been
    CONFIRMED since a prior calc. Should drift to LLM fallback so RAG
    can answer."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED  # post-calc sticky state
    p.last_offer = None  # SMS already fired, offer cleared

    utt = "Спасибо. Последний вопрос. Кто владелец Вашей компании?"
    co = ClassifierOutput.model_validate(
        {
            "intent": "RAG",
            "is_confirmation": True,  # cued by "Спасибо"
        },
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=20)
    assert isinstance(action, FireLLMFallback), (
        f"post-SMS thank-you+question must NOT re-fire calc, "
        f"got {type(action).__name__}"
    )


def test_post_calc_explicit_recalculate_intent_still_fires_calc():
    """Counterpart to Issue 4: an explicit recalculate signal in the
    post-calc CONFIRMED state still fires FireCalc. Without this guard
    a user saying 'давай пересчитаем' would get LLM-narration instead
    of an actual re-run."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_action import FireCalc
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED
    p.last_offer = None

    utt = "Давай пересчитаем"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "is_confirmation": True,
            "action": "recalculate",
        },
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=20)
    assert isinstance(action, FireCalc), (
        f"explicit recalculate intent must fire calc, "
        f"got {type(action).__name__}"
    )


def test_readback_pending_to_confirmed_in_one_turn_still_fires_calc():
    """Counterpart to Issue 4: the canonical readback-confirm flow —
    state was READBACK_PENDING at the top of the turn, classifier emits
    is_confirmation=True, step 2 transitions to CONFIRMED, step 6 fires
    FireCalc in the SAME turn. The Issue 4 gate must not break this."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_action import FireCalc
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = make_complete_profile()
    p.state = ProfileState.READBACK_PENDING  # bot just spoke readback
    p.last_offer = None

    utt = "Да"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "is_confirmation": True,
        },
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=10)
    assert isinstance(action, FireCalc), (
        f"readback→confirm canonical flow must fire calc, "
        f"got {type(action).__name__}"
    )


def test_change_intent_prepaid_pct_clarify_uses_canonical_label():
    """Fix 3: clarify message routing — prepaid_pct/prepaid_amount must
    map to the canonical "prepaid" key so build_clarification_prompt's
    existing branch fires (otherwise the renderer prints the raw field
    name to the caller)."""
    from backend.session import ClientProfile, ProfileState
    from backend.turn_dispatcher import apply_turn
    from backend.classifier_schema import ClassifierOutput

    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED
    p.last_offer = "sms"

    utt = "поменяй аванс"
    co = ClassifierOutput.model_validate(
        {
            "intent": "TOOL",
            "action": "change_param",
            "change_field": "prepaid_pct",
            "is_confirmation": False,
        },
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utt, turn_id=2)
    assert isinstance(action, EmitClarify)
    assert "prepaid" in action.missing


# ============================================================ Bug 6
# Live call e6226e5d 2026-04-29 (Stanislav 15:16:11): post-calc, the
# user said "Отправьте смс-ку" — explicit SMS request. Classifier emits
# action="sms", step 6b fires FireSMS, but last_offer="sms" stays
# sticky. User then asked unrelated договор / equipment / cities
# questions — those route to FireLLMFallback and leave last_offer
# alone. On a later bare "Да" the dispatcher matches step 5c again
# (last_offer + is_confirmation + no change_field) and re-fires SMS.
#
# Universal fix:
#   - step 6b clears last_offer before returning FireSMS, matching the
#     existing step 5c behaviour.
#   - FireLLMFallback returns clear last_offer too — by the time the
#     dispatcher reaches that path, the turn is conclusively NOT acting
#     on the prior SMS offer (bare confirms/denies are caught upstream
#     by FAST-PATH and step 5c). So any non-structural conversational
#     turn invalidates the stale offer.


def test_bug6_sms_action_step_6b_clears_last_offer():
    """Bug 6 / Stanislav 15:16:11: explicit `action=sms` request fires
    FireSMS via step 6b but must clear last_offer so a later bare "Да"
    on an unrelated LLM-narrated question can't re-fire SMS via step 5c."""
    from backend.turn_action import FireSMS

    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED
    p.last_offer = "sms"

    utt = "Отправьте смс-ку"
    co = make_classifier(
        utterance=utt,
        intent="TOOL",
        action="sms",
        is_confirmation=False,
    )
    action = apply_turn(p, co, utt, turn_id=2)
    assert isinstance(action, FireSMS), (
        f"explicit SMS request must fire FireSMS, got {type(action).__name__}"
    )
    assert p.last_offer is None, (
        "step 6b must clear last_offer to prevent stale-offer re-fire"
    )


def test_bug6_llm_fallback_clears_stale_last_offer():
    """Bug 6: when the dispatcher routes to FireLLMFallback (RAG
    question, small talk, anything non-structural), the prior post-calc
    SMS offer is invalidated. Without this, last_offer persisted across
    an arbitrary number of unrelated turns and the next bare confirm
    re-fired SMS via step 5c."""
    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED
    p.last_offer = "sms"

    utt = "Скажите, какие документы нужны?"
    co = make_classifier(
        utterance=utt,
        intent="RAG",
        is_confirmation=False,
    )
    action = apply_turn(p, co, utt, turn_id=2)
    assert isinstance(action, FireLLMFallback), (
        f"RAG question on confirmed profile must route to LLM fallback, "
        f"got {type(action).__name__}"
    )
    assert p.last_offer is None, (
        "FireLLMFallback path must clear stale last_offer so the next "
        "bare confirm cannot re-fire SMS via step 5c"
    )


def test_bug7_partial_meta_question_chto_takoe_routes_to_llm_fallback():
    """Bug 7 partial (Stanislav 15:08:23, Valery 15:29:01 — call e6226e5d
    + 19496277): user replied to the schedule clarify with "Что такое
    аннуитет?" — a meta-question, not a parameter answer. The current
    meta-question regex did not match "что такое", so step 5b re-emitted
    the same clarify prompt. Extending the regex routes the question to
    FireLLMFallback so the LLM can explain; the next user utterance
    re-enters the clarify gate with the field actually answered."""
    from backend.turn_dispatcher import _is_meta_question

    # Variants from live calls + grammatical relatives.
    assert _is_meta_question("Что такое аннуитет?")
    assert _is_meta_question("а что такое линейный?")
    assert _is_meta_question("что такое")
    # Counter-cases: must not match plain answers / general utterances.
    assert not _is_meta_question("аннуитет")
    assert not _is_meta_question("36 месяцев")
    assert not _is_meta_question("")


def test_bug7_partial_chto_takoe_routes_via_step_5b_meta_question_gate():
    """End-to-end Bug 7 partial: with intent=CONVERSATION (NOT RAG —
    RAG turns skip step 5b by design after live regression 5e6f4c48),
    on an incomplete profile, the meta-question gate inside step 5b
    must route to FireLLMFallback when the user replies to a clarify
    with "что такое X". Use an utterance that doesn't ground any
    enum field so the classifier doesn't accidentally complete the
    profile before step 5b runs."""
    p = make_partial_profile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=80000.0,
        currency="USD",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        # type_schedule deliberately missing — this is what the bot is
        # clarifying when the user replies with the meta-question.
    )
    p.state = ProfileState.COLLECTING
    # Avoid the "Бот спросил тип графика — клиент ответил 'аннуитет'"
    # grounding pitfall by phrasing the meta-question generically.
    utt = "А что такое лизинг вообще?"
    co = make_classifier(
        utterance=utt, intent="CONVERSATION", is_confirmation=False
    )
    action = apply_turn(p, co, utt, turn_id=1)
    assert isinstance(action, FireLLMFallback), (
        f"meta-question on incomplete profile must route to LLM "
        f"fallback via step 5b's meta-question gate, "
        f"got {type(action).__name__}"
    )


def test_bug6_canonical_da_after_sms_offer_still_works():
    """Sanity guard: Bug 6's clearing must NOT break the canonical
    "Да → SMS" path. Bare confirm on a fresh post-calc SMS offer must
    still fire SMS — last_offer is cleared by step 5c itself in that
    flow, so no behaviour regression."""
    from backend.turn_action import FireSMS

    p = make_complete_profile()
    p.state = ProfileState.CONFIRMED
    p.last_offer = "sms"

    utt = "Да"
    co = make_classifier(utterance=utt, intent="TOOL", is_confirmation=True)
    action = apply_turn(p, co, utt, turn_id=2)
    assert isinstance(action, FireSMS)
    assert p.last_offer is None


# ============================================================ Bug 10 follow-up
# Live call ce1a0ad6 2026-05-03 17:22:04: user said "А давай всё-таки
# поменяем цену на 20 косарей" — slang for 20 000. The classifier
# extracted change_field=cost, change_value=20 with intent=TOOL.
# `_grounded_proposed_patches` accepts the (cf, cv) pair unconditionally
# on intent=TOOL (Polish A 2026-04-27 trust signal), so cost=20 lands
# in proposed patches. Bot read back "Меняю стоимость на 20.0" and the
# calc fired with cost=20 BYN — silent data loss.
#
# Batch 1 added slang stems to extract_cost_from_utterance, but that
# function was only consulted by _apply_utterance_fallbacks (which
# skips when classifier already proposed the field). The fix layer
# missing in Batch 1: an active corrective override — when the
# slang-aware extractor returns a value that disagrees with the
# classifier's cost, prefer the extractor.


def test_bug10_followup_slang_overrides_classifier_bare_digit_top_level():
    """Top-level cost field: classifier emits cost=20 from "20 косарей".
    Slang extractor returns 20000. Corrective layer must override to
    20000 so downstream readback / calc sees the right amount."""
    from backend.turn_dispatcher import _grounded_proposed_patches

    co = make_classifier(
        utterance="давай 20 косарей",
        intent="TOOL",
        cost=20.0,
    )
    proposed = _grounded_proposed_patches(co, "давай 20 косарей")
    assert proposed.get("cost") == 20000, (
        f"slang multiplier must override classifier's bare-digit cost; "
        f"got {proposed.get('cost')!r}"
    )


def test_bug10_followup_slang_overrides_classifier_change_value():
    """Change-flow: classifier emits change_field=cost change_value=20
    on "поменяем цену на 20 косарей". Same correction must apply to
    the change pair, since that's the actual production failure mode."""
    from backend.turn_dispatcher import _grounded_proposed_patches

    co = make_classifier(
        utterance="поменяем цену на 20 косарей",
        intent="TOOL",
        change_field="cost",
        change_value="20",
    )
    proposed = _grounded_proposed_patches(co, "поменяем цену на 20 косарей")
    assert proposed.get("cost") == 20000, (
        f"slang multiplier must override change_value when extractor disagrees; "
        f"got {proposed.get('cost')!r}"
    )


def test_bug10_followup_correct_classifier_cost_kept():
    """Sanity: when classifier matches the slang interpretation, the
    override is a no-op. cost=80000 + utterance "80 тысяч долларов"
    extractor=80000 → no change."""
    from backend.turn_dispatcher import _grounded_proposed_patches

    co = make_classifier(
        utterance="за 80 тысяч долларов",
        intent="TOOL",
        cost=80000.0,
    )
    proposed = _grounded_proposed_patches(co, "за 80 тысяч долларов")
    assert proposed.get("cost") == 80000


def test_bug10_followup_no_slang_signal_does_not_override():
    """Sanity: utterance has no cost mention extractor parses; classifier
    cost stands. (Today's behaviour where classifier carries through
    its own value when extractor cannot help.) Use a bare-digit-only
    cost utterance to verify."""
    from backend.turn_dispatcher import _grounded_proposed_patches

    co = make_classifier(
        utterance="за 80000 долларов",
        intent="TOOL",
        cost=80000.0,
    )
    proposed = _grounded_proposed_patches(co, "за 80000 долларов")
    assert proposed.get("cost") == 80000
