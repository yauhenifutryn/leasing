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
    profile = make_complete_profile()
    profile.state = ProfileState.COLLECTING
    classifier = make_classifier(intent="CONVERSATION", is_confirmation=False)
    action = apply_turn(profile, classifier, utterance="Аннуитетный график")
    assert isinstance(action, EmitReadback)
    assert profile.state == ProfileState.READBACK_PENDING
    assert action.snapshot.cost == 80000.0


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
    assert action.snapshot.cost == 80000.0
    assert action.snapshot.currency == "USD"
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


def test_e7b_subject_flip_forcing_client_type_emits_paired_confirm() -> None:
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
    assert "client_type" in action.changes
    assert action.changes["subject"]["new"] == "Грузовой автомобиль"
    assert action.changes["client_type"]["old"] == "Физическое лицо"
    assert action.changes["client_type"]["new"] == "Юридическое лицо"
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
    profile.state = ProfileState.CONFIRMED
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
