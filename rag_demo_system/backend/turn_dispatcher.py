"""apply_turn — single transaction per user turn.

Consumes (profile, classifier_output, utterance) and returns ONE
TurnAction. Mutates `profile` in-place for captured fields (step 5) and
state-machine transitions (steps 1, 2, 5a). Never mutates for change
proposals — those go through EmitChangeConfirm (step 4) and mutate only
on next-turn confirmation.

Dispatch order is documented in spec §5. Phase 3.C of the apply_turn
refactor.
"""
from __future__ import annotations

from typing import Any, Optional

from .classifier_schema import ClassifierOutput, value_grounded
from .profile_state import (
    build_snapshot,
    partition_patches,
    derive_implied_flips,
    build_calc_params,
)
from .session import ClientProfile, ProfileState
from .turn_action import (
    ProfileSnapshot,
    TurnAction,
    EmitReadback,
    EmitClarify,
    EmitChangeConfirm,
    FireCalc,
    FireLLMFallback,
    FireOORMessage,
    Noop,
)


# Fields apply_turn considers at pre-compute time when sweeping
# classifier top-level values into proposed_patches.
_GROUNDED_FIELDS: tuple[str, ...] = (
    "client_type",
    "subject",
    "cost",
    "currency",
    "condition_new",
    "age_years",
    "prepaid_pct",
    "prepaid_amount",
    "term_months",
    "type_schedule",
)


def _grounded_proposed_patches(
    classifier_output: ClassifierOutput,
    utterance: str,
) -> dict[str, Any]:
    """Collect classifier-proposed patches that pass `value_grounded`.

    Includes both top-level field values AND the change_field /
    change_value pair. A hallucinated value (Qwen drift) without a
    matching utterance cue fails grounding and is dropped silently.
    """
    proposed: dict[str, Any] = {}

    for field_name in _GROUNDED_FIELDS:
        value = getattr(classifier_output, field_name, None)
        if value is None:
            continue
        if value_grounded(field_name, value, utterance):
            proposed[field_name] = value

    # Explicit change_field / change_value pair. Treated identically to
    # a top-level field for routing purposes — partition_patches decides
    # whether it's a first-time capture or a delta.
    cf = classifier_output.change_field
    cv = classifier_output.change_value
    if cf and cv is not None and value_grounded(cf, cv, utterance):
        proposed[cf] = cv

    return proposed


def _project_snapshot(
    profile: ClientProfile,
    patches: dict[str, Any],
) -> ProfileSnapshot:
    """Build a snapshot as-if `patches` were applied — without mutating
    the profile. Used as EmitChangeConfirm.snapshot so the UI / LLM
    renderer sees the proposed end-state, not the current state.
    """
    return ProfileSnapshot(
        client_type=patches.get("client_type", profile.client_type),
        subject=patches.get("subject", profile.subject),
        cost=patches.get("cost", profile.cost),
        currency=patches.get("currency", profile.currency),
        original_cost=profile.original_cost,
        original_currency=profile.original_currency,
        condition_new=patches.get("condition_new", profile.condition_new),
        age_years=patches.get("age_years", profile.age_years),
        prepaid_pct=patches.get("prepaid_pct", profile.prepaid_pct),
        prepaid_amount=patches.get("prepaid_amount", profile.prepaid_amount),
        term_months=patches.get("term_months", profile.term_months),
        type_schedule=patches.get("type_schedule", profile.type_schedule),
    )


# Sentinel reasons for Noop-as-redispatch-signal from _dispatch_once
# back into apply_turn's loop. Step 1 / step 3 transitions consume the
# classifier's confirmation semantics and want the top of the dispatch
# to re-run against the now-mutated state (so e.g. CHANGE_PENDING+confirm
# can cascade into FireCalc in the same turn).
_REDISPATCH_REASONS = frozenset({
    "redispatch_change",
    "redispatch_deny",
})


def apply_turn(
    profile: ClientProfile,
    classifier_output: ClassifierOutput,
    utterance: str,
    *,
    turn_id: Optional[int] = None,
) -> TurnAction:
    """Dispatch one user turn. Returns exactly one TurnAction.

    Mutates `profile` in-place when appropriate:
      - first-time captures (step 5)
      - state-machine transitions (steps 1, 2, 5a)
    Never mutates on change proposals (step 4); mutation happens only
    when the user confirms on the next turn, re-entering step 1.

    Re-dispatch bound: at most two iterations (spec §5). The second
    pass cannot re-enter steps 1 or 3 because the state transition
    that enabled them is consumed on the first pass.
    """
    action: TurnAction = Noop(reason="uninitialized")
    for _ in range(2):
        action = _dispatch_once(profile, classifier_output, utterance)
        if isinstance(action, Noop) and action.reason in _REDISPATCH_REASONS:
            continue
        break
    return action


def _dispatch_once(
    profile: ClientProfile,
    classifier_output: ClassifierOutput,
    utterance: str,
) -> TurnAction:
    """Single-iteration body of the apply_turn dispatch. Returns a
    Noop with reason ∈ _REDISPATCH_REASONS when the caller's loop
    should re-enter; otherwise returns a terminal TurnAction.
    """
    # STEP 1 (post-change apply): CHANGE_PENDING + is_confirmation →
    # apply the staged change, transition to CONFIRMED, re-dispatch
    # so the mutated state can unlock step 6 FireCalc.
    if (
        profile.state == ProfileState.CHANGE_PENDING
        and classifier_output.is_confirmation
        and profile.pending_change
    ):
        changes = profile.pending_change.get("changes", {}) or {}
        # Legacy single-field payload support (ClientProfile.pending_change
        # allows either {"field":..,"new_value":..} or {"changes": {...}}).
        if not changes and "field" in profile.pending_change:
            field_name = profile.pending_change["field"]
            new_value = profile.pending_change.get(
                "new_value",
                profile.pending_change.get("new"),
            )
            changes = {field_name: {"old": getattr(profile, field_name, None),
                                    "new": new_value}}
        for field_name, change in changes.items():
            if hasattr(profile, field_name):
                setattr(profile, field_name, change["new"])
        profile.pending_change = None
        profile.state = ProfileState.CONFIRMED
        return Noop(reason="redispatch_change")

    # STEP 2: READBACK_PENDING + is_confirmation → CONFIRMED. No
    # return — we fall through to step 6 in the same iteration so
    # calc fires immediately after confirmation.
    if (
        profile.state == ProfileState.READBACK_PENDING
        and classifier_output.is_confirmation
    ):
        profile.state = ProfileState.CONFIRMED

    # -------- pre-compute: grounded patches + implied flips + partition
    proposed = _grounded_proposed_patches(classifier_output, utterance)
    proposed.update(derive_implied_flips(profile, proposed))
    first_time, delta = partition_patches(profile, proposed)

    # STEP 4 (E6 fix): any delta on a captured field → EmitChangeConfirm.
    # Covers explicit change_field pairs AND top-level field flips on
    # captured fields (E7b uniformity) AND implied cross-field flips
    # (derive_implied_flips rule table). Profile fields stay untouched;
    # mutation happens only on next-turn confirm (step 1).
    if delta:
        projected_patches = dict(first_time)
        for field_name, change in delta.items():
            projected_patches[field_name] = change["new"]
        profile.state = ProfileState.CHANGE_PENDING
        profile.pending_change = {"changes": delta}
        return EmitChangeConfirm(
            changes=delta,
            snapshot=_project_snapshot(profile, projected_patches),
        )

    # STEP 5: apply first-time patches in place (additive captures;
    # no user confirmation required under the capture-first principle).
    if first_time:
        for field_name, value in first_time.items():
            setattr(profile, field_name, value)

    # STEP 5a (E5 fix): profile just complete + COLLECTING + not
    # is_confirmation → deterministic readback. Classifier `intent`
    # label is IRRELEVANT at this branch — that's the whole point of
    # the E5 fix. On live call cc7fc318 Qwen labeled the "Аннуитетный
    # график" turn as CONVERSATION and the old gate skipped; now we
    # always emit.
    if (
        profile.is_complete_for_calc()
        and profile.state == ProfileState.COLLECTING
        and not classifier_output.is_confirmation
    ):
        profile.state = ProfileState.READBACK_PENDING
        return EmitReadback(snapshot=build_snapshot(profile))

    # STEP 6 (E8a): CONFIRMED + is_confirmation + calc-ready → FireCalc.
    # Profile is already validated; build calc params from profile state.
    # Post-calc narration is rendered by execute_action's FireCalc
    # handler via render_calc_result(result) — LLM is never involved.
    if (
        profile.state == ProfileState.CONFIRMED
        and classifier_output.is_confirmation
        and profile.is_complete_for_calc()
    ):
        return FireCalc(
            snapshot=build_snapshot(profile),
            calc_params=build_calc_params(profile),
        )

    # STEP 5b: profile incomplete AND state is COLLECTING → EmitClarify
    # with missing-fields list + snapshot anchor. Fires whether or not
    # patches were applied this turn (a no-classifier-info turn still
    # needs a clarifying question). Skipped in READBACK_PENDING /
    # CHANGE_PENDING / CONFIRMED since those have their own follow-up
    # paths (the user's response is interpreted as confirm/deny, not
    # as additional field-fill).
    if (
        not profile.is_complete_for_calc()
        and profile.state == ProfileState.COLLECTING
    ):
        return EmitClarify(
            missing=sorted(profile.missing_fields()),
            snapshot=build_snapshot(profile),
        )

    return Noop(reason="no_dispatch_branch_matched")
