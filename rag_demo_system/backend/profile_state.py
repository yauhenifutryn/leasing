"""Pure functions for turn-level profile state transforms.

No I/O, no async. Every function here is referentially transparent
(no side effects beyond mutating the `profile` argument when explicitly
contracted to do so). Imported by `turn_dispatcher.apply_turn` and
`turn_dispatcher.execute_action`.

Phase 3.A of the apply_turn refactor. Design spec:
`docs/superpowers/specs/2026-04-21-apply-turn-refactor-design.md`.
"""
from __future__ import annotations

from .session import ClientProfile
from .turn_action import ProfileSnapshot


def build_snapshot(profile: ClientProfile) -> ProfileSnapshot:
    """Project the mutable ClientProfile into an immutable snapshot.

    Snapshot fields are a strict subset of ClientProfile — state-machine
    bookkeeping (confirmed_at, readback_emitted_at, pending_change,
    locked_fields, etc.) is deliberately excluded so renderers and LLM
    prompts cannot couple to them.
    """
    return ProfileSnapshot(
        client_type=profile.client_type,
        subject=profile.subject,
        cost=profile.cost,
        currency=profile.currency,
        original_cost=profile.original_cost,
        original_currency=profile.original_currency,
        condition_new=profile.condition_new,
        age_years=profile.age_years,
        prepaid_pct=profile.prepaid_pct,
        prepaid_amount=profile.prepaid_amount,
        term_months=profile.term_months,
        type_schedule=profile.type_schedule,
    )


def partition_patches(
    profile: ClientProfile,
    proposed: dict,
) -> tuple[dict, dict]:
    """Partition proposed field patches against the current profile.

    Returns (first_time_patches, delta_patches):
      - first_time_patches: {field: value} for fields where
        `profile.<field>` is currently None. These are additive
        captures; applying them does not require user confirmation
        (capture-first principle only confirms changes to already-
        captured data).
      - delta_patches: {field: {"old": X, "new": Y}} for fields where
        `profile.<field>` is not None AND differs from the proposed
        value. These are changes and must flow through
        EmitChangeConfirm — apply_turn routes them to step 4.
      - No-op proposals (field already at the proposed value) are
        dropped silently; they contribute to neither dict.

    Note: a proposed value of None on a currently-captured field IS a
    delta (used for implied flips like clearing age_years when
    condition_new flips to 1).
    """
    first_time: dict = {}
    delta: dict = {}
    for field_name, new_value in proposed.items():
        current = getattr(profile, field_name, None)
        if current is None and new_value is not None:
            first_time[field_name] = new_value
        elif current != new_value:
            delta[field_name] = {"old": current, "new": new_value}
        # else: no-op, drop
    return first_time, delta
