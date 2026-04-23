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
