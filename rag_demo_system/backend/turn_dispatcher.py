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

from typing import Optional

from .classifier_schema import ClassifierOutput
from .profile_state import (
    build_snapshot,
    partition_patches,
    derive_implied_flips,
    build_calc_params,
)
from .session import ClientProfile, ProfileState
from .turn_action import (
    TurnAction,
    EmitReadback,
    EmitClarify,
    EmitChangeConfirm,
    FireCalc,
    FireLLMFallback,
    FireOORMessage,
    Noop,
)


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
    """
    # Phase 3.C scaffolding — subsequent tasks implement each step.
    return Noop(reason="not_yet_implemented")
