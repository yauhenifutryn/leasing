"""TurnAction ADT — the single output of `apply_turn`.

Every variant is a frozen dataclass so instances can be safely passed
through async boundaries and compared by value in tests.
`execute_action` consumes a TurnAction and dispatches to the appropriate
IO path.

Design invariants (spec §3.1, §7.2):
- `apply_turn` returns exactly one variant per turn.
- LLM is invoked ONLY when the returned variant is `FireLLMFallback`.
  This is the structural E5/E8 guarantee.
- Profile mutation happens in `apply_turn`'s body (Option C, mutable
  dataclass); actions carry only the immutable `ProfileSnapshot`
  projection for rendering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class ProfileSnapshot:
    """Read-only projection of ClientProfile used by action payloads
    as anti-hallucination anchor in LLM prompts (E7) and as input to
    deterministic renderers (E8).
    """
    client_type: Optional[str]
    subject: Optional[str]
    cost: Optional[float]
    currency: Optional[str]
    original_cost: Optional[float]
    original_currency: Optional[str]
    condition_new: Optional[int]
    age_years: Optional[int]
    prepaid_pct: Optional[float]
    prepaid_amount: Optional[float]
    term_months: Optional[int]
    type_schedule: Optional[str]
    # Captured client name — anchor for FireLLMFallback so the LLM does
    # not re-greet with a garbled STT name when the classifier spuriously
    # re-emits `name` on later turns (live ac0e35d6 turn 14 emitted
    # "Здравствуйте, Боянс!" on top of a profile already holding
    # "Евгений"). Defaults to None for the common no-name case and for
    # backward compat with ProfileSnapshot callers that don't thread it.
    name: Optional[str] = None


@dataclass(frozen=True)
class EmitReadback:
    """Emit the deterministic readback of the captured profile.

    Fires when the profile first becomes complete (§5 step 5a); the
    classifier's `intent` label is deliberately ignored — this is the
    structural E5 fix.
    """
    snapshot: ProfileSnapshot


@dataclass(frozen=True)
class EmitClarify:
    """Ask for one or more missing fields. `snapshot` is included as
    anti-hallucination anchor so the clarify prompt never asks for a
    field that is already captured (E7 fix)."""
    missing: list[str]
    snapshot: ProfileSnapshot


@dataclass(frozen=True)
class EmitChangeConfirm:
    """Confirmation gate for every user-initiated change to a captured
    field (explicit `change_field` pair OR top-level delta OR implied
    cross-field flip). Profile is NOT mutated when this action is
    emitted — mutation happens only if the user confirms on the next
    turn, routing through apply_turn step 1.
    """
    changes: dict               # {field: {"old": X, "new": Y}, ...}
    snapshot: ProfileSnapshot   # projected post-change snapshot


@dataclass(frozen=True)
class FireCalc:
    """Run the calculator with `calc_params` and ship the deterministic
    post-calc narration to TTS.

    Post-calc narration text is built by `execute_action` AFTER the
    calculator returns, via `profile_prompts.render_calc_result(result)`.
    The E8 invariant (LLM never paraphrases the post-calc narration)
    lives in `execute_action`'s FireCalc handler, NOT in this payload.
    """
    snapshot: ProfileSnapshot
    calc_params: dict


@dataclass(frozen=True)
class FireLLMFallback:
    """Freeform path — freeform leasing question, small-talk, anything
    not covered by the structured dispatch. The ONLY variant that
    causes `execute_action` to invoke the LLM backend. `rag_context`
    and `snapshot` are optional anchors populated by the orchestrator.
    """
    user_utterance: str
    rag_context: Optional[str] = None
    snapshot: Optional[ProfileSnapshot] = None


@dataclass(frozen=True)
class FireOORMessage:
    """Deterministic out-of-range response (cost bounds violation,
    unsupported currency, etc.). Text is baked into the action — no
    LLM paraphrase, no renderer indirection."""
    message: str


@dataclass(frozen=True)
class EmitCalcDetail:
    """Speak the FULL breakdown (выкупной / общая сумма / удорожание) of
    the most recent successful calculator result.

    Bug 25 (ANALYSIS.md §8): the default post-calc readback is now terse
    (cost / term / prepaid / monthly only) and the deeper figures are
    revealed only when the caller asks for them. The handler reads
    `session.tool_calls_history`, finds the latest calculator entry with
    `result.ok=True`, and ships
    `profile_prompts.render_calc_result(result, detailed=True)` to TTS.
    Falls back to a Russian "пока нечего расшифровать" line when no
    prior calc exists. LLM is NEVER invoked — the deterministic-numbers
    invariant (E8) extends to this path.
    """


@dataclass(frozen=True)
class EmitSMSOffer:
    """Speak the deterministic SMS-offer line and stamp last_offer="sms".

    Fired when the user declines the just-asked detail offer
    ("Хотите услышать подробный расчёт?" → "нет спасибо"). Without this
    pivot the dispatcher would fall to EndCall and the caller never gets
    the chance to take the SMS — losing the SMS conversion at the very
    last step of the funnel. Live transcript 2026-05-08.

    Handler speaks "Хорошо, отправить график платежей по СМС?" via TTS
    and sets `session.client_profile.last_offer = "sms"` so the next
    bare "Да" routes through STEP 5c-sms → FireSMS, and a follow-up
    decline routes through the normal EndCall path.
    """


@dataclass(frozen=True)
class FireSMS:
    """Send the last successful calculator result as SMS to the caller.

    Fires when the classifier emits action='sms' (explicit "по смс" /
    "отправь смс" keywords) on a turn where the session already has a
    successful calculator result. execute_action's handler looks up
    the most recent OK calc from session.tool_calls_history, formats
    the SMS body via calculator.format_sms_body, invokes send_sms
    with session.client_phone, and speaks a deterministic confirmation.

    Closes the apply_turn vocabulary gap: prior to this action, SMS-
    intent turns dispatched FireLLMFallback and never reached the
    SMS sender.
    """
    snapshot: ProfileSnapshot


@dataclass(frozen=True)
class Noop:
    """No emission this turn. Used when a state transition consumed
    the user's confirmation with no follow-up action (e.g.
    READBACK_PENDING → CONFIRMED on a profile that is NOT calc-ready)
    and for stale-turn discard. `reason` is a short tag for logs /
    telemetry.
    """
    reason: str = ""


@dataclass(frozen=True)
class EndCall:
    """Speak a farewell line and hang up the SIP leg.

    Bug 22 (2026-04-29 client review): bot has no deterministic way to
    end the call when the caller says "до свидания" / "пока" / "спасибо,
    всё". Without this action the dispatcher fell through to
    FireLLMFallback and the LLM occasionally narrated a goodbye but the
    SIP leg stayed open until the carrier timeout.

    The dispatcher emits EndCall ONLY when both are true:
      1. Classifier signals intent=END_CALL OR utterance matches the
         narrow goodbye list (`до свидания`, `пока`, `спасибо, всё`,
         `всего доброго`, `всё, спасибо`).
      2. No pending state is active (READBACK_PENDING / CHANGE_PENDING)
         and no half-completed change is sitting on the profile —
         hanging up mid-state would lose the user's progress.

    `farewell` is the line spoken before the hangup verb is sent.
    `reason` is a log tag (e.g. "user_goodbye", "consent_denied") so
    session_analyzer can distinguish caller-initiated end from system-
    initiated end. Voice path (execute_action) speaks `farewell` via
    TTS, awaits drain, then sends Jambonz `{"type": "disconnect"}`.
    Text path (Chat Widget Scope B) emits the farewell text and a
    `call_ended` SIP-monitor event without an actual SIP teardown.
    """
    farewell: str
    reason: str = "user_goodbye"


TurnAction = Union[
    EmitReadback,
    EmitClarify,
    EmitChangeConfirm,
    EmitCalcDetail,
    FireCalc,
    FireLLMFallback,
    FireOORMessage,
    FireSMS,
    EndCall,
    Noop,
]
