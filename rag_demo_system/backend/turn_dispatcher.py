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

from typing import Any, AsyncGenerator, Optional

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
    FireSMS,
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
        name=patches.get("name", profile.name),
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

# Classifier outputs that signal the user is actively asking for a
# calculation (or changing calc params). Step 5b's clarify-missing-fields
# branch gates on this so conversational turns ("меня зовут X", "алло",
# "подожди") don't get routed into a robotic "give me all 8 fields"
# reply when the user hasn't shown calc intent yet.
_CALC_INTENT_ACTIONS = frozenset({
    "calculate",
    "recalculate",
    "change_param",
    "clarify_client_type",
})


# MVP currency + subject policy for Физ лицо. Lifted verbatim from the
# legacy DirectTool preprocessing (app.py:2336-2395) so the apply_turn
# path produces the same OOR messages and the same USD→BYN behaviour.
_PHYS_REJECT_CURRENCIES = frozenset({"EUR", "RUB", "RUR", "CNY"})
_PHYS_ALLOWED_SUBJECTS = frozenset({"легковой автомобиль", "прочий транспорт"})


def _preflight_calc_policy(profile: ClientProfile) -> Optional[TurnAction]:
    """Apply the MVP currency + subject policy just before FireCalc fires.
    Returns a non-FireCalc TurnAction (FireOORMessage) when the profile
    is not calc-eligible; returns None when the profile is fine and the
    caller should proceed with FireCalc.

    Also performs the USD→BYN conversion as a profile mutation so
    `build_calc_params(profile)` ships BYN cost to the calculator and
    `render_calc_result` picks up the USD disclosure prefix.
    """
    from .profile_prompts import _get_usd_byn_rate  # lazy — reuses cache

    # NOTE: unlike legacy app.py:2345-2346 we do NOT clear `original_*`
    # here. Legacy recomputed per turn from `_direct_params` and left
    # `profile.currency` as-is, so a re-dispatch naturally re-converted.
    # apply_turn mutates profile.currency to BYN in the conversion
    # branch below, so on the second confirmation turn (re-calc with
    # same params) the profile already holds the BYN cost AND the
    # original USD stash — clearing here would drop the USD disclosure
    # prefix from the re-calc readback. Instead, the USD→BYN switch
    # triggered by a user-initiated change (step 1 apply-patches) is
    # responsible for clearing `original_*` — see ClientProfile.
    is_phys = profile.client_type == "Физическое лицо"

    # (1) Reject unsupported currencies for Физ лицо.
    if is_phys and profile.currency in _PHYS_REJECT_CURRENCIES:
        return FireOORMessage(message=(
            f"Для физических лиц сейчас поддерживаются расчёты в белорусских "
            f"рублях и в долларах. Валюта {profile.currency} временно не "
            f"поддерживается. Уточните, пожалуйста, стоимость в BYN или USD."
        ))

    # (2) Reject non-individual subjects for Физ лицо. Spec §6 E8 — only
    #     "Легковой автомобиль" / "Прочий транспорт" are lease-eligible
    #     for individuals; everything else is a ЮЛ-only line of business.
    subject_lower = (profile.subject or "").lower().strip()
    if is_phys and subject_lower and subject_lower not in _PHYS_ALLOWED_SUBJECTS:
        return FireOORMessage(message=(
            f"Для физических лиц доступен лизинг только легковых автомобилей "
            f"и прочего транспорта. {profile.subject} доступен для "
            f"юридических лиц и ИП."
        ))

    # (3) USD → BYN conversion for Физ лицо.
    if is_phys and profile.currency == "USD" and profile.cost is not None:
        rate = _get_usd_byn_rate()
        old_cost = float(profile.cost)
        new_cost = round(old_cost * rate, 2)
        profile.cost = new_cost
        profile.currency = "BYN"
        profile.original_cost = old_cost
        profile.original_currency = "USD"

    return None


def _is_calc_intent(classifier_output: ClassifierOutput) -> bool:
    """True when the classifier signals the turn is about a calculation
    (intent=TOOL) or carries an explicit calc-path action. Everything
    else (RAG/CONVERSATION without a calc action) falls through to
    FireLLMFallback so the LLM can respond naturally.
    """
    if classifier_output.intent == "TOOL":
        return True
    if classifier_output.action in _CALC_INTENT_ACTIONS:
        return True
    return False


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
    # apply the staged change via ClientProfile.apply_pending_change (which
    # preserves the prepaid_pct/prepaid_amount slot invariant + locked_fields
    # guard — Codex adversarial 2026-04-24 high #1 fix).
    if (
        profile.state == ProfileState.CHANGE_PENDING
        and classifier_output.is_confirmation
        and profile.pending_change
    ):
        changes = profile.pending_change.get("changes", {}) or {}
        # Snapshot the change keys BEFORE apply_pending_change clears
        # pending_change, so the USD→BYN stash logic below can inspect them.
        if not changes and "field" in profile.pending_change:
            changes = {profile.pending_change["field"]: {}}
        profile.apply_pending_change()
        # Clear USD→BYN disclosure stash when the user actively switched
        # currency (or cost) away from a prior USD capture. Without this
        # the next calc re-emits "Стоимость 80000 долларов..." even
        # though the user has explicitly moved to a different currency.
        if "currency" in changes or "cost" in changes:
            if profile.currency != "USD":
                profile.original_cost = None
                profile.original_currency = None
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

    # Bug 1 loop guard (live call 6ca0eaca, 2026-04-25): while
    # CHANGE_PENDING is staged, the classifier sometimes re-emits the
    # SAME change_field/change_value pair on the next turn. Without this
    # guard, step 4 below re-stages an identical pending_change and the
    # bot asks "Меняю срок на 48, всё верно?" forever. Drop already-
    # staged identical deltas so step 4 sees only NEW corrections (e.g.
    # user replying "нет, 60" still re-stages with 60).
    if (
        profile.state == ProfileState.CHANGE_PENDING
        and profile.pending_change
        and delta
    ):
        staged = profile.pending_change.get("changes", {}) or {}
        delta = {
            field: change for field, change in delta.items()
            if not (
                field in staged
                and staged[field].get("new") == change["new"]
            )
        }

    # Name capture (first-time only). Not a calc-grounded field so it
    # lives outside `_GROUNDED_FIELDS`; the classifier extracts name
    # from surface patterns ("меня зовут X") and we mirror the legacy
    # app.py:1414-1416 semantics: accept only when currently empty,
    # ignore stale re-emissions on later turns (prevents the garbled
    # STT on live ac0e35d6 turn 14 from overwriting "Евгений" with
    # "Боянс"). Snapshot then carries the captured name into the
    # FireLLMFallback prompt anchor.
    sa_name = (getattr(classifier_output, "name", None) or "").strip()
    if sa_name and not (profile.name or "").strip():
        profile.name = sa_name

    # Mixed-category clarify: when the classifier explicitly signals
    # `action == "clarify"` AND a client_type delta is present AND the
    # classifier's `subject` value was dropped by grounding (so the
    # utterance gestured at a subject category we couldn't identify),
    # ask instead of silently staging only half the change. Live call
    # f7e5aa1d turn 11 regression: user said "для юрлица коммерческие
    # автомобили" — classifier emitted client_type=Юр + subject=Грузовой,
    # subject grounding dropped "коммерческие" (no regex overlap with the
    # commercial-subject cue list), and step 4 staged only the client_type
    # half — calc then ran with Легковой + Юр (wrong subject).
    _utt_lower = (utterance or "").lower()
    _commercial_gesture_words = (
        "коммерч",
        "грузов",
        "спецтехник",
        "оборудовани",
    )
    _mentions_commercial = any(tok in _utt_lower for tok in _commercial_gesture_words)
    if (
        classifier_output.action == "clarify"
        and "client_type" in delta
        and classifier_output.subject is None
        and _mentions_commercial
    ):
        return EmitClarify(
            missing=["subject"],
            snapshot=build_snapshot(profile),
        )

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

    # STEP 5: apply first-time patches in place via the slot-aware helper
    # (preserves locked_fields + prepaid sibling-clear — Codex high #1).
    if first_time:
        profile.apply_additive_patches(first_time)

    # STEP 5a (E5 fix): profile just complete + COLLECTING + not
    # is_confirmation → deterministic readback. Classifier `intent`
    # label is IRRELEVANT at this branch — that's the whole point of
    # the E5 fix. On live call cc7fc318 Qwen labeled the "Аннуитетный
    # график" turn as CONVERSATION and the old gate skipped; now we
    # always emit.
    # Preflight runs BEFORE the readback so RUB/EUR + Физ лицо or
    # commercial-subject + Физ лицо produces FireOORMessage instead of
    # speaking unsupported params as "confirmed" (live regression
    # f7e5aa1d 2026-04-24: "стоимость 10000 RUB" in readback).
    # Side effect: USD→BYN conversion mutates profile.cost/currency so
    # the readback (happy path) speaks BYN with USD disclosure prefix.
    if (
        profile.is_complete_for_calc()
        and profile.state == ProfileState.COLLECTING
        and not classifier_output.is_confirmation
    ):
        policy_action = _preflight_calc_policy(profile)
        if policy_action is not None:
            return policy_action
        profile.state = ProfileState.READBACK_PENDING
        return EmitReadback(snapshot=build_snapshot(profile))

    # STEP 6 (E8a): CONFIRMED + is_confirmation + calc-ready → FireCalc.
    # Profile is already validated; build calc params from profile state.
    # Post-calc narration is rendered by execute_action's FireCalc
    # handler via render_calc_result(result) — LLM is never involved.
    #
    # Legacy DirectTool preprocessing (app.py:2336-2395) runs inline here:
    # 1. Unsupported-currency reject (EUR/RUB for Физ лицо) → FireOORMessage.
    # 2. Subject-restriction reject (non-individual subject for Физ лицо).
    # 3. USD→BYN conversion for Физ лицо (profile.cost becomes BYN,
    #    profile.original_cost / original_currency stash the USD figures
    #    so render_calc_result emits the disclosure prefix).
    # Without this preprocessing, calc is invoked with raw USD cost and
    # returns ok=False (no matching rates), producing "?" placeholder
    # output. Live regression observed on session ac0e35d6 (2026-04-24).
    if (
        profile.state == ProfileState.CONFIRMED
        and classifier_output.is_confirmation
        and profile.is_complete_for_calc()
    ):
        policy_action = _preflight_calc_policy(profile)
        if policy_action is not None:
            return policy_action
        return FireCalc(
            snapshot=build_snapshot(profile),
            calc_params=build_calc_params(profile),
        )

    # STEP 6b (Bug H, live call 504eace0 2026-04-26): SMS request →
    # FireSMS. apply_turn previously had no FireSMS in its vocabulary,
    # so SMS-intent turns dispatched FireLLMFallback and the orchestrator
    # `return`d at app.py:2048 before reaching the legacy SMS direct-fire
    # code — SMS could not fire under APPLY_TURN_ENABLED=1. The handler
    # in execute_action validates session has a successful calc and a
    # phone number; if not, it speaks a deterministic fallback.
    if classifier_output.action == "sms":
        return FireSMS(snapshot=build_snapshot(profile))

    # STEP 7: classifier-flagged out-of-range → FireOORMessage with a
    # deterministic text body. Fires BEFORE step 5b so an OOR during
    # a COLLECTING turn doesn't get masked by the clarify branch.
    if classifier_output.action == "invalid_param":
        return FireOORMessage(message=_default_oor_message())

    # STEP 5b: profile incomplete AND state is COLLECTING AND the user
    # has shown calculation intent → EmitClarify with missing-fields
    # list + snapshot anchor.
    #
    # The calc-intent gate is the fix for the 2026-04-24 live regression:
    # without it, apply_turn returns EmitClarify on every non-calc turn
    # (name capture, small talk, push-back like "подожди") because the
    # profile is always incomplete on a fresh session. Legacy path had
    # this gate implicitly — its clarify branch lived inside an
    # `if needs_tool:` block at app.py:~2020. Skipped in READBACK_PENDING
    # / CHANGE_PENDING / CONFIRMED since those have their own follow-up
    # paths (the user's response is interpreted as confirm/deny, not as
    # additional field-fill).
    if (
        not profile.is_complete_for_calc()
        and profile.state == ProfileState.COLLECTING
        and _is_calc_intent(classifier_output)
    ):
        return EmitClarify(
            missing=sorted(profile.missing_fields()),
            snapshot=build_snapshot(profile),
        )

    # STEP 8 (catch-all): freeform question, state-pending deny without
    # correction, or any non-structural turn → FireLLMFallback. Snapshot
    # included when any field is captured so LLM prompt has E7 anchor.
    # `name` counts as a captured field so the LLM knows who it's
    # talking to on later turns (prevents the "Здравствуйте, Боянс!"
    # regression from live ac0e35d6 turn 14).
    any_captured = (
        bool((profile.name or "").strip())
        or any(
            getattr(profile, f, None) is not None
            for f in (
                "client_type", "subject", "cost", "currency",
                "condition_new", "age_years", "prepaid_pct",
                "prepaid_amount", "term_months", "type_schedule",
            )
        )
    )
    return FireLLMFallback(
        user_utterance=utterance,
        rag_context=None,   # orchestrator populates pre-dispatch
        snapshot=build_snapshot(profile) if any_captured else None,
    )


def _default_oor_message() -> str:
    """Generic out-of-range message. Callers that need a more specific
    text (cost bounds, unsupported currency) should look up the OOR
    reason on the classifier output and dispatch to a more targeted
    string in execute_action. Kept generic here so apply_turn stays
    classifier-schema-agnostic."""
    return (
        "Извините, введённые параметры выходят за допустимый диапазон. "
        "Уточните, пожалуйста, стоимость и валюту."
    )


# ====================================================================
# execute_action — the IO-side of the dispatcher.
# ====================================================================
#
# Phase 3.D: async generator that consumes a TurnAction and drives TTS,
# the calculator, RAG retrieval, and the LLM stream as required by the
# action's variant. The critical structural invariants from spec §7.2:
#
#   #1 RAG overlap: rag_future is awaited ONLY on FireLLMFallback.
#   #2 Sentence queue: FireLLMFallback uses the existing streaming
#      path (ported verbatim in Task 18).
#   #3 Tool-call history: FireCalc appends to session.voice_session
#      (Task 20).
#   #4 Circuit breaker: 3 consecutive calc failures → FireOORMessage
#      routing (Task 20).
#   #5 Barge-in: every emit path checks session.interrupted at phrase
#      boundary (Task 19).
#   #6 Turn-id stale guard: orchestrator filters stale classifier
#      results BEFORE calling apply_turn; execute_action assumes
#      fresh input (Task 21).
#   #7 Deterministic FireCalc narration: render_calc_result output
#      goes straight to TTS, LLM is NEVER invoked.
#
# This Task 16 lands only the FireCalc handler (invariant #7).
# Subsequent tasks add the other variants.


async def execute_action(
    action: TurnAction,
    *,
    ws,
    session,
    backend,
    tts,
    calc,
    rag_future,
) -> AsyncGenerator[str, None]:
    """Dispatch a TurnAction to IO. Async generator of TTS chunks.

    The `ws`, `session`, `backend`, `tts`, `calc`, `rag_future`
    collaborators are injected so the handler stays testable in
    isolation with fakes. In production, the orchestrator supplies
    real WebSocket / voice_session / LLM backend / TTS sink / calc
    client / speculative-RAG future instances.
    """
    if isinstance(action, FireCalc):
        # Spec §7.2 invariants #3, #4, #7:
        #   - LLM bypassed: renderer drives the spoken text verbatim.
        #   - Tool-call history: success path appends to
        #     `session.tool_calls_this_turn` so SMS replay + prior-result
        #     reuse (app.py:2142-2155 / app.py:2332-2355) still see the
        #     record after the refactor.
        #   - Circuit breaker: repeated same-signature failures route
        #     to a deterministic OOR message on the third attempt
        #     instead of re-raising into the user's ear.
        from .profile_prompts import render_calc_result  # lazy import

        calc_sig = _calc_signature(action.calc_params)
        try:
            result = await calc.calculate(action.calc_params)
        except Exception:
            _bump_calc_failure(session, calc_sig)
            if _circuit_open(session):
                oor = _CALC_CIRCUIT_OOR
                await tts.say(oor)
                yield oor
                return
            raise

        # Attach the USD disclosure block to the result BEFORE rendering
        # so `render_calc_result`'s `conv_prefix` fires. Snapshot carries
        # the original USD figures when apply_turn step 6 performed the
        # USD→BYN conversion. Legacy parity with app.py:2423-2424.
        if (
            isinstance(result, dict)
            and action.snapshot.original_currency == "USD"
            and action.snapshot.original_cost is not None
        ):
            result.setdefault("currency_conversion", {
                "from": "USD",
                "to": "BYN",
                "amount_from": action.snapshot.original_cost,
                "amount_to": action.snapshot.cost,
                "rate": _infer_usd_byn_rate(action.snapshot),
                "rate_source": "apply_turn step 6",
            })

        if _result_ok(result):
            _reset_calc_failure(session, calc_sig)
            _append_tool_call(session, action.calc_params, result)
            spoken = render_calc_result(result)
        else:
            _bump_calc_failure(session, calc_sig)
            if _circuit_open(session):
                oor = _CALC_CIRCUIT_OOR
                await tts.say(oor)
                yield oor
                return
            # Don't render the "? USD" placeholder line on an API error.
            # Surface the calculator's error text (or a generic fallback)
            # so the user hears something actionable. Live regression
            # ac0e35d6 (2026-04-24): handler previously called
            # render_calc_result on ok=False results, producing "Аванс
            # 30%: ? USD..." placeholders when USD wasn't converted.
            spoken = _format_calc_error(result)

        await tts.say(spoken)
        yield spoken
        return

    if isinstance(action, EmitReadback):
        # Deterministic readback from captured snapshot. LLM bypassed.
        from .profile_prompts import build_readback_text
        spoken = build_readback_text(action.snapshot)
        await tts.say(spoken)
        yield spoken
        return

    if isinstance(action, EmitClarify):
        # Ask for the missing fields. build_clarification_prompt is the
        # existing Fix 1.5 / 1.11 / 1.13-aware renderer; it expects a
        # set[str] of field names plus a profile-shaped object. Snapshot
        # duck-types as profile (same field attributes).
        from .profile_prompts import build_clarification_prompt
        spoken = build_clarification_prompt(set(action.missing), action.snapshot)
        await tts.say(spoken)
        yield spoken
        return

    if isinstance(action, EmitChangeConfirm):
        # Confirm the staged changes. build_change_confirm_text consumes
        # a pending_change dict; we pass the multi-field shape that
        # apply_turn step 4 produced.
        from .profile_prompts import build_change_confirm_text
        spoken = build_change_confirm_text({"changes": action.changes})
        await tts.say(spoken)
        yield spoken
        return

    if isinstance(action, FireOORMessage):
        # Deterministic OOR text — no renderer needed, payload IS the text.
        await tts.say(action.message)
        yield action.message
        return

    if isinstance(action, FireSMS):
        # Bug H — send_sms tool dispatch under apply_turn (flag=1).
        # Validates that there is a successful calc result in session
        # history and a phone number on file; otherwise speaks a
        # deterministic fallback rather than firing a no-op tool call.
        import asyncio as _asyncio_sms
        from .tools import get_tool as _get_tool

        _calls = (
            list(getattr(session, "tool_calls_history", []) or [])
            + list(getattr(session, "tool_calls_this_turn", []) or [])
        )
        _last_calc = next(
            (
                tc for tc in reversed(_calls)
                if tc.get("tool") == "calculator"
                and (tc.get("result") or {}).get("ok")
            ),
            None,
        )
        _phone = getattr(session, "client_phone", None)
        if not _last_calc or not _phone:
            spoken = (
                "Извините, мне пока нечего отправить. "
                "Давайте сначала рассчитаем условия."
            )
            await tts.say(spoken)
            yield spoken
            return

        _calc_tool = _get_tool("calculator")
        _sms_tool = _get_tool("send_sms")
        try:
            _sms_body = _calc_tool.format_sms_body(_last_calc["result"])
        except Exception:
            _sms_body = ""
        if not _sms_tool or not _sms_body:
            spoken = (
                "Извините, не удалось подготовить СМС. Пожалуйста, попробуйте позже."
            )
            await tts.say(spoken)
            yield spoken
            return

        _sms_params = {"phone": _phone, "message": _sms_body}
        try:
            await ws.send_json({
                "type": "tool_call.start",
                "tool": "send_sms",
                "params": _sms_params,
            })
        except Exception:
            pass
        try:
            _sms_result = await _asyncio_sms.to_thread(
                _sms_tool.execute, _sms_params, {}
            )
            _ok = bool(_sms_result.get("ok", False)) if isinstance(_sms_result, dict) else False
        except Exception:
            _ok = False
            _sms_result = {"ok": False}
        try:
            await ws.send_json({
                "type": "tool_call.done",
                "tool": "send_sms",
                "ok": _ok,
            })
        except Exception:
            pass
        try:
            session.tool_calls_this_turn.append({
                "tool": "send_sms",
                "params": _sms_params,
                "result": _sms_result if isinstance(_sms_result, dict) else {"ok": _ok},
            })
        except Exception:
            pass

        if _ok:
            spoken = f"Отправила график платежей по СМС на номер {_phone}."
        else:
            spoken = (
                "Извините, не удалось отправить СМС. "
                "Попробуйте, пожалуйста, позже или уточните номер."
            )
        await tts.say(spoken)
        yield spoken
        return

    if isinstance(action, Noop):
        # Intentional silence: stale-turn discard, state-transition with
        # no follow-up, etc. No TTS, no LLM.
        return

    if isinstance(action, FireLLMFallback):
        # Spec §7.2 invariants #1 + #2 + #5: RAG overlap preserved by
        # awaiting the speculative future HERE and only HERE; sentence
        # queue + phrase-level TTS dispatch mirror app.py:2513-2778;
        # barge-in short-circuits at each sentence boundary.
        rag_context: Optional[str] = None
        if rag_future is not None:
            try:
                rag_context = await rag_future.result()
            except Exception:
                rag_context = None
        async for chunk in _stream_llm_to_tts(
            utterance=action.user_utterance,
            rag_context=rag_context or action.rag_context,
            snapshot=action.snapshot,
            backend=backend,
            tts=tts,
            session=session,
        ):
            yield chunk
        return

    return
    yield  # unreachable; keeps the function classified as async generator


# ---- Circuit breaker + tool-call history helpers (Task 20) ----

# Deterministic message for FireCalc's 3rd consecutive same-signature
# failure (spec §7.2 #4). Kept as a module constant so tests can
# substring-match without hardcoding the phrasing in two places.
_CALC_CIRCUIT_OOR = (
    "Извините, не могу посчитать, давайте перепроверим параметры."
)

# Threshold for circuit-open. Matches legacy semantics: on the third
# consecutive failure the handler stops trying and speaks the OOR line.
_CALC_FAILURE_THRESHOLD = 3


def _calc_signature(params) -> str:
    """Signature used to detect same-params-failing-again. Matches the
    legacy shape at app.py:2639 / app.py:2346.
    """
    if isinstance(params, dict):
        return str(sorted(params.items()))
    return ""


def _bump_calc_failure(session, sig: str) -> None:
    """Increment the consecutive-failure counter. Same-signature failure
    increments; different-signature failure resets to 1. Mirrors
    app.py:2641-2648 verbatim. No-op when `session` is None (unit-test
    shortcut)."""
    if session is None or not sig:
        return
    prev_sig = getattr(session, "last_calc_signature", "")
    if prev_sig == sig:
        session.consecutive_calc_failures = (
            getattr(session, "consecutive_calc_failures", 0) + 1
        )
    else:
        session.consecutive_calc_failures = 1
        session.last_calc_signature = sig


def _reset_calc_failure(session, sig: str) -> None:
    """Zero the consecutive-failure counter on a successful calc and
    record the new signature. Mirrors app.py:2643-2645."""
    if session is None:
        return
    session.consecutive_calc_failures = 0
    if sig:
        session.last_calc_signature = sig


def _circuit_open(session) -> bool:
    """True once consecutive same-signature failures hit the threshold.
    When open, the FireCalc handler yields the OOR line instead of
    propagating the error."""
    if session is None:
        return False
    return (
        getattr(session, "consecutive_calc_failures", 0)
        >= _CALC_FAILURE_THRESHOLD
    )


def _result_ok(result) -> bool:
    return isinstance(result, dict) and bool(result.get("ok", True))


def _infer_usd_byn_rate(snapshot: ProfileSnapshot) -> float:
    """Reconstruct the USD→BYN rate from the snapshot's original/final
    cost pair so `render_calc_result`'s disclosure prefix narrates a
    rate consistent with what apply_turn step 6 actually used.
    Falls back to the configured rate when the snapshot can't produce
    a positive ratio."""
    from .profile_prompts import _get_usd_byn_rate  # lazy — reuses cache
    if (
        snapshot.cost is not None
        and snapshot.original_cost
        and snapshot.original_cost > 0
    ):
        return snapshot.cost / snapshot.original_cost
    return _get_usd_byn_rate()


def _format_calc_error(result) -> str:
    """Build a user-facing error phrase from a calc result with ok=False.
    Calculator's `error` field already holds a Russian OOR message
    when the API rejects the params; passthrough. Otherwise use a
    generic fallback so the user doesn't hear "?" placeholders."""
    if isinstance(result, dict):
        err = result.get("error")
        if err:
            return str(err)
    return (
        "К сожалению, не удалось рассчитать по этим параметрам. "
        "Попробуем уточнить — стоимость, срок или аванс?"
    )


def _append_tool_call(session, params, result) -> None:
    """Append a calculator invocation to the voice_session's per-turn
    tool-call log. Orchestrator rolls this into `chat_history` at turn
    end (voice_session.reset_turn_state). Spec §7.2 #3."""
    if session is None:
        return
    log = getattr(session, "tool_calls_this_turn", None)
    if log is None:
        return
    log.append({
        "tool": "calculator",
        "params": params,
        "result": result,
        "ok": _result_ok(result),
    })


def _session_interrupted(session) -> bool:
    """Barge-in check. Tolerates `session=None` (used by unit tests that
    bypass the real voice_session)."""
    return session is not None and bool(getattr(session, "interrupted", False))


def _snapshot_anchor_lines(snap: ProfileSnapshot) -> list[str]:
    """Render the captured fields of a snapshot as prompt anchor lines.
    Fields with `None` are omitted so the LLM does not see placeholder
    slots it might try to fill.
    """
    pairs: list[tuple[str, object]] = [
        ("name", snap.name),
        ("client_type", snap.client_type),
        ("subject", snap.subject),
        ("cost", snap.cost),
        ("currency", snap.currency),
        ("original_cost", snap.original_cost),
        ("original_currency", snap.original_currency),
        ("condition_new", snap.condition_new),
        ("age_years", snap.age_years),
        ("prepaid_pct", snap.prepaid_pct),
        ("prepaid_amount", snap.prepaid_amount),
        ("term_months", snap.term_months),
        ("type_schedule", snap.type_schedule),
    ]
    return [f"- {k}: {v}" for k, v in pairs if v is not None]


def _build_fallback_messages(
    utterance: str,
    rag_context: Optional[str],
    snapshot: Optional[ProfileSnapshot],
) -> list[dict]:
    """Assemble the chat messages for the FireLLMFallback stream.

    Injects:
      - `snapshot` as anti-hallucination anchor (E7). LLM sees captured
        fields and is instructed not to re-ask them.
      - `rag_context` as KB grounding block.
    The production orchestrator prepends its own system prompt via the
    `backend.stream(system_prompt=...)` kwarg; this function deliberately
    avoids hardcoding the system prompt path so tests run without it.
    """
    anchor_block = ""
    if snapshot is not None:
        lines = _snapshot_anchor_lines(snapshot)
        if lines:
            anchor_block = (
                "Уже уточнено у клиента (НЕ переспрашивай эти поля):\n"
                + "\n".join(lines) + "\n\n"
            )
    kb_block = ""
    if rag_context:
        kb_block = (
            "Фрагменты из базы знаний (единственный источник фактов. "
            "Адреса, числа, ставки бери ТОЛЬКО отсюда):\n\n"
            + str(rag_context) + "\n\n"
        )
    user_content = f"{anchor_block}{kb_block}Сообщение клиента: {utterance}"
    return [{"role": "user", "content": user_content}]


async def _stream_llm_to_tts(
    *,
    utterance: str,
    rag_context: Optional[str],
    snapshot: Optional[ProfileSnapshot],
    backend,
    tts,
    session,
) -> AsyncGenerator[str, None]:
    """LLM stream → SentenceDetector → phrase-level TTS.

    Ported from `app.py:2513-2778`. Behavioural contract:
      - tokens flow through `backend.stream(messages=...)`;
      - `SentenceDetector.feed` yields full sentences at boundary;
      - each sentence is cleaned, shipped to `tts.say`, then yielded
        so the caller can accumulate full-turn text for chat history;
      - `session.interrupted` is checked at each sentence boundary —
        on flip, the loop exits without flushing the remainder
        (matches legacy phrase-boundary barge-in semantics);
      - on normal end, the trailing buffer is flushed as a final
        sentence.

    Failure isolation (Codex adversarial 2026-04-24 high #2): any
    exception from backend.stream(...) or the inner per-sentence emit is
    caught, logged, and replaced with a deterministic fallback sentence
    so the turn closes cleanly instead of tearing down the session.
    """
    from .sentence_detector import SentenceDetector
    from .text_utils import clean_answer

    messages = _build_fallback_messages(utterance, rag_context, snapshot)
    detector = SentenceDetector()
    _fallback = (
        "Извините, сейчас технические неполадки. Попробуйте, пожалуйста, "
        "повторить вопрос."
    )

    async def _emit(text: str) -> Optional[str]:
        cleaned = clean_answer(text)
        if not cleaned:
            return None
        await tts.say(cleaned)
        return cleaned

    try:
        async for token in backend.stream(messages=messages):
            if _session_interrupted(session):
                return
            if not token:
                continue
            for sentence in detector.feed(token):
                if _session_interrupted(session):
                    return
                out = await _emit(sentence)
                if out:
                    yield out
    except Exception as exc:  # noqa: BLE001 — graceful degradation per spec
        print(
            f"[_stream_llm_to_tts] backend.stream failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        if _session_interrupted(session):
            return
        try:
            out = await _emit(_fallback)
            if out:
                yield out
        except Exception as exc2:
            print(
                f"[_stream_llm_to_tts] fallback emit also failed: {type(exc2).__name__}: {exc2}",
                flush=True,
            )
        return

    # Success path: flush trailing partial sentence.
    if _session_interrupted(session):
        return
    remaining = detector.flush()
    if remaining:
        out = await _emit(remaining)
        if out:
            yield out
