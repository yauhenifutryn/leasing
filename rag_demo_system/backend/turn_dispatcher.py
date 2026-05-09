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

import re
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
    EmitCalcDetail,
    EmitSMSOffer,
    EndCall,
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
        # Polish C 2026-04-27: type_schedule is the only enum field where
        # users describe BEHAVIOR ("равные платежи", "уменьшающиеся")
        # rather than NAMING the schedule. Verbatim cue grounding fights
        # the classifier prompt's semantic instructions. Trust the
        # classifier on intent=TOOL — same shape as the change-pair fix
        # below and the schema-layer fix in classifier_schema.py. Pydantic
        # Literal["0", "1"] still enforces enum validity. Other enum
        # fields keep verbatim grounding because users name them directly.
        if (
            field_name == "type_schedule"
            and classifier_output.intent == "TOOL"
        ):
            proposed[field_name] = value
            continue
        if value_grounded(field_name, value, utterance):
            proposed[field_name] = value

    # Explicit change_field / change_value pair. Treated identically to
    # a top-level field for routing purposes — partition_patches decides
    # whether it's a first-time capture or a delta.
    #
    # Polish A (live call eb3d0a3d 2026-04-27): the verbatim-utterance
    # grounding check that used to gate this assignment dropped legitimate
    # cross-turn-reasoning cases. Example: bot explained "линейный график
    # обычно дешевле, аннуитет ровнее"; user replied "Давай тот, что
    # дешевле." Classifier correctly resolved to change_field=type_schedule,
    # change_value="1" using its own prior turn — but value_grounded
    # returned False because "1"/"линей" isn't in the user verbatim, so
    # the pair was dropped and the dispatcher fell through to
    # FireLLMFallback (LLM then asked for type_schedule again).
    #
    # Universal fix: trust the (change_field, change_value) pair when the
    # classifier marked this turn as intent=TOOL — the structural signal
    # that the user is performing a tool/parameter action. Hallucination
    # guard remains for intent=CONVERSATION / intent=RAG: bare "ну хорошо"
    # with a phantom change_value still requires verbatim grounding so
    # phantom pairs from non-action turns get dropped (test
    # test_e6_ungrounded_change_value_drops_silently).
    #
    # Additional safety nets that apply in all cases:
    #   - classifier_schema step 3 (canonicalize_change_value) already
    #     enforces enum membership and type correctness; uncanonical
    #     values null both fields before reaching this function.
    #   - EmitChangeConfirm is a confirmation step, not a write — the user
    #     gets to reject before profile mutation.
    cf = classifier_output.change_field
    cv = classifier_output.change_value
    if cf and cv is not None:
        if classifier_output.intent == "TOOL" or value_grounded(cf, cv, utterance):
            proposed[cf] = cv

    # Bug 10 follow-up (live call ce1a0ad6 2026-05-03): when the user
    # says a slang-multiplied amount ("20 косарей" = 20000), the 4B
    # classifier sometimes captures the bare digit as cost=20 (or
    # change_value=20) and intent=TOOL skips value_grounded. Without a
    # corrective layer the bot reads back "Меняю стоимость на 20" and
    # the calc fires with cost=20 BYN — silent data loss.
    # extract_cost_from_utterance is slang-aware (косар/тонн/куск/штук
    # stems plus тысяч/миллион); when it returns a value that disagrees
    # with what the classifier proposed, prefer the extractor — the
    # range gate (10_000 ≤ cost ≤ 100_000_000) ensures it only fires on
    # genuine cost-shaped utterances.
    if "cost" in proposed:
        from . import utterance_grounding as ug
        try:
            slang_value = ug.extract_cost_from_utterance(utterance or "")
        except Exception:  # noqa: BLE001
            slang_value = None
        if slang_value is not None:
            try:
                cls_cost = int(proposed["cost"])
            except (TypeError, ValueError):
                cls_cost = None
            if cls_cost != slang_value:
                proposed["cost"] = slang_value

    return proposed


def _apply_utterance_fallbacks(
    profile: ClientProfile,
    proposed: dict[str, Any],
    utterance: str,
) -> None:
    """Belt-and-suspenders: when the small classifier omits a slot the
    user clearly named, run a deterministic utterance regex pass to
    fill the gap. Fires only when (a) the classifier did not propose
    the field, and (b) the profile field is currently empty — so a
    legitimate explicit-change turn cannot be overridden.

    Recovers Codex CP-3.6 P1: e.g. classifier returns intent=RAG on
    "Я думаю взять себе машину" and skips slot extraction; the regex
    fallback still seeds profile.subject = "Легковой автомобиль".

    Year-form fields (age_years, term_months) are intentionally
    excluded — the dedicated phase-aware year-form disambiguation in
    the caller is the sole writer for those slots.
    """
    from . import utterance_grounding as ug

    _FALLBACKS: tuple[tuple[str, Any], ...] = (
        ("subject", ug.extract_subject_from_utterance),
        ("client_type", ug.extract_client_type_from_utterance),
        ("condition_new", ug.extract_condition_new_from_utterance),
        ("currency", ug.extract_currency_from_utterance),
        ("prepaid_pct", ug.extract_prepaid_pct_from_utterance),
        ("type_schedule", ug.extract_type_schedule_from_utterance),
        # Issue 1 (live call 5fa0bb3d 2026-04-26): RU-numeral cost form
        # ("Сто десять тысяч долларов") — Qwen3-4B silently drops cost.
        # Reuses parse_ru_number; conservative range gate excludes leakage.
        ("cost", ug.extract_cost_from_utterance),
    )
    for field_name, extractor in _FALLBACKS:
        if field_name in proposed:
            continue
        if getattr(profile, field_name, None) is not None:
            continue
        try:
            value = extractor(utterance or "")
        except Exception:  # noqa: BLE001
            value = None
        if value is None or value == "":
            continue
        # currency fallback is suppressed when the same utterance carries
        # a cost — the classifier's combined cost+currency capture is
        # authoritative on those turns and the regex must not second-guess.
        if field_name == "currency" and "cost" in proposed:
            continue
        proposed[field_name] = value


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


# Subject policy for Физ лицо. The currency reject list (EUR/RUB/CNY)
# was removed 2026-05-09 in favour of FX drift to BYN — see
# `_preflight_calc_policy` step (3). Subject restriction kept: only
# "Легковой автомобиль" / "Прочий транспорт" are lease-eligible for
# individuals (Spec §6 E8).
_PHYS_ALLOWED_SUBJECTS = frozenset({"легковой автомобиль", "прочий транспорт"})


# Bug 2 (live call bf7a95a8 2026-04-26): clarify-gate meta-question
# detector. When the bot has emitted a clarify ("Подскажите тип графика
# — аннуитет или линейный") and the user replies with a meta-question
# instead of an answer ("А какой лучше?"), step 5b previously re-emitted
# the same prompt verbatim, looping until the user gave up. Generic
# across all clarify variants, not regex-per-prompt: detect the user is
# asking ABOUT the open question and route the turn to FireLLMFallback
# so the LLM can explain. The next user utterance returns to the
# clarify gate with the field actually answered.
_META_QUESTION_RE = re.compile(
    r"\b(?:"
    r"как(?:ой|ая|ое|ие)\s+(?:луч\w+|выгодн\w+|дешев\w+|правильн\w+|"
        r"разниц\w*|подойд\w+|подход\w+)|"
    # Bug 7 (live calls e6226e5d 15:08:23 + 19496277 15:29:01): "Что
    # такое аннуитет?" / "что такое линейный?" — meta-questions about
    # the open clarify, not parameter answers. Without this branch the
    # dispatcher re-emitted the same clarify prompt and looped the user.
    r"что\s+так(?:ое|ие)|"
    r"что\s+(?:луч\w+|выбра\w+|выгодн\w+|дешев\w+|посовет\w+|порекоменд\w+)|"
    r"в\s+ч[её]м\s+разниц\w*|"
    r"чем\s+отлич\w+|"
    r"не\s+поним\w+|"
    r"объясн\w+|"
    r"поясн\w+|"
    r"посовет\w+"
    r")",
    re.IGNORECASE,
)


# Bug 22 (2026-04-29 client review): the bot has no deterministic
# end-of-call action. Without this list, "до свидания" / "пока" turns
# fall to FireLLMFallback and the SIP leg stays open. The list is
# intentionally narrow — bare "хорошо" / "ладно" / "понятно" are
# acknowledgements, NOT goodbye signals, and stay in the SKIP_CLASSIFIER
# small-talk path. Anchored to ^ and tolerant of trailing punctuation
# / pleasantries so "До свидания, спасибо!" matches but "до свидания, а
# что насчёт..." does not (still ambiguous, let the LLM handle).
_GOODBYE_RE = re.compile(
    r"^\s*(?:"
    r"до\s+свидан\w+|"
    r"всего\s+доброго|всего\s+хорошего|"
    r"спасибо\s*,?\s*вс[её](?:\s+пока)?|"
    r"вс[её]\s*,?\s*спасибо|"
    r"больше\s+ничего\s+не\s+нужно|"
    r"пока\b|пока-пока|"
    r"всем\s+пока"
    r")[\s\.,!?]*$",
    re.IGNORECASE,
)


def _is_goodbye_utterance(utterance: str) -> bool:
    """True when the utterance is a clean caller-initiated goodbye.

    Conservative: only matches the canonical farewell forms. Mixed
    utterances ("до свидания, а ещё один вопрос") are deliberately not
    matched — those carry pending intent and should not trigger a
    hangup. The classifier's intent=END_CALL signal covers richer cases
    semantically; this regex is the deterministic fallback.
    """
    if not utterance:
        return False
    return bool(_GOODBYE_RE.match(utterance))


def _is_meta_question(utterance: str) -> bool:
    """True when the utterance is a meta-question about the open clarify
    rather than a concrete answer. Conservative: returns False on empty
    input, on plain affirmations, and on straightforward field answers
    (those have their own grounded cues that the classifier or fallback
    regex picks up before this gate is consulted)."""
    if not utterance:
        return False
    return bool(_META_QUESTION_RE.search(utterance))


def _preflight_calc_policy(profile: ClientProfile) -> Optional[TurnAction]:
    """Apply the MVP currency + subject policy just before FireCalc fires.
    Returns a non-FireCalc TurnAction (FireOORMessage) when the profile
    is not calc-eligible; returns None when the profile is fine and the
    caller should proceed with FireCalc.

    Also performs the USD→BYN conversion as a profile mutation so
    `build_calc_params(profile)` ships BYN cost to the calculator and
    `render_calc_result` picks up the USD disclosure prefix.
    """
    from .profile_prompts import _get_nbrb_rate  # lazy — reuses cache

    # NOTE: unlike legacy app.py:2345-2346 we do NOT clear `original_*`
    # here. Legacy recomputed per turn from `_direct_params` and left
    # `profile.currency` as-is, so a re-dispatch naturally re-converted.
    # apply_turn mutates profile.currency to BYN in the conversion
    # branch below, so on the second confirmation turn (re-calc with
    # same params) the profile already holds the BYN cost AND the
    # original FX stash — clearing here would drop the disclosure
    # prefix from the re-calc readback. Instead, the FX switch
    # triggered by a user-initiated change (step 1 apply-patches) is
    # responsible for clearing `original_*` — see ClientProfile.
    is_phys = profile.client_type == "Физическое лицо"

    # (2) Reject non-individual subjects for Физ лицо. Spec §6 E8 — only
    #     "Легковой автомобиль" / "Прочий транспорт" are lease-eligible
    #     for individuals; everything else is a ЮЛ-only line of business.
    # Bug M (live call 8741ad68 2026-04-26): the prior message was purely
    # informational ("X недоступен для физлиц") and left the user to
    # figure out the next step alone. Make it actionable: list the two
    # concrete branches ("выбрать легковой / сменить статус на юрлицо")
    # so the next user utterance maps cleanly to a single field change.
    subject_lower = (profile.subject or "").lower().strip()
    if is_phys and subject_lower and subject_lower not in _PHYS_ALLOWED_SUBJECTS:
        return FireOORMessage(message=(
            f"{profile.subject} для физических лиц не финансируется. "
            f"Хотите выбрать легковой автомобиль, или сменить статус "
            f"на юридическое лицо?"
        ))

    # (3) FX → BYN drift for Физ лицо. Any non-BYN currency converts to
    # BYN at the per-currency NBRB rate, including USD (legacy path),
    # EUR, RUB, and any other code NBRB publishes (CNY, GBP, PLN…).
    # User intent (2026-05-09): keep the conversation moving rather
    # than rejecting EUR/RUB with a "specify in BYN or USD" loop —
    # individuals always get BYN figures, the original currency is
    # stashed for the disclosure prefix in render_calc_result.
    if (
        is_phys
        and profile.currency
        and profile.currency != "BYN"
        and profile.cost is not None
    ):
        rate = _get_nbrb_rate(profile.currency)
        old_cost = float(profile.cost)
        new_cost = round(old_cost * rate, 2)
        old_currency = profile.currency
        profile.cost = new_cost
        profile.currency = "BYN"
        profile.original_cost = old_cost
        profile.original_currency = old_currency

    # (4) Bug 27 (live call 5746bfec 2026-05-03 18:42:14): user asked
    #     for prepaid 60%; calc API returned FAIL. The KB section
    #     `minimum-advance-and-no-advance` documents the rule (max 40%)
    #     AND the workaround (the rest goes as a первый платёж по
    #     графику). Surface both inline so the caller has a concrete
    #     next step instead of a generic OOR. The helper is shaped to
    #     absorb future range checks (term_months, age_years, cost) so
    #     this dispatcher gate stays a one-liner.
    #
    # Bug 27 follow-up (live call b5d70d6a 2026-05-03 19:49): when the
    # user said "Удобно" to accept the workaround, the bot LLM-narrated
    # agreement but calc never re-fired. Stamp last_offer here so the
    # next is_confirmation turn auto-applies prepaid_pct=40 (the "40%
    # авансом" half of the workaround that the calculator API can
    # actually accept; the "20% первым платежом" half is a verbal
    # arrangement with the client, outside the calc params).
    from .preflight_calc import validate_calc_inputs as _validate
    msg = _validate(profile)
    if msg:
        profile.last_offer = "prepaid_workaround_40"
        return FireOORMessage(message=msg)

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
    # Capture pre-turn state ONCE, before any iteration mutates it. This
    # is the snapshot Step 6's FireCalc gate uses to distinguish "user
    # entered the turn in a pre-calc state (READBACK_PENDING / CHANGE_
    # PENDING) and a same-turn transition advanced us to CONFIRMED" from
    # "state has been CONFIRMED since a prior turn (calc already fired)".
    # Capturing here (not inside _dispatch_once) is critical because the
    # redispatch loop's second iteration sees state=CONFIRMED — the
    # transition already happened in iteration #1.
    pre_turn_state = profile.state
    action: TurnAction = Noop(reason="uninitialized")
    for _ in range(2):
        action = _dispatch_once(
            profile, classifier_output, utterance,
            pre_turn_state=pre_turn_state,
        )
        if isinstance(action, Noop) and action.reason in _REDISPATCH_REASONS:
            continue
        break
    return action


def _dispatch_once(
    profile: ClientProfile,
    classifier_output: ClassifierOutput,
    utterance: str,
    *,
    pre_turn_state: ProfileState | None = None,
) -> TurnAction:
    """Single-iteration body of the apply_turn dispatch. Returns a
    Noop with reason ∈ _REDISPATCH_REASONS when the caller's loop
    should re-enter; otherwise returns a terminal TurnAction.

    `pre_turn_state` is the profile.state snapshot from BEFORE the
    redispatch loop. Used by step 6's FireCalc gate (Issue 4 fix).
    Defaults to current profile.state when called directly (test
    fixtures, isolated invocations).
    """
    if pre_turn_state is None:
        pre_turn_state = profile.state

    # STEP 0 (Bug 22, 2026-04-29 client review): EndCall — caller-
    # initiated goodbye. Placed at the top of dispatch so the user's
    # signal to leave wins over field-collection logic (otherwise STEP
    # 5b clarify on an incomplete profile would emit a clarify prompt
    # for a caller who's literally saying goodbye).
    #
    # Two signals trigger:
    #   (a) classifier emits intent=END_CALL semantically (richer
    #       phrasings like "Хорошо, спасибо, на этом всё, до свидания").
    #   (b) FAST-PATH for narrow goodbye forms via _is_goodbye_utterance.
    #       Existing SKIP_CLASSIFIER list at app.py:1010 routes those
    #       through with intent=CONVERSATION; the regex fallback here
    #       lifts them without a classifier round-trip.
    #
    # Suppressed when:
    #   - pre_turn_state in {READBACK_PENDING, CHANGE_PENDING}: the user
    #     is mid-state, hanging up loses progress. Let STEP 1/2 handle
    #     the turn first.
    #   - classifier_output.change_field is set: a staged change is
    #     in flight ("поменяй срок на 48, до свидания"); apply the
    #     change first, the user can goodbye next turn.
    _is_goodbye_intent = (
        classifier_output.intent == "END_CALL"
        or _is_goodbye_utterance(utterance or "")
    )
    _pending_state = pre_turn_state in (
        ProfileState.READBACK_PENDING,
        ProfileState.CHANGE_PENDING,
    )
    _change_in_flight = bool(classifier_output.change_field)
    if _is_goodbye_intent and not _pending_state and not _change_in_flight:
        # Live transcript 2026-05-08: after the terse calc render the bot
        # asked "Хотите услышать подробный расчёт?" (last_offer="detail").
        # User replied "нет спасибо" — classifier emitted intent=END_CALL
        # and the bot hung up without ever offering SMS. The detail
        # decline is NOT a goodbye, it's a "skip the breakdown, what's
        # next?". Pivot to the SMS offer so the funnel completes; the
        # next turn's decline (now last_offer="sms") becomes the real
        # EndCall.
        if (
            getattr(profile, "last_offer", None) == "detail"
            and pre_turn_state == ProfileState.CONFIRMED
        ):
            profile.last_offer = "sms"
            return EmitSMSOffer()
        # Short, feminine-grammar farewell. No AI-disclosure repeat.
        farewell = "Хорошего дня! Обращайтесь, если возникнут вопросы."
        return EndCall(farewell=farewell, reason="user_goodbye")

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
        # Codex adversarial 2026-04-26 (CP-3.6 high #1): honour the boolean
        # return. apply_pending_change() returns False when every field in
        # the staged payload is locked or unknown, and preserves
        # pending_change for retry. Advancing to CONFIRMED on a False
        # return would tell the user "applied" while the calculator runs
        # on the previous, unmodified terms. Re-emit the change-confirm
        # prompt instead so the state stays in CHANGE_PENDING and the
        # user is asked again.
        applied_ok = profile.apply_pending_change()
        if not applied_ok:
            print(
                f"[apply_turn] CHANGE_PENDING confirm rejected: "
                f"apply_pending_change returned False (locked/unknown fields). "
                f"Re-emitting change-confirm.",
                flush=True,
            )
            return EmitChangeConfirm(
                changes=(profile.pending_change or {}).get("changes", {}) or {},
                snapshot=build_snapshot(profile),
            )
        # Clear USD→BYN disclosure stash when the user actively switched
        # currency (or cost) away from a prior USD capture. Without this
        # the next calc re-emits "Стоимость 80000 долларов..." even
        # though the user has explicitly moved to a different currency.
        if "currency" in changes or "cost" in changes:
            if profile.currency != "USD":
                profile.original_cost = None
                profile.original_currency = None
        profile.state = ProfileState.CONFIRMED
        # Bug J (live call 69941ab4 2026-04-26): clear the stale post-calc
        # SMS offer flag. Without this, the redispatch into STEP 5c would
        # match (last_offer="sms" + is_confirmation + no change_field) and
        # fire FireSMS instead of the FireCalc that step 6 should produce
        # for the just-applied change. The user's "Да" is consuming the
        # change-confirm, NOT the prior SMS offer.
        profile.last_offer = None
        return Noop(reason="redispatch_change")

    # STEP 1.5 (Bug 25, ANALYSIS.md §8): detail_request — caller asked
    # for the full calc breakdown after the terse readback. Routed
    # through EmitCalcDetail so the deterministic renderer speaks
    # выкупной / общая сумма / удорожание without LLM paraphrase.
    #
    # detail_request is a freeform query about the prior result —
    # semantically incompatible with mutating state. The guard MUST
    # defer to any classifier signal that proposes a parameter change
    # so a "поменяй срок на 12, подробнее" turn (live regression call
    # 356221c6 turn 11 2026-05-03) routes to the change-confirm path
    # instead of replaying the prior calc detail. We block on:
    #   - is_confirmation / is_stop_request (turn-shape semantics)
    #   - change_field set (explicit change pair)
    #   - any top-level _GROUNDED_FIELDS slot non-None (parameter
    #     mutation in flight, e.g. cost=50000 piggybacked on detail)
    #   - mutating actions (change_param / calculate / recalculate /
    #     sms / confirm); pure clarify / invalid_param / None do not
    #     block — they don't progress state mutation.
    if getattr(classifier_output, "detail_request", False):
        _blocks_detail = (
            classifier_output.is_confirmation
            or classifier_output.is_stop_request
            or classifier_output.change_field is not None
            or classifier_output.action in (
                "change_param", "calculate", "recalculate",
                "sms", "confirm",
            )
            or any(
                getattr(classifier_output, _f, None) is not None
                for _f in _GROUNDED_FIELDS
            )
        )
        if not _blocks_detail:
            return EmitCalcDetail()

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
    # Codex CP-3.6 P1: if the small classifier dropped a slot the user
    # clearly named, the deterministic regex fallback fills the gap so
    # downstream first_time / clarify routing sees the captured field.
    # Kept on all intents — it is a safety net for the "classifier
    # mis-labeled an obvious TOOL utterance as RAG" failure mode
    # (e.g. "Я думаю взять себе машину" → classifier=RAG, fallback
    # correctly recovers subject=Легковой автомобиль).
    _apply_utterance_fallbacks(profile, proposed, utterance)

    # Year-form disambiguation (Bugs Q + S, 2026-04-26).
    # Russian "X лет/года/год" carries either age-of-vehicle ("Сколько
    # лет вашему транспорту?" → "Два года") or term ("на три года",
    # "три года срок"). Surface form is identical; meaning depends on
    # what the bot is currently asking. Conversation state is the
    # reliable signal — utterance regex is not.
    #
    # Two states:
    #   (A) Age-collection: condition_new=0 AND age_years is None.
    #       Bot just asked age. Year-form grounds age_years; any
    #       classifier-emitted term_months from the same utterance is
    #       misattribution and gets dropped.
    #   (B) Term-collection: term_months is None AND we're past age
    #       (age_years filled OR condition_new=1). Year-form grounds
    #       term_months; classifier-emitted age_years on this utterance
    #       gets dropped.
    try:
        from .utterance_grounding import extract_age_years_from_utterance
        _utt_year = extract_age_years_from_utterance(utterance or "")
    except Exception:  # noqa: BLE001
        _utt_year = None
    if _utt_year is not None:
        _age_phase = (
            getattr(profile, "condition_new", None) == 0
            and getattr(profile, "age_years", None) is None
        )
        _term_phase = (
            getattr(profile, "term_months", None) is None
            and (
                getattr(profile, "age_years", None) is not None
                or getattr(profile, "condition_new", None) == 1
            )
        )
        if _age_phase:
            if "age_years" not in proposed:
                proposed["age_years"] = _utt_year
            # Drop misattributed term_months whose value matches _utt_year*12.
            _tm = proposed.get("term_months")
            if (
                isinstance(_tm, int)
                and _tm > 0
                and _tm % 12 == 0
                and _tm // 12 == _utt_year
            ):
                proposed.pop("term_months", None)
        elif _term_phase:
            if "term_months" not in proposed:
                proposed["term_months"] = _utt_year * 12
            _ag = proposed.get("age_years")
            if isinstance(_ag, int) and _ag == _utt_year:
                proposed.pop("age_years", None)

    proposed.update(derive_implied_flips(profile, proposed))

    # Bug 1 (live call bf7a95a8 2026-04-26): flip the subject-grounded
    # flag on any turn where subject lands in `proposed` — that means it
    # passed `_grounded_proposed_patches` (utterance cue match) or
    # `_apply_utterance_fallbacks` (extract_subject_from_utterance regex).
    # Either way the user just produced explicit evidence, so the gate
    # in `ClientProfile.missing_fields()` should stop treating subject
    # as silently inferred. Fires BEFORE partition_patches so a no-op
    # re-mention ("да, машину") still flips the flag even though it
    # produces no first_time / delta entry.
    if proposed.get("subject"):
        profile.subject_user_grounded = True

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
    # Bug O (live call 42b6e6bf 2026-04-26): when the user addresses
    # the bot ("Привет, Ксения, подскажи..."), the classifier extracts
    # the bot's persona name. The accept guard below rejects the bot
    # name unless the utterance shows an explicit self-introduction.
    # Real clients named Ксения/Ксюша still work via "я Ксения" /
    # "меня зовут Ксения" patterns.
    import re as _re_name
    _bot_names = ("ксения", "ксюша")

    def _name_acceptable(candidate: str) -> bool:
        if candidate.lower() not in _bot_names:
            return True
        _utt_lower = (utterance or "").lower()
        return bool(_re_name.search(
            r"\b(я|меня\s+зовут|меня\s+звать|зовите\s+меня|это\s+я)"
            r"\s+(ксения|ксюша)\b",
            _utt_lower,
        ))

    sa_name_change = (
        getattr(classifier_output, "name_change", None) or ""
    ).strip()
    sa_name = (getattr(classifier_output, "name", None) or "").strip()
    # Bug 8 (ANALYSIS.md §2): name_change is the explicit correction
    # signal — overwrite profile.name even when one is already set. The
    # plain `name` field keeps first-time-only semantics (legacy
    # app.py:1414-1416 / live ac0e35d6 turn 14: garbled STT must not
    # silently rewrite a real capture). Bot-name guard applies to both.
    if sa_name_change and _name_acceptable(sa_name_change):
        profile.name = sa_name_change
    elif (
        sa_name
        and not (profile.name or "").strip()
        and _name_acceptable(sa_name)
    ):
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

    # Issues 2 + 3 (live call 5fa0bb3d 2026-04-26): the user signaled
    # change intent (action="change_param" or change_field set) but no
    # grounded value landed. Two regressions to close:
    #   (Fix 2) last_offer="sms" must be cleared so a follow-up bare-Да
    #     can't slot into step 5c → FireSMS. Step 4 already clears
    #     last_offer when delta exists; this widens the rule to the
    #     no-grounded-value case.
    #   (Fix 3) when change_field is set but didn't ground into proposed,
    #     emit a deterministic clarify keyed by change_field instead of
    #     falling through to FireLLMFallback. Without this, the LLM
    #     hallucinates a change-confirm and the user confirms a change
    #     that was never staged.
    # prepaid_pct / prepaid_amount route to the canonical "prepaid" key
    # so build_clarification_prompt's existing branch fires.
    _change_intent = (
        classifier_output.action == "change_param"
        or classifier_output.change_field is not None
    )
    if _change_intent:
        profile.last_offer = None
    if (
        classifier_output.action == "change_param"
        and classifier_output.change_field is not None
        and classifier_output.change_field not in proposed
    ):
        cf = classifier_output.change_field
        missing_label = (
            "prepaid" if cf in ("prepaid_pct", "prepaid_amount") else cf
        )
        return EmitClarify(
            missing=[missing_label],
            snapshot=build_snapshot(profile),
        )

    # STEP 4 (E6 fix): any delta on a captured field → EmitChangeConfirm.
    # Covers explicit change_field pairs AND top-level field flips on
    # captured fields (E7b uniformity) AND implied cross-field flips
    # (derive_implied_flips rule table). Profile fields stay untouched;
    # mutation happens only on next-turn confirm (step 1).
    if delta:
        # Failure B (call 0897572e 2026-04-28): if a previous change-confirm
        # is still pending and the user issues a NEW change before confirming,
        # MERGE the new delta into the existing pending instead of replacing.
        # Without this, the prior staged field is silently lost on confirm.
        existing_changes: dict = {}
        if (
            profile.state == ProfileState.CHANGE_PENDING
            and profile.pending_change
        ):
            existing_changes = dict(profile.pending_change.get("changes") or {})
        merged_delta = {**existing_changes, **delta}  # new delta wins on key collision
        projected_patches = dict(first_time)
        for field_name, change in merged_delta.items():
            projected_patches[field_name] = change["new"]
        profile.state = ProfileState.CHANGE_PENDING
        profile.pending_change = {"changes": merged_delta}
        # Bug J: starting a change cycle supersedes any pending post-calc
        # SMS offer. Without clearing here, the next "Да" (which confirms
        # the change) would slot into STEP 5c → FireSMS instead of going
        # through STEP 1's apply-pending-change → STEP 6 FireCalc path.
        profile.last_offer = None
        return EmitChangeConfirm(
            changes=merged_delta,
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
    # Bug L (live call 2635bb30 2026-04-26 + codex-rescue): the original
    # `not is_confirmation` gate skipped readback even when the user was
    # supplying brand-new slots in the same breath as an affirmation
    # ("36 мес, 38%, аннуитет, пожалуйста"). Result: state stayed
    # COLLECTING, fell through to FireLLMFallback, the LLM narrated a
    # fake readback question, and the deterministic readback only fired
    # one or two turns later. Refined gate: skip readback only when this
    # is a BARE confirmation (no new slots captured this turn). When
    # first_time is non-empty, the user can't be confirming a readback
    # they haven't seen — emit it.
    if (
        profile.is_complete_for_calc()
        and profile.state == ProfileState.COLLECTING
        and not (classifier_output.is_confirmation and not first_time)
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
    # Legacy DirectTool preprocessing (app.py:2336-2395) runs inline here.
    # After the 2026-05-09 currency drift change:
    # 1. Subject-restriction reject (non-individual subject for Физ лицо)
    #    → FireOORMessage. The unsupported-currency reject was removed.
    # 2. FX → BYN drift for Физ лицо: any non-BYN currency (USD, EUR,
    #    RUB, CNY, …) converts to BYN at the per-currency NBRB rate;
    #    profile.cost becomes BYN, profile.original_cost +
    #    original_currency stash the source figures so render_calc_result
    #    can emit the disclosure prefix.
    # Without this preprocessing, calc is invoked with raw USD cost and
    # returns ok=False (no matching rates), producing "?" placeholder
    # output. Live regression observed on session ac0e35d6 (2026-04-24).
    # STEP 5c (live call cdbcf56b 2026-04-26): bare "Да" after the
    # post-calc SMS-or-change offer means "yes, send the SMS". The
    # FireCalc handler stamps profile.last_offer = "sms" right after
    # speaking the offer; the next turn's classifier emits is_confirmation=
    # True with no change_field. Without this branch the same input would
    # match STEP 6 below (CONFIRMED + is_confirmation + complete) and
    # re-fire FireCalc with identical params instead of sending SMS.
    # Issue 5 (live call d5174335 2026-04-27): SMS fired on "Да." that
    # was actually confirming a change-confirm, not the prior post-calc
    # SMS offer. Bug J's last_offer=None clear (in step 1 + step 4) is
    # the primary defense, but a barge-in-induced state corruption made
    # last_offer stay "sms" through the change-confirm turn. Defensive
    # gate: SMS only fires when this turn entered with state CONFIRMED
    # (clean post-calc), NOT CHANGE_PENDING (just-applied change). Ensures
    # SMS can never piggyback off a change-confirm "Да." regardless of
    # what last_offer was. Pure post-calc SMS-confirm flow keeps working
    # because pre_turn_state is CONFIRMED there.
    # Bug 8 follow-up (2026-05-07): "detail" branch of step 5c.
    # First post-calc turn stamps last_offer="detail" because the terse
    # render now asks "Хотите услышать подробный расчёт?" (not the old
    # detail-or-SMS combined offer). A bare "Да" / "давай" must route to
    # the detail render, NOT to FireSMS — otherwise "давай" sends an SMS
    # for a calc the user hasn't seen the breakdown for. Mirrors the
    # post-CONFIRMED guard so a stale "detail" offer can't piggyback off
    # a change-confirm turn.
    if (
        getattr(profile, "last_offer", None) == "detail"
        and classifier_output.is_confirmation
        and not classifier_output.change_field
        and classifier_output.action != "change_param"
        and pre_turn_state == ProfileState.CONFIRMED
    ):
        profile.last_offer = None
        return EmitCalcDetail()

    if (
        getattr(profile, "last_offer", None) == "sms"
        and classifier_output.is_confirmation
        and not classifier_output.change_field
        and classifier_output.action != "change_param"
        and pre_turn_state == ProfileState.CONFIRMED
    ):
        profile.last_offer = None
        return FireSMS(snapshot=build_snapshot(profile))

    # STEP 5d (Bug 27 follow-up, live call b5d70d6a 2026-05-03 19:49):
    # the prior turn fired the prepaid-over-40 workaround OOR
    # ("максимум 40 процентов... 40% авансом + 20% первым платежом"
    # — see backend/preflight_calc.py). When the user accepts ("Удобно"
    # / "Да"), auto-apply prepaid_pct = 40 and fire calc on this turn,
    # rather than waiting for the user to manually re-state the
    # parameter. The "20% первым платежом" half is informational only
    # — calculator API doesn't accept it as a parameter — the bot's
    # acceptance message ("оформим 40% авансом, 20% первым платежом")
    # carries the verbal arrangement.
    if (
        getattr(profile, "last_offer", None) == "prepaid_workaround_40"
        and classifier_output.is_confirmation
        and not classifier_output.change_field
        and classifier_output.action != "change_param"
    ):
        profile.last_offer = None
        profile.prepaid_pct = 40.0
        # Re-run preflight (covers a pathological profile where another
        # OOR field would now surface; in normal flow this returns None).
        policy_action = _preflight_calc_policy(profile)
        if policy_action is not None:
            return policy_action
        return FireCalc(
            snapshot=build_snapshot(profile),
            calc_params=build_calc_params(profile),
        )

    # Issue 4 (live call 2ab41112 2026-04-26): post-SMS, the user said
    # "Спасибо. Кто владелец Вашей компании?" — the small classifier
    # emitted is_confirmation=True (cued by "Спасибо") with intent=RAG.
    # The bare CONFIRMED+is_confirmation gate matched and re-fired calc
    # with stale params instead of routing the RAG question to the LLM.
    # Universal fix: only fire calc when EITHER (a) we entered the turn
    # in READBACK_PENDING (the canonical readback→confirm flow that
    # transitions to CONFIRMED inside step 2 of THIS turn), or (b) the
    # classifier explicitly asked for calc / recalc. Bare is_confirmation
    # in a sticky CONFIRMED state isn't enough signal — "спасибо" / "ок"
    # / "хорошо" are routinely classified as is_confirmation=true.
    # READBACK_PENDING (canonical readback→confirm flow, transitions in
    # step 2) and CHANGE_PENDING (change-confirm flow, transitions in
    # step 1 via redispatch) are the two legitimate pre-calc states.
    # Both advance to CONFIRMED inside this turn.
    _came_from_pre_calc_state = pre_turn_state in (
        ProfileState.READBACK_PENDING,
        ProfileState.CHANGE_PENDING,
    )
    _explicit_calc_intent = classifier_output.action in ("calculate", "recalculate")
    if (
        profile.state == ProfileState.CONFIRMED
        and classifier_output.is_confirmation
        and profile.is_complete_for_calc()
        and (_came_from_pre_calc_state or _explicit_calc_intent)
    ):
        policy_action = _preflight_calc_policy(profile)
        if policy_action is not None:
            return policy_action
        return FireCalc(
            snapshot=build_snapshot(profile),
            calc_params=build_calc_params(profile),
        )

    # STEP 6b (Bug H, live call 504eace0 2026-04-26): SMS request →
    # FireSMS. Prior to this action, apply_turn dispatched FireLLMFallback
    # for SMS-intent turns and SMS never fired. The handler in
    # execute_action validates session has a successful calc and a phone
    # number; if not, it speaks a deterministic fallback.
    # Bug 6 (live call e6226e5d 2026-04-29 Stanislav 15:16:11): clear
    # last_offer here too, mirroring step 5c. Without this, an explicit
    # "Отправьте смс" left last_offer="sms" sticky across subsequent
    # unrelated LLM-narrated turns; a later bare "Да" then matched step
    # 5c again and re-fired SMS.
    if classifier_output.action == "sms":
        profile.last_offer = None
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
    # Bug P (live call 42b6e6bf 2026-04-26): step 5b previously only fired
    # on COLLECTING. After a successful change-confirm cycle (e.g. user
    # changed subject from Грузовой → Легковой), state transitions to
    # CONFIRMED. If a downstream slot is still missing (e.g. type_schedule
    # was never captured because subject change reset the implicit
    # confirmation chain), apply_turn previously fell through to
    # FireLLMFallback — bot LLM-narrated about params instead of asking
    # the deterministic "аннуитет или линейный?" prompt. Allow CONFIRMED
    # + incomplete to also enter clarify so the missing-slot ask fires
    # deterministically.
    # Bug R (live call 730d3aab 2026-04-26): a bare affirmation like
    # "Давай" classified as intent=CONVERSATION/RAG fails _is_calc_intent
    # but the user is clearly mid-leasing-flow. When profile has any
    # core field captured (subject/client_type/cost), treat the user as
    # being in calc context and emit clarify for the missing slot. This
    # prevents the LLM-fallback from inventing defaults like "аннуитет"
    # for type_schedule that the user never specified.
    _has_any_core_field = any(
        getattr(profile, f, None) is not None
        for f in ("client_type", "subject", "cost", "currency",
                  "condition_new", "term_months", "prepaid_pct",
                  "prepaid_amount", "type_schedule")
    )
    # Live regression 5e6f4c48 (2026-04-26): Bug R's _has_any_core_field
    # gate over-fires when the classifier (or a permissive grounding
    # cue) silently captures a field on a RAG turn — every subsequent
    # RAG question loops the clarify prompt instead of drifting to LLM.
    # When the classifier explicitly labels the turn as RAG, the user is
    # asking about company info, not progressing the leasing flow; skip
    # step 5b and let the LLM/RAG path answer. Bug R's bare-affirmation
    # case ("Давай") is intent=CONVERSATION, not RAG, so it is preserved.
    _intent_label = (getattr(classifier_output, "intent", None) or "").upper()
    _is_rag_turn = _intent_label == "RAG"
    if (
        not _is_rag_turn
        and not profile.is_complete_for_calc()
        and profile.state in (ProfileState.COLLECTING, ProfileState.CONFIRMED)
        and (_is_calc_intent(classifier_output) or _has_any_core_field)
    ):
        # Bug 2 (live call bf7a95a8 2026-04-26): when the user replies to
        # an open clarify with a meta-question ("А какой лучше?", "в чём
        # разница?", "не понимаю"), re-emitting the same prompt loops the
        # caller. Detect the meta-question pattern and drift to LLM
        # fallback so the model can explain. The snapshot anchor is
        # included so the LLM sees the captured fields and answers in
        # context. The next user utterance re-enters the clarify gate
        # with the field actually answered.
        if _is_meta_question(utterance):
            # Bug 6: meta-question on a stale post-calc SMS offer means
            # the user moved on. Clear last_offer so a later bare confirm
            # cannot re-fire SMS via step 5c.
            profile.last_offer = None
            return FireLLMFallback(
                user_utterance=utterance,
                rag_context=None,
                snapshot=build_snapshot(profile),
            )
        return EmitClarify(
            missing=sorted(profile.missing_fields()),
            snapshot=build_snapshot(profile),
        )

    # Issue 3a (live call d5174335 2026-04-27): fail-closed safety net.
    # If profile is calc-complete AND user is confirming AND we entered
    # this turn from a pre-calc state, the user is ALWAYS waiting for a
    # calc result — not for the LLM to chat. Falling through to
    # FireLLMFallback let the LLM fabricate calc numbers
    # ("аванс составит 30 тысяч долларов, ежемесячный платеж 4167")
    # without ever invoking the calculator tool.
    # If we landed here despite all prior step gates, the safer action
    # is FireCalc — worst case it produces the same result twice; best
    # case it covers the race / barge-in / state-mutation gap that put
    # us in the catch-all. Layered with the LLM prompt anti-fabrication
    # clause shipped in the same wave.
    # Pre-calc states only — sticky CONFIRMED is excluded because Fix 4
    # (call 2ab41112) explicitly catches the "post-calc thank-you-plus-
    # question" pattern there ("Спасибо. Кто владелец?") and routes to
    # RAG / FireLLMFallback, NOT a calc re-fire. The safety net is for
    # the readback→confirm and change-pending→confirm flows where the
    # user is actively trying to get a calc result and a barge-in /
    # state race would otherwise drop them into FireLLMFallback.
    if (
        profile.is_complete_for_calc()
        and classifier_output.is_confirmation
        and pre_turn_state in (
            ProfileState.READBACK_PENDING,
            ProfileState.CHANGE_PENDING,
        )
        and classifier_output.intent != "RAG"
    ):
        profile.state = ProfileState.CONFIRMED
        return FireCalc(
            calc_params=build_calc_params(profile),
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
    # Bug 6 (live call e6226e5d 2026-04-29 Stanislav 15:16:11): when
    # the dispatcher reaches the catch-all FireLLMFallback, the turn is
    # conclusively non-structural — bare confirms/denies are caught
    # upstream by FAST-PATH and step 5c. So a stale post-calc SMS offer
    # is invalidated by any non-trivial conversation; clearing it here
    # prevents a bare "Да" many turns later from re-firing SMS via the
    # step 5c gate.
    profile.last_offer = None
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
            # Bug 8 follow-up (2026-05-07): the terse render now asks about
            # DETAIL only (not detail-or-SMS combined). Stamp last_offer="detail"
            # so a bare "Да" on the next turn routes through apply_turn STEP
            # 5c-detail → EmitCalcDetail, NOT FireSMS. After detail is
            # rendered, last_offer flips to "sms" (see EmitCalcDetail handler
            # below) so the next "Да" then routes to SMS.
            try:
                session.client_profile.last_offer = "detail"
            except Exception:  # noqa: BLE001
                pass
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

    if isinstance(action, EmitCalcDetail):
        # Bug 25 (ANALYSIS.md §8): caller asked "подробнее" after the
        # terse calc readback. Look up the most recent successful calc
        # in session.tool_calls_history and ship the detailed render.
        # LLM is NEVER invoked — the deterministic-numbers invariant
        # extends to this path. When no prior calc exists (caller asked
        # for detail before the bot ever computed anything), speak a
        # short Russian explanation rather than failing silently.
        from .profile_prompts import render_calc_result
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
        if _last_calc is None:
            spoken = (
                "Я ещё не делала расчёт. "
                "Назовите параметры — стоимость, срок, аванс — и я посчитаю."
            )
        else:
            spoken = render_calc_result(_last_calc["result"], detailed=True)
            # Bug 8 follow-up (2026-05-07): the detailed render now asks
            # "Отправить график платежей по СМС?". Stamp last_offer="sms"
            # so the next bare "Да" / "давай" routes to FireSMS via step 5c.
            try:
                session.client_profile.last_offer = "sms"
            except Exception:  # noqa: BLE001
                pass
        await tts.say(spoken)
        yield spoken
        return

    if isinstance(action, FireOORMessage):
        # Deterministic OOR text — no renderer needed, payload IS the text.
        await tts.say(action.message)
        yield action.message
        return

    if isinstance(action, EmitSMSOffer):
        # Pivot from declined detail to SMS offer (live transcript
        # 2026-05-08). last_offer is already set to "sms" by apply_turn,
        # so a follow-up "Да" routes through STEP 5c-sms → FireSMS, and
        # a follow-up decline goes through the normal EndCall path.
        spoken = "Хорошо, отправить график платежей по СМС?"
        await tts.say(spoken)
        yield spoken
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
        # Differentiate the two failure modes — debugging the chat path
        # was muddled by both branches reading "нечего отправить" in
        # session_analyzer. (Bug surfaced 2026-05-07 calc_smoke run.)
        if not _last_calc:
            spoken = (
                "Извините, мне пока нечего отправить. "
                "Давайте сначала рассчитаем условия."
            )
            await tts.say(spoken)
            yield spoken
            return
        if not _phone:
            spoken = (
                "Чтобы отправить график по СМС, мне нужен ваш номер телефона. "
                "Подскажите, на какой номер прислать?"
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
            # Bug 21 (ANALYSIS.md §8): the bot used to trail into silence
            # after the SMS confirmation, leaving the caller to volunteer
            # any next step. Append the canonical continuation offer so
            # the call stays open for follow-up questions.
            spoken = (
                f"Отправила график платежей по СМС на номер {_phone}. "
                "Чем ещё могу помочь по лизингу?"
            )
        else:
            spoken = (
                "Извините, не удалось отправить СМС. "
                "Попробуйте, пожалуйста, позже или уточните номер."
            )
        await tts.say(spoken)
        yield spoken
        return

    if isinstance(action, EndCall):
        # Bug 22: speak the farewell, drain the audio buffer, then send
        # the Jambonz hangup verb. tts.disconnect() (defined in
        # backend/execute_adapters.py:TtsSink) handles the drain timing
        # and the {"type": "disconnect"} send on the same websocket the
        # rest of the voice path uses. For text mode (Chat Widget Scope
        # B), TtsSink will be subclassed; the subclass overrides
        # disconnect() to emit a `call_ended` SIP-monitor event without
        # touching a real SIP leg.
        try:
            await tts.say(action.farewell)
        except Exception:  # noqa: BLE001
            pass
        yield action.farewell
        # Drain + hangup verb (best-effort; never raises).
        try:
            disc = getattr(tts, "disconnect", None)
            if callable(disc):
                await disc()
        except Exception:  # noqa: BLE001
            pass
        # Stamp the session for analyzer / monitor.
        try:
            session.call_ended_reason = action.reason  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
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
        # Codex CP-3.6 P2: orchestrator stamps `session.memory_block`
        # before dispatch so follow-up RAG / conversation turns retain
        # prior dialogue context. Tolerate absence (unit tests).
        _memory_block = getattr(session, "memory_block", None) if session is not None else None
        async for chunk in _stream_llm_to_tts(
            utterance=action.user_utterance,
            rag_context=rag_context or action.rag_context,
            snapshot=action.snapshot,
            backend=backend,
            tts=tts,
            session=session,
            memory_block=_memory_block,
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
    "Извините, не могу посчитать с этими параметрами. "
    "Хотите изменить срок, аванс или стоимость?"
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

    Polish D-lite (live call 56c0e2f9 2026-04-27): the calculator's
    `error` text used to passthrough verbatim (e.g. "По заданным
    параметрам условия лизинга не найдены."), ending in a period with
    no follow-up. The bot then went silent and the user had to invoke
    the bot again. Universal fix: every calc-fail spoken line ends with
    a concrete question so the user always has an obvious next move.
    Structured per-constraint guidance (which param actually broke the
    request) is deferred to Section 3.5 (Calculator Funnel API
    Integration); this is just the silence guard.
    """
    follow_up = "Хотите изменить срок, аванс или стоимость?"
    if isinstance(result, dict):
        err = result.get("error")
        if err:
            base = str(err).strip().rstrip(".!?")
            return f"{base}. {follow_up}"
    return f"Не удалось рассчитать по этим параметрам. {follow_up}"


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
    memory_block: Optional[str] = None,
    tool_calls_history: Optional[list[dict]] = None,
) -> list[dict]:
    """Assemble the chat messages for the FireLLMFallback stream.

    Injects:
      - `memory_block` (Codex CP-3.6 P2) as recent dialogue context so
        follow-up RAG / conversation turns retain prior-turn anchors.
      - `snapshot` as anti-hallucination anchor (E7). LLM sees captured
        fields and is instructed not to re-ask them.
      - `rag_context` as KB grounding block.
      - `tool_calls_history` (B1 fix, 2026-05-09): SMS-status anchor.
        Without this, the LLM can fabricate "СМС отправлено" on a turn
        where the user asked about prior actions ("ты графика
        отправила?") because nothing else in the prompt grounds the
        truth. Conservative default: when None or empty, emit the
        "no SMS yet" guard so legacy call-sites are safe-by-default.
    The production orchestrator prepends its own system prompt via the
    `backend.stream(system_prompt=...)` kwarg; this function deliberately
    avoids hardcoding the system prompt path so tests run without it.
    """
    memory_prefix = ""
    if memory_block:
        memory_prefix = str(memory_block) + "\n\n"
    anchor_block = ""
    if snapshot is not None:
        lines = _snapshot_anchor_lines(snapshot)
        if lines:
            anchor_block = (
                "Уже уточнено у клиента (НЕ переспрашивай эти поля):\n"
                + "\n".join(lines) + "\n\n"
            )

    # B1 fix: SMS-status anchor. Counts only successful (ok=True) entries
    # — failed dispatches must NOT flip the guard, otherwise a Twilio 5xx
    # silently turns the LLM into a confident liar.
    sms_sent = False
    for _entry in (tool_calls_history or []):
        if (
            isinstance(_entry, dict)
            and _entry.get("tool") == "send_sms"
            and _entry.get("ok") is True
        ):
            sms_sent = True
            break
    if sms_sent:
        sms_anchor = (
            "СМС уже отправлено в этом звонке. Если клиент спрашивает "
            "про СМС — подтверди отправку. Не предлагай отправить ещё раз "
            "без явной просьбы.\n\n"
        )
    else:
        sms_anchor = (
            "СМС ещё не отправлено в этом звонке. Если клиент спрашивает "
            "«ты график/расчёт прислала?» или похожее — НЕ говори, что "
            "отправила. Можно предложить отправить сейчас.\n\n"
        )
    # Anti-hallucination role guard (live call 8cb0bfaf 2026-04-28):
    # the LLM fallback hallucinated "Меняем стоимость на 110k и график
    # на аннуитетный. Подтвердите?" when the classifier dropped the
    # multi-field change. State machine never staged the change, but
    # the user heard a confirm-prompt and said "Да" on the next turn —
    # silent data loss when the calc fired with old values.
    #
    # Architectural truth: change-confirm wording is ALWAYS produced by
    # backend.profile_prompts.build_change_confirm_text via the
    # EmitChangeConfirm action. The LLM is NEVER the source of change-
    # confirm sentences. This block tells the LLM that contract so it
    # cannot accidentally fabricate one. If the LLM thinks the user
    # asked for a change but doesn't understand what, the right move
    # is to ask, not to make up a confirmation.
    role_guard = (
        "ВАЖНО — твоя роль: НЕ пиши формулировки вида «Меняю X на Y, "
        "всё верно?», «Подтверждаю изменение», «Подтвердите параметры». "
        "Подтверждение изменений и итоговый readback — работа отдельной "
        "системы, а не твоя. Если клиент попросил изменение и ты не "
        "уверен какое именно поле и какое значение — переспроси "
        "конкретно («что именно меняем — срок, аванс, или стоимость?»). "
        "Лучше уточнить, чем выдумать подтверждение.\n\n"
        # Currency policy guard (live call b31925a8 2026-05-09):
        # the LLM fallback hallucinated "юани не принимаем" / "только в
        # белорусских и долларах" when classifier dropped CNY. After the
        # 2026-05-09 currency-drift change, the system supports ANY
        # currency for Физ лицо by converting to BYN at the NBRB rate.
        # The LLM must NOT speak the old "только BYN/USD" policy.
        "ВАЖНО — валютная политика: для физических лиц система принимает "
        "ЛЮБУЮ валюту (юани, фунты, злотые, евро, рубли — что угодно) и "
        "автоматически пересчитывает в белорусские рубли по курсу НБ РБ. "
        "НЕ говори клиенту «только BYN» / «только в рублях или долларах» / "
        "«юани не принимаем» — это устаревшая политика. Юр.лица — "
        "BYN/USD/EUR/RUB напрямую. Если не знаешь, что ответить про валюту — "
        "просто переспроси сумму, не объявляй ограничений.\n\n"
    )
    kb_block = ""
    if rag_context:
        kb_block = (
            "Фрагменты из базы знаний (единственный источник фактов. "
            "Адреса, числа, ставки бери ТОЛЬКО отсюда):\n\n"
            + str(rag_context) + "\n\n"
        )
    user_content = (
        f"{memory_prefix}{anchor_block}{sms_anchor}{role_guard}{kb_block}"
        f"Сообщение клиента: {utterance}"
    )
    return [{"role": "user", "content": user_content}]


async def _stream_llm_to_tts(
    *,
    utterance: str,
    rag_context: Optional[str],
    snapshot: Optional[ProfileSnapshot],
    backend,
    tts,
    session,
    memory_block: Optional[str] = None,
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
    from .text_utils import clean_answer, validate_addresses

    # B1: pass session tool history so the prompt builder can inject the
    # SMS-status anchor (anti-hallucination guard for "ты график прислала?"
    # turns where the LLM previously fabricated "Да, отправила").
    _tool_history = list(getattr(session, "tool_calls_history", []) or [])
    messages = _build_fallback_messages(
        utterance, rag_context, snapshot,
        memory_block=memory_block,
        tool_calls_history=_tool_history,
    )
    detector = SentenceDetector()
    _fallback = (
        "Извините, сейчас технические неполадки. Попробуйте, пожалуйста,"
        " повторить вопрос."
    )

    # Bug 11 partial (live call 14:41:57 2026-04-29): the LLM-fallback
    # output spoke "ваш юридический адрес, указанный ранее в договоре,
    # улица Первомайская дом 12" — no договор is in session and the KB
    # didn't supply that street. validate_addresses() existed but was
    # never wired into the fallback emit path; only clean_answer() ran.
    # Pass rag_context as the single context-chunk source. When
    # rag_context is None or empty, validate_addresses returns text
    # unchanged (legacy behaviour preserved for non-RAG turns).
    _context_chunks: list[str] = [rag_context] if rag_context else []

    async def _emit(text: str) -> Optional[str]:
        cleaned = clean_answer(text)
        if not cleaned:
            return None
        cleaned = validate_addresses(cleaned, _context_chunks)
        if not cleaned.strip():
            return None
        await tts.say(cleaned)
        return cleaned

    # Latency instrumentation — restore the per-stage marker that
    # commit a1e53f4 dropped when apply_turn became sole orchestrator.
    # Without these, scripts/analyze_latency.sh has nothing to read and
    # we can't tell whether a perceived 3-4s turn lives in LLM, TTS, or
    # something else. Two timestamps gate the worst stage in this
    # function — first token from vLLM and first TTS audio shipped.
    import time as _time
    _t_start = _time.monotonic()
    _t_first_token: float | None = None
    _t_first_tts: float | None = None
    _sid_short = ""
    try:
        _sid = getattr(session, "session_id", None) or getattr(session, "id", None)
        if _sid:
            _sid_short = str(_sid)[:8]
    except Exception:  # noqa: BLE001
        pass

    try:
        async for token in backend.stream(messages=messages):
            if _t_first_token is None:
                _t_first_token = _time.monotonic()
            if _session_interrupted(session):
                return
            if not token:
                continue
            for sentence in detector.feed(token):
                if _session_interrupted(session):
                    return
                out = await _emit(sentence)
                if out:
                    if _t_first_tts is None:
                        _t_first_tts = _time.monotonic()
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
            if _t_first_tts is None:
                _t_first_tts = _time.monotonic()
            yield out

    # Emit the per-stage marker that scripts/analyze_latency.sh reads.
    # Done AFTER the yield-loop so we don't add any per-token overhead.
    try:
        _t_end = _time.monotonic()
        _llm_first_ms = (
            int((_t_first_token - _t_start) * 1000)
            if _t_first_token is not None else -1
        )
        _llm_total_ms = int((_t_end - _t_start) * 1000)
        _tts_first_ms = (
            int((_t_first_tts - _t_first_token) * 1000)
            if _t_first_token is not None and _t_first_tts is not None else -1
        )
        print(
            f"[LATENCY:{_sid_short}] "
            f"llm_first_ms={_llm_first_ms} "
            f"llm_total_ms={_llm_total_ms} "
            f"tts_first_ms={_tts_first_ms} "
            f"path=fallback_llm",
            flush=True,
        )
    except Exception as _lat_exc:  # noqa: BLE001
        print(f"[LATENCY:{_sid_short}] log failed: {_lat_exc}", flush=True)
