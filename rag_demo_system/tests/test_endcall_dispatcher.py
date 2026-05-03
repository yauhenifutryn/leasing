"""Bug 22 — EndCall TurnAction.

Caller-initiated goodbye must produce a deterministic SIP teardown:
the bot speaks a brief farewell, drains TTS, and sends Jambonz
{"type": "disconnect"}. Without this action, "до свидания" turns fell
to FireLLMFallback and the SIP leg stayed open until carrier timeout.

Two trigger paths are tested:
  (a) classifier emits intent=END_CALL semantically.
  (b) FAST-PATH narrow goodbye regex (_is_goodbye_utterance) for SKIP'd
      utterances that never reach the classifier.

Suppression cases lock in the safety: the dispatcher MUST NOT emit
EndCall when a pending state is active or a change is in flight.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.classifier_schema import ClassifierOutput  # noqa: E402
from backend.session import ClientProfile, ProfileState  # noqa: E402
from backend.turn_action import EndCall, EmitReadback, EmitChangeConfirm  # noqa: E402
from backend.turn_dispatcher import (  # noqa: E402
    apply_turn,
    _is_goodbye_utterance,
)


# ── _is_goodbye_utterance: surface-form regex fallback ────────────────

def test_goodbye_regex_matches_canonical_forms() -> None:
    for utt in (
        "до свидания",
        "До свидания.",
        "До свидания!",
        "Пока",
        "пока-пока",
        "всего доброго",
        "всего хорошего",
        "спасибо, всё",
        "Спасибо, всё.",
        "всё, спасибо",
        "больше ничего не нужно",
    ):
        assert _is_goodbye_utterance(utt), f"goodbye not matched: {utt!r}"


def test_goodbye_regex_rejects_ambiguous_phrasings() -> None:
    for utt in (
        "до свидания, а ещё один вопрос",     # has follow-up intent
        "хорошо, спасибо",                     # acknowledgement, not goodbye
        "пока думаю",                           # пока ≠ farewell here
        "ладно",                                # SKIP'd small-talk, not goodbye
        "спасибо большое за помощь",           # gratitude, not farewell
    ):
        assert not _is_goodbye_utterance(utt), f"false positive: {utt!r}"


# ── Dispatcher: classifier intent=END_CALL ────────────────────────────

def _make_profile(state: ProfileState = ProfileState.COLLECTING, **fields) -> ClientProfile:
    p = ClientProfile()
    p.state = state
    for k, v in fields.items():
        setattr(p, k, v)
    return p


def _co(intent: str | None = "CONVERSATION", **kw) -> ClassifierOutput:
    base = dict(intent=intent, is_confirmation=False, is_stop_request=False)
    base.update(kw)
    return ClassifierOutput.model_validate(
        {k: v for k, v in base.items() if v is not None},
        context={"utterance": kw.get("_utterance", "")},
    )


def test_intent_end_call_yields_endcall_in_collecting_state() -> None:
    profile = _make_profile()
    co = _co(intent="END_CALL")
    action = apply_turn(profile, co, utterance="хорошо, спасибо, на этом всё, до свидания")
    assert isinstance(action, EndCall)
    assert action.farewell  # non-empty
    assert action.reason == "user_goodbye"


def test_goodbye_utterance_yields_endcall_when_classifier_says_conversation() -> None:
    # FAST-PATH: SKIP_CLASSIFIER routes "до свидания" through with
    # intent=CONVERSATION; the dispatcher's regex fallback lifts it.
    profile = _make_profile()
    co = _co(intent="CONVERSATION")
    action = apply_turn(profile, co, utterance="до свидания")
    assert isinstance(action, EndCall)


def test_goodbye_in_confirmed_state_still_ends_call() -> None:
    # After a successful calc, "до свидания" is the natural close.
    profile = _make_profile(
        state=ProfileState.CONFIRMED,
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=50000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        type_schedule="0",
    )
    co = _co(intent="END_CALL")
    action = apply_turn(profile, co, utterance="спасибо, до свидания")
    assert isinstance(action, EndCall)


# ── Suppression cases ─────────────────────────────────────────────────

def test_goodbye_suppressed_in_readback_pending() -> None:
    # Mid-readback "до свидания" must NOT hang up — the user might be
    # confused and waiting on the readback to land. Let the LLM handle.
    profile = _make_profile(
        state=ProfileState.READBACK_PENDING,
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=50000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        type_schedule="0",
    )
    co = _co(intent="END_CALL")
    action = apply_turn(profile, co, utterance="до свидания")
    assert not isinstance(action, EndCall)


def test_goodbye_suppressed_when_change_field_is_in_flight() -> None:
    # User staging a change ("поменяй срок на 48, до свидания") must
    # not hang up — apply the change first.
    profile = _make_profile(
        state=ProfileState.CONFIRMED,
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=50000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=36,
        type_schedule="0",
    )
    co = _co(
        intent="END_CALL",
        change_field="term_months",
        change_value=48,
        action="change_param",
    )
    action = apply_turn(profile, co, utterance="поменяй срок на 48, до свидания")
    assert not isinstance(action, EndCall)


def test_non_goodbye_does_not_yield_endcall() -> None:
    profile = _make_profile()
    co = _co(intent="CONVERSATION")
    action = apply_turn(profile, co, utterance="расскажите про условия лизинга")
    assert not isinstance(action, EndCall)
