"""Tests for SMS-intent detection.

Bug 4 (live call 6dd5880b, 2026-04-25): bot prompted "Хотите ... отправить
график платежей по СМС?" — user said "Да, открой пожалуйста" and "Да".
`has_sms_intent` only matched explicit SMS keywords ("отправ", "смс", "sms",
"пришли"), so the direct-send path at app.py:1820-1867 didn't fire. Qwen3.5
narrated "график отправлен" without actually invoking the send_sms tool.

Fix: extend detection so a short affirmation right after a successful
calculator call is treated as SMS intent (user accepting the bot's offer).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.sms_intent import detect_sms_intent


def _ok_calc():
    return [{"tool": "calculator", "result": {"ok": True, "monthly_payment": 1000}}]


def _failed_calc():
    return [{"tool": "calculator", "result": {"ok": False}}]


def _no_calc():
    return []


# ---------- Explicit keyword path (existing behaviour) ----------


def test_explicit_keyword_sms() -> None:
    assert detect_sms_intent("отправь смс", _ok_calc()) is True


def test_explicit_keyword_sms_no_calc() -> None:
    # Even without a calc, explicit intent still counts (legacy parity).
    assert detect_sms_intent("пришли смс", _no_calc()) is True


def test_explicit_keyword_otprav() -> None:
    assert detect_sms_intent("отправь график", _ok_calc()) is True


# ---------- Affirmation-after-calc path (new behaviour) ----------


def test_bare_da_after_calc_is_sms() -> None:
    """Bot just offered SMS; user said 'Да'. This is the live regression."""
    assert detect_sms_intent("Да.", _ok_calc()) is True


def test_da_otkroy_after_calc_is_sms() -> None:
    """Live call 6dd5880b user utterance verbatim."""
    assert detect_sms_intent("Да, открой, пожалуйста.", _ok_calc()) is True


def test_konechno_after_calc_is_sms() -> None:
    assert detect_sms_intent("Конечно, давай.", _ok_calc()) is True


def test_davai_after_calc_is_sms() -> None:
    assert detect_sms_intent("Давай.", _ok_calc()) is True


# ---------- Negative cases (must NOT fire SMS) ----------


def test_da_with_no_prior_calc_not_sms() -> None:
    """No calc in history → 'Да' is just a confirmation of something else."""
    assert detect_sms_intent("Да.", _no_calc()) is False


def test_da_after_failed_calc_not_sms() -> None:
    """Failed calc shouldn't trigger SMS — there's nothing to send."""
    assert detect_sms_intent("Да.", _failed_calc()) is False


def test_da_with_change_request_not_sms() -> None:
    """User saying 'Да, давай изменим' is asking for params change, not SMS."""
    assert detect_sms_intent("Да, давай изменим параметры.", _ok_calc()) is False


def test_no_with_change_words_not_sms() -> None:
    assert detect_sms_intent("Нет, поменяй срок.", _ok_calc()) is False


def test_long_utterance_not_sms() -> None:
    """Long sentence unlikely to be a bare SMS confirmation."""
    long_msg = (
        "Да я хотел уточнить ещё пару моментов про условия и страховку перед "
        "тем как принять решение о подписании договора"
    )
    assert detect_sms_intent(long_msg, _ok_calc()) is False


def test_question_not_sms() -> None:
    assert detect_sms_intent("А что входит в общую сумму?", _ok_calc()) is False


def test_empty_message_not_sms() -> None:
    assert detect_sms_intent("", _ok_calc()) is False


def test_only_change_keyword_not_sms() -> None:
    assert detect_sms_intent("измени срок на 36", _ok_calc()) is False


# ---------- Mixed / boundary cases ----------


def test_da_pozhaluysta_after_calc_is_sms() -> None:
    assert detect_sms_intent("Да, пожалуйста.", _ok_calc()) is True


def test_horosho_after_calc_is_sms() -> None:
    assert detect_sms_intent("Хорошо.", _ok_calc()) is True


def test_ok_after_calc_is_sms() -> None:
    assert detect_sms_intent("Ок.", _ok_calc()) is True


def test_nyet_after_calc_not_sms() -> None:
    """'Нет' alone should not trigger SMS."""
    assert detect_sms_intent("Нет.", _ok_calc()) is False


# ---------- Profile-state gating (Bug 15, live call 1cae210d 2026-04-25) ----
# User said "Да" to confirm a CHANGE_PENDING (type_schedule линейный →
# аннуитет). detect_sms_intent's existing logic saw last-call=calc-OK and
# bare "Да" → returned True. The bot fired send_sms with stale params
# instead of re-running the calculator with the new schedule.


def test_da_during_change_pending_not_sms() -> None:
    """CHANGE_PENDING + 'Да' = confirm-the-change, NOT send SMS."""
    from backend.session import ProfileState
    assert detect_sms_intent(
        "Да.",
        _ok_calc(),
        profile_state=ProfileState.CHANGE_PENDING,
    ) is False


def test_da_during_readback_pending_not_sms() -> None:
    """READBACK_PENDING + 'Да' = confirm-the-readback, NOT send SMS."""
    from backend.session import ProfileState
    assert detect_sms_intent(
        "Да.",
        _ok_calc(),
        profile_state=ProfileState.READBACK_PENDING,
    ) is False


def test_da_during_confirmed_is_sms() -> None:
    """Post-calc CONFIRMED state — 'Да' here IS the SMS confirm."""
    from backend.session import ProfileState
    assert detect_sms_intent(
        "Да.",
        _ok_calc(),
        profile_state=ProfileState.CONFIRMED,
    ) is True


def test_da_during_collecting_with_calc_history_is_sms() -> None:
    """COLLECTING with calc history (post-calc, pre-state-update) = SMS."""
    from backend.session import ProfileState
    assert detect_sms_intent(
        "Да.",
        _ok_calc(),
        profile_state=ProfileState.COLLECTING,
    ) is True


def test_explicit_sms_keyword_during_change_pending_blocked() -> None:
    """Even explicit 'отправь смс' during CHANGE_PENDING is blocked.
    Sending now would deliver stale params; finish the change-confirm first."""
    from backend.session import ProfileState
    assert detect_sms_intent(
        "Отправь смс.",
        _ok_calc(),
        profile_state=ProfileState.CHANGE_PENDING,
    ) is False


def test_no_state_arg_preserves_legacy_behaviour() -> None:
    """Backwards-compat: omitting profile_state matches prior behaviour."""
    assert detect_sms_intent("Да.", _ok_calc()) is True
