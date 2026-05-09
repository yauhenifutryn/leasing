"""B1 — phantom SMS claim.

Live call (HANDOVER reference, but verified independently):
user said "Ксения, ты графика отправила?" — bot replied "График платежей
я отправила в виде СМС" before any send_sms tool had actually fired.

Architectural fix shape: the LLM-fallback prompt builder must inject an
anti-hallucination anchor stating whether send_sms has actually fired
during this call. Mirrors the existing role_guard contract (which
forbids the LLM from fabricating change-confirms).

These are STRUCTURAL tests — they verify the prompt builder injects the
guard text. Behavioral verification (LLM stops claiming SMS-sent)
requires live-call observation; the structural pin is the falsifiable
boundary in unit-test land.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.turn_dispatcher import _build_fallback_messages  # noqa: E402


def test_fallback_prompt_includes_no_sms_yet_anchor_by_default():
    """Default: tool_calls_history is empty/None → SMS guard fires."""
    msgs = _build_fallback_messages(
        utterance="Ксения, ты графика отправила?",
        rag_context=None,
        snapshot=None,
        tool_calls_history=[],
    )
    content = msgs[0]["content"]
    assert "СМС" in content or "SMS" in content, (
        "no-SMS-yet anchor missing entirely"
    )
    # Must explicitly tell the LLM no SMS has been sent.
    assert (
        "СМС ещё не отправлено" in content
        or "СМС не отправлено" in content
    ), f"anti-hallucination 'no SMS sent' guard missing in:\n{content}"


def test_fallback_prompt_omits_no_sms_anchor_after_send_sms_fired():
    """After a successful send_sms tool call, the SMS guard must NOT
    appear — otherwise the LLM is told the opposite of what happened.
    """
    history = [
        {"tool": "send_sms", "ok": True, "args": {"phone": "+375290000000"}},
    ]
    msgs = _build_fallback_messages(
        utterance="а график вы прислали?",
        rag_context=None,
        snapshot=None,
        tool_calls_history=history,
    )
    content = msgs[0]["content"]
    assert "СМС ещё не отправлено" not in content
    assert "СМС не отправлено" not in content
    # Positive anchor: tell the LLM SMS WAS sent so it can confirm
    # truthfully when asked.
    assert "СМС отправлено" in content or "СМС уже отправлено" in content, (
        f"post-SMS positive anchor missing:\n{content}"
    )


def test_fallback_prompt_failed_send_sms_keeps_no_sms_guard():
    """Failed tool call (ok=False) must NOT count as 'SMS sent'.
    Otherwise a failed dispatch silently flips the guard and the LLM
    starts claiming SMS-sent on a turn where none reached the customer.
    """
    history = [
        {"tool": "send_sms", "ok": False, "error": "twilio_5xx"},
    ]
    msgs = _build_fallback_messages(
        utterance="ты график прислала?",
        rag_context=None,
        snapshot=None,
        tool_calls_history=history,
    )
    content = msgs[0]["content"]
    assert (
        "СМС ещё не отправлено" in content
        or "СМС не отправлено" in content
    ), f"failed send_sms must keep the no-SMS guard active:\n{content}"


def test_fallback_prompt_other_tools_in_history_do_not_flip_sms_anchor():
    """Sanity: a successful calculator call earlier in the call must
    not flip the SMS anchor (history entry is for a different tool)."""
    history = [
        {"tool": "calc", "ok": True, "args": {"cost": 80000}},
    ]
    msgs = _build_fallback_messages(
        utterance="ты график прислала?",
        rag_context=None,
        snapshot=None,
        tool_calls_history=history,
    )
    content = msgs[0]["content"]
    assert (
        "СМС ещё не отправлено" in content
        or "СМС не отправлено" in content
    )


def test_fallback_prompt_tool_calls_history_optional_for_callers():
    """Backward-compat: callers that don't pass tool_calls_history (or
    pass None) still get a usable prompt — default to 'no SMS yet' guard
    so legacy call-sites are safe-by-default."""
    msgs = _build_fallback_messages(
        utterance="а ты графика отправила?",
        rag_context=None,
        snapshot=None,
    )
    content = msgs[0]["content"]
    # Conservative default = no SMS guard (matches behavior at call-start).
    assert (
        "СМС ещё не отправлено" in content
        or "СМС не отправлено" in content
    )
