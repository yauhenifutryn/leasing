"""Working-hours predicate for the specialist-handoff gate (Bug 19).

Mikro Leasing's specialists are reachable Mon-Fri 09:00-18:00 Minsk
time. Outside that window, any "I'll connect you to a specialist now"
promise is broken on the spot — the client hangs up expecting a transfer
that never happens. The fix wraps the handoff offer in a check against
this window and surfaces an alternative wording for non-business hours.

The helper is a pure function on a `now` parameter so tests can pin the
clock without monkeypatching `datetime.now`. The default falls back to
`datetime.now(tz=ZoneInfo("Europe/Minsk"))` so production callers can
omit the argument.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo


MINSK_TZ = ZoneInfo("Europe/Minsk")
WORK_START_HOUR = 9
WORK_END_HOUR = 18  # exclusive — 18:00 sharp is already after-hours.


def is_working_hours(now: Optional[datetime] = None) -> bool:
    """True when `now` (or the current Minsk time) falls inside the
    Mon-Fri 09:00-18:00 specialist availability window.

    The boundary is half-open: [09:00, 18:00). 09:00:00 is in; 17:59:59
    is in; 18:00:00 is out. Anything before 09:00 is out. Saturday and
    Sunday are always out, regardless of hour.

    A timezone-naive `now` is treated as Minsk-local. A timezone-aware
    `now` is converted to Minsk time before the comparison so callers
    can pass UTC timestamps from logs.
    """
    if now is None:
        now = datetime.now(tz=MINSK_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=MINSK_TZ)
    else:
        now = now.astimezone(MINSK_TZ)

    if now.weekday() >= 5:  # 5 = Sat, 6 = Sun.
        return False
    return WORK_START_HOUR <= now.hour < WORK_END_HOUR


def working_hours_marker(now: Optional[datetime] = None) -> str:
    """One-line system-context tag the LLM consumes to decide handoff
    phrasing. Returned shape locks in the rule-text the prompt expects.
    Kept here (next to the predicate) so any change to the marker text
    happens in one place.
    """
    if is_working_hours(now):
        return (
            "[Время сейчас: РАБОЧЕЕ время. Специалисты на связи. "
            "Если клиент просит соединить — оформите передачу как обычно.]"
        )
    return (
        "[Время сейчас: НЕРАБОЧЕЕ время (специалисты доступны Пн-Пт "
        "09:00-18:00 по Минску). Если клиент просит специалиста — "
        "НЕ обещайте мгновенное переключение. Скажите: \"Сейчас наши "
        "специалисты не на связи. Я зафиксирую заявку, и они перезвонят "
        "в рабочее время.\"]"
    )


def augment_system_prompt_with_working_hours(
    system_prompt: str,
    now: Optional[datetime] = None,
) -> str:
    """Prepend the working-hours context marker to a system prompt.

    The marker becomes the first line so the LLM sees the handoff rule
    before any other instruction. Idempotent: re-applying does not stack
    multiple markers, so callers can safely augment a prompt that was
    already augmented (the prior marker is stripped before the new one
    is prepended).
    """
    marker = working_hours_marker(now)
    body = system_prompt
    if body.startswith("[Время сейчас:"):
        nl = body.find("\n")
        body = body[nl + 1:].lstrip("\n") if nl >= 0 else ""
    return f"{marker}\n\n{body}" if body else marker
