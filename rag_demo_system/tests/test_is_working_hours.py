"""Bug 19 — working-hours predicate boundary cases."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.is_working_hours import (  # noqa: E402
    MINSK_TZ,
    augment_system_prompt_with_working_hours,
    is_working_hours,
    working_hours_marker,
)


def _minsk(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=MINSK_TZ)


# 2026-05-04 is a Monday; 2026-05-09 is Saturday; 2026-05-10 is Sunday.

def test_weekday_at_open_is_working() -> None:
    assert is_working_hours(_minsk(2026, 5, 4, 9, 0)) is True


def test_weekday_inside_window_is_working() -> None:
    assert is_working_hours(_minsk(2026, 5, 4, 13, 30)) is True


def test_weekday_at_close_is_after_hours() -> None:
    # Boundary is half-open: 18:00 sharp is already after-hours.
    assert is_working_hours(_minsk(2026, 5, 4, 18, 0)) is False


def test_weekday_one_minute_after_close_is_after_hours() -> None:
    assert is_working_hours(_minsk(2026, 5, 4, 18, 1)) is False


def test_weekday_before_open_is_after_hours() -> None:
    assert is_working_hours(_minsk(2026, 5, 4, 8, 59)) is False


def test_saturday_any_hour_is_after_hours() -> None:
    for hour in (9, 12, 17):
        assert is_working_hours(_minsk(2026, 5, 9, hour, 0)) is False


def test_sunday_any_hour_is_after_hours() -> None:
    for hour in (9, 12, 17):
        assert is_working_hours(_minsk(2026, 5, 10, hour, 0)) is False


def test_naive_datetime_treated_as_minsk_local() -> None:
    naive = datetime(2026, 5, 4, 10, 0)
    assert is_working_hours(naive) is True


def test_utc_time_converted_to_minsk_before_check() -> None:
    # 2026-05-04 06:00 UTC = 2026-05-04 09:00 Minsk → in.
    utc = datetime(2026, 5, 4, 6, 0, tzinfo=ZoneInfo("UTC"))
    assert is_working_hours(utc) is True
    # 2026-05-04 15:01 UTC = 2026-05-04 18:01 Minsk → out.
    utc = datetime(2026, 5, 4, 15, 1, tzinfo=ZoneInfo("UTC"))
    assert is_working_hours(utc) is False


def test_marker_says_рабочее_when_inside_window() -> None:
    s = working_hours_marker(_minsk(2026, 5, 4, 13, 0))
    assert "РАБОЧЕЕ" in s
    # Off-hours wording must NOT appear inside the working-hours marker.
    assert "перезвонят в рабочее время" not in s


def test_marker_says_нерабочее_outside_window() -> None:
    s = working_hours_marker(_minsk(2026, 5, 4, 20, 0))
    assert "НЕРАБОЧЕЕ" in s
    assert "перезвонят в рабочее время" in s
    # The off-hours marker must explicitly mention the canonical hours.
    assert "09:00-18:00" in s


# ── Augmentation: marker prepended to system prompt ────────────────────

def test_augment_prepends_marker_to_prompt() -> None:
    out = augment_system_prompt_with_working_hours(
        "# Role\nВы Ксения...",
        now=_minsk(2026, 5, 4, 13, 0),
    )
    assert out.startswith("[Время сейчас:")
    assert "РАБОЧЕЕ" in out
    assert "Вы Ксения..." in out  # body preserved


def test_augment_outside_hours_inserts_off_marker() -> None:
    out = augment_system_prompt_with_working_hours(
        "# Role\nВы Ксения...",
        now=_minsk(2026, 5, 4, 20, 0),
    )
    assert "НЕРАБОЧЕЕ" in out
    assert "перезвонят в рабочее время" in out
    assert "Вы Ксения..." in out


def test_augment_is_idempotent() -> None:
    base = "# Role\nВы Ксения..."
    once = augment_system_prompt_with_working_hours(
        base, now=_minsk(2026, 5, 4, 13, 0)
    )
    twice = augment_system_prompt_with_working_hours(
        once, now=_minsk(2026, 5, 4, 13, 0)
    )
    # Re-applying must not stack two markers.
    assert twice.count("[Время сейчас:") == 1
    assert twice == once


def test_augment_swaps_marker_when_hours_flip() -> None:
    base = "# Role\nВы Ксения..."
    in_hours = augment_system_prompt_with_working_hours(
        base, now=_minsk(2026, 5, 4, 13, 0)
    )
    # Re-augmenting with an off-hours `now` strips the prior marker
    # and inserts the off-hours one — only one marker remains.
    flipped = augment_system_prompt_with_working_hours(
        in_hours, now=_minsk(2026, 5, 4, 20, 0)
    )
    assert flipped.count("[Время сейчас:") == 1
    assert "НЕРАБОЧЕЕ" in flipped
    assert "РАБОЧЕЕ" not in flipped or flipped.find("НЕРАБОЧЕЕ") < flipped.find("РАБОЧЕЕ")
