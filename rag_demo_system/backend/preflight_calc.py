"""Pre-tool validation gate (Bug 27, Batch 3).

Live call 5746bfec at 18:42:14 (2026-05-03): the user asked for prepaid
60%, the dispatcher passed that through to the calculator API, the API
returned FAIL, and the bot then surfaced a generic "out of range"
error. The topical KB section `minimum-advance-and-no-advance` already
documents the rule and the legitimate workaround:

    "Допустимый аванс — от 0% до 40% стоимости. Программа устанавливает
     ограничение не более 40%. Если клиент хочет внести больше —
     невозможно как аванс, но можно как первый платёж по графику
     (например, 40% авансом + 20% первым платежом)."

So the right user experience is to (a) catch the OOR before the calc
fires and (b) surface the workaround inline. This module is the gate.

Today it only handles `prepaid_pct > 40`. The signature is intentionally
shaped to absorb future range checks without forcing a callsite churn:
`validate_calc_inputs(profile) -> Optional[str]` returns either the
Russian explanation to speak to the caller or None when the profile is
calc-eligible. New range checks (term_months, age_years, cost minimum)
should land here next to the existing prepaid one.
"""
from __future__ import annotations

from typing import Any, Optional


PREPAID_PCT_MAX = 40.0


def _prepaid_oor_message(value: float) -> str:
    # No em-dashes (CLAUDE.md §7); voice-bot rules forbid them.
    return (
        f"Аванс по нашей программе максимум 40 процентов от стоимости, "
        f"вы назвали {int(value)} процентов. Но если хотите внести "
        f"больше, это можно оформить как первый платёж по графику. "
        f"Например, 40 процентов авансом и {int(value) - 40} процентов "
        f"первым платежом. Так удобно?"
    )


def validate_calc_inputs(profile: Any) -> Optional[str]:
    """Inspect a ClientProfile and return a user-facing Russian message
    when any required calculator input is out of range; None when the
    profile is OK to feed to FireCalc.

    Today only `prepaid_pct` is checked. Future bugs (term_months 12-84,
    age_years 18-75, cost minimum) extend this helper rather than adding
    new gates around the dispatcher.
    """
    pct = getattr(profile, "prepaid_pct", None)
    if pct is not None:
        try:
            f = float(pct)
        except (TypeError, ValueError):
            return None  # bad type is caught downstream by tools/calculator
        if f > PREPAID_PCT_MAX:
            return _prepaid_oor_message(f)

    return None
