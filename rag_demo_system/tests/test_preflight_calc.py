"""Bug 27 — pre-tool validation gate for prepaid_pct."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.preflight_calc import validate_calc_inputs  # noqa: E402


def _profile(**overrides):
    base = dict(
        prepaid_pct=20,
        term_months=36,
        cost=50000,
        age_years=None,
        condition_new=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_in_range_prepaid_returns_none() -> None:
    assert validate_calc_inputs(_profile(prepaid_pct=20)) is None


def test_boundary_prepaid_40_is_in_range() -> None:
    # 40% exactly is still allowed — the rule says "не более 40%".
    assert validate_calc_inputs(_profile(prepaid_pct=40)) is None


def test_prepaid_41_triggers_oor() -> None:
    msg = validate_calc_inputs(_profile(prepaid_pct=41))
    assert msg is not None
    assert "максимум 40" in msg


def test_prepaid_60_triggers_oor_with_workaround() -> None:
    msg = validate_calc_inputs(_profile(prepaid_pct=60))
    assert msg is not None
    assert "максимум 40" in msg
    # Workaround must be in the message — caller should hear that paying
    # >40% is possible as a первый платёж, not as аванс.
    assert "первым платежом" in msg
    # The workaround should also state the concrete split (40 + 20).
    assert "40" in msg and "20" in msg


def test_prepaid_100_triggers_oor() -> None:
    msg = validate_calc_inputs(_profile(prepaid_pct=100))
    assert msg is not None
    assert "максимум 40" in msg
    assert "первым платежом" in msg


def test_missing_prepaid_returns_none() -> None:
    # Profile may not have prepaid_pct yet — that's a clarify case,
    # not an OOR case. Validation must not fire here.
    assert validate_calc_inputs(_profile(prepaid_pct=None)) is None


def test_prepaid_amount_only_returns_none() -> None:
    # When prepaid is in absolute currency rather than %, the gate
    # is not concerned with this bug's scope.
    p = _profile(prepaid_pct=None)
    p.prepaid_amount = 5000
    assert validate_calc_inputs(p) is None


def test_oor_message_is_speakable() -> None:
    msg = validate_calc_inputs(_profile(prepaid_pct=60))
    assert msg is not None
    # Voice-bot rules: no markdown, no emoji, no URLs.
    for forbidden in ("**", "##", "http://", "https://", "—", "*"):
        # "—" is em-dash — voice-bot rules forbid it. Use commas/colons.
        if forbidden == "—":
            assert forbidden not in msg, f"em-dash forbidden in voice text: {msg}"
        else:
            assert forbidden not in msg
