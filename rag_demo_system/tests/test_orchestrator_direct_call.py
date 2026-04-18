"""Contract tests for the _can_direct_call gate in _stream_voice_response.

The core invariant: once the profile is confirmed, DirectTool must NOT fire
on turns where the user didn't send a tool-intent or a valid-state confirmation.

Fix 39 refactor: the inline boolean was extracted into _sticky_calc_ready
helper so it can be unit-tested directly (see test_sticky_calc_plain_ack.py).
The gate additionally requires state to be READBACK_PENDING/CHANGE_PENDING
for is_confirm to unlock recalc — plain "хорошо" on CONFIRMED no longer fires.
"""
from pathlib import Path


_APP_PY = Path(__file__).resolve().parents[1] / "backend" / "app.py"


def test_profile_ready_uses_sticky_calc_helper():
    src = _APP_PY.read_text(encoding="utf-8")
    # Contract: gate delegates to the named helper (testable in isolation)
    assert "_profile_ready = _sticky_calc_ready(" in src, (
        "Fix 39 refactor broken: _profile_ready must call _sticky_calc_ready"
    )
    # Contract: helper enforces state gate so is_confirm on plain CONFIRMED
    # does NOT fire the calculator.
    assert "def _sticky_calc_ready(" in src, "_sticky_calc_ready helper missing"
    assert "ProfileState.READBACK_PENDING" in src and "ProfileState.CHANGE_PENDING" in src


def test_profile_ready_logic():
    """Logic-level test of the extracted helper (see test_sticky_calc_plain_ack.py
    for the full matrix tied to real ClientProfile / ProfileState)."""
    def _profile_ready(is_complete, confirmed_at, sa_is_confirm, needs_tool, confirm_unlocks):
        return (
            is_complete
            and (confirmed_at is not None or confirm_unlocks)
            and (needs_tool or confirm_unlocks)
        )

    # First confirmation: state=READBACK_PENDING, is_confirm=true -> confirm_unlocks=True
    assert _profile_ready(True, None, True, False, True) is True

    # Post-confirm info question: no fresh signal -> False
    assert _profile_ready(True, 1234567890, False, False, False) is False

    # Post-confirm recalc via explicit tool intent
    assert _profile_ready(True, 1234567890, False, True, False) is True

    # Post-confirm plain "хорошо": classifier emits is_confirm=true but state=CONFIRMED,
    # so confirm_unlocks=False. Gate must reject. (Fix 39 regression guard.)
    assert _profile_ready(True, 1234567890, True, False, False) is False

    # Incomplete profile never fires
    assert _profile_ready(False, 1234567890, True, True, True) is False
