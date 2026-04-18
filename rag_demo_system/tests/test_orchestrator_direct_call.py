"""Contract tests for the _can_direct_call gate in _stream_voice_response.

The core invariant: once the profile is confirmed, DirectTool must NOT fire
on turns where the user didn't send a tool-intent or a confirmation signal.
"""
from pathlib import Path


_APP_PY = Path(__file__).resolve().parents[1] / "backend" / "app.py"


def test_profile_ready_requires_fresh_signal():
    src = _APP_PY.read_text(encoding="utf-8")
    # The _profile_ready expression must include the fresh-signal clause.
    marker = "_profile_ready = ("
    assert marker in src, "refactor broke this contract test"
    idx = src.index(marker)
    body = src[idx : idx + 500]
    # Contract: fresh-signal clause is present (needs_tool or _sa_is_confirm)
    assert "needs_tool or _sa_is_confirm" in body, (
        "Fix 15 regression: _profile_ready no longer requires needs_tool or "
        "_sa_is_confirm, so the calculator will re-fire on every post-confirm "
        "turn (info questions, chit-chat) — the sticky-tool bug is back."
    )


def test_profile_ready_logic():
    """Logic-level test of the boolean expression."""
    def _profile_ready(is_complete, confirmed_at, sa_is_confirm, needs_tool):
        return (
            is_complete
            and (confirmed_at is not None or sa_is_confirm)
            and (needs_tool or sa_is_confirm)
        )

    # First confirmation: profile complete, confirmed_at None yet, user says "Верно"
    assert _profile_ready(True, None, True, False) is True  # _sa_is_confirm=True

    # Post-confirm info question "кто директор": confirmed_at set, but no fresh signal
    assert _profile_ready(True, 1234567890, False, False) is False  # THE BUG — must be False

    # Post-confirm recalc "пересчитай со сроком 60": needs_tool=True via classifier action
    assert _profile_ready(True, 1234567890, False, True) is True

    # Post-confirm change-confirm "Да": _sa_is_confirm=True
    assert _profile_ready(True, 1234567890, True, False) is True

    # Profile not complete: never fires
    assert _profile_ready(False, 1234567890, True, True) is False

    # Never confirmed, no explicit confirmation this turn: doesn't fire
    assert _profile_ready(True, None, False, True) is False
