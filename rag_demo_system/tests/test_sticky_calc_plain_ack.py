"""Fix 39 #2 — sticky-calc gate must not re-fire on plain acknowledgments.

Observed: after the calculator runs and state=CONFIRMED, utterances like
"Хорошо." or "Ладно, спасибо." make the classifier emit is_confirmation=true,
which used to flip the gate open and re-call the calculator three more times
in session aba110a8 (2026-04-18). The gate now requires state to be
READBACK_PENDING or CHANGE_PENDING for is_confirmation to unlock recalc.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import _sticky_calc_ready  # noqa: E402
from backend.session import ClientProfile, ProfileState  # noqa: E402


def _complete_profile(state: ProfileState, confirmed: bool = False) -> ClientProfile:
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=50000,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20,
        term_months=36,
        type_schedule="0",
    )
    p.state = state
    if confirmed:
        p.confirmed_at = time.time()
    return p


# ── Plain-ack sticky-calc guard (the actual Fix 39 regression) ─────────

def test_plain_ack_on_confirmed_does_not_fire() -> None:
    profile = _complete_profile(ProfileState.CONFIRMED, confirmed=True)
    assert _sticky_calc_ready(profile, sa_is_confirm=True, needs_tool=False) is False


def test_plain_ack_on_collecting_does_not_fire() -> None:
    profile = _complete_profile(ProfileState.COLLECTING, confirmed=False)
    assert _sticky_calc_ready(profile, sa_is_confirm=True, needs_tool=False) is False


# ── Legitimate confirmation paths must still work ──────────────────────

def test_readback_pending_confirm_fires() -> None:
    profile = _complete_profile(ProfileState.READBACK_PENDING, confirmed=False)
    assert _sticky_calc_ready(profile, sa_is_confirm=True, needs_tool=False) is True


def test_change_pending_confirm_fires() -> None:
    profile = _complete_profile(ProfileState.CHANGE_PENDING, confirmed=False)
    assert _sticky_calc_ready(profile, sa_is_confirm=True, needs_tool=False) is True


# ── Tool-intent recalc on CONFIRMED still works ───────────────────────

def test_explicit_tool_intent_on_confirmed_fires() -> None:
    profile = _complete_profile(ProfileState.CONFIRMED, confirmed=True)
    assert _sticky_calc_ready(profile, sa_is_confirm=False, needs_tool=True) is True


def test_no_confirm_no_tool_on_confirmed_does_not_fire() -> None:
    profile = _complete_profile(ProfileState.CONFIRMED, confirmed=True)
    assert _sticky_calc_ready(profile, sa_is_confirm=False, needs_tool=False) is False


# ── Incomplete profile never fires regardless ─────────────────────────

def test_incomplete_profile_never_fires() -> None:
    p = ClientProfile(client_type="Физическое лицо", subject="Легковой автомобиль", cost=50000)
    p.state = ProfileState.READBACK_PENDING
    assert _sticky_calc_ready(p, sa_is_confirm=True, needs_tool=True) is False
