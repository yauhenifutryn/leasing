"""VoiceSession carries ClientProfile and listen_mode fields by default."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.session import ClientProfile  # noqa: E402
from backend.voice_session import VoiceSession  # noqa: E402


def test_voice_session_has_client_profile() -> None:
    vs = VoiceSession(session_id="test")
    assert isinstance(vs.client_profile, ClientProfile)
    assert vs.client_profile.is_complete_for_calc() is False


def test_voice_session_listen_mode_default_off() -> None:
    vs = VoiceSession(session_id="test")
    assert vs.listen_mode is False
    assert vs.listen_mode_until == 0.0


def test_voice_session_profile_independent_per_instance() -> None:
    """Each VoiceSession gets its own ClientProfile (default_factory, not shared)."""
    vs1 = VoiceSession(session_id="a")
    vs2 = VoiceSession(session_id="b")
    vs1.client_profile.name = "Сергей"
    assert vs2.client_profile.name is None
