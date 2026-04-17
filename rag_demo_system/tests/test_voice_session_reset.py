from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.voice_session import VoiceSession


def test_reset_turn_state_moves_to_history():
    s = VoiceSession(session_id="s1")
    s.tool_calls_this_turn.append({"tool": "calculator", "ok": True})
    s.reset_turn_state()
    assert s.tool_calls_this_turn == []
    assert len(s.tool_calls_history) == 1
    assert s.tool_calls_history[0]["tool"] == "calculator"


def test_reset_turn_state_no_op_when_empty():
    s = VoiceSession(session_id="s2")
    s.reset_turn_state()
    assert s.tool_calls_this_turn == []
    assert s.tool_calls_history == []


def test_reset_preserves_other_state():
    s = VoiceSession(session_id="s3")
    s.client_name = "Никита"
    s.listen_mode = True
    s.last_calc_signature = "sig_abc"
    s.consecutive_calc_failures = 2
    s.tool_calls_this_turn.append({"tool": "calculator"})
    s.reset_turn_state()
    assert s.client_name == "Никита"
    assert s.listen_mode is True
    assert s.last_calc_signature == "sig_abc"
    assert s.consecutive_calc_failures == 2
    assert s.tool_calls_this_turn == []
