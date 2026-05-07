import json
import time
from pathlib import Path

from backend.chat_persistence import save_chat_turn, mark_session_ended


def test_save_chat_turn_creates_file_on_first_turn(tmp_path: Path):
    save_chat_turn(
        session_id="chat-test1",
        transcript=[{"role": "user", "text": "Hi"}, {"role": "assistant", "text": "Hello"}],
        tool_calls=[],
        name="Иван",
        phone="+375291234567",
        state_dir=tmp_path,
    )
    out = tmp_path / "transcripts" / "chat-test1.json"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["session_id"] == "chat-test1"
    assert data["transport"] == "chat"
    assert data["name"] == "Иван"
    assert data["phone"] == "+375291234567"
    assert data["client_id"] is None
    assert data["turn_count"] == 2
    assert data["ended_at"] is None
    assert "started_at" in data
    assert "last_turn_at" in data


def test_save_chat_turn_overwrites_with_growing_transcript(tmp_path: Path):
    started_first: str | None = None
    last_turn_first: str | None = None
    for i in range(3):
        save_chat_turn(
            session_id="chat-test2",
            transcript=[{"role": "user", "text": f"msg{j}"} for j in range(i + 1)],
            tool_calls=[],
            name="",
            phone="",
            state_dir=tmp_path,
        )
        if i == 0:
            first = json.loads(
                (tmp_path / "transcripts" / "chat-test2.json").read_text(encoding="utf-8")
            )
            started_first = first["started_at"]
            last_turn_first = first["last_turn_at"]
            # Sleep just over 1s so the second-precision timestamps differ
            # and the preservation assertion is meaningful.
            time.sleep(1.1)
    data = json.loads((tmp_path / "transcripts" / "chat-test2.json").read_text())
    assert data["turn_count"] == 3
    # started_at must be preserved across overwrites; last_turn_at must advance.
    assert data["started_at"] == started_first
    assert data["last_turn_at"] != last_turn_first


def test_mark_session_ended_sets_timestamp(tmp_path: Path):
    save_chat_turn(
        session_id="chat-test3",
        transcript=[{"role": "user", "text": "Bye"}],
        tool_calls=[], name="", phone="", state_dir=tmp_path,
    )
    mark_session_ended("chat-test3", state_dir=tmp_path)
    data = json.loads((tmp_path / "transcripts" / "chat-test3.json").read_text())
    assert data["ended_at"] is not None


def test_save_chat_turn_includes_tool_calls(tmp_path: Path):
    save_chat_turn(
        session_id="chat-test4",
        transcript=[{"role": "user", "text": "Calc"}],
        tool_calls=[{"tool": "calculator", "params": {"cost": 10000}, "result": {"ok": True}}],
        name="", phone="", state_dir=tmp_path,
    )
    data = json.loads((tmp_path / "transcripts" / "chat-test4.json").read_text())
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool"] == "calculator"
    assert data["tool_call_count"] == 1


def test_save_chat_turn_rejects_traversal_session_id(tmp_path):
    """Codex finding: ../sessions would overwrite StateStore. Must raise."""
    import pytest
    with pytest.raises(ValueError, match="outside transcripts dir"):
        save_chat_turn(
            session_id="../sessions",
            transcript=[],
            tool_calls=[],
            name="",
            phone="",
            state_dir=tmp_path,
        )


def test_mark_session_ended_rejects_traversal_session_id(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="outside transcripts dir"):
        mark_session_ended("../sessions", state_dir=tmp_path)
