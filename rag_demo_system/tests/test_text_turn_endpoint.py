"""Tests for the chat-widget /api/text-turn endpoint.

Three targeted tests (Task 4 of the chat-widget plan):

  1. Empty / whitespace-only message returns ok=False without invoking the
     dispatcher (must pass even with no LLM running).
  2. Missing session_id is auto-assigned with the chat- prefix.
  3. Response shape contains every documented key.

All three must pass without a running classifier vLLM. The endpoint's
classifier try/except synthesizes a CONVERSATION fallback on connection
failure so apply_turn always receives a valid ClassifierOutput.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient with chat persistence redirected to tmp_path so test
    runs don't write into the real .state/ directory."""
    import backend.app as app_mod
    monkeypatch.setattr(
        app_mod, "_TEST_STATE_DIR_OVERRIDE", tmp_path, raising=False,
    )
    return TestClient(app_mod.app)


def test_text_turn_rejects_empty_message(client):
    resp = client.post("/api/text-turn", json={
        "message": "  ",
        "session_id": "chat-test-empty",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "error" in data


def test_text_turn_assigns_session_id_when_missing(client):
    resp = client.post("/api/text-turn", json={"message": "Hi"})
    # The endpoint may succeed (classifier reachable) or fall back to a
    # synthesized CONVERSATION action (classifier unavailable). In either
    # case we must get HTTP 200 and a chat- prefixed session id.
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"].startswith("chat-")


def test_text_turn_response_shape(client):
    resp = client.post("/api/text-turn", json={
        "message": "Здравствуйте",
        "session_id": "chat-shape-1",
        "name": "Иван",
    })
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "ok", "session_id", "reply", "action", "ended",
        "profile_state", "missing",
    ):
        assert key in data, f"missing key in response: {key}"


def test_text_turn_multi_turn_resets_per_turn_state(client, monkeypatch):
    """Two turns on the same session_id; verify reset_turn_state is called
    so the second turn doesn't see the first turn's tool_calls_this_turn.
    This is the regression test for the stuck-in-calculator loop (C1)."""
    import backend.app as app_mod

    captured_states: list[list] = []

    # Spy on reset_turn_state to record each call so we can assert it
    # fired on every turn (without coupling to the dispatcher internals).
    real_reset = app_mod.VoiceSession.reset_turn_state

    def _spy(self):
        captured_states.append(list(self.tool_calls_this_turn))
        return real_reset(self)

    monkeypatch.setattr(app_mod.VoiceSession, "reset_turn_state", _spy)

    sid = "chat-multi-1"
    r1 = client.post(
        "/api/text-turn",
        json={"message": "Здравствуйте", "session_id": sid},
    )
    r2 = client.post(
        "/api/text-turn",
        json={"message": "Расскажи", "session_id": sid},
    )
    assert r1.status_code == 200 and r1.json()["ok"] is True
    assert r2.status_code == 200 and r2.json()["ok"] is True
    # Both turns should have invoked reset_turn_state at least once.
    assert len(captured_states) >= 2, (
        f"expected >= 2 reset_turn_state calls, got {len(captured_states)}"
    )


def test_text_turn_stamps_memory_block_for_llm_context(client):
    """Verify voice_session.memory_block is populated after a turn so
    FireLLMFallback can prepend it to the LLM prompt. Regression test
    for the lost-conversational-memory bug (C2)."""
    import backend.app as app_mod

    sid = "chat-mem-1"
    r = client.post(
        "/api/text-turn",
        json={"message": "Здравствуйте", "session_id": sid},
    )
    assert r.status_code == 200
    vs = app_mod.voice_sessions.get(sid)
    # If the session ended on this turn (EndCall), it would have been
    # popped (I1). For "Здравствуйте" the dispatcher should NOT EndCall,
    # so the entry must still exist with memory_block stamped.
    assert vs is not None, "voice_session should still be in voice_sessions"
    # memory_block should exist (may be empty string on turn 1 if the
    # transcript hasn't grown, but the attribute MUST be set, not None).
    assert hasattr(vs, "memory_block")
    assert vs.memory_block is not None  # was None before C2 fix
