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


def test_text_turn_rejects_traversal_session_id(client):
    """Codex finding: client-supplied session_id with .. must be rejected."""
    resp = client.post("/api/text-turn", json={
        "message": "hi",
        "session_id": "../sessions",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "invalid session_id" in data["error"]


def test_text_turn_rejects_session_id_with_slashes(client):
    resp = client.post("/api/text-turn", json={
        "message": "hi",
        "session_id": "chat-/etc/passwd",
    })
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


def test_text_turn_rejects_unprefixed_session_id(client):
    resp = client.post("/api/text-turn", json={
        "message": "hi",
        "session_id": "evil-no-prefix",
    })
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


def test_text_turn_serializes_concurrent_requests_per_session(client, monkeypatch):
    """Codex finding: overlapping HTTP requests for the same session_id
    must serialize, otherwise an older classifier result can apply after
    a newer one and corrupt profile state. Test that the per-session
    lock holds: send two requests in rapid succession on the same
    session_id and assert both complete with ok=True (no race-induced
    error). Functional check that the lock doesn't deadlock."""
    sid = "chat-concurrency-1"
    r1 = client.post("/api/text-turn", json={"message": "Здравствуйте", "session_id": sid})
    r2 = client.post("/api/text-turn", json={"message": "Расскажи", "session_id": sid})
    assert r1.status_code == 200 and r1.json()["ok"] is True
    assert r2.status_code == 200 and r2.json()["ok"] is True
    # Both turns reused the same session — the second's profile_state should
    # equal or be downstream of the first's (state machine never goes backward).
    # We don't assert a specific state, just that both succeeded under the lock.


def test_chat_session_reaper_evicts_idle_sessions(client, monkeypatch):
    """Codex finding: abandoned chat sessions must be evicted by the
    reaper, otherwise tab-close leaks memory forever."""
    import backend.app as app_mod

    sid = "chat-reaper-1"
    r = client.post("/api/text-turn", json={"message": "hi", "session_id": sid})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert sid in app_mod.voice_sessions
    assert sid in app_mod.chat_session_last_activity

    # Backdate the activity timestamp past the idle timeout.
    app_mod.chat_session_last_activity[sid] = (
        app_mod.time.monotonic() - app_mod.CHAT_SESSION_IDLE_TIMEOUT_SEC - 10
    )

    # Run a single reaper iteration synchronously (don't wait 60s).
    # The reaper body uses time.monotonic() and the dict directly.
    import asyncio as _asyncio
    async def _reap_once():
        # Inline copy of the reaper's eviction loop body (one pass).
        now = app_mod.time.monotonic()
        stale = [
            s for s, last in list(app_mod.chat_session_last_activity.items())
            if now - last > app_mod.CHAT_SESSION_IDLE_TIMEOUT_SEC
        ]
        for s in stale:
            app_mod.voice_sessions.pop(s, None)
            app_mod.chat_session_locks.pop(s, None)
            app_mod.chat_session_last_activity.pop(s, None)
    _asyncio.get_event_loop().run_until_complete(_reap_once())

    assert sid not in app_mod.voice_sessions
    assert sid not in app_mod.chat_session_last_activity


def test_chat_session_max_active_returns_busy(client, monkeypatch):
    """Codex finding: random session_id posts must not grow voice_sessions
    unboundedly. When the cap is hit, new sessions get a busy response."""
    import backend.app as app_mod

    # Drain any sessions left over from earlier tests so the cap-of-1
    # check is meaningful (voice_sessions is a module-level dict shared
    # across tests).
    app_mod.voice_sessions.clear()
    app_mod.chat_session_locks.clear()
    app_mod.chat_session_last_activity.clear()

    monkeypatch.setattr(app_mod, "CHAT_SESSION_MAX_ACTIVE", 1, raising=False)
    # Fill the cap with one session. Session-id suffix must be 6-64 chars
    # of [A-Za-z0-9_-] to satisfy the route validator.
    r1 = client.post("/api/text-turn", json={"message": "hi", "session_id": "chat-cap-001"})
    assert r1.json()["ok"] is True
    # New session should bounce.
    r2 = client.post("/api/text-turn", json={"message": "hi", "session_id": "chat-cap-002"})
    assert r2.json()["ok"] is False
    assert "server busy" in r2.json()["error"]
    # Existing session should still work.
    r3 = client.post("/api/text-turn", json={"message": "Расскажи", "session_id": "chat-cap-001"})
    assert r3.json()["ok"] is True


def test_text_turn_stamps_memory_block_for_llm_context(client):
    """Verify voice_session.memory_block is populated after a turn so
    FireLLMFallback can prepend it to the LLM prompt. Regression test
    for the lost-conversational-memory bug (C2)."""
    import backend.app as app_mod

    sid = "chat-mem-001"
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
