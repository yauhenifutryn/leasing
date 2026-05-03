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
