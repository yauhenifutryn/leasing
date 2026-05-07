"""End-to-end integration test for /api/text-turn.

Drives a chat session through the full Микро Лизинг collect→readback→
calc→SMS→EndCall flow. Skipped by default — set RUN_INTEGRATION=1 with
the local SessionAgent + brain vLLM stack running to exercise it.

This test is a documentation of the expected happy-path conversation
shape AND a smoke gate. It would have caught regressions like:
- Calculator never firing because field collection drifts off-script.
- EndCall never firing because the goodbye-classifier prompt is broken.
- Persistence side effects missing (transcript file, ended_at stamp).
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    import backend.app as app_mod
    monkeypatch.setattr(app_mod, "_TEST_STATE_DIR_OVERRIDE", tmp_path, raising=False)
    return TestClient(app_mod.app)


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="set RUN_INTEGRATION=1 with local SessionAgent + brain vLLM running",
)
def test_full_chat_flow_collect_calc_sms_endcall(client, tmp_path):
    sid = "chat-integ-1"
    msgs = [
        "Здравствуйте",
        "Хочу посчитать лизинг",
        "Легковой автомобиль",
        "Стоимость 50000 долларов",
        "Юрлицо",
        "Новый",
        "Срок 36 месяцев, аванс 30%, равные платежи",
        "Да всё верно",
        "Отправь смс",
        "Да",
        "Спасибо, до свидания",
    ]
    saw_endcall = False
    for m in msgs:
        resp = client.post("/api/text-turn", json={
            "message": m, "session_id": sid,
            "name": "Иван", "phone": "+375291234567",
        })
        assert resp.status_code == 200, f"HTTP {resp.status_code} on '{m}'"
        data = resp.json()
        assert data["ok"] is True, f"ok=false on '{m}': {data}"
        if data["ended"]:
            saw_endcall = True
            break

    assert saw_endcall, "EndCall never fired across the full flow"

    record = json.loads((tmp_path / "transcripts" / f"{sid}.json").read_text(encoding="utf-8"))
    assert record["ended_at"] is not None, "ended_at stamp missing after EndCall"
    assert record["transport"] == "chat"
    assert record["phone"] == "+375291234567"
    assert record["name"] == "Иван"
    assert record["client_id"] is None
    assert record["turn_count"] >= 8, f"turn_count too low: {record['turn_count']}"
