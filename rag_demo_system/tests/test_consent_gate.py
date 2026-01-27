from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import app
from backend import router as router_mod


def test_chat_requires_consent_before_llm(monkeypatch) -> None:
    called = {"llm": False}

    def fake_call(*args, **kwargs):
        called["llm"] = True
        raise RuntimeError("LLM should not be called without consent")

    monkeypatch.setattr(router_mod, "call_openai_compatible", fake_call)

    client = TestClient(app)
    resp = client.post("/api/chat", json={"message": "привет"})
    data = resp.json()
    assert data.get("consent") == "needed"
    assert called["llm"] is False
