from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import app, engine
from backend.llm import LLMResponse
from backend import llm as llm_mod


def test_chat_includes_memory_block(monkeypatch) -> None:
    captured = {"prompt": None}

    def fake_retrieve(message: str, fast: bool = False, voice_fast: bool = False, session_id: str | None = None):
        return {
            "ok": True,
            "final": [
                {
                    "chunk_id": "c1",
                    "text": "Факт из базы знаний.",
                    "heading_path": [],
                    "source": "kb",
                    "doc_name": "kb",
                    "start_char": 0,
                    "end_char": 10,
                }
            ],
            "weak": False,
            "normalized_query": message,
            "rewritten_query": message,
            "top_rerank_score": 1.0,
            "candidates": [],
        }

    def fake_call(*args, **kwargs):
        captured["prompt"] = kwargs.get("user_prompt")
        return LLMResponse(text="ok", raw={})

    monkeypatch.setattr(engine, "retrieve", fake_retrieve)
    monkeypatch.setattr(llm_mod, "call_openai_compatible", fake_call)

    client = TestClient(app)
    consent_resp = client.post("/api/chat", json={"message": "да"})
    session_id = consent_resp.json()["session_id"]

    client.post("/api/chat", json={"message": "Первый вопрос", "session_id": session_id})
    client.post("/api/chat", json={"message": "Второй вопрос", "session_id": session_id})

    prompt = captured["prompt"] or ""
    assert "Контекст диалога" in prompt
    assert "Клиент: Первый вопрос" in prompt
