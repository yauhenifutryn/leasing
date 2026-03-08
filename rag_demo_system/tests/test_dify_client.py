from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import rag_backends


class FakeResponse:
    def __init__(self, lines=None, json_data=None):
        self._lines = lines or []
        self._json = json_data or {}

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = True):
        return iter(self._lines)

    def json(self):
        return self._json


def test_dify_stream_collects_answer_and_retriever_resources(monkeypatch) -> None:
    import importlib

    dify_client = importlib.import_module("backend.dify_client")
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        calls.append({"url": url, "headers": headers, "json": json, "stream": stream})
        return FakeResponse(
            lines=[
                'data: {"event":"message","answer":"Добрый день"}',
                'data: {"event":"message","answer":"!","conversation_id":"conv-1","task_id":"task-1"}',
                'data: {"event":"message_end","metadata":{"retriever_resources":[{"document_name":"kb.md","segment_position":2,"content":"Фрагмент","score":0.8}]}}',
            ]
        )

    monkeypatch.setattr(dify_client.requests, "post", fake_post)

    response = dify_client.chat_once(
        base_url="http://dify.local/v1",
        api_key="secret",
        query="Какие условия?",
        user="session-1",
        inputs={"backend": "dify_rag"},
    )

    assert calls[0]["url"] == "http://dify.local/v1/chat-messages"
    assert calls[0]["stream"] is True
    assert calls[0]["json"]["inputs"] == {"backend": "dify_rag"}
    assert response.answer == "Добрый день!"
    assert response.backend == "dify_rag"
    assert response.conversation_ref == {
        "conversation_id": "conv-1",
        "task_id": "task-1",
    }
    assert response.used_knowledge == [
        {
            "chunk_id": "kb.md:2",
            "doc_name": "kb.md",
            "heading_path": [],
            "snippet": "Фрагмент",
            "score": 0.8,
        }
    ]


def test_dify_stop_generation_uses_task_id(monkeypatch) -> None:
    import importlib

    dify_client = importlib.import_module("backend.dify_client")
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        calls.append({"url": url, "headers": headers, "json": json, "stream": stream})
        return FakeResponse(json_data={"result": "success"})

    monkeypatch.setattr(dify_client.requests, "post", fake_post)

    dify_client.stop_generation(
        base_url="http://dify.local/v1",
        api_key="secret",
        task_id="task-9",
        user="session-1",
    )

    assert calls == [
        {
            "url": "http://dify.local/v1/chat-messages/task-9/stop",
            "headers": {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            "json": {"user": "session-1"},
            "stream": False,
        }
    ]


def test_map_dify_retriever_resources_is_shared_contract() -> None:
    used = rag_backends.map_dify_retriever_resources(
        [
            {
                "document_name": "kb.md",
                "segment_position": 4,
                "content": "Текст",
                "score": 0.3,
            }
        ]
    )

    assert used[0]["chunk_id"] == "kb.md:4"
