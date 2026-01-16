from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import app


def test_chat_stream_flag_in_body_triggers_sse() -> None:
    client = TestClient(app)
    resp = client.post("/api/chat", json={"message": "да", "stream": True})
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "data:" in resp.text
