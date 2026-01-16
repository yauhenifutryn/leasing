from pathlib import Path
import json
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


def test_streamed_consent_response_has_final_type() -> None:
    client = TestClient(app)
    resp = client.post("/api/chat", json={"message": "да", "stream": True})
    lines = [line for line in resp.text.splitlines() if line.startswith("data:")]
    assert lines, "streamed response missing data line"
    payload_text = lines[0].split("data:", 1)[1].strip()
    payload_text = payload_text.replace("\\r", "").replace("\\n", "")
    payload = json.loads(payload_text)
    assert payload.get("type") == "final"
