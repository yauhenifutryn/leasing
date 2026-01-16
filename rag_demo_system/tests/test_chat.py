import os

import pytest
import requests

BASE_URL = os.getenv("RAG_DEMO_BASE_URL", "http://127.0.0.1:8000")


def _server_up() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _server_up(), reason="RAG demo backend not running")
def test_chat_used_knowledge():
    # index KB (skip if qdrant not running)
    try:
        resp = requests.post(f"{BASE_URL}/api/index", timeout=30)
        if resp.status_code != 200:
            pytest.skip("Index failed; Qdrant likely down")
    except Exception:
        pytest.skip("Index failed; Qdrant likely down")

    # consent
    requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": "да, согласен"},
        timeout=10,
    )

    # chat
    resp = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": "Какие требования к лизингу грузового транспорта?"},
        timeout=30,
    )
    data = resp.json()
    assert data.get("used_knowledge"), "used_knowledge is missing"
    assert data["used_knowledge"][0].get("chunk_id"), "chunk_id is missing"
