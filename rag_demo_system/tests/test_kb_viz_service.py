"""Tests for kb_viz_service.

Substitutes fakes for the embedder, UMAP reducer, and Qdrant client so the
test does not require sentence-transformers or a live Qdrant. Exercises:
  - /health returns expected shape
  - /overlay_query returns a well-formed response with query_id
  - /overlay_query with kind="2d" returns a 2-coordinate position
  - bearer token auth accepts the correct token and rejects everything else
  - 404 on HTML paths before render has run
  - /feedback appends a JSONL record
  - /feedback requires a comment when verdict == "wrong"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

SERVICES_DIR = Path(__file__).resolve().parents[1] / "services"
if str(SERVICES_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR.parent))


class _FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeReducer:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def transform(self, arr):
        return [[float(i) for i in range(self.n_components)]]


class _FakeHit:
    def __init__(self, id_: str, score: float, payload: dict) -> None:
        self.id = id_
        self.score = score
        self.payload = payload


class _FakeQdrant:
    def search(self, collection_name, query_vector, limit, with_payload, with_vectors):
        return [
            _FakeHit(
                id_=f"chunk-{i}",
                score=0.9 - i * 0.1,
                payload={
                    "chunk_id": f"chunk-{i}",
                    "text": "Long Russian text about leasing " * 10,
                    "heading_path": ["pricing", "monthly-fee"],
                },
            )
            for i in range(limit)
        ]


def _reload_service() -> None:
    mod_name = "services.kb_viz_service"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    if "services" in sys.modules:
        del sys.modules["services"]


def _fresh(monkeypatch, tmp_path, token: str | None):
    if token is None:
        monkeypatch.delenv("KB_VIZ_OVERLAY_TOKEN", raising=False)
    else:
        monkeypatch.setenv("KB_VIZ_OVERLAY_TOKEN", token)
    monkeypatch.setenv("KB_VIZ_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("KB_VIZ_STATE_DIR", str(tmp_path / ".state"))
    _reload_service()
    from services import kb_viz_service as svc

    svc.STATE._embedder = _FakeEmbedder()
    svc.STATE._reducers = {"2d": _FakeReducer(2), "3d": _FakeReducer(3)}
    svc.STATE._qdrant = _FakeQdrant()
    return TestClient(svc.app), svc


@pytest.fixture
def open_client(monkeypatch, tmp_path):
    return _fresh(monkeypatch, tmp_path, token=None)


@pytest.fixture
def secured_client(monkeypatch, tmp_path):
    return _fresh(monkeypatch, tmp_path, token="secret-xyz")


def test_health(open_client) -> None:
    client, _ = open_client
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["service"] == "kb_viz"
    assert body["token_required"] is False
    # Must expose the Qdrant collection so operators can confirm the viz is
    # reading the same index the voice pipeline is writing.
    assert body["qdrant_coll"] == "micro_leasing_kb"
    assert body["embed_model"] == "intfloat/multilingual-e5-large"


def test_overlay_query_3d_returns_query_id(open_client) -> None:
    client, _ = open_client
    res = client.post("/overlay_query", json={"text": "сколько стоит", "kind": "3d", "top_k": 3})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["kind"] == "3d"
    assert len(body["position"]) == 3
    assert len(body["top_k"]) == 3
    assert isinstance(body.get("query_id"), str) and len(body["query_id"]) >= 16
    first = body["top_k"][0]
    assert first["chunk_id"].startswith("chunk-")
    assert 0.0 <= first["score"] <= 1.0
    # Section comes from heading_path[1] now (heading_path[0] is the
    # doc-level root and was the same for every chunk — useless as a label).
    assert first["section"] == "monthly-fee"
    assert first["text_preview"].endswith("…")
    # Full text is also returned so the UI can click-to-expand without a
    # second round-trip to the server.
    assert first["text_full"]
    assert len(first["text_full"]) >= len(first["text_preview"])


def test_overlay_query_2d(open_client) -> None:
    client, _ = open_client
    res = client.post("/overlay_query", json={"text": "что нужно", "kind": "2d"})
    assert res.status_code == 200
    body = res.json()
    assert body["kind"] == "2d"
    assert len(body["position"]) == 2


def test_auth_rejects_missing_token(secured_client) -> None:
    client, _ = secured_client
    res = client.post("/overlay_query", json={"text": "q", "kind": "3d"})
    assert res.status_code == 401


def test_auth_rejects_wrong_token(secured_client) -> None:
    client, _ = secured_client
    res = client.post(
        "/overlay_query",
        json={"text": "q", "kind": "3d"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert res.status_code == 401


def test_auth_accepts_right_token(secured_client) -> None:
    client, _ = secured_client
    res = client.post(
        "/overlay_query",
        json={"text": "q", "kind": "3d"},
        headers={"Authorization": "Bearer secret-xyz"},
    )
    assert res.status_code == 200


def test_static_404_before_render(open_client) -> None:
    client, _ = open_client
    assert client.get("/").status_code == 404
    assert client.get("/3d").status_code == 404


def test_input_validation(open_client) -> None:
    client, _ = open_client
    assert client.post("/overlay_query", json={"text": "", "kind": "3d"}).status_code == 422
    assert client.post("/overlay_query", json={"text": "ok", "kind": "4d"}).status_code == 422
    assert client.post("/overlay_query", json={"text": "ok", "kind": "2d", "top_k": 100}).status_code == 422


def test_feedback_correct_appends_jsonl(open_client) -> None:
    client, svc = open_client
    q = client.post("/overlay_query", json={"text": "стоимость", "kind": "3d"}).json()
    payload = {
        "query_id": q["query_id"],
        "query_text": "стоимость",
        "kind": "3d",
        "verdict": "correct",
        "top_k": [
            {"chunk_id": m["chunk_id"], "section": m["section"], "score": m["score"]}
            for m in q["top_k"]
        ],
        "client_id": "demo-client-1",
    }
    res = client.post("/feedback", json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ok"] is True

    log = Path(body["log_path"])
    assert log.exists()
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["verdict"] == "correct"
    assert record["query_text"] == "стоимость"
    assert record["client_id"] == "demo-client-1"
    assert len(record["top_k"]) == 5
    assert record["comment"] is None
    assert "ts" in record


def test_feedback_wrong_requires_comment(open_client) -> None:
    client, _ = open_client
    q = client.post("/overlay_query", json={"text": "x", "kind": "3d"}).json()
    bad = client.post(
        "/feedback",
        json={
            "query_id": q["query_id"],
            "query_text": "x",
            "kind": "3d",
            "verdict": "wrong",
            "top_k": [],
        },
    )
    assert bad.status_code == 422

    good = client.post(
        "/feedback",
        json={
            "query_id": q["query_id"],
            "query_text": "x",
            "kind": "3d",
            "verdict": "wrong",
            "comment": "Нашло не то — я спрашивал про ИП, а показало про физлицо.",
            "top_k": [],
        },
    )
    assert good.status_code == 200


def test_feedback_invalid_verdict(open_client) -> None:
    client, _ = open_client
    res = client.post(
        "/feedback",
        json={
            "query_id": "q",
            "query_text": "q",
            "kind": "3d",
            "verdict": "maybe",
            "top_k": [],
        },
    )
    assert res.status_code == 422


def test_feedback_respects_token(secured_client) -> None:
    client, _ = secured_client
    payload = {
        "query_id": "q",
        "query_text": "q",
        "kind": "3d",
        "verdict": "correct",
        "top_k": [],
    }
    assert client.post("/feedback", json=payload).status_code == 401
    assert (
        client.post(
            "/feedback",
            json=payload,
            headers={"Authorization": "Bearer secret-xyz"},
        ).status_code
        == 200
    )


def test_feedback_appends_two_lines(open_client) -> None:
    client, _ = open_client
    for i in range(2):
        q = client.post("/overlay_query", json={"text": f"q{i}", "kind": "3d"}).json()
        res = client.post(
            "/feedback",
            json={
                "query_id": q["query_id"],
                "query_text": f"q{i}",
                "kind": "3d",
                "verdict": "correct",
                "top_k": [],
            },
        )
        assert res.status_code == 200
    log = Path(res.json()["log_path"])
    lines = [json.loads(ln) for ln in log.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["query_text"] == "q0"
    assert lines[1]["query_text"] == "q1"


def test_coverage_empty(open_client) -> None:
    client, _ = open_client
    res = client.get("/coverage")
    assert res.status_code == 200
    body = res.json()
    assert body["total_feedback"] == 0
    assert body["unique_chunks_validated"] == 0
    assert body["per_chunk"] == {}
    assert body["per_section"] == {}


def test_coverage_aggregates_feedback(open_client) -> None:
    client, _ = open_client

    def submit(verdict: str, chunk_ids: list[str], section: str, comment: str | None = None) -> None:
        payload = {
            "query_id": f"q-{verdict}-{'-'.join(chunk_ids)}",
            "query_text": "test query",
            "kind": "3d",
            "verdict": verdict,
            "top_k": [
                {"chunk_id": cid, "section": section, "score": 0.5}
                for cid in chunk_ids
            ],
        }
        if comment:
            payload["comment"] = comment
        res = client.post("/feedback", json=payload)
        assert res.status_code == 200, res.text

    submit("correct", ["c-1", "c-2"], "Стоимость")
    submit("correct", ["c-2", "c-3"], "Стоимость")
    submit("wrong", ["c-4"], "Документы", comment="Ответ про паспорт, а надо про УНП.")

    res = client.get("/coverage")
    assert res.status_code == 200
    body = res.json()
    assert body["total_feedback"] == 3
    assert body["unique_chunks_validated"] == 4

    pc = body["per_chunk"]
    assert pc["c-1"]["correct"] == 1 and pc["c-1"]["wrong"] == 0
    assert pc["c-2"]["correct"] == 2 and pc["c-2"]["wrong"] == 0
    assert pc["c-3"]["correct"] == 1
    assert pc["c-4"]["wrong"] == 1 and pc["c-4"]["correct"] == 0
    assert pc["c-4"]["last_verdict"] == "wrong"
    assert pc["c-1"]["section"] == "Стоимость"

    ps = body["per_section"]
    assert ps["Стоимость"]["correct"] == 2
    assert ps["Стоимость"]["wrong"] == 0
    assert ps["Стоимость"]["unique_chunks"] == 3
    assert ps["Документы"]["wrong"] == 1
    assert ps["Документы"]["unique_chunks"] == 1


def test_coverage_ignores_malformed_jsonl_lines(open_client, tmp_path) -> None:
    client, svc = open_client
    log = svc.STATE.feedback_log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "\n".join(
            [
                "",
                "this is not json",
                json.dumps(
                    {
                        "ts": "2026-04-21T10:00:00Z",
                        "verdict": "correct",
                        "top_k": [{"chunk_id": "x-1", "section": "A"}],
                    }
                ),
                "   ",
            ]
        ),
        encoding="utf-8",
    )
    body = client.get("/coverage").json()
    assert body["total_feedback"] == 1
    assert body["unique_chunks_validated"] == 1
    assert body["per_chunk"]["x-1"]["correct"] == 1


def test_coverage_per_user_breakdown(open_client) -> None:
    client, _ = open_client

    def submit(user: str, verdict: str, chunk_ids: list[str], section: str, comment: str | None = None) -> None:
        payload = {
            "query_id": f"q-{user}-{verdict}-{'-'.join(chunk_ids)}",
            "query_text": f"q from {user}",
            "kind": "3d",
            "verdict": verdict,
            "client_id": user,
            "top_k": [{"chunk_id": cid, "section": section, "score": 0.5} for cid in chunk_ids],
        }
        if comment:
            payload["comment"] = comment
        res = client.post("/feedback", json=payload)
        assert res.status_code == 200, res.text

    # sasha validates 2 chunks correct, 1 wrong
    submit("sasha", "correct", ["c-1", "c-2"], "Стоимость")
    submit("sasha", "wrong", ["c-5"], "Документы", comment="wrong section returned")
    # john validates 2 chunks correct (one overlaps with sasha)
    submit("john", "correct", ["c-2", "c-3"], "Стоимость")
    # anonymous (no client_id) adds one more correct
    res = client.post(
        "/feedback",
        json={
            "query_id": "q-anon",
            "query_text": "anon",
            "kind": "3d",
            "verdict": "correct",
            "top_k": [{"chunk_id": "c-4", "section": "Офисы", "score": 0.4}],
        },
    )
    assert res.status_code == 200

    body = client.get("/coverage").json()
    assert body["total_feedback"] == 4

    users = body["per_user"]
    assert "sasha" in users
    assert "john" in users
    # anonymous becomes "ip:<addr>" via remote_addr, or "anon" in test env
    other_keys = [k for k in users.keys() if k not in ("sasha", "john")]
    assert len(other_keys) == 1

    assert users["sasha"]["total_events"] == 2
    assert sorted(users["sasha"]["correct_chunks"]) == ["c-1", "c-2"]
    assert users["sasha"]["wrong_chunks"] == ["c-5"]

    assert users["john"]["total_events"] == 1
    assert sorted(users["john"]["correct_chunks"]) == ["c-2", "c-3"]
    assert users["john"]["wrong_chunks"] == []

    # Aggregate consistency: c-2 should have correct=2 (sasha + john)
    assert body["per_chunk"]["c-2"]["correct"] == 2


def test_profiles_empty(open_client) -> None:
    client, _ = open_client
    res = client.get("/profiles")
    assert res.status_code == 200
    assert res.json() == {"profiles": []}


def test_profiles_upsert_create_then_touch(open_client) -> None:
    client, svc = open_client
    r1 = client.post("/profiles", json={"name": "sasha"})
    assert r1.status_code == 200, r1.text
    rec1 = r1.json()
    assert rec1["name"] == "sasha"
    assert rec1["created_ts"]
    assert rec1["last_seen_ts"] == rec1["created_ts"]

    # Same name touches last_seen but preserves created_ts
    r2 = client.post("/profiles", json={"name": "sasha"})
    assert r2.status_code == 200
    rec2 = r2.json()
    assert rec2["created_ts"] == rec1["created_ts"]
    assert rec2["last_seen_ts"] >= rec1["last_seen_ts"]

    client.post("/profiles", json={"name": "john"})
    listing = client.get("/profiles").json()["profiles"]
    names = [p["name"] for p in listing]
    assert sorted(names) == ["john", "sasha"]


def test_profiles_auto_touch_on_feedback(open_client) -> None:
    client, svc = open_client
    q = client.post(
        "/overlay_query",
        json={"text": "сколько", "kind": "3d", "client_id": "maria"},
    ).json()
    client.post(
        "/feedback",
        json={
            "query_id": q["query_id"],
            "query_text": "сколько",
            "kind": "3d",
            "verdict": "correct",
            "client_id": "maria",
            "top_k": [],
        },
    )
    listing = client.get("/profiles").json()["profiles"]
    names = [p["name"] for p in listing]
    assert "maria" in names


def test_profiles_reject_blank(open_client) -> None:
    client, _ = open_client
    res = client.post("/profiles", json={"name": "   "})
    # Either 422 from pydantic min_length (after strip happens upstream in client)
    # or 422 from upsert ValueError (the body passes min_length=1 because it is
    # whitespace, but upsert strips and rejects).
    assert res.status_code == 422


def test_profiles_respects_token(secured_client) -> None:
    """Both GET and POST must require the bearer when a token is configured,
    so a cross-origin page cannot harvest participant names via the open
    wildcard CORS policy.
    """
    client, _ = secured_client
    assert client.post("/profiles", json={"name": "x"}).status_code == 401
    assert client.get("/profiles").status_code == 401
    headers = {"Authorization": "Bearer secret-xyz"}
    assert (
        client.post("/profiles", json={"name": "x"}, headers=headers).status_code
        == 200
    )
    assert client.get("/profiles", headers=headers).status_code == 200


def test_coverage_respects_token(secured_client) -> None:
    """/coverage leaks per-user + IP info, must be gated when token is set."""
    client, _ = secured_client
    assert client.get("/coverage").status_code == 401
    assert (
        client.get("/coverage", headers={"Authorization": "Bearer secret-xyz"}).status_code
        == 200
    )
