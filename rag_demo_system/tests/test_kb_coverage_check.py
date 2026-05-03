"""Unit tests for kb_coverage_check.

Section 7 Phase A.3 deliverable. Mock retriever — runs without network.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent.parent
    script_path = repo_root / "rag_demo_system" / "scripts" / "kb_coverage_check.py"
    spec = importlib.util.spec_from_file_location("kb_coverage_check", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kb_coverage_check"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


kcc = _load_module()


def test_classify_thresholds():
    assert kcc.classify(0.40, miss=0.65, cover=0.80) == "MISS"
    assert kcc.classify(0.64, miss=0.65, cover=0.80) == "MISS"
    # boundary inclusive on the lower side of PARTIAL
    assert kcc.classify(0.65, miss=0.65, cover=0.80) == "PARTIAL"
    assert kcc.classify(0.79, miss=0.65, cover=0.80) == "PARTIAL"
    # boundary inclusive on the lower side of COVERED
    assert kcc.classify(0.80, miss=0.65, cover=0.80) == "COVERED"
    assert kcc.classify(0.95, miss=0.65, cover=0.80) == "COVERED"


def test_load_queries_plain_text(tmp_path: Path):
    q_file = tmp_path / "q.txt"
    q_file.write_text(
        "\n".join(
            [
                "сколько стоит лизинг?",
                "",
                "# this is a comment",
                "  адрес офиса в Бресте  ",
            ],
        ),
        encoding="utf-8",
    )
    queries = kcc.load_queries(q_file)
    assert queries == ["сколько стоит лизинг?", "адрес офиса в Бресте"]


def test_load_queries_jsonl(tmp_path: Path):
    q_file = tmp_path / "q.jsonl"
    q_file.write_text(
        "\n".join(
            [
                json.dumps({"query": "test query 1"}),
                json.dumps({"q": "test query 2"}),
                json.dumps({"text": "test query 3"}),
                json.dumps({"user_text": "test query 4"}),
                json.dumps({"utterance": "test query 5"}),
                json.dumps({"unrelated": "should be skipped silently"}),
            ],
        ),
        encoding="utf-8",
    )
    queries = kcc.load_queries(q_file)
    assert queries == [
        "test query 1",
        "test query 2",
        "test query 3",
        "test query 4",
        "test query 5",
    ]


def test_load_queries_mixed_format_falls_through_for_non_json(tmp_path: Path):
    """A line that starts with `{` but isn't valid JSON falls through to plain text."""
    q_file = tmp_path / "q.txt"
    q_file.write_text("{not actually json\nplain query\n", encoding="utf-8")
    queries = kcc.load_queries(q_file)
    assert queries == ["{not actually json", "plain query"]


def test_parse_retrieve_response_with_top_rerank_score():
    payload = {
        "ok": True,
        "top_rerank_score": 0.87,
        "final": [
            {
                "text": "Условия автолизинга для физлиц.",
                "heading_path": ["Условия", "Автолизинг"],
                "rerank_score": 0.87,
            },
            {"text": "second chunk", "heading_path": ["Other"]},
        ],
    }
    r = kcc.parse_retrieve_response("какие условия?", payload)
    assert r.top_score == 0.87
    assert r.n_candidates == 2
    assert "Условия автолизинга" in r.top_chunk_excerpt
    assert r.top_heading_path == "Условия / Автолизинг"
    assert r.error is None


def test_parse_retrieve_response_falls_back_to_candidate_score():
    payload = {
        "ok": True,
        "candidates": [
            {"text": "chunk text", "score": 0.72, "heading_path": ["A"]},
        ],
    }
    r = kcc.parse_retrieve_response("q", payload)
    assert r.top_score == 0.72
    assert r.n_candidates == 1
    assert "chunk text" in r.top_chunk_excerpt


def test_parse_retrieve_response_handles_empty_response():
    r = kcc.parse_retrieve_response("q", {"ok": True})
    assert r.top_score == 0.0
    assert r.n_candidates == 0
    assert r.top_chunk_excerpt == ""


def test_run_coverage_with_mock_retriever():
    queries = ["q1", "q2", "q3"]

    def fake(q: str) -> kcc.RetrievalResult:
        return kcc.RetrievalResult(
            query=q, top_score=0.7, top_chunk_excerpt="ex", top_heading_path="h", n_candidates=1,
        )

    results = kcc.run_coverage(queries, fake, miss=0.65, cover=0.80)
    assert len(results) == 3
    assert all(r.top_score == 0.7 for r in results)


def test_render_report_classifies_and_includes_summary():
    results = [
        kcc.RetrievalResult(query="hit-q", top_score=0.92, top_chunk_excerpt="ex1", top_heading_path="A", n_candidates=1),
        kcc.RetrievalResult(query="partial-q", top_score=0.72, top_chunk_excerpt="ex2", top_heading_path="B", n_candidates=1),
        kcc.RetrievalResult(query="miss-q", top_score=0.30, top_chunk_excerpt="ex3", top_heading_path="C", n_candidates=1),
        kcc.RetrievalResult(
            query="err-q", top_score=-1.0, top_chunk_excerpt="", top_heading_path="", n_candidates=0,
            error="connection refused",
        ),
    ]
    out = kcc.render_report(
        results,
        miss=0.65,
        cover=0.80,
        queries_path=Path("/tmp/q.txt"),
        retriever_label="mock",
    )
    assert "KB Coverage Check" in out
    assert "**MISS:** 1" in out
    assert "**PARTIAL:** 1" in out
    assert "**COVERED:** 1" in out
    assert "**ERROR:** 1" in out
    assert "miss-q" in out
    assert "partial-q" in out
    assert "hit-q" in out
    assert "connection refused" in out
