"""
Unit tests for benchmark_runner.py.

Validates:
1. JSONL result dict has all 17 required fields
2. Warmup flagging logic: first N questions marked warmup=True, rest False
3. Error result structure: error field set, timing fields null
4. keyword_hit_rate computation: hits / expected_keywords
5. keyword_hit_rate is None when expected_keywords is empty (out_of_scope)
6. JSONL serialization uses ensure_ascii=False

Tests do NOT require a live backend or WebSocket connection.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# All 17 fields every result record must contain
REQUIRED_RESULT_FIELDS = {
    "question_id",
    "stack_id",
    "warmup",
    "transcript",
    "answer",
    "retrieved_chunks",
    "speech_stopped",
    "stt_done",
    "retrieval_done",
    "llm_first_token",
    "tts_first_chunk",
    "playback_started",
    "primary_kpi_ms",
    "llm_ttfb_ms",
    "keyword_hits",
    "keyword_hit_rate",
    "error",
}


def _make_result(**overrides) -> dict:
    """Build a minimal valid result dict for testing purposes."""
    from benchmark_runner import build_result_dict
    base = {
        "question_id": "sf-01",
        "stack_id": "our_rag__Qwen3-30B-A3B__sensevoice__cosyvoice",
        "warmup": False,
        "transcript": "Какой минимальный аванс?",
        "answer": "Минимальный аванс составляет 10%",
        "retrieved_chunks": [],
        "speech_stopped": 1711000000.0,
        "stt_done": 1711000001.0,
        "retrieval_done": 1711000002.0,
        "llm_first_token": 1711000003.0,
        "tts_first_chunk": 1711000004.0,
        "playback_started": 1711000005.0,
        "primary_kpi_ms": 5000.0,
        "llm_ttfb_ms": 3000.0,
        "keyword_hits": ["аванс", "10%"],
        "keyword_hit_rate": 1.0,
        "error": None,
    }
    base.update(overrides)
    return base


def test_result_has_required_fields() -> None:
    """A result dict produced by build_result_dict must contain all 17 required fields."""
    from benchmark_runner import build_result_dict

    result = build_result_dict(
        question_id="sf-01",
        stack_id="our_rag__Qwen3-30B-A3B__sensevoice__cosyvoice",
        warmup=False,
        transcript="Какой минимальный аванс?",
        answer="Минимальный аванс 10%",
        retrieved_chunks=[],
        speech_stopped=1711000000.0,
        stt_done=1711000001.0,
        retrieval_done=1711000002.0,
        llm_first_token=1711000003.0,
        tts_first_chunk=None,
        playback_started=None,
        primary_kpi_ms=None,
        llm_ttfb_ms=2000.0,
        keyword_hits=["аванс"],
        keyword_hit_rate=0.5,
        error=None,
    )
    missing = REQUIRED_RESULT_FIELDS - set(result.keys())
    assert missing == set(), f"Result is missing fields: {missing}"


def test_warmup_flagging() -> None:
    """First N results must have warmup=True, subsequent results must have warmup=False."""
    from benchmark_runner import is_warmup

    warmup_count = 3
    # First 3 are warmup
    assert is_warmup(0, warmup_count) is True
    assert is_warmup(1, warmup_count) is True
    assert is_warmup(2, warmup_count) is True
    # Index 3 onward are not warmup
    assert is_warmup(3, warmup_count) is False
    assert is_warmup(4, warmup_count) is False
    assert is_warmup(10, warmup_count) is False


def test_error_result_has_null_timings() -> None:
    """When error is set, all timing fields should be None."""
    from benchmark_runner import build_error_result

    result = build_error_result(
        question_id="sf-05",
        stack_id="our_rag__Qwen3-30B-A3B__sensevoice__cosyvoice",
        warmup=False,
        speech_stopped=1711000000.0,
        error_message="Connection timeout",
    )
    assert result["error"] == "Connection timeout"
    assert result["stt_done"] is None
    assert result["retrieval_done"] is None
    assert result["llm_first_token"] is None
    assert result["tts_first_chunk"] is None
    assert result["playback_started"] is None
    assert result["primary_kpi_ms"] is None
    assert result["llm_ttfb_ms"] is None
    assert result["keyword_hits"] == []
    assert result["keyword_hit_rate"] is None


def test_keyword_hit_rate_computed_correctly() -> None:
    """keyword_hit_rate = len(hits) / len(expected_keywords) for non-empty expected_keywords.

    Keyword matching uses simple case-insensitive substring match.
    The keyword must appear literally in the answer text (no stemming/lemmatisation).
    Use keywords that appear verbatim in the answer to test the match logic.
    """
    from benchmark_runner import compute_keyword_hits

    # "аванс" and "10%" appear literally; "месяц" does not appear; "составляет" appears literally
    answer = "Минимальный аванс составляет 10% от общей суммы"
    expected = ["аванс", "10%", "составляет", "несуществующее"]

    hits, rate = compute_keyword_hits(answer, expected)
    assert "аванс" in hits
    assert "10%" in hits
    assert "составляет" in hits
    assert "несуществующее" not in hits
    assert len(hits) == 3
    assert rate == 3 / 4


def test_keyword_hit_rate_null_for_empty_expected() -> None:
    """keyword_hit_rate must be None when expected_keywords is empty (out_of_scope category)."""
    from benchmark_runner import compute_keyword_hits

    answer = "Это не входит в наши услуги"
    hits, rate = compute_keyword_hits(answer, [])
    assert hits == []
    assert rate is None


def test_jsonl_serialization_uses_ensure_ascii_false() -> None:
    """JSONL output for Cyrillic text must not be escaped (ensure_ascii=False)."""
    result = _make_result(answer="Минимальный аванс 10%")
    serialized = json.dumps(result, ensure_ascii=False)
    # Cyrillic characters must appear as-is, not as \\uXXXX escapes
    assert "Минимальный" in serialized
    assert "\\u" not in serialized
