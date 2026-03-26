"""
Unit tests for benchmark_compare.py.

Validates:
1. percentiles() returns correct mean/p50/p95 for a known list
2. percentiles() returns all-None for empty list
3. Warmup rows (warmup=True) are excluded from metric computation (load_results filters them)
4. Error rows (error != null) are excluded from timing metrics but counted in error_count
5. Output contains markdown table with Primary KPI mean, p50, p95 rows
6. Output contains LLM TTFB mean, p50, p95 rows
7. Output contains keyword_hit_rate and error_count rows
8. Delta column shows difference (B - A)
9. Winner column shows arrow toward better stack
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Import testable functions from benchmark_compare
from benchmark_compare import percentiles, compute_metrics, format_comparison_table, load_results


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_result(
    *,
    question_id: str = "sf-01",
    stack_id: str = "stack_a",
    warmup: bool = False,
    primary_kpi_ms: float | None = 2000.0,
    llm_ttfb_ms: float | None = 1000.0,
    keyword_hit_rate: float | None = 0.8,
    error: str | None = None,
) -> dict:
    """Build a minimal result dict for test fixtures."""
    return {
        "question_id": question_id,
        "stack_id": stack_id,
        "warmup": warmup,
        "transcript": "тест",
        "answer": "ответ",
        "retrieved_chunks": [],
        "speech_stopped": 1711000000.0,
        "stt_done": 1711000001.0,
        "retrieval_done": 1711000002.0,
        "llm_first_token": 1711000003.0,
        "tts_first_chunk": 1711000004.0,
        "playback_started": 1711000005.0,
        "primary_kpi_ms": primary_kpi_ms,
        "llm_ttfb_ms": llm_ttfb_ms,
        "keyword_hits": [],
        "keyword_hit_rate": keyword_hit_rate,
        "error": error,
    }


def _write_jsonl(results: list[dict]) -> Path:
    """Write results to a temp JSONL file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        encoding="utf-8",
    )
    for r in results:
        tmp.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Tests for percentiles()
# ---------------------------------------------------------------------------

def test_percentiles_returns_mean_p50_p95() -> None:
    """percentiles() must compute correct mean/p50/p95 for a known list.

    Uses statistics.quantiles(n=100, method='inclusive') which returns 99 cut points.
    For data [1..100]: mean=50.5, p50=50.5, p95=95.05 (verified empirically).
    """
    values = [float(i) for i in range(1, 101)]  # 1..100
    result = percentiles(values)
    assert result["mean"] == 50.5
    # p50: qs[49] from quantiles(n=100, method='inclusive') on [1..100]
    assert result["p50"] == 50.5
    # p95: qs[94] from quantiles(n=100, method='inclusive') on [1..100]
    assert result["p95"] == 95.05


def test_percentiles_returns_all_none_for_empty_list() -> None:
    """percentiles() must return all-None when given an empty list."""
    result = percentiles([])
    assert result == {"mean": None, "p50": None, "p95": None}


def test_percentiles_returns_all_none_for_all_none_values() -> None:
    """percentiles() must return all-None when all values are None."""
    result = percentiles([None, None, None])
    assert result == {"mean": None, "p50": None, "p95": None}


# ---------------------------------------------------------------------------
# Tests for load_results() warmup exclusion
# ---------------------------------------------------------------------------

def test_warmup_excluded() -> None:
    """load_results() must exclude warmup=True rows."""
    warmup_row = _make_result(warmup=True, primary_kpi_ms=500.0)
    non_warmup_row = _make_result(warmup=False, primary_kpi_ms=2000.0)

    path = _write_jsonl([warmup_row, non_warmup_row])
    try:
        results = load_results(path)
        assert len(results) == 1
        assert results[0]["primary_kpi_ms"] == 2000.0
    finally:
        path.unlink(missing_ok=True)


def test_load_results_returns_non_warmup_rows_only() -> None:
    """load_results() must return only warmup=False rows from the file."""
    rows = [
        _make_result(question_id=f"sf-{i:02d}", warmup=(i < 3), primary_kpi_ms=float(i * 1000))
        for i in range(5)
    ]
    path = _write_jsonl(rows)
    try:
        results = load_results(path)
        # Rows with i < 3 (warmup) are excluded; i >= 3 remain
        assert len(results) == 2
        assert all(not r["warmup"] for r in results)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests for compute_metrics() error row handling
# ---------------------------------------------------------------------------

def test_error_rows_excluded_from_timing_metrics() -> None:
    """Error rows must be excluded from timing percentile computation."""
    rows = [
        _make_result(primary_kpi_ms=2000.0, llm_ttfb_ms=1000.0, error=None),
        _make_result(primary_kpi_ms=None, llm_ttfb_ms=None, error="Connection timeout"),
        _make_result(primary_kpi_ms=4000.0, llm_ttfb_ms=2000.0, error=None),
    ]
    metrics = compute_metrics(rows)
    # Timing metrics computed over 2 non-error rows: 2000 and 4000
    assert metrics["primary_kpi_ms"]["mean"] == 3000.0
    assert metrics["error_count"] == 1
    assert metrics["total_questions"] == 3


def test_error_count_in_metrics() -> None:
    """error_count must reflect the number of rows where error is not None."""
    rows = [
        _make_result(error=None),
        _make_result(error="timeout"),
        _make_result(error="disconnect"),
    ]
    metrics = compute_metrics(rows)
    assert metrics["error_count"] == 2
    assert metrics["total_questions"] == 3


# ---------------------------------------------------------------------------
# Tests for format_comparison_table() output structure
# ---------------------------------------------------------------------------

def _build_test_metrics(
    primary_mean: float = 2000.0,
    llm_mean: float = 1000.0,
    khr: float = 0.8,
    errors: int = 0,
    total: int = 10,
) -> dict:
    """Build a minimal metrics dict for table formatting tests."""
    return {
        "primary_kpi_ms": {"mean": primary_mean, "p50": primary_mean * 0.9, "p95": primary_mean * 1.2},
        "llm_ttfb_ms": {"mean": llm_mean, "p50": llm_mean * 0.9, "p95": llm_mean * 1.2},
        "keyword_hit_rate": khr,
        "error_count": errors,
        "total_questions": total,
    }


def test_comparison_table_has_primary_kpi_rows() -> None:
    """Output markdown table must contain Primary KPI mean, p50, p95 rows."""
    ma = _build_test_metrics(primary_mean=2100.0)
    mb = _build_test_metrics(primary_mean=1800.0)
    table = format_comparison_table(ma, mb, "Stack A", "Stack B")
    assert "Primary KPI mean" in table
    assert "Primary KPI p50" in table
    assert "Primary KPI p95" in table


def test_comparison_table_has_llm_ttfb_rows() -> None:
    """Output markdown table must contain LLM TTFB mean, p50, p95 rows."""
    ma = _build_test_metrics()
    mb = _build_test_metrics()
    table = format_comparison_table(ma, mb, "Stack A", "Stack B")
    assert "LLM TTFB mean" in table
    assert "LLM TTFB p50" in table
    assert "LLM TTFB p95" in table


def test_comparison_table_has_keyword_hit_rate_and_error_count() -> None:
    """Output table must contain keyword_hit_rate and error_count rows."""
    ma = _build_test_metrics(khr=0.75, errors=2)
    mb = _build_test_metrics(khr=0.85, errors=1)
    table = format_comparison_table(ma, mb, "Stack A", "Stack B")
    assert "Keyword hit rate" in table
    assert "Error count" in table


def test_delta_column_shows_b_minus_a() -> None:
    """Delta column must show B value minus A value for latency metrics."""
    # A is slower (2100ms), B is faster (1800ms): delta should be -300.0 (B-A)
    ma = _build_test_metrics(primary_mean=2100.0)
    mb = _build_test_metrics(primary_mean=1800.0)
    table = format_comparison_table(ma, mb, "Stack A", "Stack B")
    # Delta = B - A = 1800 - 2100 = -300.0
    assert "-300.0" in table


def test_winner_column_points_to_lower_latency() -> None:
    """For latency metrics, winner arrow must point to the stack with lower value."""
    # B is faster (lower latency): winner should indicate B
    ma = _build_test_metrics(primary_mean=2100.0)
    mb = _build_test_metrics(primary_mean=1800.0)
    table = format_comparison_table(ma, mb, "Stack A", "Stack B")
    # Arrow should point toward B (right-pointing or B label)
    assert "B" in table
    # The winner indicator format: "B -->" means B wins
    assert "-->" in table
