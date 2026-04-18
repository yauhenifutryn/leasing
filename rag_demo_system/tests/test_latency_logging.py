"""Contract tests for per-turn latency summary logging.

If this regresses, we lose the ability to diagnose outliers.
"""
from pathlib import Path


_APP_PY = Path(__file__).resolve().parents[1] / "backend" / "app.py"


def test_latency_summary_log_present():
    src = _APP_PY.read_text(encoding="utf-8")
    assert "[LATENCY:{session_id[:8]}]" in src, (
        "Fix 22 regression: per-turn LATENCY summary log removed"
    )
    # Required keys must all be present in the summary.
    for key in ("classifier_ms", "rag_ms", "llm_first_ms", "tts_first_ms",
                "total_e2e_ms", "user_len", "out_tokens", "path"):
        assert f"{key}=" in src, f"Fix 22 regression: latency summary missing {key}"


def test_rag_ms_variable_captured():
    src = _APP_PY.read_text(encoding="utf-8")
    # _t_rag_ms must be assigned as a standalone variable so the summary can read it.
    assert "_t_rag_ms = " in src, (
        "Fix 22 regression: _t_rag_ms not captured — summary cannot read RAG time"
    )


def test_summary_handles_missing_fields():
    """If classifier was skipped (fast-path), summary must emit -1 not crash."""
    src = _APP_PY.read_text(encoding="utf-8")
    # The `if '_t_classify_ms' in locals()` check protects fast-path turns.
    assert "'_t_classify_ms' in locals()" in src, (
        "Fix 22 regression: summary not defensive against missing classifier timing"
    )
