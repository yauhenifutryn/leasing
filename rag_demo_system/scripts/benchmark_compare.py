"""
Benchmark comparison script.

Reads two JSONL result files produced by benchmark_runner.py and outputs
a side-by-side markdown metrics table for latency and quality comparison.

Usage:
    python benchmark_compare.py results_a.jsonl results_b.jsonl [--output comparison.md]

Warmup rows (warmup=True) are automatically excluded from all metrics per D-05.
Error rows are excluded from timing percentiles but counted in error_count.

Output format per D-07 and D-08:
    | Metric                  | Stack A | Stack B | Delta  | Winner  |
    |-------------------------|---------|---------|--------|---------|
    | Primary KPI mean (ms)   | 2100.0  | 1800.0  | -300.0 | B -->   |
    ...

For latency metrics: lower is better (arrow points to lower value).
For keyword hit rate: higher is better (arrow points to higher value).
For error count: lower is better (arrow points to lower value).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Pure helper functions (importable without side effects)
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL result file and return non-warmup rows only.

    Each line is a JSON object; lines with warmup=True are excluded per D-05.
    """
    results = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("warmup", False):
                results.append(row)
    return results


def percentiles(values: list[float | None]) -> dict[str, float | None]:
    """Compute mean, p50, and p95 for a list of values.

    None values in the list are ignored (treated as missing data).
    If the cleaned list is empty, all results are None.

    Uses statistics.mean for mean and statistics.quantiles(method='inclusive')
    for p50/p95. The 'inclusive' method matches the expected percentile
    boundaries for benchmark reporting.
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "p50": None, "p95": None}
    qs = statistics.quantiles(clean, n=100, method="inclusive")
    return {
        "mean": statistics.mean(clean),
        "p50": qs[49],   # 50th percentile (index 49 in 0-based 100-quantile list)
        "p95": qs[94],   # 95th percentile (index 94 in 0-based 100-quantile list)
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute benchmark metrics from a list of non-warmup result records.

    Timing metrics exclude rows where error is not None.
    Error count includes all rows with error != None.
    keyword_hit_rate mean excludes rows where keyword_hit_rate is None
    (i.e., out_of_scope questions with empty expected_keywords).
    """
    non_error = [r for r in results if r.get("error") is None]
    error_count = sum(1 for r in results if r.get("error") is not None)

    primary_kpi_values = [r["primary_kpi_ms"] for r in non_error]
    llm_ttfb_values = [r["llm_ttfb_ms"] for r in non_error]
    khr_values = [r["keyword_hit_rate"] for r in results if r.get("keyword_hit_rate") is not None]

    khr_mean: float | None = None
    if khr_values:
        khr_mean = statistics.mean(khr_values)

    return {
        "primary_kpi_ms": percentiles(primary_kpi_values),
        "llm_ttfb_ms": percentiles(llm_ttfb_values),
        "keyword_hit_rate": khr_mean,
        "error_count": error_count,
        "total_questions": len(results),
    }


def _fmt_ms(value: float | None) -> str:
    """Format a millisecond value to 1 decimal place, or 'N/A' if None."""
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def _fmt_rate(value: float | None) -> str:
    """Format a rate (0..1) to 2 decimal places, or 'N/A' if None."""
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _winner_latency(a: float | None, b: float | None) -> str:
    """Return winner indicator for a latency metric (lower is better)."""
    if a is None or b is None:
        return "N/A"
    if b < a:
        return "B -->"
    if a < b:
        return "<-- A"
    return "tie"


def _winner_quality(a: float | None, b: float | None) -> str:
    """Return winner indicator for a quality metric (higher is better)."""
    if a is None or b is None:
        return "N/A"
    if b > a:
        return "B -->"
    if a > b:
        return "<-- A"
    return "tie"


def _delta_ms(a: float | None, b: float | None) -> str:
    """Return delta string (B - A) for latency metrics, or 'N/A'."""
    if a is None or b is None:
        return "N/A"
    return f"{b - a:.1f}"


def _delta_rate(a: float | None, b: float | None) -> str:
    """Return delta string (B - A) for rate metrics, or 'N/A'."""
    if a is None or b is None:
        return "N/A"
    return f"{b - a:.2f}"


def _delta_int(a: int | None, b: int | None) -> str:
    """Return delta string (B - A) for integer metrics, or 'N/A'."""
    if a is None or b is None:
        return "N/A"
    return str(b - a)


def format_comparison_table(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    label_a: str = "Stack A",
    label_b: str = "Stack B",
) -> str:
    """Build a markdown side-by-side comparison table.

    Columns: Metric, Stack A, Stack B, Delta (B-A), Winner
    Rows: Primary KPI mean/p50/p95, LLM TTFB mean/p50/p95,
          Keyword hit rate, Error count, Total questions.

    For latency: lower is better.
    For keyword hit rate: higher is better.
    For error count: lower is better.
    """
    kpi_a = metrics_a["primary_kpi_ms"]
    kpi_b = metrics_b["primary_kpi_ms"]
    ttfb_a = metrics_a["llm_ttfb_ms"]
    ttfb_b = metrics_b["llm_ttfb_ms"]
    khr_a = metrics_a["keyword_hit_rate"]
    khr_b = metrics_b["keyword_hit_rate"]
    err_a = metrics_a["error_count"]
    err_b = metrics_b["error_count"]
    total_a = metrics_a["total_questions"]
    total_b = metrics_b["total_questions"]

    rows = [
        (
            "Primary KPI mean (ms)",
            _fmt_ms(kpi_a["mean"]),
            _fmt_ms(kpi_b["mean"]),
            _delta_ms(kpi_a["mean"], kpi_b["mean"]),
            _winner_latency(kpi_a["mean"], kpi_b["mean"]),
        ),
        (
            "Primary KPI p50 (ms)",
            _fmt_ms(kpi_a["p50"]),
            _fmt_ms(kpi_b["p50"]),
            _delta_ms(kpi_a["p50"], kpi_b["p50"]),
            _winner_latency(kpi_a["p50"], kpi_b["p50"]),
        ),
        (
            "Primary KPI p95 (ms)",
            _fmt_ms(kpi_a["p95"]),
            _fmt_ms(kpi_b["p95"]),
            _delta_ms(kpi_a["p95"], kpi_b["p95"]),
            _winner_latency(kpi_a["p95"], kpi_b["p95"]),
        ),
        (
            "LLM TTFB mean (ms)",
            _fmt_ms(ttfb_a["mean"]),
            _fmt_ms(ttfb_b["mean"]),
            _delta_ms(ttfb_a["mean"], ttfb_b["mean"]),
            _winner_latency(ttfb_a["mean"], ttfb_b["mean"]),
        ),
        (
            "LLM TTFB p50 (ms)",
            _fmt_ms(ttfb_a["p50"]),
            _fmt_ms(ttfb_b["p50"]),
            _delta_ms(ttfb_a["p50"], ttfb_b["p50"]),
            _winner_latency(ttfb_a["p50"], ttfb_b["p50"]),
        ),
        (
            "LLM TTFB p95 (ms)",
            _fmt_ms(ttfb_a["p95"]),
            _fmt_ms(ttfb_b["p95"]),
            _delta_ms(ttfb_a["p95"], ttfb_b["p95"]),
            _winner_latency(ttfb_a["p95"], ttfb_b["p95"]),
        ),
        (
            "Keyword hit rate",
            _fmt_rate(khr_a),
            _fmt_rate(khr_b),
            _delta_rate(khr_a, khr_b),
            _winner_quality(khr_a, khr_b),
        ),
        (
            "Error count",
            str(err_a),
            str(err_b),
            _delta_int(err_a, err_b),
            _winner_latency(float(err_a), float(err_b)),  # lower is better
        ),
        (
            "Total questions",
            str(total_a),
            str(total_b),
            _delta_int(total_a, total_b),
            "N/A",
        ),
    ]

    # Build the markdown table
    header = f"| Metric | {label_a} | {label_b} | Delta | Winner |"
    separator = "|--------|---------|---------|-------|--------|"
    lines = [header, separator]
    for metric, val_a, val_b, delta, winner in rows:
        lines.append(f"| {metric} | {val_a} | {val_b} | {delta} | {winner} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two benchmark JSONL result files and produce a markdown metrics table.",
    )
    parser.add_argument(
        "file_a",
        type=Path,
        help="First result JSONL file (Stack A)",
    )
    parser.add_argument(
        "file_b",
        type=Path,
        help="Second result JSONL file (Stack B)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output markdown file path (default: stdout)",
    )

    args = parser.parse_args()

    if not args.file_a.exists():
        print(f"Error: file not found: {args.file_a}", file=sys.stderr)
        sys.exit(1)
    if not args.file_b.exists():
        print(f"Error: file not found: {args.file_b}", file=sys.stderr)
        sys.exit(1)

    results_a = load_results(args.file_a)
    results_b = load_results(args.file_b)

    metrics_a = compute_metrics(results_a)
    metrics_b = compute_metrics(results_b)

    label_a = args.file_a.stem
    label_b = args.file_b.stem

    table = format_comparison_table(metrics_a, metrics_b, label_a, label_b)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
        print(f"Comparison table written to: {args.output}")
    else:
        print(table)
