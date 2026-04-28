#!/usr/bin/env python3
"""Dump per-metric latency distributions from current backend log format.

Reads:
  .state/logs.jsonl   (event=chat with `timings` sub-object)
  .state/backend.log  (classifier print line: `[Classifier] result: (NNNms)`)

Usage (from rag_demo_system/):
  python3 scripts/dump_latency.py
  python3 scripts/dump_latency.py --since-minutes 60   # only the last hour
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

LOGS_JSONL = ".state/logs.jsonl"
BACKEND_LOG = ".state/backend.log"
CLASSIFIER_RE = re.compile(r"\[Classifier\] result: \(([0-9.]+)ms")


def load_metrics(since_ts: float | None) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    if os.path.exists(LOGS_JSONL):
        with open(LOGS_JSONL) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("event") != "chat":
                    continue
                if since_ts is not None:
                    ts = d.get("ts") or d.get("timestamp")
                    if isinstance(ts, (int, float)) and ts < since_ts:
                        continue
                for k, v in (d.get("timings") or {}).items():
                    if isinstance(v, (int, float)) and v > 0:
                        metrics.setdefault(k, []).append(float(v))

    classifier: list[float] = []
    if os.path.exists(BACKEND_LOG):
        with open(BACKEND_LOG) as f:
            for line in f:
                m = CLASSIFIER_RE.search(line)
                if m:
                    classifier.append(float(m.group(1)))
    if classifier:
        metrics["classifier_ms (backend.log)"] = classifier
    return metrics


def percentile(arr: list[float], q: float) -> float:
    n = len(arr)
    return arr[min(int(n * q), n - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-minutes", type=int, default=None,
                    help="Only include records from the last N minutes")
    args = ap.parse_args()

    since_ts = None
    if args.since_minutes:
        since_ts = time.time() - args.since_minutes * 60

    metrics = load_metrics(since_ts)
    if not metrics:
        print("No metrics found. Files checked:")
        print(f"  {LOGS_JSONL} exists={os.path.exists(LOGS_JSONL)}")
        print(f"  {BACKEND_LOG} exists={os.path.exists(BACKEND_LOG)}")
        return 1

    print(f"{'metric':<28} {'n':>5} {'p50':>7} {'p90':>7} {'p95':>7} {'p99':>7} {'max':>7}")
    print("-" * 72)
    for k in sorted(metrics):
        a = sorted(metrics[k])
        n = len(a)
        print(
            f"{k:<28} {n:>5} {percentile(a, 0.5):>7.0f} "
            f"{percentile(a, 0.9):>7.0f} {percentile(a, 0.95):>7.0f} "
            f"{percentile(a, 0.99):>7.0f} {a[-1]:>7.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
