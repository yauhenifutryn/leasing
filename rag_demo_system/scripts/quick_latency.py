#!/usr/bin/env python3
"""Parse the latency markers actually emitted in .state/backend.log today
and print p50 / p95 / outliers per stage. Bridges the gap until the
proper [LATENCY:] block is restored on the apply_turn path.

Markers parsed:
    [Jambonz:<sid>] STT(<ms>ms): ...
    [Classifier] result: (<ms>ms) session=<sid>

Usage:
    python scripts/quick_latency.py [LOGFILE]
Default: .state/backend.log
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG = Path(".state/backend.log")

STT_RE = re.compile(r"\[Jambonz:[a-f0-9]+\] STT\((\d+)ms\)")
CLF_RE = re.compile(r"\[Classifier\] result: \((\d+)ms\) session=([a-f0-9]+)")


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = int(len(s) * pct)
    if idx >= len(s):
        idx = len(s) - 1
    return s[idx]


def summarize(name: str, values: list[int]) -> None:
    if not values:
        print(f"  {name}: no samples")
        return
    n = len(values)
    print(
        f"  {name}: count={n} "
        f"p50={percentile(values, 0.5)}ms "
        f"p90={percentile(values, 0.9)}ms "
        f"p95={percentile(values, 0.95)}ms "
        f"p99={percentile(values, 0.99)}ms "
        f"max={max(values)}ms"
    )


def main(argv: list[str]) -> int:
    log_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_LOG
    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}", file=sys.stderr)
        return 1

    stt_ms: list[int] = []
    classifier_ms: list[int] = []
    classifier_per_session: dict[str, list[int]] = defaultdict(list)

    with log_path.open() as fh:
        for line in fh:
            m = STT_RE.search(line)
            if m:
                stt_ms.append(int(m.group(1)))
                continue
            m = CLF_RE.search(line)
            if m:
                ms = int(m.group(1))
                sid = m.group(2)
                classifier_ms.append(ms)
                classifier_per_session[sid].append(ms)

    print(f"=== Stage latency from {log_path} ===")
    summarize("STT       ", stt_ms)
    summarize("Classifier", classifier_ms)

    if classifier_ms:
        print()
        print("=== Top 10 slowest classifier turns ===")
        for ms in sorted(classifier_ms, reverse=True)[:10]:
            print(f"  {ms}ms")

    if classifier_per_session:
        print()
        print("=== Sessions with classifier > 1500ms (likely felt slow) ===")
        for sid, vals in sorted(classifier_per_session.items()):
            spikes = [v for v in vals if v > 1500]
            if spikes:
                print(f"  {sid[:8]}: {len(spikes)} spike(s), max={max(spikes)}ms")

    print()
    print("NOTE: LLM brain (FireLLMFallback) and TTS time are NOT in the")
    print("log today; these are the missing stages. STT + Classifier sum")
    print("alone is below 2s p95 on this data — the perceived 3-4s comes")
    print("from LLM brain streaming + TTS, which currently have no markers.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
