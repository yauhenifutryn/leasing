#!/usr/bin/env python3
"""Aggregate KB gaps across all session analysis reports.

Reads .state/analysis/session_reports.jsonl and produces a ranked list
of topics that clients asked about but the KB could not answer.

Usage:
    python scripts/kb_gap_report.py
    python scripts/kb_gap_report.py --min-count 2 --reports .state/analysis/session_reports.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default=".state/analysis/session_reports.jsonl")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum occurrences to show")
    args = parser.parse_args()

    reports_path = Path(args.reports)
    if not reports_path.exists():
        print(f"No reports file: {reports_path}")
        return

    gap_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    total_sessions = 0
    total_scores: list[float] = []

    for line in reports_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue

        if report.get("skipped") or report.get("error"):
            continue

        total_sessions += 1
        score = report.get("overall_score")
        if score is not None:
            total_scores.append(float(score))

        for gap in report.get("kb_gaps", []):
            gap_counter[gap.strip().lower()] += 1

        for issue in report.get("issues", []):
            severity = issue.get("severity", "minor")
            issue_type = issue.get("type", "unknown")
            issue_counter[f"[{severity}] {issue_type}"] += 1

    if not total_sessions:
        print("No valid session reports found")
        return

    avg_score = sum(total_scores) / len(total_scores) if total_scores else 0

    print(f"Sessions analyzed: {total_sessions}")
    print(f"Average quality score: {avg_score:.1f}/10")
    print()

    gaps = [(topic, count) for topic, count in gap_counter.items() if count >= args.min_count]
    gaps.sort(key=lambda x: x[1], reverse=True)

    if gaps:
        print(f"KB Gaps (topics clients asked about, no KB answer):")
        print(f"{'Topic':<50s} Count")
        print("-" * 60)
        for topic, count in gaps:
            print(f"  {topic:<48s} {count}")
    else:
        print("No KB gaps detected (all questions answered from KB)")

    print()

    issues = [(desc, count) for desc, count in issue_counter.items() if count >= args.min_count]
    issues.sort(key=lambda x: x[1], reverse=True)

    if issues:
        print(f"Recurring issues:")
        print(f"{'Issue':<50s} Count")
        print("-" * 60)
        for desc, count in issues:
            print(f"  {desc:<48s} {count}")
    else:
        print("No recurring issues")


if __name__ == "__main__":
    main()
