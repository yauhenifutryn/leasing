#!/usr/bin/env python3
"""Quality trend report from session analysis data.

Shows per-session scores, flags low-quality sessions, and computes trends.

Usage:
    python scripts/quality_report.py
    python scripts/quality_report.py --threshold 5 --reports .state/analysis/session_reports.jsonl
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default=".state/analysis/session_reports.jsonl")
    parser.add_argument("--threshold", type=float, default=5.0, help="Flag sessions below this score")
    args = parser.parse_args()

    reports_path = Path(args.reports)
    if not reports_path.exists():
        print(f"No reports file: {reports_path}")
        return

    sessions: list[dict] = []
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
        sessions.append(report)

    if not sessions:
        print("No valid session reports found")
        return

    scores = [s.get("overall_score", 0) for s in sessions]
    avg = sum(scores) / len(scores)

    print(f"Total sessions: {len(sessions)}")
    print(f"Average score:  {avg:.1f}/10")
    print(f"Best:           {max(scores):.0f}/10")
    print(f"Worst:          {min(scores):.0f}/10")
    print()

    # Trend: compare first half vs second half
    if len(scores) >= 4:
        mid = len(scores) // 2
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (len(scores) - mid)
        delta = second_half - first_half
        trend = "improving" if delta > 0.5 else "degrading" if delta < -0.5 else "stable"
        print(f"Trend: {trend} (first half: {first_half:.1f}, second half: {second_half:.1f}, delta: {delta:+.1f})")
        print()

    # Per-session detail
    flagged = [s for s in sessions if s.get("overall_score", 10) < args.threshold]
    if flagged:
        print(f"Flagged sessions (score < {args.threshold}):")
        print(f"{'Session ID':<40s} {'Score':>5s} {'Turns':>5s} {'Transport':<10s} Summary")
        print("-" * 100)
        for s in flagged:
            sid = s.get("session_id", "?")[:36]
            score = s.get("overall_score", 0)
            turns = s.get("turn_count", 0)
            transport = s.get("transport", "browser")
            summary = s.get("summary", "")[:50]
            print(f"  {sid:<38s} {score:>5.0f} {turns:>5d} {transport:<10s} {summary}")
    else:
        print(f"No sessions below threshold ({args.threshold})")

    print()

    # Score breakdown averages
    score_keys = ["banned_phrases", "specialist_overuse", "humor_and_tone",
                  "answer_completeness", "response_variety", "tool_use_quality"]
    print("Average score breakdown:")
    for key in score_keys:
        vals = [s.get("scores", {}).get(key, 0) for s in sessions if key in s.get("scores", {})]
        if vals:
            print(f"  {key:<25s} {sum(vals)/len(vals):.1f}/10")


if __name__ == "__main__":
    main()
