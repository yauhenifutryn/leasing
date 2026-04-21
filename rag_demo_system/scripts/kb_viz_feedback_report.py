"""Aggregate kb_viz_feedback.jsonl into a human-readable report.

Matches the shape of kb_gap_report.py from the self-improvement pipeline:
prints counts, top-reported chunks, free-text comments, and section-level
coverage. Intended to be run periodically by the dev, not the client.

Usage:
    python kb_viz_feedback_report.py
    python kb_viz_feedback_report.py --log custom/path/kb_viz_feedback.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path("rag_demo_system/.state/kb_viz_feedback.jsonl")


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def report(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No feedback recorded yet.\n"

    content_recs = [r for r in records if r.get("signal_type", "content") == "content"]
    relevance_recs = [r for r in records if r.get("signal_type") == "relevance"]

    lines: list[str] = []
    n = len(records)
    correct = sum(1 for r in content_recs if r.get("verdict") == "correct")
    wrong = sum(1 for r in content_recs if r.get("verdict") == "wrong")
    relevant = sum(1 for r in relevance_recs if r.get("verdict") == "relevant")
    not_relevant = sum(1 for r in relevance_recs if r.get("verdict") == "not_relevant")
    lines.append(f"Total feedback events: {n}")
    lines.append(f"  Content signals:   {len(content_recs)} ({correct} ✓ / {wrong} ✗)")
    lines.append(f"  Relevance signals: {len(relevance_recs)} ({relevant} ◯ / {not_relevant} ⊘)")

    # Per-section rollup — content only; relevance is analyzed separately.
    per_section: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0})
    chunk_wrong: dict[str, list[str]] = defaultdict(list)
    chunk_correct: dict[str, int] = defaultdict(int)
    chunk_not_relevant: dict[str, list[str]] = defaultdict(list)
    comments: list[tuple[str, str, str]] = []  # (query_text, comment, sections)
    for r in content_recs:
        verdict = r.get("verdict", "")
        sections_seen: set[str] = set()
        for m in r.get("top_k") or []:
            section = str(m.get("section", "")) or "Без раздела"
            sections_seen.add(section)
            cid = str(m.get("chunk_id", ""))
            if verdict == "wrong":
                chunk_wrong[cid].append(r.get("query_text", ""))
            elif verdict == "correct":
                chunk_correct[cid] += 1
        for s in sections_seen:
            if verdict in ("correct", "wrong"):
                per_section[s][verdict] += 1
        if verdict == "wrong" and r.get("comment"):
            comments.append((
                str(r.get("query_text", "")),
                str(r.get("comment", "")),
                ", ".join(sorted(sections_seen)),
            ))

    for r in relevance_recs:
        if r.get("verdict") != "not_relevant":
            continue
        for m in r.get("top_k") or []:
            cid = str(m.get("chunk_id", ""))
            chunk_not_relevant[cid].append(r.get("query_text", ""))

    lines.append("")
    lines.append("Per-section tally (correct / wrong events):")
    for section in sorted(per_section.keys()):
        c = per_section[section]["correct"]
        w = per_section[section]["wrong"]
        lines.append(f"  {section}: {c} ✓ / {w} ✗")

    lines.append("")
    lines.append("Most-reported wrong chunks (top 10):")
    wrong_ranked = sorted(chunk_wrong.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
    if not wrong_ranked:
        lines.append("  (none)")
    else:
        for cid, qs in wrong_ranked:
            lines.append(f"  {cid}: {len(qs)} wrong feedback(s)")
            for q in qs[:3]:
                lines.append(f"    - query: {q}")
            if len(qs) > 3:
                lines.append(f"    ... +{len(qs) - 3} more")

    lines.append("")
    lines.append("Client comments (content=wrong only):")
    if not comments:
        lines.append("  (none)")
    else:
        for q, c, sections in comments:
            lines.append(f"  [{sections}] query: {q}")
            lines.append(f"    comment: {c}")

    lines.append("")
    lines.append("Most-often-flagged irrelevant chunks (relevance signal, top 10):")
    nr_ranked = sorted(chunk_not_relevant.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
    if not nr_ranked:
        lines.append("  (none)")
    else:
        for cid, qs in nr_ranked:
            lines.append(f"  {cid}: {len(qs)} not-relevant flag(s)")
            for q in qs[:3]:
                lines.append(f"    - query: {q}")
            if len(qs) > 3:
                lines.append(f"    ... +{len(qs) - 3} more")

    return "\n".join(lines) + "\n"


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate kb_viz_feedback.jsonl.")
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    args = parser.parse_args(argv)

    records = _load(Path(args.log))
    sys.stdout.write(report(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
