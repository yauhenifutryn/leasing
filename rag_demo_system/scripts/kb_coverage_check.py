#!/usr/bin/env python3
"""Check KB coverage of real-user queries via the prod /api/retrieve endpoint.

Section 7 Phase A.3 of the master plan
(`.planning/master_plan_2026_04_18/07_kb_refinement.md`). Read-only.

Loads a query corpus (one query per line, or JSONL with `{"query": "..."}` shape),
calls a configurable retriever (default: HTTP POST to `/api/retrieve`), and
classifies each query into:
- **MISS**     top score < `--miss-threshold` (default 0.65) — KB has no good match
- **PARTIAL**  `--miss-threshold` ≤ top score < `--cover-threshold` (default 0.80)
- **COVERED**  top score ≥ `--cover-threshold`

Emits `docs/superpowers/kb-coverage-<date>.md` ranked MISS-first since those are
the actionable queries — the topical sections in Phase C must explicitly cover
them via examples + synonym entries.

Threshold defaults match the spec but should be tuned based on the observed
score distribution (which depends on whether the retriever returns cosine
similarity, cross-encoder logits, or rerank-normalized scores).

Usage examples:
    # Live run against the dev server
    python rag_demo_system/scripts/kb_coverage_check.py \\
        --queries .state/kb_viz_queries.jsonl \\
        --url http://localhost:8000/api/retrieve

    # Live run against a remote prod (via SSH tunnel or direct):
    python rag_demo_system/scripts/kb_coverage_check.py \\
        --queries /tmp/queries.txt \\
        --url http://38.128.232.83:8000/api/retrieve

    # Mock mode (smoke test, no network):
    python rag_demo_system/scripts/kb_coverage_check.py \\
        --queries /tmp/queries.txt --mode mock
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_PATH = REPO_ROOT / "docs" / "superpowers" / f"kb-coverage-{date.today().isoformat()}.md"
DEFAULT_URL = "http://localhost:8000/api/retrieve"


@dataclass
class RetrievalResult:
    query: str
    top_score: float
    top_chunk_excerpt: str  # first ~140 chars of the top chunk text
    top_heading_path: str
    n_candidates: int
    raw_response_keys: list[str] = field(default_factory=list)
    error: str | None = None


def load_queries(path: Path) -> list[str]:
    """Load queries from a JSONL or plain-text file. Skips blanks and comments."""
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # JSONL form: {"query": "..."} or {"q": "..."} or {"text": "..."}
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # not valid JSON, treat as plain text
                out.append(line)
                continue
            for key in ("query", "q", "text", "user_text", "utterance"):
                if isinstance(obj.get(key), str) and obj[key].strip():
                    out.append(obj[key].strip())
                    break
            continue
        out.append(line)
    return out


def http_retriever(url: str, timeout_s: float = 15.0) -> Callable[[str], RetrievalResult]:
    """Build a retriever that POSTs to /api/retrieve."""

    def call(query: str) -> RetrievalResult:
        body = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            return RetrievalResult(
                query=query,
                top_score=-1.0,
                top_chunk_excerpt="",
                top_heading_path="",
                n_candidates=0,
                error=str(exc),
            )
        except json.JSONDecodeError as exc:
            return RetrievalResult(
                query=query,
                top_score=-1.0,
                top_chunk_excerpt="",
                top_heading_path="",
                n_candidates=0,
                error=f"non-JSON response: {exc}",
            )
        return parse_retrieve_response(query, payload)

    return call


def parse_retrieve_response(query: str, payload: dict) -> RetrievalResult:
    """Extract top score + excerpt from a /api/retrieve response.

    Prefers `top_rerank_score` (post-reranker, production-faithful). Falls
    back to the first candidate's score field. Tolerates several possible
    response shapes since the endpoint has evolved over time.
    """
    top_score = payload.get("top_rerank_score")
    candidates = payload.get("final") or payload.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []

    if top_score is None and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            top_score = first.get("rerank_score") or first.get("score") or first.get("similarity")

    excerpt = ""
    heading = ""
    if candidates:
        first = candidates[0]
        if isinstance(first, dict):
            text = first.get("text") or first.get("chunk") or ""
            excerpt = text[:140].replace("\n", " ").strip()
            hp = first.get("heading_path") or first.get("headings") or []
            if isinstance(hp, list):
                heading = " / ".join(str(x) for x in hp)
            else:
                heading = str(hp)

    return RetrievalResult(
        query=query,
        top_score=float(top_score) if top_score is not None else 0.0,
        top_chunk_excerpt=excerpt,
        top_heading_path=heading,
        n_candidates=len(candidates),
        raw_response_keys=sorted(payload.keys()) if isinstance(payload, dict) else [],
    )


def classify(score: float, miss: float, cover: float) -> str:
    if score < miss:
        return "MISS"
    if score < cover:
        return "PARTIAL"
    return "COVERED"


def render_report(
    results: list[RetrievalResult],
    miss: float,
    cover: float,
    queries_path: Path,
    retriever_label: str,
) -> str:
    by_class: dict[str, list[RetrievalResult]] = {"MISS": [], "PARTIAL": [], "COVERED": [], "ERROR": []}
    for r in results:
        if r.error:
            by_class["ERROR"].append(r)
            continue
        by_class[classify(r.top_score, miss, cover)].append(r)

    lines: list[str] = []
    lines.append(f"# KB Coverage Check — {date.today().isoformat()}")
    lines.append("")
    lines.append(f"Queries: `{queries_path}` ({len(results)} queries)")
    lines.append(f"Retriever: {retriever_label}")
    lines.append(f"Thresholds: MISS < `{miss}`, PARTIAL `[{miss}, {cover})`, COVERED ≥ `{cover}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **MISS:** {len(by_class['MISS'])} ({_pct(by_class['MISS'], results)})")
    lines.append(f"- **PARTIAL:** {len(by_class['PARTIAL'])} ({_pct(by_class['PARTIAL'], results)})")
    lines.append(f"- **COVERED:** {len(by_class['COVERED'])} ({_pct(by_class['COVERED'], results)})")
    if by_class["ERROR"]:
        lines.append(f"- **ERROR:** {len(by_class['ERROR'])} (retriever failed)")
    lines.append("")

    # Top-score histogram (excluding errors)
    lines.append("## Top-score distribution (excludes errors)")
    lines.append("")
    bins = [-0.01, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    counts = [0] * (len(bins) - 1)
    for r in results:
        if r.error:
            continue
        for k in range(len(bins) - 1):
            if bins[k] <= r.top_score < bins[k + 1]:
                counts[k] += 1
                break
    lines.append("| range | count |")
    lines.append("|---|---:|")
    for k, c in enumerate(counts):
        lines.append(f"| `[{bins[k]:.2f}, {bins[k+1]:.2f})` | {c} |")
    lines.append("")

    for klass in ("ERROR", "MISS", "PARTIAL", "COVERED"):
        items = by_class[klass]
        if not items:
            continue
        lines.append(f"## {klass} — {len(items)} queries")
        lines.append("")
        lines.append("| score | query | top heading | top chunk excerpt |")
        lines.append("|---:|---|---|---|")
        # Sort MISS/PARTIAL by lowest score (most actionable); COVERED by highest
        reverse = klass == "COVERED"
        for r in sorted(items, key=lambda x: x.top_score, reverse=reverse):
            score_cell = f"{r.top_score:.3f}" if not r.error else "—"
            query_cell = r.query[:80].replace("|", "\\|")
            heading_cell = r.top_heading_path[:60].replace("|", "\\|") if r.top_heading_path else ""
            if r.error:
                excerpt_cell = f"ERROR: {r.error[:120]}"
            else:
                excerpt_cell = r.top_chunk_excerpt.replace("|", "\\|") if r.top_chunk_excerpt else ""
            lines.append(f"| {score_cell} | {query_cell} | {heading_cell} | {excerpt_cell} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def _pct(group: list, all_items: list) -> str:
    if not all_items:
        return "0%"
    return f"{100.0 * len(group) / len(all_items):.0f}%"


def run_coverage(
    queries: list[str],
    retriever: Callable[[str], RetrievalResult],
    *,
    miss: float,
    cover: float,
) -> list[RetrievalResult]:
    """Run the retriever over all queries and return a list of RetrievalResult.

    Threshold args here are unused inside this function (classification is in
    the report renderer) but kept in the signature so callers can document the
    thresholds at the same call site as the run.
    """
    del miss, cover  # documented-only
    return [retriever(q) for q in queries]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check KB coverage of real-user queries (Section 7 Phase A.3).",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Path to query corpus (JSONL or plain text, one query per line)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--url", default=DEFAULT_URL, help="Retrieve endpoint URL")
    parser.add_argument(
        "--mode",
        choices=("http", "mock"),
        default="http",
        help="http = real call to --url; mock = stub for smoke testing",
    )
    parser.add_argument("--miss-threshold", type=float, default=0.65)
    parser.add_argument("--cover-threshold", type=float, default=0.80)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    if not args.queries.exists():
        print(f"Queries file not found: {args.queries}", file=sys.stderr)
        sys.exit(1)

    queries = load_queries(args.queries)
    if not queries:
        print(f"No queries loaded from {args.queries}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "mock":
        retriever_label = "mock (returns score=0.5 for every query)"

        def retriever(q: str) -> RetrievalResult:
            return RetrievalResult(
                query=q,
                top_score=0.5,
                top_chunk_excerpt="(mock excerpt)",
                top_heading_path="(mock heading)",
                n_candidates=1,
            )
    else:
        retriever_label = f"HTTP POST {args.url}"
        retriever = http_retriever(args.url, args.timeout)

    print(f"Running {len(queries)} queries against {retriever_label}…", file=sys.stderr)
    results = run_coverage(queries, retriever, miss=args.miss_threshold, cover=args.cover_threshold)

    counts = Counter(
        (r.error and "ERROR")
        or classify(r.top_score, args.miss_threshold, args.cover_threshold)
        for r in results
    )
    print(
        f"MISS={counts.get('MISS', 0)}  PARTIAL={counts.get('PARTIAL', 0)}  "
        f"COVERED={counts.get('COVERED', 0)}  ERROR={counts.get('ERROR', 0)}",
        file=sys.stderr,
    )

    report = render_report(
        results,
        miss=args.miss_threshold,
        cover=args.cover_threshold,
        queries_path=args.queries,
        retriever_label=retriever_label,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
