#!/usr/bin/env python3
"""Cluster KB entries by embedding similarity to surface duplicates.

Section 7 Phase A.1 of the master plan
(`.planning/master_plan_2026_04_18/07_kb_refinement.md`). Read-only.

Loads `knowledge_base/kb_faq_ru.yaml`, embeds each entry's
`canonical_question + best_answer` with the production embedding model
(`intfloat/multilingual-e5-large`, normalized), runs single-linkage
union-find on the cosine-similarity graph at a tunable threshold, and
emits a markdown report with multi-member clusters highlighted for the
Phase B surgical pass.

The embedding fidelity matches `rag_demo_system/backend/engine.py`:
- same model name (`intfloat/multilingual-e5-large`)
- `normalize_embeddings=True`
- NO `passage:` prefix (engine.py line 130 does not prefix passages
  either, even though e5 docs recommend it; we mirror prod, not docs).

Usage:
    python rag_demo_system/scripts/kb_cluster.py
    python rag_demo_system/scripts/kb_cluster.py --threshold 0.82
    python rag_demo_system/scripts/kb_cluster.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_KB_PATH = REPO_ROOT / "knowledge_base" / "kb_faq_ru.yaml"
DEFAULT_OUT_PATH = REPO_ROOT / "docs" / "superpowers" / f"kb-clusters-{date.today().isoformat()}.md"
DEFAULT_MODEL = "intfloat/multilingual-e5-large"  # mirrors rag_demo_system/backend/engine.py


def load_entries(yaml_path: Path) -> list[dict]:
    """Load the flat list of entry dicts from kb_faq_ru.yaml."""
    import yaml  # local import keeps the module importable without pyyaml

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        # Some YAML variants wrap the list in `entries:` or similar
        for key in ("entries", "items", "faq", "data"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        raise SystemExit(
            f"Unexpected YAML root: dict with keys {list(raw.keys())[:5]}; expected a list",
        )
    if not isinstance(raw, list):
        raise SystemExit(f"Unexpected YAML root type: {type(raw).__name__}")
    return raw


def entry_text(entry: dict) -> str:
    """Production-faithful text used for clustering.

    `canonical_question + best_answer` — the unit that distinguishes one
    intent from another. No prefix, matching engine.py passage encoding.
    """
    parts = [entry.get("canonical_question", ""), entry.get("best_answer", "")]
    return "\n".join(p.strip() for p in parts if p)


def cluster_by_threshold(sims, threshold: float) -> list[list[int]]:
    """Single-linkage union-find on the threshold graph.

    Two entries share a cluster iff there exists a path of edges with
    cosine_sim >= threshold connecting them. Diagonal is ignored.

    Returns clusters sorted by descending size (ties broken by smallest
    member index, for stable output).
    """
    n = len(sims)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i][j] >= threshold:
                union(i, j)

    by_root: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        by_root[find(i)].append(i)
    return sorted(by_root.values(), key=lambda c: (-len(c), c[0]))


def _short(text: str, limit: int = 140) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def render_report(
    clusters: list[list[int]],
    sims,
    entries: list[dict],
    threshold: float,
    model_name: str,
    yaml_path: Path,
) -> str:
    lines: list[str] = []
    lines.append(f"# KB Cluster Report — {date.today().isoformat()}")
    lines.append("")
    lines.append(f"Source: `{yaml_path.relative_to(REPO_ROOT)}` ({len(entries)} entries)")
    lines.append(f"Embedding model: `{model_name}` (normalized)")
    lines.append(
        f"Cluster threshold: cosine ≥ `{threshold}` (single-linkage union-find on entry pairs)",
    )
    lines.append("")

    multi = [c for c in clusters if len(c) >= 2]
    triple_plus = [c for c in clusters if len(c) >= 3]
    singletons = sum(1 for c in clusters if len(c) == 1)
    largest = max((len(c) for c in clusters), default=0)

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Multi-member clusters (≥2): **{len(multi)}**")
    lines.append(f"- Surgical-pass priority clusters (≥3 members): **{len(triple_plus)}**")
    lines.append(f"- Singleton entries: {singletons}")
    lines.append(f"- Largest cluster size: {largest}")
    lines.append("")

    # similarity histogram (off-diagonal upper triangle)
    n = len(entries)
    bins = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    counts = [0] * (len(bins) - 1)
    for i in range(n):
        for j in range(i + 1, n):
            s = sims[i][j]
            for k in range(len(bins) - 1):
                if bins[k] <= s < bins[k + 1]:
                    counts[k] += 1
                    break
    lines.append("## Pairwise cosine similarity histogram (upper triangle, off-diagonal)")
    lines.append("")
    lines.append("| range | count |")
    lines.append("|---|---:|")
    for k, c in enumerate(counts):
        lines.append(f"| `[{bins[k]:.2f}, {bins[k+1]:.2f})` | {c} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Multi-member clusters")
    lines.append("")
    if not multi:
        lines.append("_(No clusters found at this threshold. Try a lower --threshold.)_")
        lines.append("")

    for ci, members in enumerate(multi, start=1):
        priority = " — SURGICAL-PASS PRIORITY" if len(members) >= 3 else ""
        lines.append(f"### Cluster {ci} — {len(members)} entries{priority}")
        lines.append("")

        # within-cluster pairwise stats (skip diagonal)
        pair_sims = [sims[a][b] for idx_a, a in enumerate(members) for b in members[idx_a + 1 :]]
        if pair_sims:
            mean_sim = sum(pair_sims) / len(pair_sims)
            min_sim = min(pair_sims)
            max_sim = max(pair_sims)
            lines.append(
                f"Within-cluster cosine: mean **{mean_sim:.3f}**, "
                f"min {min_sim:.3f}, max {max_sim:.3f}.",
            )
            lines.append("")

        for idx in members:
            entry = entries[idx]
            intent = entry.get("intent") or "(no intent)"
            cq = _short(entry.get("canonical_question", ""))
            ba_first = (entry.get("best_answer") or "").strip().splitlines()[:1]
            ba_short = _short(ba_first[0] if ba_first else "")
            lines.append(f"- **`{intent}`** — *{cq}*")
            lines.append(f"  - delta: {ba_short}")
        lines.append("")

    return "\n".join(lines) + "\n"


def encode_entries(texts: list[str], model_name: str, device: str):
    """Load the embedder lazily and encode. Returns a 2-D numpy array."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        msg = (
            f"Missing dependency: {exc.name}.\n"
            "Install the production stack into the active env:\n"
            "    pip install sentence-transformers==3.4.1 pyyaml==6.0.2 numpy\n"
            "(first encode also downloads ~2.2GB for intfloat/multilingual-e5-large.)\n"
            "Or run this script on the prod server where the model is already cached.",
        )
        raise SystemExit(msg) from exc

    print(f"Loading embedding model {model_name} on {device}…", file=sys.stderr)
    model = SentenceTransformer(model_name, device=device)
    print(f"Encoding {len(texts)} entries (batch=32)…", file=sys.stderr)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")


def cosine_sim_matrix(vectors):
    """Pairwise cosine similarity. Vectors must be L2-normalized."""
    import numpy as np

    sims = vectors @ vectors.T
    np.fill_diagonal(sims, 0.0)
    # Return as plain Python lists so downstream code works without numpy
    return sims.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster kb_faq_ru.yaml entries by embedding similarity (Section 7 Phase A.1).",
    )
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB_PATH, help="Path to KB YAML")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Output markdown path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine cluster threshold")
    parser.add_argument("--device", default="cpu", help="Encode device: cpu | cuda | mps")
    args = parser.parse_args()

    if not args.kb.exists():
        print(f"KB file not found: {args.kb}", file=sys.stderr)
        sys.exit(1)

    entries = load_entries(args.kb)
    texts = [entry_text(e) for e in entries]
    blanks = sum(1 for t in texts if not t)
    if blanks:
        print(
            f"WARN: {blanks} entries have empty canonical_question + best_answer (still embedded as empty)",
            file=sys.stderr,
        )

    vectors = encode_entries(texts, args.model, args.device)
    sims = cosine_sim_matrix(vectors)
    clusters = cluster_by_threshold(sims, args.threshold)

    report = render_report(clusters, sims, entries, args.threshold, args.model, args.kb)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    multi = sum(1 for c in clusters if len(c) >= 2)
    triple = sum(1 for c in clusters if len(c) >= 3)
    print(f"Wrote {args.out.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(
        f"Clusters: {multi} multi-member, {triple} surgical-priority (≥3), "
        f"{len(entries) - sum(len(c) for c in clusters if len(c) >= 2)} singletons.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
