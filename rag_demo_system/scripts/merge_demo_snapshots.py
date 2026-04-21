"""Merge N per-query demo snapshots into ONE unified demo HTML, locally.

Use this when you already have:
  - N snapshot JSONs (e.g. kb_viz_demo_snapshot_1.json .. _6.json)
  - At least one rendered demo HTML (e.g. kb_viz_3d_demo_1.html) to use
    as a template
...and the server is no longer available. No network calls, no embeddings,
no UMAP recomputation — the template HTML already carries the base plot
and all the overlay JS, we just swap the baked-in ``__KB_VIZ_DEMO__``
variable for a unified {user, coverage, queries: [...]} bundle so the
output HTML shows a pill picker instead of a single frozen query.

Usage:
    python scripts/merge_demo_snapshots.py \\
        --snapshots "results/kb_viz_demo_snapshot_*.json" \\
        --template-3d results/kb_viz_3d_demo_1.html \\
        --template-2d results/kb_viz_2d_demo_1.html \\
        --out-dir results

Defaults auto-detect both patterns in --out-dir (the common case — you
downloaded everything into one folder). Outputs overwrite / create
kb_viz_3d_demo.html and kb_viz_2d_demo.html in --out-dir.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any


def _json_for_script(value: Any) -> str:
    """Same `</` escape render_viz uses so the JSON can sit inside <script>."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _load_snapshots(paths: list[str]) -> list[dict[str, Any]]:
    expanded: list[Path] = []
    for pat in paths:
        matches = sorted(glob.glob(pat))
        if not matches:
            raise SystemExit(f"No snapshots matched pattern: {pat}")
        for m in matches:
            expanded.append(Path(m))
    snaps: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for p in expanded:
        data = json.loads(p.read_text(encoding="utf-8"))
        # Support both per-query (legacy) and already-bundled snapshots.
        if "queries" in data and isinstance(data["queries"], list):
            for q in data["queries"]:
                qid = q.get("query_id") or q.get("query_text")
                if qid in seen_ids:
                    continue
                seen_ids.add(qid)
                snaps.append({"_file": str(p), "_user": data.get("user"),
                              "_coverage": data.get("coverage"), **q})
        elif "top_k" in data:
            qid = data.get("query_id") or data.get("query_text")
            if qid in seen_ids:
                continue
            seen_ids.add(qid)
            snaps.append({
                "_file": str(p),
                "_user": data.get("user"),
                "_coverage": data.get("coverage"),
                "query_id": data.get("query_id"),
                "query_text": data.get("query_text"),
                "top_k": data.get("top_k", []),
            })
        else:
            print(f"[merge] skipping {p}: not a recognised snapshot shape", file=sys.stderr)
    if not snaps:
        raise SystemExit("No usable snapshots found.")
    return snaps


def _rewrite_demo_var(html: str, new_bundle: dict[str, Any]) -> str:
    """Replace the single-line ``var __KB_VIZ_DEMO__ = {...};`` in an
    already-rendered HTML with a new bundle. The replacement matches from
    that specific var declaration up to (but not including) the next
    ``var feedbackUrl`` line, which both the original rendering and the
    multi-query rendering emit right after it.
    """
    new_js = "var __KB_VIZ_DEMO__ = " + _json_for_script(new_bundle) + ";"
    pattern = re.compile(
        r"var\s+__KB_VIZ_DEMO__\s*=\s*.*?;(?=\s*var\s+feedbackUrl)",
        re.DOTALL,
    )
    replaced, n = pattern.subn(lambda _m: new_js, html, count=1)
    if n == 0:
        raise SystemExit(
            "Could not find `var __KB_VIZ_DEMO__ = ...;` in template. "
            "Is this HTML from an older render? Re-render with current code."
        )
    return replaced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--snapshots",
        nargs="+",
        default=None,
        help="Glob pattern(s) for snapshot JSONs. Defaults to "
             "<out-dir>/kb_viz_demo_snapshot_*.json.",
    )
    parser.add_argument("--template-3d", default=None, help="Template 3D HTML (default: <out-dir>/kb_viz_3d_demo_1.html).")
    parser.add_argument("--template-2d", default=None, help="Template 2D HTML (default: <out-dir>/kb_viz_2d_demo_1.html). Optional.")
    parser.add_argument("--out-dir", default="rag_demo_system/results")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_patterns = args.snapshots or [str(out_dir / "kb_viz_demo_snapshot_*.json")]
    snaps = _load_snapshots(snapshot_patterns)

    # Use the coverage + user from the first snapshot (they all share the
    # same session state if they were captured in one batch run).
    user = next((s.get("_user") for s in snaps if s.get("_user")), "Demo")
    coverage = next((s.get("_coverage") for s in snaps if s.get("_coverage")), {
        "per_chunk": {}, "per_section": {}, "per_user": {},
        "total_feedback": 0, "unique_chunks_validated": 0,
    })
    queries = [
        {"query_id": s.get("query_id"), "query_text": s.get("query_text"), "top_k": s.get("top_k", [])}
        for s in snaps
    ]
    bundle = {"user": user, "coverage": coverage, "queries": queries}

    merged_path = out_dir / "kb_viz_demo_snapshot.json"
    merged_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] wrote unified snapshot -> {merged_path}  ({len(queries)} queries)")

    t3d_default = out_dir / "kb_viz_3d_demo_1.html"
    t2d_default = out_dir / "kb_viz_2d_demo_1.html"
    t3d = Path(args.template_3d) if args.template_3d else t3d_default
    t2d = Path(args.template_2d) if args.template_2d else (t2d_default if t2d_default.exists() else None)

    if not t3d.exists():
        raise SystemExit(f"Template 3D HTML not found: {t3d}")

    for template, out_name in [(t3d, "kb_viz_3d_demo.html")] + ([(t2d, "kb_viz_2d_demo.html")] if t2d else []):
        html = template.read_text(encoding="utf-8")
        merged = _rewrite_demo_var(html, bundle)
        target = out_dir / out_name
        target.write_text(merged, encoding="utf-8")
        print(f"[merge] {target}  ({target.stat().st_size // 1024} KB)")

    print("\n[merge] Done. Open kb_viz_3d_demo.html and use the pill picker at the top of the panel to switch queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
