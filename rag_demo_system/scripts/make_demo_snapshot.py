"""Freeze the current KB-viz overlay state into self-contained demo HTMLs.

Run this BEFORE killing the overlay server. For each input query it hits
the live endpoints once, bakes the response into a kb_viz_{2d,3d}_demo_N.html
pair, and emits a zip-friendly folder you can hand to a client who does
not have network access to the server. The resulting HTMLs have the
entire visualisation pre-populated (tether lines, numbered top-K, verdict
halos, match cards) and all interactive actions neutralised.

Single-query usage:
    python scripts/make_demo_snapshot.py \\
        --url http://38.80.122.98:8500 \\
        --token "$KB_VIZ_OVERLAY_TOKEN" \\
        --query "как оплатить через ерип" \\
        --out-dir results

Multi-query batch (repeat --query, or feed a file with one per line):
    python scripts/make_demo_snapshot.py --url ... \\
        --query "как оплатить через ерип" \\
        --query "что такое лизинг без прав" \\
        --query "график работы офисов"

    # OR
    python scripts/make_demo_snapshot.py --url ... \\
        --queries-file scripts/demo_queries.example.txt

Per-query outputs (N = 1, 2, ...):
    kb_viz_demo_snapshot_N.json     raw data used for bake N
    kb_viz_2d_demo_N.html           self-contained 2D preview
    kb_viz_3d_demo_N.html           self-contained 3D preview

Plus a combined index:
    kb_viz_demo_index.json          {queries: [{n, query, files}], ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib import request, error

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import render_viz  # noqa: E402


def _http_json(url: str, token: str | None, body: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = request.Request(url, data=data, headers=headers, method="POST" if body else "GET")
    try:
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}: {detail}") from exc
    except error.URLError as exc:
        raise SystemExit(f"Could not reach {url}: {exc}") from exc


def _load_queries(args: argparse.Namespace) -> list[str]:
    queries: list[str] = list(args.query or [])
    if args.queries_file:
        path = Path(args.queries_file)
        if not path.exists():
            raise SystemExit(f"queries-file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            queries.append(stripped)
    # Dedupe while preserving order so repeating --query doesn't waste calls.
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    if not unique:
        raise SystemExit("no queries provided: pass --query or --queries-file")
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Live overlay base URL, e.g. http://38.80.122.98:8500")
    parser.add_argument("--token", default=None, help="Bearer token if KB_VIZ_OVERLAY_TOKEN is set on the server")
    parser.add_argument("--query", action="append", default=None, help="Example query to bake. Pass multiple times for a batch.")
    parser.add_argument("--queries-file", default=None, help="Path to a text file with one query per line.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--user", default="Demo", help="User label baked into the demo (cosmetic only)")
    parser.add_argument("--embeddings", default="rag_demo_system/results/embeddings.json")
    parser.add_argument("--out-dir", default="rag_demo_system/results")
    args = parser.parse_args(argv)

    queries = _load_queries(args)
    base = args.url.rstrip("/")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict[str, Any]] = []
    for n, query in enumerate(queries, start=1):
        print(f"\n[demo {n}/{len(queries)}] query={query!r}", flush=True)
        print(f"[demo {n}]   POST {base}/overlay_query", flush=True)
        q_resp = _http_json(
            f"{base}/overlay_query",
            args.token,
            {"text": query, "kind": "3d", "top_k": args.top_k, "client_id": args.user},
        )
        print(f"[demo {n}]   GET  {base}/coverage", flush=True)
        cov = _http_json(f"{base}/coverage", args.token)

        snapshot = {
            "query_id": q_resp.get("query_id"),
            "query_text": query,
            "top_k": q_resp.get("top_k", []),
            "coverage": cov,
            "user": args.user,
        }

        suffix = f"_demo_{n}"
        snapshot_path = out_dir / f"kb_viz_demo_snapshot_{n}.json"
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[demo {n}]   snapshot -> {snapshot_path}", flush=True)

        rendered = render_viz.render(
            embeddings_path=Path(args.embeddings),
            out_dir=out_dir,
            overlay_url="https://demo.invalid/overlay_query",
            overlay_token=None,
            demo_snapshot=snapshot,
            html_suffix=suffix,
        )
        files = {k: str(p) for k, p in rendered.items() if k.endswith("_html")}
        for k, p in files.items():
            print(f"[demo {n}]   {k}: {p}", flush=True)
        index.append({"n": n, "query": query, "files": files, "snapshot": str(snapshot_path)})

    # Combined index for easy browsing.
    index_path = out_dir / "kb_viz_demo_index.json"
    index_path.write_text(
        json.dumps({"count": len(index), "queries": index}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[demo] {len(queries)} demo HTML pairs written to {out_dir}/:")
    for row in index:
        three_d = row["files"].get("3d_html", "")
        print(f"  #{row['n']}: {Path(three_d).name}   <- {row['query']}")
    print(f"\n[demo] index: {index_path}")
    print("[demo] open any kb_viz_3d_demo_N.html in a browser — no server needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
