"""Freeze the current KB-viz overlay state into ONE self-contained demo HTML.

Run this BEFORE killing the overlay server. It hits the live endpoints
once per requested query, plus /coverage once, and bakes everything into
a single ``kb_viz_3d_demo.html`` (plus a 2D sibling). The resulting HTML
carries a pill picker — the client clicks any example question to see
the corresponding tether lines, numbered top-K, and match cards, all
without a server.

Single-query usage:
    python scripts/make_demo_snapshot.py \\
        --url http://38.80.122.98:8500 \\
        --token "$KB_VIZ_OVERLAY_TOKEN" \\
        --query "как оплатить через ерип" \\
        --out-dir results

Batch (repeat --query, or feed a file with one per line):
    python scripts/make_demo_snapshot.py --url ... \\
        --query "как оплатить через ерип" \\
        --query "что такое лизинг без прав" \\
        --query "график работы офисов"

    # OR
    python scripts/make_demo_snapshot.py --url ... \\
        --queries-file scripts/demo_queries.example.txt

Outputs:
    kb_viz_demo_snapshot.json   raw data (coverage + all queries)
    kb_viz_2d_demo.html         self-contained 2D preview (single file)
    kb_viz_3d_demo.html         self-contained 3D preview (single file)
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

    per_query: list[dict[str, Any]] = []
    for n, query in enumerate(queries, start=1):
        print(f"\n[demo {n}/{len(queries)}] query={query!r}", flush=True)
        print(f"[demo {n}]   POST {base}/overlay_query", flush=True)
        q_resp = _http_json(
            f"{base}/overlay_query",
            args.token,
            {"text": query, "kind": "3d", "top_k": args.top_k, "client_id": args.user},
        )
        per_query.append({
            "query_id": q_resp.get("query_id"),
            "query_text": query,
            "top_k": q_resp.get("top_k", []),
        })

    # Coverage is session-global; fetch it once at the end so it reflects
    # the cumulative state including any feedback the baker submitted
    # while poking around before snapshotting.
    print(f"\n[demo] GET {base}/coverage", flush=True)
    cov = _http_json(f"{base}/coverage", args.token)

    snapshot = {
        "user": args.user,
        "coverage": cov,
        "queries": per_query,
    }

    snapshot_path = out_dir / "kb_viz_demo_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[demo] snapshot -> {snapshot_path}", flush=True)

    print("[demo] rendering demo HTMLs...", flush=True)
    rendered = render_viz.render(
        embeddings_path=Path(args.embeddings),
        out_dir=out_dir,
        overlay_url="https://demo.invalid/overlay_query",
        overlay_token=None,
        demo_snapshot=snapshot,
        html_suffix="_demo",
    )
    for k, p in rendered.items():
        print(f"  {k}: {p}")
    print(
        f"\n[demo] open {out_dir / 'kb_viz_3d_demo.html'} in any browser — "
        f"{len(queries)} example {'query' if len(queries) == 1 else 'queries'} baked in, "
        "no server needed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
