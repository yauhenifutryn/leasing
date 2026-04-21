"""Freeze the current KB-viz overlay state into a self-contained demo HTML.

Run this BEFORE killing the overlay server. It hits the live endpoints
once, bakes the response into a new ``kb_viz_{2d,3d}_demo.html`` pair,
and emits a single zip-friendly folder you can hand to a client who
doesn't have network access to the server. The resulting HTMLs have the
entire visualisation pre-populated (tether lines, numbered top-K, verdict
halos, match cards) and all interactive actions neutralised.

Usage:
    python -m rag_demo_system.scripts.make_demo_snapshot \\
        --url http://38.80.122.98:8500 \\
        --token "$KB_VIZ_OVERLAY_TOKEN" \\
        --query "как оплатить через ерип" \\
        --out-dir rag_demo_system/results

Outputs (in --out-dir):
    kb_viz_demo_snapshot.json   raw data used for the bake
    kb_viz_2d_demo.html         self-contained 2D preview
    kb_viz_3d_demo.html         self-contained 3D preview
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Live overlay base URL, e.g. http://38.80.122.98:8500")
    parser.add_argument("--token", default=None, help="Bearer token if KB_VIZ_OVERLAY_TOKEN is set on the server")
    parser.add_argument("--query", required=True, help="Example query to pre-bake into the demo")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--user", default="Demo", help="User label baked into the demo (cosmetic only)")
    parser.add_argument("--embeddings", default="rag_demo_system/results/embeddings.json")
    parser.add_argument("--out-dir", default="rag_demo_system/results")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    print(f"[demo] POST {base}/overlay_query with query={args.query!r}", flush=True)
    q = _http_json(
        f"{base}/overlay_query",
        args.token,
        {"text": args.query, "kind": "3d", "top_k": args.top_k, "client_id": args.user},
    )
    print(f"[demo] GET  {base}/coverage", flush=True)
    cov = _http_json(f"{base}/coverage", args.token)

    snapshot = {
        "query_id": q.get("query_id"),
        "query_text": args.query,
        "top_k": q.get("top_k", []),
        "coverage": cov,
        "user": args.user,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / "kb_viz_demo_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[demo] wrote snapshot -> {snapshot_path}", flush=True)

    print("[demo] rendering demo HTMLs...", flush=True)
    # A placeholder URL is needed so the overlay <script> block emits; the
    # HTML will never actually call it in demo mode.
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
        f"\nOpen {out_dir / 'kb_viz_3d_demo.html'} in any browser — no server needed.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
