"""Dump all points from the live Qdrant KB collection to a JSON file.

Run on the GPU server (or any host with network access to Qdrant). The output
JSON is consumed by render_viz.py and kb_viz_service.py.

Defaults align with rag_demo_system/backend/settings.py:
    QDRANT_URL        http://localhost:6333
    QDRANT_COLLECTION micro_leasing_kb

Usage:
    python dump_kb_embeddings.py
    python dump_kb_embeddings.py --qdrant-url http://localhost:6333 \
        --collection micro_leasing_kb --out results/embeddings.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "micro_leasing_kb"
DEFAULT_OUT = "rag_demo_system/results/embeddings.json"
SCROLL_BATCH = 256


def dump(qdrant_url: str, collection: str, out_path: Path, model_name: str | None = None) -> dict[str, Any]:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)

    info = client.get_collection(collection_name=collection)
    vector_dim = _extract_vector_dim(info)

    points: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not batch:
            break
        for rec in batch:
            vector = rec.vector
            if isinstance(vector, dict):
                if not vector:
                    continue
                vector = next(iter(vector.values()))
            if vector is None:
                continue
            points.append(
                {
                    "id": str(rec.id),
                    "vector": list(vector),
                    "payload": dict(rec.payload or {}),
                }
            )
        if offset is None:
            break

    if not points:
        raise RuntimeError(f"Collection '{collection}' at {qdrant_url} is empty. Index the KB first.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": collection,
        "qdrant_url": qdrant_url,
        "model_name": model_name or os.getenv("KB_VIZ_MODEL_NAME") or "intfloat/multilingual-e5-large",
        "vector_dim": vector_dim or (len(points[0]["vector"]) if points else 0),
        "count": len(points),
        "points": points,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"count": len(points), "vector_dim": payload["vector_dim"], "path": str(out_path)}


def _extract_vector_dim(info: Any) -> int | None:
    """Qdrant collection info has different shapes across client versions."""
    try:
        params = info.config.params
        vectors = params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            return int(getattr(first, "size", 0)) or None
    except Exception:
        pass
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump Qdrant KB embeddings to JSON.")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model-name", default=None, help="Embed model label to store alongside points.")
    args = parser.parse_args(argv)

    try:
        result = dump(
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            out_path=Path(args.out),
            model_name=args.model_name,
        )
    except Exception as exc:
        sys.stderr.write(f"Dump failed: {exc}\n")
        return 1
    print(f"Wrote {result['count']} points (dim={result['vector_dim']}) to {result['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
