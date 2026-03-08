from __future__ import annotations

from typing import Any


def filter_vector_hits(vector_hits: list[dict[str, Any]], score_threshold: float) -> list[dict[str, Any]]:
    return [hit for hit in vector_hits if float(hit.get("score", 0.0)) >= score_threshold]
