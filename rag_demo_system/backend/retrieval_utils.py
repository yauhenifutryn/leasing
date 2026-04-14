from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def filter_vector_hits(vector_hits: list[dict[str, Any]], score_threshold: float) -> list[dict[str, Any]]:
    return [hit for hit in vector_hits if float(hit.get("score", 0.0)) >= score_threshold]


def dedup_chunks(candidates: list[dict[str, Any]], threshold: float = 0.85) -> list[dict[str, Any]]:
    """Remove near-duplicate chunks, keeping the highest-scored from each group.

    Compares chunk texts pairwise using SequenceMatcher. When two chunks have
    similarity >= threshold, the one with the lower rerank_score is dropped.

    Input must be sorted by rerank_score descending (highest first). Output
    preserves that order.
    """
    if len(candidates) <= 1:
        return list(candidates)

    dropped: set[int] = set()

    for i in range(len(candidates)):
        if i in dropped:
            continue
        text_i = candidates[i].get("text", "")
        for j in range(i + 1, len(candidates)):
            if j in dropped:
                continue
            text_j = candidates[j].get("text", "")
            ratio = SequenceMatcher(None, text_i, text_j).ratio()
            if ratio >= threshold:
                dropped.add(j)

    return [c for idx, c in enumerate(candidates) if idx not in dropped]
