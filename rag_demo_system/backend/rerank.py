from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    chunk_id: str
    score: float


class Reranker:
    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[RerankResult]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        results: list[RerankResult] = []
        for cand, score in zip(candidates, scores):
            results.append(RerankResult(chunk_id=cand["chunk_id"], score=float(score)))
        return results
