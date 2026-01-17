from backend.engine import effective_retrieval
from backend.settings import RetrievalConfig


def test_effective_retrieval_defaults():
    cfg = RetrievalConfig(
        vector_top_k=8,
        bm25_top_k=8,
        final_top_n=6,
        score_threshold=0.35,
        min_rerank_score=0.1,
        context_max_tokens=1800,
        fast_vector_top_k=4,
        fast_bm25_top_k=4,
        fast_final_top_n=3,
        fast_context_max_tokens=900,
    )

    regular = effective_retrieval(cfg, fast=False)
    assert regular["vector_top_k"] == 8
    assert regular["bm25_top_k"] == 8
    assert regular["final_top_n"] == 6
    assert regular["context_max_tokens"] == 1800


def test_effective_retrieval_fast():
    cfg = RetrievalConfig(
        vector_top_k=8,
        bm25_top_k=8,
        final_top_n=6,
        score_threshold=0.35,
        min_rerank_score=0.1,
        context_max_tokens=1800,
        fast_vector_top_k=3,
        fast_bm25_top_k=2,
        fast_final_top_n=2,
        fast_context_max_tokens=700,
    )

    fast = effective_retrieval(cfg, fast=True)
    assert fast["vector_top_k"] == 3
    assert fast["bm25_top_k"] == 2
    assert fast["final_top_n"] == 2
    assert fast["context_max_tokens"] == 700
