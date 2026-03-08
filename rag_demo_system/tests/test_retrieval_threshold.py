from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.retrieval_utils import filter_vector_hits


def test_filter_vector_hits_applies_score_threshold() -> None:
    hits = [
        {"chunk_id": "high", "score": 0.82},
        {"chunk_id": "low", "score": 0.12},
        {"chunk_id": "missing"},
    ]

    filtered = filter_vector_hits(hits, score_threshold=0.35)

    assert filtered == [{"chunk_id": "high", "score": 0.82}]
