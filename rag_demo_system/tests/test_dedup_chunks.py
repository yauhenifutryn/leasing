from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.retrieval_utils import dedup_chunks


def test_dedup_removes_near_duplicate_keeping_highest_score() -> None:
    """Near-duplicate chunks (>= 0.85 similarity) should be collapsed,
    keeping only the one with the highest rerank_score."""
    candidates = [
        {"chunk_id": "dir1", "text": "Директор компании Дедков Иван Петрович", "rerank_score": 0.84},
        {"chunk_id": "dir2", "text": "Директор компании Дедков Иван Петрович, генеральный", "rerank_score": 0.82},
        {"chunk_id": "owner", "text": "Владелец Mikro Kapital Management S.A. Люксембург", "rerank_score": 0.81},
        {"chunk_id": "deps", "text": "Заместители директора и руководители отделов", "rerank_score": 0.79},
    ]
    result = dedup_chunks(candidates, threshold=0.85)

    ids = [c["chunk_id"] for c in result]
    assert "dir1" in ids, "highest-scored duplicate should survive"
    assert "dir2" not in ids, "near-duplicate with lower score should be removed"
    assert "owner" in ids
    assert "deps" in ids
    assert len(result) == 3


def test_dedup_no_false_positives_on_distinct_chunks() -> None:
    """Chunks with different content should all survive."""
    candidates = [
        {"chunk_id": "a", "text": "Условия оперативного лизинга для юридических лиц", "rerank_score": 0.9},
        {"chunk_id": "b", "text": "График работы офисов в Минске и Гомеле", "rerank_score": 0.8},
        {"chunk_id": "c", "text": "Контактные телефоны и адреса филиалов", "rerank_score": 0.7},
    ]
    result = dedup_chunks(candidates, threshold=0.85)
    assert len(result) == 3


def test_dedup_empty_input() -> None:
    assert dedup_chunks([], threshold=0.85) == []


def test_dedup_single_chunk() -> None:
    candidates = [{"chunk_id": "x", "text": "some text", "rerank_score": 0.5}]
    result = dedup_chunks(candidates, threshold=0.85)
    assert len(result) == 1


def test_dedup_preserves_order() -> None:
    """Output order should match input order (already sorted by rerank_score desc)."""
    candidates = [
        {"chunk_id": "a", "text": "Первый уникальный текст про лизинг автомобилей", "rerank_score": 0.9},
        {"chunk_id": "b", "text": "Второй текст про условия договора", "rerank_score": 0.8},
        {"chunk_id": "c", "text": "Третий текст про страхование", "rerank_score": 0.7},
    ]
    result = dedup_chunks(candidates, threshold=0.85)
    scores = [c["rerank_score"] for c in result]
    assert scores == sorted(scores, reverse=True)


def test_dedup_triple_duplicate_keeps_one() -> None:
    """Three near-identical chunks should collapse to one (the highest scored)."""
    base = "Директор ООО Микро Лизинг Дедков Иван Петрович назначен приказом"
    candidates = [
        {"chunk_id": "d1", "text": base, "rerank_score": 0.85},
        {"chunk_id": "d2", "text": base + " номер 42", "rerank_score": 0.83},
        {"chunk_id": "d3", "text": base + " от 2020 года", "rerank_score": 0.80},
        {"chunk_id": "other", "text": "Совершенно другой текст про калькулятор", "rerank_score": 0.70},
    ]
    result = dedup_chunks(candidates, threshold=0.85)
    ids = [c["chunk_id"] for c in result]
    assert ids.count("d1") == 1
    assert "d2" not in ids
    assert "d3" not in ids
    assert "other" in ids
