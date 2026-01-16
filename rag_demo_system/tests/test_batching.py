from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.engine import iter_batches


def test_iter_batches_splits_items() -> None:
    items = list(range(10))
    batches = list(iter_batches(items, 4))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert batches[0] == [0, 1, 2, 3]
    assert batches[-1] == [8, 9]
