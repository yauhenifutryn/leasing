import json
from pathlib import Path
from typing import Any, List


def maybe_log_progress(index: int, total: int, every: int) -> bool:
    if every <= 0:
        return False
    if index % every != 0 and index != total:
        return False
    print(f"KB progress: {index}/{total}", flush=True)
    return True


def maybe_write_checkpoint(
    index: int,
    total: int,
    every: int,
    out_path: Path,
    partial_path: Path,
    data: List[Any],
) -> bool:
    if every <= 0:
        return False
    if index % every != 0 and index != total:
        return False
    partial_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True
