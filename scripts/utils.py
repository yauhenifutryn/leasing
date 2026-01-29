import os
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import orjson  # type: ignore
    _HAS_ORJSON = True
except Exception:
    orjson = None
    _HAS_ORJSON = False

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

def _json_default(obj: Any) -> Any:
    if np is not None and hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def read_json(path: str | Path) -> dict:
    if _HAS_ORJSON:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str | Path, obj: Any) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    if _HAS_ORJSON:
        with open(path, "wb") as f:
            f.write(
                orjson.dumps(
                    obj,
                    option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
                )
            )
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)

def list_audio(in_dir: str | Path) -> List[Path]:
    exts = {".wav", ".mp3", ".m4a", ".flac"}
    return [p for p in Path(in_dir).glob("*") if p.suffix.lower() in exts]

def list_files(in_dir: str | Path, suffix: str) -> List[Path]:
    return sorted(Path(in_dir).glob(f"*{suffix}"))

def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def should_skip(path: str | Path, overwrite: bool) -> bool:
    return Path(path).exists() and not overwrite

def chunked(items: Iterable[Path], size: int) -> List[List[Path]]:
    collection = list(items)
    if size <= 0:
        return [collection]
    return [collection[i : i + size] for i in range(0, len(collection), size)]
