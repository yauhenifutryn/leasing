import os
import json
import orjson
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

def read_json(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return orjson.loads(f.read())

def write_json(path: str | Path, obj: Any) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))

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
