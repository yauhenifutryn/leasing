from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import yaml


def load_abbreviations(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {k.lower(): v for k, v in (payload.get("expansions") or {}).items()}


def normalize_query(text: str, expansions: dict[str, str]) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    tokens = cleaned.split(" ")
    out = []
    for t in tokens:
        key = re.sub(r"[^\wа-яё]+", "", t)
        if key in expansions:
            out.append(expansions[key])
        else:
            out.append(t)
    return " ".join(out).strip()
