from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

_WORD_RE = re.compile(r"[A-Za-zА-Яа-я0-9]{3,}")


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _clean_list(items: Any) -> List[str]:
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, Iterable):
        return []
    cleaned: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text:
            cleaned.append(text)
    return cleaned


def _derive_keywords(entry: Dict[str, Any], limit: int = 8) -> List[str]:
    source = " ".join(
        [
            _clean_str(entry.get("intent")),
            _clean_str(entry.get("canonical_question")),
        ]
    )
    words = _WORD_RE.findall(source)
    unique: List[str] = []
    for word in words:
        if word not in unique:
            unique.append(word)
        if len(unique) >= limit:
            break
    return unique


def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(entry)
    category = _clean_str(entry.get("category"))
    subtopic = _clean_str(entry.get("subtopic"))

    if not category:
        category = _clean_str(entry.get("intent")) or "Общее"
    if not subtopic:
        subtopic = "Разное"

    keywords = _clean_list(entry.get("keywords"))
    if not keywords:
        keywords = _derive_keywords(entry)

    out["category"] = category
    out["subtopic"] = subtopic
    out["keywords"] = keywords
    out["tags"] = _clean_list(entry.get("tags"))
    out["references"] = _clean_list(entry.get("references"))
    return out
