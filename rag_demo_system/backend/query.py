from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import yaml

# Synonym groups: when any word in a group appears in the query,
# all synonyms are appended to improve embedding retrieval coverage.
_SYNONYM_GROUPS: list[list[str]] = [
    ["директор", "руководство", "руководитель", "управляющий", "глава", "начальник"],
    ["аванс", "первоначальный взнос", "предоплата"],
    ["машина", "автомобиль", "авто", "транспорт"],
    ["грузовой", "тягач", "полуприцеп", "фура", "грузовик"],
    ["страховка", "страхование", "каско", "полис"],
    ["платёж", "платеж", "оплата", "взнос"],
    ["расторгнуть", "расторжение", "прекратить", "разорвать"],
    ["досрочно", "досрочное", "раньше срока", "погасить"],
    ["документы", "бумаги", "справки", "пакет документов"],
    ["офис", "отделение", "филиал", "представительство"],
    ["телефон", "номер", "позвонить", "контакт"],
]


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


def expand_synonyms(query: str) -> str:
    """Append synonyms to the query for better embedding retrieval.

    If any word in the query matches a synonym group, all other words
    from that group are appended. This runs after normalize_query.
    """
    query_lower = query.lower()
    additions: list[str] = []
    for group in _SYNONYM_GROUPS:
        for word in group:
            if word in query_lower:
                for synonym in group:
                    if synonym not in query_lower:
                        additions.append(synonym)
                break
    if not additions:
        return query
    return query + " " + " ".join(additions)
