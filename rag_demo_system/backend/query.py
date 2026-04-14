from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import yaml

# Synonym groups: when any word in a group appears in the query,
# all synonyms are appended to improve embedding retrieval coverage.
_SYNONYM_GROUPS: list[list[str]] = [
    ["директор", "руководство", "руководитель", "управляющий", "глава", "начальник"],
    ["владелец", "учредитель", "собственник", "акционер", "основатель", "управляется"],
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
    ["валюта", "рубли", "доллары", "евро", "BYN", "USD", "EUR"],
    ["юридическое лицо", "юрлицо", "ИП", "предприниматель", "организация", "компания"],
    ["физическое лицо", "физлицо", "частное лицо", "гражданин"],
    ["недвижимость", "помещение", "здание", "офис", "склад", "коммерческая недвижимость"],
    ["ставка", "процент", "удорожание", "переплата"],
    ["срок", "период", "длительность", "на сколько лет"],
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


def llm_expand_query(query: str, base_url: str, model: str) -> str:
    """Use LLM to rewrite query with synonyms for better retrieval.

    Adds ~150-250ms latency but dramatically improves recall when the user's
    phrasing doesn't match KB terminology.
    """
    if not base_url or not model:
        return query
    try:
        from .llm import call_openai_compatible
        resp = call_openai_compatible(
            base_url=base_url,
            model=model,
            system_prompt=(
                "Ты помощник по поиску в базе знаний лизинговой компании «Микро Лизинг». "
                "Перепиши вопрос клиента как набор ключевых слов и синонимов для поиска. "
                "Если вопрос короткий или неполный, добавь очевидные уточнения. "
                "Например: «адрес» → «адрес офиса Минск Гомель Брест контакты расположение». "
                "«ставка» → «процентная ставка процент удорожание лизинговое вознаграждение». "
                "Верни ТОЛЬКО ключевые слова через пробел. Максимум 20 слов."
            ),
            user_prompt=query,
            temperature=0.0,
            max_tokens=50,
            timeout_sec=5,
        )
        expanded = resp.text.strip().strip('"').strip("'")
        if expanded and 3 < len(expanded) < 200:
            return f"{query} {expanded}"
    except Exception:  # noqa: BLE001
        pass  # Fallback to original query on any error
    return query


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
