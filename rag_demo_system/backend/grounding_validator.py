"""Post-generation validator that strips hallucinated high-risk facts.

Runs only on RAG-intent turns (information questions). Extracts Belarusian
phone numbers, street addresses, and Russian tri-part personal names from
the LLM response. For each extracted fact, checks whether it appears in
the retrieved chunks. If not, replaces the span with a safe fallback.
"""

from __future__ import annotations

import re
from typing import Any

FALLBACK = ", уточните, пожалуйста, у специалиста по телефону +375 17 322 77 00"

_PHONE_RE = re.compile(r"\+?375[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")
_STREET_RE = re.compile(
    r"(?:ул\.?|улица|пр-т|проспект|шоссе|бульвар|переулок|пер\.?)\s+"
    r"[А-ЯЁ][а-яёА-ЯЁ\-]+"
    r"(?:,\s*\d+[а-яё]?)?",
    re.IGNORECASE,
)
_PATRONYMIC = r"(?:ович|евич|инич|ьевич|иевич|овна|евна|инична|ьевна)"
_NAME_RE = re.compile(
    rf"[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+{_PATRONYMIC}\s+[А-ЯЁ][а-яё]+"
)


def extract_high_risk_facts(text: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for m in _PHONE_RE.finditer(text or ""):
        facts.append({"type": "phone", "value": m.group(0), "span": m.span()})
    for m in _STREET_RE.finditer(text or ""):
        facts.append({"type": "street_address", "value": m.group(0), "span": m.span()})
    for m in _NAME_RE.finditer(text or ""):
        facts.append({"type": "personal_name", "value": m.group(0), "span": m.span()})
    return facts


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def check_grounded(fact: dict[str, Any], chunks: list[str]) -> bool:
    value_norm = _normalize(fact.get("value", ""))
    if not value_norm:
        return True
    haystack = _normalize(" ".join(chunks or []))
    if value_norm in haystack:
        return True
    # Fuzzy: check with "ул." / "улица" normalized together
    value_alt = value_norm.replace("ул. ", "улица ").replace("пр-т ", "проспект ")
    if value_alt in haystack:
        return True
    return False


def replace_ungrounded(response: str, chunks: list[str]) -> str:
    facts = extract_high_risk_facts(response)
    if not facts:
        return response
    # Sort by start position descending so replacements don't shift earlier spans.
    facts.sort(key=lambda f: f["span"][0], reverse=True)
    out = response
    for f in facts:
        if check_grounded(f, chunks):
            continue
        start, end = f["span"]
        print(f"[Grounding] replaced ungrounded {f['type']}: {f['value']!r} -> fallback", flush=True)
        if f["type"] == "personal_name":
            out = out[:start] + out[end:]  # strip entirely
        else:
            out = out[:start] + FALLBACK + out[end:]
    return out
