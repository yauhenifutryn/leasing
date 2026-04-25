"""Utterance-level fallback grounding.

Extracts slot values directly from the user's utterance text when the
classifier omits them. Runs as a deterministic safety net for the
small Qwen3-4B classifier, which sometimes returns intent=RAG /
CONVERSATION with no slot extraction even though the utterance
clearly named a category.

Issue 7 (live call 77cfa127, 2026-04-25): user said "Я думаю взять
себе машину." — Qwen3-4B returned `intent: RAG` with no `subject`
field. classifier_schema's `_subject_value_grounded` operates on the
classifier's emitted VALUE, so it had nothing to ground. Profile
stayed `subj=-` and the orchestrator legitimately asked for subject
on the next turn (annoying re-ask).

Reuses the same regex tables from classifier_schema.py so the
fallback obeys identical category cues. Conservative by design:
returns None on ambiguous utterances.
"""
from __future__ import annotations

import re
from typing import Optional

# Mirror classifier_schema._SUBJECT_VALUE_CUES priority order: most-specific
# categories first so "грузовая машина" doesn't get caught by the bare-car
# fallback below.
_SPECIFIC_SUBJECT_CUES: list[tuple[str, re.Pattern[str]]] = [
    (
        "Грузовой автомобиль",
        re.compile(
            r"\b(грузов\w*|грузовик\w*|фур\w+|тягач\w*|самосвал\w*|"
            r"микроавтобус\w*|камаз|уаз)",
            re.IGNORECASE,
        ),
    ),
    (
        "Спецтехника",
        re.compile(
            r"\b(спецтехник\w*|погрузчик\w*|экскаватор\w*|бульдозер\w*|"
            r"кран\w*|каток\w*|трактор\w*|комбайн\w*)",
            re.IGNORECASE,
        ),
    ),
    (
        "Оборудование",
        re.compile(
            r"\b(оборудовани\w*|станк\w+|установк\w+)|\bлиния\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Недвижимость",
        re.compile(
            r"\b(недвижимост\w*|квартир\w+|здани\w+|помещени\w+|"
            r"склад\w*|офис\w*)",
            re.IGNORECASE,
        ),
    ),
    (
        "Прочий транспорт",
        re.compile(
            r"\b(автобус\w*|прицеп\w*|мотоцикл\w*|скутер\w*)",
            re.IGNORECASE,
        ),
    ),
    # "Легковой" specific cues run before the bare-car fallback.
    (
        "Легковой автомобиль",
        re.compile(
            r"\b("
            r"легков\w*|седан\w*|внедорожник\w*|кроссовер\w*|"
            r"bmw|mercedes|mercedes-benz|toyota|kia|hyundai|audi|volkswagen|vw|"
            r"lexus|mazda|renault|peugeot|ford|lada|skoda|fiat|chevrolet|"
            r"nissan|honda|"
            r"мерседес|тойот\w+|киа|хендай|ауди|фольксваген|лексус|мазд\w+|"
            r"фольцваген|рено|пежо|форд|лад\w+|шкод\w+|ниссан|хонд\w+"
            r")",
            re.IGNORECASE,
        ),
    ),
]

_GENERIC_CAR_RE = re.compile(r"\b(машин\w*|автомобил\w*|авто)\b", re.IGNORECASE)


def extract_subject_from_utterance(utterance: str) -> Optional[str]:
    """Return the most likely subject value from the utterance, or None.

    Order:
      1. Specific category cues (Грузовой / Спецтехника / Оборудование /
         Недвижимость / Прочий транспорт / Легковой brand+category words).
         First match wins — they're mutually exclusive in practice.
      2. Bare-car fallback ("машина" / "автомобиль" / "авто") → Легковой,
         only when no specific cue fired.

    Returns None when nothing matches or the utterance is empty.
    """
    if not utterance:
        return None
    utt = utterance
    for value, pattern in _SPECIFIC_SUBJECT_CUES:
        if pattern.search(utt):
            return value
    if _GENERIC_CAR_RE.search(utt):
        return "Легковой автомобиль"
    return None
