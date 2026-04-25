"""Tests for utterance-level fallback grounding.

Issue 7 (live call 77cfa127, 2026-04-25): user said "Я думаю взять
себе машину" — Qwen3-4B classifier returned intent=RAG with NO subject
extraction. The bare-машина grounding in classifier_schema.py operates
on classifier OUTPUT, so it had nothing to ground. Profile stayed
subj=- and the orchestrator legitimately re-asked for subject on the
next turn.

Fix: utterance-level fallback grounding that runs when the classifier
omits a slot. Returns the most likely slot value from the utterance
text alone, using the same regex / cue rules already in
classifier_schema.py.

Constraints:
  - Conservative: never override an explicit classifier value.
  - Drop on competing categories (don't ground "грузовик" as Легковой).
  - Drop when utterance is ambiguous (no clear cue).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.utterance_grounding import extract_subject_from_utterance


# ---------- Bare-car path (live regression) ----------


def test_bare_mashinu_grounds_legkovoy() -> None:
    """The live regression: classifier omitted subject on this exact utterance."""
    assert extract_subject_from_utterance("Я думаю взять себе машину.") == "Легковой автомобиль"


def test_bare_avtomobil_grounds_legkovoy() -> None:
    assert extract_subject_from_utterance("хочу автомобиль") == "Легковой автомобиль"


def test_bare_avto_grounds_legkovoy() -> None:
    assert extract_subject_from_utterance("давай посчитаем авто") == "Легковой автомобиль"


# ---------- Explicit category cues ----------


def test_gruzovik_grounds_gruzovoy() -> None:
    assert extract_subject_from_utterance("хочу грузовик") == "Грузовой автомобиль"


def test_truck_synonyms_ground_gruzovoy() -> None:
    assert extract_subject_from_utterance("нужен тягач") == "Грузовой автомобиль"
    assert extract_subject_from_utterance("ищу самосвал") == "Грузовой автомобиль"


def test_spectekh_grounds_spectekhnika() -> None:
    assert extract_subject_from_utterance("нужна спецтехника") == "Спецтехника"
    assert extract_subject_from_utterance("ищу экскаватор") == "Спецтехника"


def test_oborudovanie_grounds_oborudovanie() -> None:
    assert extract_subject_from_utterance("оборудование для производства") == "Оборудование"


# ---------- Negative cases (must NOT ground) ----------


def test_competing_category_blocks_bare_car() -> None:
    """If utterance has both 'машина' AND 'грузовик', do NOT ground as Легковой."""
    res = extract_subject_from_utterance("у меня грузовая машина")
    # Should ground to Грузовой (specific), NOT Легковой (generic).
    assert res == "Грузовой автомобиль"


def test_no_subject_signal_returns_none() -> None:
    assert extract_subject_from_utterance("привет, как дела") is None


def test_empty_utterance_returns_none() -> None:
    assert extract_subject_from_utterance("") is None


def test_only_name_returns_none() -> None:
    assert extract_subject_from_utterance("Я Никита.") is None


def test_question_about_terms_returns_none() -> None:
    assert extract_subject_from_utterance("какие у вас условия") is None


def test_address_question_returns_none() -> None:
    assert extract_subject_from_utterance("какой адрес в Минске") is None


# ---------- Boundary cases ----------


def test_legkovaya_mashina_grounds_legkovoy() -> None:
    """Explicit Легковая cue takes priority over generic машина."""
    assert extract_subject_from_utterance("легковая машина") == "Легковой автомобиль"


def test_kommercheskiy_subject_does_not_misground() -> None:
    """Bare "коммерческий транспорт" without a specific vehicle cue is
    ambiguous — return None and let the orchestrator ask. Keeps the
    fallback conservative."""
    res = extract_subject_from_utterance("коммерческий транспорт")
    assert res != "Легковой автомобиль"  # must not misground as car
    # Either None or "Прочий транспорт" / "Грузовой автомобиль" is acceptable.
