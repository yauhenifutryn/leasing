"""B5 — subject-aware age question.

Live trace: subject = "Недвижимость" but bot asked "Сколько лет вашему
транспорту?" — wrong noun for a real-estate lease. The clarify gate must
branch on `profile.subject` and emit the matching dative-case noun.

Mapping:
    Легковой автомобиль  → транспорту
    Грузовой автомобиль  → транспорту
    Прочий транспорт     → транспорту
    Спецтехника          → технике
    Оборудование         → оборудованию
    Недвижимость         → объекту
    None / unknown       → транспорту  (legacy default — most common case)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_prompts import build_clarification_prompt  # noqa: E402
from backend.session import ClientProfile  # noqa: E402


def _ask_age(subject: str | None) -> str:
    profile = ClientProfile(subject=subject) if subject else ClientProfile()
    return build_clarification_prompt({"age_years"}, profile)


def test_age_for_legkovoy_uses_транспорту():
    assert "транспорту" in _ask_age("Легковой автомобиль")


def test_age_for_gruzovoy_uses_транспорту():
    assert "транспорту" in _ask_age("Грузовой автомобиль")


def test_age_for_prochiy_transport_uses_транспорту():
    assert "транспорту" in _ask_age("Прочий транспорт")


def test_age_for_spetstekhnika_uses_технике():
    out = _ask_age("Спецтехника")
    assert "технике" in out, f"expected 'технике', got: {out!r}"
    # Wrong noun must not leak.
    assert "транспорту" not in out


def test_age_for_oborudovanie_uses_оборудованию():
    out = _ask_age("Оборудование")
    assert "оборудованию" in out, f"expected 'оборудованию', got: {out!r}"
    assert "транспорту" not in out


def test_age_for_nedvizhimost_uses_объекту():
    """Live regression: realestate lease asked age in vehicle wording."""
    out = _ask_age("Недвижимость")
    assert "объекту" in out, f"expected 'объекту' for Недвижимость, got: {out!r}"
    assert "транспорту" not in out


def test_age_for_unknown_subject_defaults_to_транспорту():
    """Legacy fallback when subject is None — most common path is auto."""
    out = _ask_age(None)
    assert "транспорту" in out
