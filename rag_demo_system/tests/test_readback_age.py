"""Fix 1.11 — age_years must appear in the readback when condition_new=0.

Live call 22028754 on 2026-04-19 exposed that the б/у readback went:

    "Легковой автомобиль, б/у, стоимость 15000 долларов (это 45000
     белорусских рублей по курсу 3 к 1), Физическое лицо, срок 24
     месяцев, аванс 20%, график аннуитет. Всё верно?"

The profile carried age_years=5 and the calculator was invoked with
age=5, but the spoken confirmation never mentioned it. Client answered
"Верно" on a parameter set they never heard. Violates the "no silent
inputs to the calculator before confirmation" contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_prompts import build_readback_text, _age_noun
from backend.session import ClientProfile


def _profile(**overrides):
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=45000.0,
        currency="BYN",
        condition_new=0,
        age_years=5,
        term_months=24,
        type_schedule="0",
        prepaid_pct=20.0,
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def test_used_readback_mentions_age() -> None:
    txt = build_readback_text(_profile())
    assert "5" in txt and "лет" in txt, f"age missing from readback: {txt}"
    assert "возраст" in txt.lower(), f"age label missing: {txt}"


def test_new_readback_no_age_phrase() -> None:
    txt = build_readback_text(_profile(condition_new=1, age_years=None))
    assert "возраст" not in txt.lower()


def test_used_but_unset_age_no_phrase() -> None:
    # condition_new=0 but age not yet captured: do not fabricate an age
    # line. The clarify (Fix 1.5) should have caught the missing age
    # before we got here, so seeing this in practice is unusual.
    txt = build_readback_text(_profile(age_years=None))
    assert "возраст" not in txt.lower()


def test_age_noun_russian_agreement() -> None:
    # 1 год, 2-4 года, 5-20 лет, 21 год, 22 года, 25 лет, etc.
    assert _age_noun(1) == "год"
    assert _age_noun(2) == "года"
    assert _age_noun(3) == "года"
    assert _age_noun(4) == "года"
    assert _age_noun(5) == "лет"
    assert _age_noun(11) == "лет"
    assert _age_noun(14) == "лет"
    assert _age_noun(21) == "год"
    assert _age_noun(22) == "года"
    assert _age_noun(25) == "лет"


def test_used_readback_age_with_usd_disclosure() -> None:
    # Combine Fix 1.6 (USD dual disclosure) with Fix 1.11 (age line).
    # Live call 22028754 used this exact shape (USD + б/у).
    p = _profile(currency="USD", cost=15000.0)
    txt = build_readback_text(p)
    # USD disclosure
    assert "долларов" in txt and "рубл" in txt
    # Age disclosure
    assert "возраст 5 лет" in txt
