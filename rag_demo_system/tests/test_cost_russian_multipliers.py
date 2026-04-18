"""Fix 34: `has_field_signal` must recognise Russian spelled multipliers
("80 тысяч" = 80000, "3 миллиона" = 3000000) for cost / prepaid_amount.

Observed 2026-04-18 session 600a3858:
  User: "Давай я всё-таки хочу легковой автомобиль за 80 тысяч рублей"
  Bot:  "Меняю предмет лизинга на Легковой автомобиль, остальное оставляю"
                                      ^^^ cost 80000 was dropped by Fix 31's
                                          strict literal-digit check

Fix 34 accepts "80 тысяч" / "80 тыс" / "3 миллиона" / "3 млн" patterns.
"""

from __future__ import annotations

from backend.profile_hygiene import has_field_signal


def test_cost_80_тысяч_matches():
    assert has_field_signal("cost", 80000, "за 80 тысяч рублей") is True


def test_cost_150_тысяч_matches():
    assert has_field_signal("cost", 150000, "стоимость 150 тысяч") is True


def test_cost_80_тыс_short_matches():
    assert has_field_signal("cost", 80000, "80 тыс") is True


def test_cost_3_миллиона_matches():
    assert has_field_signal("cost", 3000000, "3 миллиона") is True


def test_cost_1_миллион_matches():
    assert has_field_signal("cost", 1000000, "1 миллион") is True


def test_cost_2_млн_short_matches():
    assert has_field_signal("cost", 2000000, "2 млн") is True


def test_cost_literal_still_matches():
    assert has_field_signal("cost", 80000, "80000") is True
    assert has_field_signal("cost", 150000, "150 000 рублей") is True


def test_cost_no_signal_still_rejects():
    # No mention of cost in utterance at all → reject
    assert has_field_signal("cost", 80000, "срок 36 месяцев") is False


def test_prepaid_amount_also_accepts_тысяч():
    assert has_field_signal("prepaid_amount", 16000, "16 тысяч") is True


def test_term_months_accepts_digits_or_years():
    # Fix 40b: term now accepts both "48 месяцев" and "4 года" (years-to-months
    # conversion). Without the years path, multi-field changes like
    # "грузовик за 50 тысяч на 7 лет" dropped the term patch because "84"
    # was never in the utterance.
    assert has_field_signal("term_months", 48, "48 месяцев") is True
    assert has_field_signal("term_months", 48, "срок на 4 года") is True
    # Wrong year count still rejects
    assert has_field_signal("term_months", 48, "срок на 3 года") is False
    # Non-whole-year months still require literal digits
    assert has_field_signal("term_months", 30, "срок на 2 года") is False  # 24 != 30


def test_prepaid_pct_still_requires_literal_digits():
    assert has_field_signal("prepaid_pct", 20, "20 процентов") is True


def test_cost_wrong_multiplier_value_rejects():
    # User said "80 тысяч" but hint is 150000 — mismatch, reject
    assert has_field_signal("cost", 150000, "80 тысяч") is False
