import pytest
from backend.numeric_words_ru import parse_ru_number, replace_ru_number_words


@pytest.mark.parametrize("text,expected", [
    ("двадцать тысяч", 20000),
    ("сто тысяч", 100000),
    ("пятьдесят тысяч", 50000),
    ("пять тысяч", 5000),
    ("тысяча", 1000),
    ("миллион", 1000000),
    ("два миллиона", 2000000),
    ("сто двадцать тысяч", 120000),
    ("двести пятьдесят тысяч", 250000),
    ("оставим двадцать тысяч долларов", 20000),
    ("хочу сто тысяч рублей", 100000),
])
def test_parse_ru_number_handles_common_forms(text, expected):
    assert parse_ru_number(text) == expected


@pytest.mark.parametrize("text", [
    "",
    "привет",
    "никакого числа здесь нет",
    "двадцать процентов",
])
def test_parse_ru_number_returns_none_when_no_number(text):
    assert parse_ru_number(text) is None


def test_parse_ru_number_with_digits_returns_none():
    # Digit-form is not this function's job — grounded by digit search.
    assert parse_ru_number("20000 долларов") is None


# Bug session-2026-05-08: colloquial fractional Russian numbers
# ("полмиллиона", "полтора миллиона", "полторы тысячи", "пол тысячи").
# Live transcript "за полмиллион рублей" was not understood; user typed
# the equivalent of 500,000 RUB and the bot asked for the cost again.
@pytest.mark.parametrize("text,expected", [
    # half-of-scale, single-token forms (with and without trailing -а/-ов)
    ("полмиллиона", 500000),
    ("полмиллион", 500000),
    ("полмиллионов", 500000),
    ("полмиллиарда", 500000000),
    ("полтысячи", 500),
    # half-of-scale, two-word ("пол миллиона", "пол тысячи")
    ("пол миллиона", 500000),
    ("пол тысячи", 500),
    ("пол миллиона рублей", 500000),
    # one-and-a-half-scale ("полтора миллиона" / "полторы тысячи")
    ("полтора миллиона", 1500000),
    ("полторы тысячи", 1500),
    ("полтора миллиона долларов", 1500000),
    # standalone (legacy slang) "полста" = 50
    ("полста", 50),
    # Mixed inside a sentence
    ("хочу купить за полмиллион рублей", 500000),
    ("за полмиллиона", 500000),
    ("полтора миллиона рублей пожалуйста", 1500000),
])
def test_parse_ru_number_handles_fractional_colloquial(text, expected):
    assert parse_ru_number(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("за полмиллион рублей", "за 500000 рублей"),
    ("полмиллиона", "500000"),
    ("полтора миллиона долларов", "1500000 долларов"),
    ("полторы тысячи", "1500"),
    ("пол тысячи рублей", "500 рублей"),
    # idempotency: digit-mixed input is not rewritten
    ("500000 рублей", "500000 рублей"),
    # not-a-number: "пол" alone (e.g., "пол полки") is left untouched
    ("деревянный пол", "деревянный пол"),
])
def test_replace_ru_number_words_handles_fractional_colloquial(text, expected):
    assert replace_ru_number_words(text) == expected
