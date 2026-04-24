import pytest
from backend.numeric_words_ru import parse_ru_number


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
