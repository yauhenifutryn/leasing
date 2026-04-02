from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tts_normalize import normalize_for_tts, transliterate_latin


def test_integer_in_text() -> None:
    assert "двадцать тысяч" in normalize_for_tts("стоимостью 20000 долларов")


def test_percentage() -> None:
    result = normalize_for_tts("аванс от 10% до 39%")
    assert "десяти" in result or "десять" in result
    assert "тридцати девяти" in result or "тридцать девять" in result
    assert "процентов" in result


def test_spaced_thousands() -> None:
    assert "двадцать тысяч" in normalize_for_tts("стоимость 20 000 рублей")


def test_decimal_number() -> None:
    result = normalize_for_tts("ставка 16.5%")
    assert "шестнадцать" in result
    assert "%" not in result


def test_year_preserved_as_number() -> None:
    result = normalize_for_tts("автомобиль 2008 года")
    assert "две тысячи восьмого" in result or "две тысячи восемь" in result


def test_no_numbers_unchanged() -> None:
    text = "Здравствуйте, чем могу помочь?"
    assert normalize_for_tts(text) == text


def test_currency_dollar() -> None:
    result = normalize_for_tts("цена $20,000")
    assert "двадцать тысяч" in result
    assert "долларов" in result


def test_mixed_content() -> None:
    result = normalize_for_tts("аванс 25% на срок 18 месяцев за 20000 долларов")
    assert "двадцать пять" in result
    assert "восемнадцать" in result
    assert "двадцать тысяч" in result


# --- Transliteration tests ---


def test_known_brand() -> None:
    assert transliterate_latin("Volkswagen Polo Sedan") == "Фольксваген Поло Седан"


def test_messenger_names() -> None:
    assert transliterate_latin("Viber") == "Вайбер"
    assert transliterate_latin("WhatsApp") == "Вотсапп"
    assert transliterate_latin("Telegram") == "Телеграм"


def test_email_term() -> None:
    assert transliterate_latin("e-mail") == "имейл"
    assert transliterate_latin("email") == "имейл"


def test_unknown_latin_fallback() -> None:
    result = transliterate_latin("Bentley")
    assert len(result) > 0
    assert all(
        c.isspace() or c == "-" or ("\u0400" <= c <= "\u04ff")
        for c in result
    )


def test_mixed_cyrillic_latin() -> None:
    result = transliterate_latin("автомобиль BMW X5")
    assert "автомобиль" in result
    assert "бэ эм вэ" in result


def test_full_pipeline_with_transliteration() -> None:
    result = normalize_for_tts("Volkswagen Polo стоит $20,000 или 39%")
    assert "Фольксваген" in result
    assert "Поло" in result
    assert "двадцать тысяч" in result
    assert "долларов" in result
    assert "тридцать девять процентов" in result
