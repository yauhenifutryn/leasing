from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.text_utils import clean_voice_output


def test_strips_emoji():
    assert clean_voice_output("Отличный выбор! 😄 Давайте рассчитаем.") == "Отличный выбор! Давайте рассчитаем."


def test_strips_multiple_emoji():
    assert clean_voice_output("Хорошо 👍🤔 Считаю!") == "Хорошо Считаю!"


def test_converts_numbered_list():
    text = "Вам нужно: 1. Паспорт 2. Права 3. Справка о доходах"
    result = clean_voice_output(text)
    assert "1." not in result
    assert "паспорт" in result.lower()


def test_converts_dash_list():
    text = "Условия:\n- аванс 30%\n- срок 36 месяцев\n- равные платежи"
    result = clean_voice_output(text)
    assert "\n-" not in result
    assert "аванс" in result.lower()


def test_strips_markdown_bold():
    assert "**" not in clean_voice_output("**Важно:** аванс 30%.")


def test_strips_markdown_headers():
    assert "##" not in clean_voice_output("## Условия лизинга")


def test_preserves_normal_text():
    text = "Ежемесячный платёж составит около 700 рублей. Хотите изменить параметры?"
    assert clean_voice_output(text) == text


def test_preserves_phone_numbers():
    text = "Звоните по номеру +375 17 322 77 00."
    assert clean_voice_output(text) == text


def test_strips_asterisk_bullets():
    text = "Документы:\n* паспорт\n* водительское удостоверение"
    result = clean_voice_output(text)
    assert "\n*" not in result
