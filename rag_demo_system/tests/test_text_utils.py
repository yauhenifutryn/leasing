from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.text_utils import clean_answer, iter_final_text, sanitize_rewrite


def test_clean_answer_extracts_final_marker() -> None:
    raw = "Размышления...\nFINAL: Здравствуйте. Чем могу помочь?"
    assert clean_answer(raw) == "Здравствуйте. Чем могу помочь?"


def test_clean_answer_strips_answer_prefix() -> None:
    raw = "**Ответ:** Да, можно."
    assert clean_answer(raw) == "Да, можно."

def test_clean_answer_strips_think_block() -> None:
    raw = "<think>Внутренние размышления</think>\nFINAL: Привет."
    assert clean_answer(raw) == "Привет."


def test_iter_final_text_skips_until_marker() -> None:
    chunks = ["Размышления ", "FINAL:", " Здравствуйте", ". Чем могу помочь?"]
    out = "".join(iter_final_text(chunks))
    assert out == " Здравствуйте. Чем могу помочь?"


def test_iter_final_text_skips_think_block() -> None:
    chunks = ["<think>шаг 1", " шаг 2</think>", "FINAL: Привет"]
    out = "".join(iter_final_text(chunks))
    assert out == " Привет"


def test_iter_final_text_fallback_after_think() -> None:
    chunks = ["<think>шаг 1</think>", "Здравствуйте."]
    out = "".join(iter_final_text(chunks))
    assert out == "Здравствуйте."


def test_sanitize_rewrite_strips_think_and_final() -> None:
    raw = "<think>шаг 1</think>\nFINAL: лизинг для ИП"
    assert sanitize_rewrite(raw) == "лизинг для ИП"


def test_sanitize_rewrite_keeps_first_line_only() -> None:
    raw = "ключевые слова\nвторая строка"
    assert sanitize_rewrite(raw) == "ключевые слова"
