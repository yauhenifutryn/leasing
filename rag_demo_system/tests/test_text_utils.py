from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.text_utils import clean_answer


def test_clean_answer_extracts_final_marker() -> None:
    raw = "Размышления...\nFINAL: Здравствуйте. Чем могу помочь?"
    assert clean_answer(raw) == "Здравствуйте. Чем могу помочь?"


def test_clean_answer_strips_answer_prefix() -> None:
    raw = "**Ответ:** Да, можно."
    assert clean_answer(raw) == "Да, можно."
