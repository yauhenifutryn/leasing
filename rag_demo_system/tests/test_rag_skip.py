from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.rag_skip import should_skip_rag


def test_pure_name_capture_is_skipped():
    assert should_skip_rag("Привет, я Вадим.", {"name": "Вадим"}, {}) is True


def test_name_plus_question_is_not_skipped():
    assert should_skip_rag("Привет, я Вадим, какие офисы в Минске?",
                           {"name": "Вадим"}, {"action": "clarify"}) is False


def test_name_plus_tool_intent_is_not_skipped():
    assert should_skip_rag("Я Вадим и хочу машину за 100 тысяч.",
                           {"name": "Вадим", "subject": "Легковой автомобиль", "cost": 100000},
                           {"action": "calculate"}) is False


def test_question_mark_blocks_skip():
    assert should_skip_rag("Я Вадим. Адрес в Минске?",
                           {"name": "Вадим"}, {}) is False


def test_no_name_patch_no_skip():
    assert should_skip_rag("Здравствуйте.", {}, {}) is False


def test_long_utterance_no_skip():
    assert should_skip_rag("Привет, я Вадим, очень рад познакомиться с вами",
                           {"name": "Вадим"}, {}) is False
