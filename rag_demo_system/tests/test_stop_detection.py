from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from backend.text_utils import contains_stop_word


@pytest.mark.parametrize("text,expected", [
    ("Стоп.", True),
    ("замолчи на секунду", True),
    ("Помолчи, я думаю.", True),
    ("подожди секунду", True),
    ("тихо, пожалуйста", True),
    ("хватит", True),
    ("Не надо говорить.", True),
    ("Ну и что?", False),
    ("Алло.", False),
    ("в нашем разговоре уже", False),
    ("нажмите стоп-кран", True),  # literal match; classifier should filter
    ("", False),
    ("а что такое нагрузка?", False),
])
def test_contains_stop_word(text, expected):
    assert contains_stop_word(text) is expected


def test_contains_stop_word_short_commands_trigger():
    assert contains_stop_word("стоп")
    assert contains_stop_word("подожди")
    assert contains_stop_word("тихо замолчи")
    assert contains_stop_word("подожди пожалуйста")


def test_contains_stop_word_discourse_markers_do_not_trigger():
    # >3 tokens: stop-word embedded as discourse marker, not a command
    assert not contains_stop_word("подожди секунду я хочу уточнить")
    assert not contains_stop_word("погоди я ещё думаю над этим")
    assert not contains_stop_word("хватит уже говорить про это мне")


def test_contains_stop_word_three_token_boundary():
    # Exactly 3 tokens: still treated as stop
    assert contains_stop_word("стоп пожалуйста замолчи")
    # 4 tokens: not a stop
    assert not contains_stop_word("стоп пожалуйста замолчи уже")
