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
