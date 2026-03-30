from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.sentence_detector import SentenceDetector


def test_simple_sentence() -> None:
    d = SentenceDetector()
    assert d.feed("Привет. ") == ["Привет."]


def test_multiple_sentences() -> None:
    d = SentenceDetector()
    assert d.feed("Первое. Второе. ") == ["Первое.", "Второе."]


def test_partial_no_emit() -> None:
    d = SentenceDetector()
    assert d.feed("Начало предложения") == []


def test_incremental_tokens() -> None:
    d = SentenceDetector()
    assert d.feed("Лизинг") == []
    assert d.feed(" доступен") == []
    assert d.feed(". ") == ["Лизинг доступен."]


def test_question_mark() -> None:
    d = SentenceDetector()
    assert d.feed("Какой аванс? ") == ["Какой аванс?"]


def test_exclamation() -> None:
    d = SentenceDetector()
    assert d.feed("Здравствуйте! ") == ["Здравствуйте!"]


def test_ellipsis() -> None:
    d = SentenceDetector()
    assert d.feed("Давайте уточним... ") == ["Давайте уточним..."]


def test_abbreviation_no_split() -> None:
    d = SentenceDetector()
    assert d.feed("т.е. это значит. ") == ["т.е. это значит."]


def test_usd_no_split() -> None:
    d = SentenceDetector()
    assert d.feed("Сумма 2000 USD. ") == ["Сумма 2000 USD."]


def test_flush_remaining() -> None:
    d = SentenceDetector()
    d.feed("Неполное")
    assert d.flush() == "Неполное"


def test_flush_empty() -> None:
    d = SentenceDetector()
    assert d.flush() is None
