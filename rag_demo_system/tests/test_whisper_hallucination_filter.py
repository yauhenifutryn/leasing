from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.whisper_server import _is_hallucination


def test_empty_not_hallucination():
    assert _is_hallucination("") is False


def test_plain_speech_not_hallucination():
    assert _is_hallucination("Привет, я Вадим") is False
    assert _is_hallucination("Хочу рассчитать лизинг") is False
    assert _is_hallucination("Аннуитет.") is False


def test_prodolzhenie_sleduet_filtered():
    assert _is_hallucination("Продолжение следует...") is True
    assert _is_hallucination("Продолжение следует") is True
    assert _is_hallucination("продолжение следует") is True
    assert _is_hallucination("ПРОДОЛЖЕНИЕ СЛЕДУЕТ...") is True


def test_prodolzhaem_filtered():
    assert _is_hallucination("Продолжаем.") is True
    assert _is_hallucination("Продолжаем") is True


def test_subtitles_boilerplate_filtered():
    assert _is_hallucination("Субтитры создавал DimaTorzok") is True
    assert _is_hallucination("субтитры создавал dimatorzok") is True


def test_spasibo_za_vnimanie_filtered():
    assert _is_hallucination("Спасибо за внимание!") is True
    assert _is_hallucination("Спасибо за просмотр") is True


def test_substring_not_false_match():
    # A real utterance that happens to CONTAIN one of these words should NOT be filtered.
    # (We match only whole-string, not substring.)
    assert _is_hallucination("Я хочу продолжение разговора") is False
    assert _is_hallucination("Субтитры не нужны") is False
