"""Tests for STT echo-filter false-positive guard.

Issue 8 (live call 77cfa127, 2026-04-25): bot asked "новая или б/у...
в рублях или долларах?" — user replied "Новая машина в долларах."
The word-overlap echo filter (threshold 0.6) saw the user's reply
sharing 4/5 words with the bot's recent speech and rejected it as
echo. Three rejections in a row → call hung → disconnect.

Root cause: legitimate user answers naturally echo bot vocabulary.
A 0.6 threshold over a short user utterance is below the noise floor
of "user answers using bot's words".

Fix:
  1. Skip word-overlap filter entirely on short utterances (≤ 5 words).
     Short replies are usually direct answers using bot vocab.
  2. Raise threshold to 0.85 for longer utterances. True acoustic echo
     produces near-identical text via Whisper; 0.85 keeps that, drops
     the false-positive ramp.
  3. Keep substring match for literal repeats — those are unambiguous.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.echo_filter import is_echo


# ---------- True positives (must filter) ----------


def test_literal_substring_is_echo() -> None:
    """User STT exactly contains bot speech — clearly echo."""
    bot = "ХОТИТЕ ИЗМЕНИТЬ ПАРАМЕТРЫ ИЛИ ОТПРАВИТЬ ГРАФИК ПО СМС"
    user = "хотите изменить параметры"
    assert is_echo(user, bot) is True


def test_high_overlap_long_utterance_is_echo() -> None:
    """User has 0.9+ word overlap with bot, 6+ words — acoustic echo."""
    bot = "ПОДСКАЖИТЕ ТИП КЛИЕНТА ФИЗИЧЕСКОЕ ИЛИ ЮРИДИЧЕСКОЕ ЛИЦО"
    user = "подскажите тип клиента физическое или юридическое"
    assert is_echo(user, bot) is True


# ---------- False positives we must NOT filter (live regression) ----------


def test_short_answer_with_bot_vocab_not_echo() -> None:
    """The live regression utterance — user's legitimate answer."""
    bot = "ДЛЯ РАСЧЕТА УТОЧНИТЕ МАШИНА НОВАЯ ИЛИ Б У И КАКАЯ У НЕЁ СТОИМОСТЬ В РУБЛЯХ ИЛИ ДОЛЛАРАХ"
    user = "Новая машина в долларах"
    assert is_echo(user, bot) is False


def test_short_yes_answer_not_echo() -> None:
    bot = "ХОТИТЕ ОТПРАВИТЬ ГРАФИК ПО СМС"
    user = "да"
    assert is_echo(user, bot) is False


def test_one_word_subject_answer_not_echo() -> None:
    bot = "ЧТО ИМЕННО ХОТИТЕ В ЛИЗИНГ ЛЕГКОВОЙ АВТОМОБИЛЬ ГРУЗОВОЙ ОБОРУДОВАНИЕ"
    user = "грузовик"
    assert is_echo(user, bot) is False


def test_short_clarification_with_bot_words_not_echo() -> None:
    """User echoes bot's slot question phrasing legitimately."""
    bot = "СРОК В МЕСЯЦАХ И ПРОЦЕНТ АВАНСА ПОЖАЛУЙСТА"
    user = "тридцать месяцев тридцать процентов"
    assert is_echo(user, bot) is False


def test_three_word_answer_not_echo() -> None:
    bot = "АНАЛИЗ ВАШИХ ПАРАМЕТРОВ ПОДОЖДИТЕ"
    user = "хорошо подожду"
    assert is_echo(user, bot) is False


# ---------- True positives over the new threshold ----------


def test_long_paraphrase_above_threshold_is_echo() -> None:
    """8+ word user utterance with 0.85+ overlap — likely acoustic feedback
    Whisper paraphrased. Filter."""
    bot = "ЗДРАВСТВУЙТЕ ВАС ПРИВЕТСТВУЕТ КОМПАНИЯ МИКРО ЛИЗИНГ ДЛЯ ПРОДОЛЖЕНИЯ РАЗГОВОРА"
    user = "здравствуйте вас приветствует компания микро лизинг для продолжения разговора"
    assert is_echo(user, bot) is True


def test_below_threshold_long_utterance_not_echo() -> None:
    """User's words happen to overlap moderately with bot but utterance is
    independent. 0.6 overlap, 7 words — should NOT be filtered now."""
    bot = "ПОДСКАЖИТЕ ТИП КЛИЕНТА ФИЗИЧЕСКОЕ ИЛИ ЮРИДИЧЕСКОЕ ЛИЦО"
    user = "клиент физическое лицо хочу взять машину"  # ~3/6 = 0.5 overlap
    assert is_echo(user, bot) is False


# ---------- Boundary ----------


def test_empty_user_is_not_echo() -> None:
    assert is_echo("", "БОТ ЧТО-ТО СКАЗАЛ") is False


def test_empty_bot_history_is_not_echo() -> None:
    assert is_echo("любой текст", "") is False


def test_single_char_user_not_echo() -> None:
    """Length guard: STT garbage like "А" or "О"."""
    assert is_echo("а", "БОТ ГОВОРИТ ЧТО-ТО О") is False
