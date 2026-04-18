"""Tests for `split_for_tts_streaming` (Fix 25 — interruptible readback TTS).

The split function is the core of Fix 25: it turns a monolithic readback
string into short phrases so `_emit_plain_assistant_response` can call
Silero once per phrase and check `session.interrupted` between calls.
Without this, one readback = one 3-4s blocking synth = barge-in blackout.
"""

from __future__ import annotations

from backend.text_utils import split_for_tts_streaming


def test_empty_and_whitespace_return_empty():
    assert split_for_tts_streaming("") == []
    assert split_for_tts_streaming("   ") == []
    assert split_for_tts_streaming("\n\t") == []


def test_single_sentence_returns_single_phrase():
    out = split_for_tts_streaming("Здравствуйте.")
    assert out == ["Здравствуйте."]


def test_readback_splits_into_short_phrases():
    # Canonical readback produced by `build_readback_text`. Each comma
    # must become its own phrase so barge-in within the readback aborts
    # synthesis at the next phrase boundary.
    text = (
        "Давайте подтвердим параметры: Грузовой автомобиль, новый, "
        "стоимость 80000 BYN, Юридическое лицо, срок 36 месяцев, "
        "аванс 20%, график аннуитет. Всё верно?"
    )
    phrases = split_for_tts_streaming(text)
    assert len(phrases) >= 8
    assert phrases[0].endswith(":")
    assert phrases[-1] == "Всё верно?"
    # No phrase is more than ~60 chars — that bounds per-phrase Silero
    # blocking time and therefore bounds barge-in blackout duration.
    assert all(len(p) <= 60 for p in phrases), phrases


def test_clarification_prompt_splits_on_colon_and_commas():
    text = "Уточните, пожалуйста, стоимость, валюта (BYN или USD), новый или б/у."
    phrases = split_for_tts_streaming(text)
    # "Уточните," "пожалуйста," "стоимость," ...
    assert len(phrases) >= 3
    assert all(any(ch.isalnum() for ch in p) for p in phrases)


def test_punctuation_only_fragments_dropped():
    # "..." on its own is not a phrase — nothing to synthesize.
    phrases = split_for_tts_streaming("Привет. . . . ещё текст.")
    assert "." not in phrases
    assert any("Привет" in p for p in phrases)
    assert any("ещё текст" in p for p in phrases)


def test_change_confirm_short_text():
    text = "Меняю срок на 48, остальное оставляю. Всё верно?"
    phrases = split_for_tts_streaming(text)
    assert len(phrases) == 3
    assert phrases[0] == "Меняю срок на 48,"
    assert phrases[1] == "остальное оставляю."
    assert phrases[2] == "Всё верно?"


def test_preserves_trailing_punctuation_for_prosody():
    phrases = split_for_tts_streaming("Вы физическое лицо, ИП или юридическое лицо?")
    # Last phrase retains question mark for Silero's question intonation.
    assert phrases[-1].endswith("?")


def test_no_punctuation_returns_whole_text():
    out = split_for_tts_streaming("без знаков препинания просто слова")
    assert out == ["без знаков препинания просто слова"]
