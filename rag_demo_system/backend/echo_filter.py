"""STT echo-filter for Jambonz audio pipeline.

Detects when Whisper transcribed the bot's own TTS audio leaking back
through the SIP line as acoustic feedback (speakerphone mode is the
common cause). False positives matter: legitimate user replies often
echo bot vocabulary ("Новая машина в долларах" answering "новая или
б/у... в рублях или долларах?"), and rejecting those locks the user
out of the conversation.

Issue 8 (live call 77cfa127, 2026-04-25): the previous in-line filter
in app.py used word-set overlap >= 0.6 and treated all matches as
echo. The user's 4-word legitimate answer scored 0.75 overlap with
bot's question and got filtered. After three rejections in a row the
call disconnected.

Heuristics here:
  1. Substring of bot speech → echo (literal repeat is unambiguous).
  2. Length-aware: skip word-overlap check entirely on short user
     utterances (≤ 5 words). Short replies are direct answers to bot
     questions and naturally use bot vocabulary.
  3. For longer utterances: require ≥ 0.85 word overlap. True acoustic
     echo produces near-identical text via Whisper; raising the
     threshold from 0.6 → 0.85 keeps that detection while letting
     paraphrased natural speech through.
  4. Empty / single-char inputs → not echo.
"""
from __future__ import annotations


_SHORT_UTTERANCE_WORD_THRESHOLD = 5
_OVERLAP_ECHO_THRESHOLD = 0.85
_MIN_USER_LEN_CHARS = 3
_MIN_USER_WORDS_FOR_OVERLAP = 2


def is_echo(user_text: str, recent_bot_text: str) -> bool:
    """Return True when `user_text` is likely echo of `recent_bot_text`.

    Both arguments are case-folded to upper inside this function so callers
    can pass either case. The empty-input cases short-circuit to False.
    """
    if not user_text or not recent_bot_text:
        return False
    user_up = user_text.upper().strip()
    bot_up = recent_bot_text.upper().strip()
    if len(user_up) < _MIN_USER_LEN_CHARS or not bot_up:
        return False

    # Path 1: literal substring of bot speech is unambiguous echo.
    if user_up in bot_up:
        return True

    # Path 2: word-overlap, only for utterances long enough that overlap is
    # meaningful. Short user replies legitimately echo bot vocabulary.
    user_words = user_up.split()
    if len(user_words) <= _SHORT_UTTERANCE_WORD_THRESHOLD:
        return False
    if len(user_words) < _MIN_USER_WORDS_FOR_OVERLAP:
        return False

    bot_words = set(bot_up.split())
    user_word_set = set(user_words)
    overlap = len(user_word_set & bot_words) / len(user_word_set)
    return overlap >= _OVERLAP_ECHO_THRESHOLD
