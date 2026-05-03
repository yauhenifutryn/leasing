"""Bug 26 — TTS phonetic dictionary entries for short Cyrillic abbreviations.

Silero TTS reads short Cyrillic uppercase tokens like ПДН by adding a
phantom trailing phoneme ("пэ-дэ-эн-эн" instead of "пэ-дэ-эн"). The
existing _ABBREV_TTS map at backend/voice_adapters.py:290 is the
canonical preprocessing surface; these tests pin the new entries so a
future cleanup can't silently drop them.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.voice_adapters import normalize_abbreviations_for_tts  # noqa: E402


def test_pdn_phonetic_spelling() -> None:
    assert normalize_abbreviations_for_tts("Что такое ПДН?") == "Что такое пэ дэ эн?"


def test_pdn_in_sentence_context() -> None:
    out = normalize_abbreviations_for_tts(
        "ПДН — это показатель долговой нагрузки."
    )
    assert "пэ дэ эн" in out
    assert "ПДН" not in out


def test_pdn_mixed_case_also_replaced() -> None:
    # LLM occasionally emits "ПДн" (mixed) — must still phoneticize.
    assert normalize_abbreviations_for_tts("Норматив ПДн = 40%") == "Норматив пэ дэ эн = 40%"


def test_rf_phonetic_spelling() -> None:
    assert normalize_abbreviations_for_tts("Лизинг в РФ не консультирую.") == \
        "Лизинг в эр эф не консультирую."


def test_rb_phonetic_spelling() -> None:
    assert normalize_abbreviations_for_tts("Регулятор НБ РБ.") == "Регулятор НБ эр бэ."


def test_ao_expanded_not_letter_by_letter() -> None:
    # АО is a corporate-form noun; spelling it letter-by-letter ("а о")
    # is awkward. Use the full noun phrase like ООО is handled.
    assert normalize_abbreviations_for_tts("Это АО.") == "Это акционерное общество."


def test_existing_entries_still_work_after_extension() -> None:
    # Regression guard: extending the dict must not break prior entries.
    assert "белорусских рублей" in normalize_abbreviations_for_tts("100 BYN")
    assert "и пэ" in normalize_abbreviations_for_tts("я ИП")
    assert "каско" in normalize_abbreviations_for_tts("оформите КАСКО")


def test_word_boundary_avoids_partial_match() -> None:
    # ПДНщик shouldn't be touched (word-boundary regex). This locks the
    # \b anchor stays in place if the dict grows.
    assert normalize_abbreviations_for_tts("ПДНщик не слово") == "ПДНщик не слово"


def test_gai_two_syllable_pronunciation() -> None:
    # Bug 26 follow-up (live call 099bfb78 2026-05-03): TTS read "ГАИ"
    # as one fast indistinct syllable. The two-syllable form "га и"
    # gives Silero a slight pause for clarity.
    out = normalize_abbreviations_for_tts(
        "Документы для ГАИ оформляются после выкупа."
    )
    assert "га и" in out
    assert "ГАИ" not in out
