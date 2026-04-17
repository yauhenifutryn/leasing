"""Russian abbreviation expansion in clean_voice_output (TTS preprocessing)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.text_utils import clean_voice_output, _expand_abbreviations  # noqa: E402


# ── Positive cases: abbreviation -> full word ──────────────────────────


def test_street_ul() -> None:
    assert _expand_abbreviations("ул. Комсомольская, 10а") == "улица Комсомольская, 10а"


def test_prospect_hyphen_t() -> None:
    assert _expand_abbreviations("пр-т Ленина, 33") == "проспект Ленина, 33"


def test_prospect_prosp() -> None:
    assert _expand_abbreviations("просп. Победителей") == "проспект Победителей"


def test_prospect_pr_before_capital() -> None:
    assert _expand_abbreviations("пр. Машерова, 6а") == "проспект Машерова, 6а"


def test_gorod_before_city() -> None:
    assert _expand_abbreviations("г. Минск") == "город Минск"


def test_oblast_expand() -> None:
    assert _expand_abbreviations("Минская обл.") == "Минская область"


def test_rayon_expand() -> None:
    assert _expand_abbreviations("Центральный р-н") == "Центральный район"


def test_tel_expand() -> None:
    assert _expand_abbreviations("тел. +375 17 322 77 00") == "телефон +375 17 322 77 00"


def test_dom_with_digit() -> None:
    assert _expand_abbreviations("д. 5") == "дом 5"
    assert _expand_abbreviations("д.5") == "дом 5"


def test_kvartira_with_digit() -> None:
    assert _expand_abbreviations("кв. 12") == "квартира 12"


def test_korpus_and_etazh() -> None:
    assert _expand_abbreviations("корп. 2, эт. 3") == "корпус 2, этаж 3"


def test_full_address_expansion() -> None:
    addr = "г. Гомель, пр-т Ленина, д. 33, эт. 1"
    expected = "город Гомель, проспект Ленина, дом 33, этаж 1"
    assert _expand_abbreviations(addr) == expected


# ── Negative cases: must NOT over-expand ──────────────────────────────


def test_pr_in_etcetera_preserved() -> None:
    """'и пр.' (etc.) must not become 'и проспект'."""
    result = _expand_abbreviations("холодильники, стиральные машины и пр.")
    assert "проспект" not in result
    # ул., пр-т and others shouldn't appear either
    assert "пр." in result  # original "и пр." preserved


def test_g_before_lowercase_preserved() -> None:
    """'г.' followed by lowercase (e.g., 'г. до н.э.') must not become 'город'."""
    result = _expand_abbreviations("5 г. до нашей эры")
    assert "город" not in result


def test_d_without_digit_preserved() -> None:
    """'д.н.э.' and similar non-address uses of 'д.' must not become 'дом'."""
    result = _expand_abbreviations("до н.э. и д.н.э.")
    assert "дом" not in result


def test_standalone_common_letters_preserved() -> None:
    """Single bare 'г' or 'д' without period/context shouldn't expand."""
    assert _expand_abbreviations("год") == "год"
    assert _expand_abbreviations("день") == "день"


# ── Integration via clean_voice_output ────────────────────────────────


def test_clean_voice_output_expands_and_strips() -> None:
    raw = "**Адрес:** г. Минск, ул. Победителей — д. 57"
    out = clean_voice_output(raw)
    assert "город Минск" in out
    assert "улица Победителей" in out
    assert "дом 57" in out
    assert "**" not in out
    assert "—" not in out


def test_clean_voice_output_handles_multiline_addresses() -> None:
    raw = "Офисы:\n- Минск: пр-т Победителей, 57\n- Могилёв: ул. Комсомольская, 10а"
    out = clean_voice_output(raw)
    assert "проспект Победителей" in out
    assert "улица Комсомольская" in out
    # bullet list was converted to commas
    assert "- " not in out
