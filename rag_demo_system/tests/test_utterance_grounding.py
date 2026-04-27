"""Tests for utterance-level fallback grounding.

Issue 7 (live call 77cfa127, 2026-04-25): user said "Я думаю взять
себе машину" — Qwen3-4B classifier returned intent=RAG with NO subject
extraction. The bare-машина grounding in classifier_schema.py operates
on classifier OUTPUT, so it had nothing to ground. Profile stayed
subj=- and the orchestrator legitimately re-asked for subject on the
next turn.

Fix: utterance-level fallback grounding that runs when the classifier
omits a slot. Returns the most likely slot value from the utterance
text alone, using the same regex / cue rules already in
classifier_schema.py.

Constraints:
  - Conservative: never override an explicit classifier value.
  - Drop on competing categories (don't ground "грузовик" as Легковой).
  - Drop when utterance is ambiguous (no clear cue).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.utterance_grounding import (
    extract_age_years_from_utterance,
    extract_client_type_from_utterance,
    extract_condition_new_from_utterance,
    extract_cost_from_utterance,
    extract_currency_from_utterance,
    extract_prepaid_pct_from_utterance,
    extract_subject_from_utterance,
    extract_term_months_from_utterance,
    extract_type_schedule_from_utterance,
)


# ---------- Bare-car path (live regression) ----------


def test_bare_mashinu_grounds_legkovoy() -> None:
    """The live regression: classifier omitted subject on this exact utterance."""
    assert extract_subject_from_utterance("Я думаю взять себе машину.") == "Легковой автомобиль"


def test_bare_avtomobil_grounds_legkovoy() -> None:
    assert extract_subject_from_utterance("хочу автомобиль") == "Легковой автомобиль"


def test_bare_avto_grounds_legkovoy() -> None:
    assert extract_subject_from_utterance("давай посчитаем авто") == "Легковой автомобиль"


# ---------- Explicit category cues ----------


def test_gruzovik_grounds_gruzovoy() -> None:
    assert extract_subject_from_utterance("хочу грузовик") == "Грузовой автомобиль"


def test_truck_synonyms_ground_gruzovoy() -> None:
    assert extract_subject_from_utterance("нужен тягач") == "Грузовой автомобиль"
    assert extract_subject_from_utterance("ищу самосвал") == "Грузовой автомобиль"


def test_spectekh_grounds_spectekhnika() -> None:
    assert extract_subject_from_utterance("нужна спецтехника") == "Спецтехника"
    assert extract_subject_from_utterance("ищу экскаватор") == "Спецтехника"


def test_oborudovanie_grounds_oborudovanie() -> None:
    assert extract_subject_from_utterance("оборудование для производства") == "Оборудование"


# ---------- Negative cases (must NOT ground) ----------


def test_competing_category_blocks_bare_car() -> None:
    """If utterance has both 'машина' AND 'грузовик', do NOT ground as Легковой."""
    res = extract_subject_from_utterance("у меня грузовая машина")
    # Should ground to Грузовой (specific), NOT Легковой (generic).
    assert res == "Грузовой автомобиль"


def test_no_subject_signal_returns_none() -> None:
    assert extract_subject_from_utterance("привет, как дела") is None


def test_empty_utterance_returns_none() -> None:
    assert extract_subject_from_utterance("") is None


def test_only_name_returns_none() -> None:
    assert extract_subject_from_utterance("Я Никита.") is None


def test_question_about_terms_returns_none() -> None:
    assert extract_subject_from_utterance("какие у вас условия") is None


def test_address_question_returns_none() -> None:
    assert extract_subject_from_utterance("какой адрес в Минске") is None


# ---------- Boundary cases ----------


def test_legkovaya_mashina_grounds_legkovoy() -> None:
    """Explicit Легковая cue takes priority over generic машина."""
    assert extract_subject_from_utterance("легковая машина") == "Легковой автомобиль"


def test_kommercheskiy_subject_does_not_misground() -> None:
    """Bare "коммерческий транспорт" without a specific vehicle cue is
    ambiguous — return None and let the orchestrator ask. Keeps the
    fallback conservative."""
    res = extract_subject_from_utterance("коммерческий транспорт")
    assert res != "Легковой автомобиль"  # must not misground as car
    # Either None or "Прочий транспорт" / "Грузовой автомобиль" is acceptable.


# ---------- age_years fallback (Issue 1, live call 3d3e17b9 2026-04-25) ----------


def test_terse_n_let_grounds_age() -> None:
    """The exact live regression: bot asked age, user replied "4 года"."""
    assert extract_age_years_from_utterance("4 года") == 4


def test_terse_5_let_grounds_age() -> None:
    assert extract_age_years_from_utterance("5 лет") == 5


def test_n_godov_grounds_age() -> None:
    assert extract_age_years_from_utterance("8 годов") == 8


def test_age_in_full_sentence() -> None:
    assert extract_age_years_from_utterance("ну, машине примерно 7 лет") == 7


def test_age_with_punctuation() -> None:
    assert extract_age_years_from_utterance("Ксения, 5 лет машине.") == 5


def test_year_abbrev_grounds_age() -> None:
    assert extract_age_years_from_utterance("3 г") == 3


def test_zero_age_grounds() -> None:
    """Edge: a brand-new vehicle saying "0 лет" — must still ground."""
    assert extract_age_years_from_utterance("0 лет") == 0


def test_age_above_50_rejected() -> None:
    """Calc-eligible ceiling is 50 years — older numbers are STT garbage."""
    assert extract_age_years_from_utterance("100 лет") is None


def test_inverted_form_let_na_n() -> None:
    """Polish E (live call 56c0e2f9 2026-04-27): user thinks aloud
    "на, например, лет на 5". The forward-form regex misses because the
    digit comes AFTER the year-unit word. This case feeds the dispatcher's
    term_months grounding (`has_field_signal` calls extract_age_years_*
    when term_months % 12 == 0). Without this fix, classifier emits
    term_months=60, regex fails, dispatcher drops the patch, bot loops
    on "Подскажите срок"."""
    assert extract_age_years_from_utterance("лет на 5") == 5
    assert extract_age_years_from_utterance("на, например, лет на 5,") == 5
    assert extract_age_years_from_utterance("года на 7") == 7
    assert extract_age_years_from_utterance("год на 1") == 1


def test_no_year_unit_returns_none() -> None:
    """Bare digits without лет/года must NOT be grounded — could be cost."""
    assert extract_age_years_from_utterance("100 тысяч") is None
    assert extract_age_years_from_utterance("60") is None


# ---------- Russian word-numerals (Bug 17, live call 9ec121bc 2026-04-25) ---
# Bot asks "Сколько лет вашему транспорту?", user answers "Два года" (with the
# word, not the digit). STT transcribes the word verbatim. Without word-numeral
# support, the regex can't extract → patch never lands → profile stays
# age_years=None → LLM hallucinates re-asking already-captured fields.


def test_word_numeral_dva_goda() -> None:
    """The exact live regression: bot asked age, user said "Два года"."""
    assert extract_age_years_from_utterance("Два года.") == 2


def test_word_numeral_tri_goda() -> None:
    assert extract_age_years_from_utterance("Три года") == 3


def test_word_numeral_pyat_let() -> None:
    assert extract_age_years_from_utterance("Пять лет") == 5


def test_word_numeral_odin_god() -> None:
    assert extract_age_years_from_utterance("Один год") == 1


def test_word_numeral_desyat_let() -> None:
    assert extract_age_years_from_utterance("Десять лет") == 10


def test_word_numeral_lowercase() -> None:
    assert extract_age_years_from_utterance("два года") == 2


def test_word_numeral_in_sentence() -> None:
    assert extract_age_years_from_utterance("Ну, наверное, три года будет.") == 3


def test_word_numeral_without_year_unit_returns_none() -> None:
    """A bare numeral word with no year-unit could be anything — don't ground."""
    assert extract_age_years_from_utterance("два") is None


def test_term_in_months_does_not_misground_as_age() -> None:
    """User answering term question in months should not poison age."""
    assert extract_age_years_from_utterance("60 месяцев") is None


def test_empty_utterance_returns_none() -> None:
    assert extract_age_years_from_utterance("") is None
    assert extract_age_years_from_utterance("  ") is None


def test_term_in_years_DOES_match_caller_must_gate() -> None:
    """Documents that the regex DOES match year-form term answers
    ("на 7 лет"). Caller is responsible for gating on
    `condition_new == 0 AND age_years is None` so this fallback never
    runs while the bot is asking about term. The function itself stays
    simple and value-honest."""
    assert extract_age_years_from_utterance("на 7 лет") == 7


# ---------- client_type fallback ----------


def test_phys_grounds_client_type() -> None:
    assert extract_client_type_from_utterance("физлицо") == "Физическое лицо"
    assert extract_client_type_from_utterance("Физическое.") == "Физическое лицо"
    assert extract_client_type_from_utterance("я физик") == "Физическое лицо"


def test_legal_grounds_client_type() -> None:
    assert extract_client_type_from_utterance("юрлицо") == "Юридическое лицо"
    assert extract_client_type_from_utterance("Юридическое.") == "Юридическое лицо"
    assert extract_client_type_from_utterance("ИП") == "Юридическое лицо"
    assert extract_client_type_from_utterance("ООО") == "Юридическое лицо"
    assert extract_client_type_from_utterance("у меня малый бизнес") == "Юридическое лицо"


def test_ambiguous_client_type_returns_none() -> None:
    assert extract_client_type_from_utterance("физическое или юридическое") is None


def test_no_client_type_signal_returns_none() -> None:
    assert extract_client_type_from_utterance("здравствуйте") is None
    assert extract_client_type_from_utterance("") is None


# ---------- condition_new fallback ----------


def test_used_grounds_zero() -> None:
    assert extract_condition_new_from_utterance("подержанный") == 0
    assert extract_condition_new_from_utterance("Поддержанная.") == 0
    assert extract_condition_new_from_utterance("б/у") == 0
    assert extract_condition_new_from_utterance("бэу") == 0


def test_new_grounds_one() -> None:
    assert extract_condition_new_from_utterance("новый") == 1
    assert extract_condition_new_from_utterance("Новая.") == 1


def test_zero_mileage_grounds_new() -> None:
    """Without пробега / нулевой пробег = new car phrasing."""
    assert extract_condition_new_from_utterance("без пробега") == 1


def test_contradictory_condition_returns_none() -> None:
    assert extract_condition_new_from_utterance("новая или подержанная") is None


def test_no_condition_signal_returns_none() -> None:
    assert extract_condition_new_from_utterance("BMW") is None
    assert extract_condition_new_from_utterance("") is None


# ---------- currency fallback ----------


def test_dollars_grounds_usd() -> None:
    assert extract_currency_from_utterance("в долларах") == "USD"
    assert extract_currency_from_utterance("USD") == "USD"


def test_byn_rubles_grounds_byn() -> None:
    assert extract_currency_from_utterance("в рублях") == "BYN"
    assert extract_currency_from_utterance("белорусских рублей") == "BYN"


def test_eur_grounds_eur() -> None:
    assert extract_currency_from_utterance("в евро") == "EUR"


def test_russian_rubles_grounds_rub() -> None:
    assert extract_currency_from_utterance("российских рублей") == "RUB"


def test_ambiguous_currency_returns_none() -> None:
    """Multiple currency cues = bot question echoed back, not a real choice."""
    assert extract_currency_from_utterance("в рублях или долларах") is None


def test_no_currency_signal_returns_none() -> None:
    assert extract_currency_from_utterance("сто тысяч") is None


# ---------- term_months fallback ----------


def test_n_months_grounds_term() -> None:
    assert extract_term_months_from_utterance("60 месяцев") == 60
    assert extract_term_months_from_utterance("36 мес.") == 36


def test_n_years_grounds_term_in_months() -> None:
    assert extract_term_months_from_utterance("на 5 лет") == 60
    assert extract_term_months_from_utterance("7 лет") == 84


def test_term_out_of_range_rejected() -> None:
    """11 months and 100 months are outside the 12-84 calc-eligible band."""
    assert extract_term_months_from_utterance("11 месяцев") is None
    assert extract_term_months_from_utterance("100 месяцев") is None


def test_no_term_signal_returns_none() -> None:
    assert extract_term_months_from_utterance("привет") is None
    assert extract_term_months_from_utterance("") is None


# ---------- prepaid_pct fallback ----------


def test_n_percent_grounds_prepaid() -> None:
    assert extract_prepaid_pct_from_utterance("20 процентов") == 20.0
    assert extract_prepaid_pct_from_utterance("аванс 30 %") == 30.0


def test_zero_prepaid_phrases() -> None:
    assert extract_prepaid_pct_from_utterance("без аванса") == 0.0
    assert extract_prepaid_pct_from_utterance("нулевой аванс") == 0.0


def test_no_prepaid_signal_returns_none() -> None:
    assert extract_prepaid_pct_from_utterance("BMW") is None
    assert extract_prepaid_pct_from_utterance("") is None


def test_prepaid_pct_above_100_rejected() -> None:
    assert extract_prepaid_pct_from_utterance("200 процентов") is None


# ---------- type_schedule fallback ----------


def test_annuity_grounds_zero_string() -> None:
    """Annuity = "0" string (matches profile schema Literal["0", "1"])."""
    assert extract_type_schedule_from_utterance("аннуитет") == "0"
    assert extract_type_schedule_from_utterance("Аннуитетный график.") == "0"


def test_linear_grounds_one_string() -> None:
    assert extract_type_schedule_from_utterance("линейный") == "1"
    assert extract_type_schedule_from_utterance("дифференцированный") == "1"


def test_ambiguous_schedule_returns_none() -> None:
    assert extract_type_schedule_from_utterance("аннуитет или линейный") is None


def test_no_schedule_signal_returns_none() -> None:
    assert extract_type_schedule_from_utterance("BMW") is None


# ---------- cost fallback (Issue 1, live call 5fa0bb3d 2026-04-26) ----------
# Classifier omitted cost on "Сто десять тысяч долларов и поддержанный".
# Digit-form is captured reliably by the classifier; the regression-prone
# case is the fully-spelled-out Russian numeral. Reuses parse_ru_number from
# numeric_words_ru so percent-only utterances and digit-only utterances are
# correctly rejected (those go through the classifier path).


def test_ru_numeral_cost_grounds() -> None:
    """The live regression case: spelled-out cost in RU numerals."""
    assert (
        extract_cost_from_utterance("Сто десять тысяч долларов и поддержанный")
        == 110000
    )


def test_ru_numeral_cost_simple_thousand() -> None:
    assert extract_cost_from_utterance("двадцать тысяч долларов") == 20000


def test_ru_numeral_cost_million() -> None:
    assert extract_cost_from_utterance("один миллион рублей") == 1000000


def test_digit_form_cost_grounds() -> None:
    """Digit form supported as a sanity check (also typically captured by
    the classifier, but the fallback should still work end-to-end)."""
    assert extract_cost_from_utterance("110000 долларов") == 110000


def test_digit_form_with_grouping_grounds() -> None:
    assert extract_cost_from_utterance("150 000 рублей") == 150000


def test_digit_with_thousand_word_grounds() -> None:
    """'80 тысяч' shape — N digits + scale word."""
    assert extract_cost_from_utterance("80 тысяч долларов") == 80000


def test_no_cost_signal_returns_none() -> None:
    assert extract_cost_from_utterance("привет, как дела") is None
    assert extract_cost_from_utterance("") is None


def test_percent_only_utterance_returns_none() -> None:
    """parse_ru_number drops percent contexts, so a pure-prepaid-pct
    utterance must NOT ground as cost."""
    assert extract_cost_from_utterance("двадцать процентов") is None


def test_age_years_does_not_ground_as_cost() -> None:
    """A small numeric like '5 лет' must not become cost=5."""
    assert extract_cost_from_utterance("5 лет") is None


def test_term_months_does_not_ground_as_cost() -> None:
    """'60 месяцев' must not ground as cost=60."""
    assert extract_cost_from_utterance("60 месяцев") is None


def test_below_min_cost_rejected() -> None:
    """Cost values below the realistic leasing range are rejected — these
    are almost always term/age/prepaid leakage."""
    assert extract_cost_from_utterance("100 долларов") is None
    assert extract_type_schedule_from_utterance("") is None
