from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_hygiene import filter_patches, has_field_signal


def test_drops_patches_from_noise_short_utterance():
    # "ОООООО" is 1 non-digit token of questionable content
    assert filter_patches({"client_type": "Юридическое лицо"}, "ОООООО") == {}


def test_drops_bot_name_as_user_name():
    patches = {"name": "Ксения"}
    out = filter_patches(patches, "Ксения, ладно, а что такое нагрузка?", bot_name="Ксения")
    assert "name" not in out


def test_drops_bot_name_with_patronymic():
    # Classifier sometimes captures "Ксения Николаевна" when user/TTS uses
    # formal address. First-token match against bot_name must reject it.
    out = filter_patches({"name": "Ксения Николаевна"}, "Ксения Николаевна, подскажите", bot_name="Ксения")
    assert "name" not in out


def test_drops_bot_name_lowercase_with_patronymic():
    out = filter_patches({"name": "ксения ивановна"}, "ксения ивановна спасибо", bot_name="Ксения")
    assert "name" not in out


def test_keeps_real_user_name_matching_different_first_token():
    out = filter_patches({"name": "Николай"}, "меня зовут Николай", bot_name="Ксения")
    assert out.get("name") == "Николай"


# ---------- Bug 20 (live call 45247512 2026-04-25) ----------
# Classifier hallucinated `name="не указано"` when user asked "А как меня
# звали?". The patch landed, then the stale-name guard later rejected the
# real "Никита" because a "name" was already on file. Filter must reject
# meta-phrases that are clearly not human names.


def test_drops_ne_ukazano_as_name():
    """The exact live regression: classifier emits name='не указано'."""
    out = filter_patches(
        {"name": "не указано"},
        "Понял, спасибо. А как меня звали?",
        bot_name="Ксения",
    )
    assert "name" not in out


def test_drops_neizvestno_as_name():
    out = filter_patches({"name": "неизвестно"}, "не помнишь меня?", bot_name="Ксения")
    assert "name" not in out


def test_drops_ne_skazano_as_name():
    out = filter_patches({"name": "не сказано"}, "ты не запомнила", bot_name="Ксения")
    assert "name" not in out


def test_drops_anonim_as_name():
    out = filter_patches({"name": "аноним"}, "просто клиент", bot_name="Ксения")
    assert "name" not in out


def test_drops_polzovatel_as_name():
    out = filter_patches({"name": "пользователь"}, "клиент", bot_name="Ксения")
    assert "name" not in out


def test_drops_klient_as_name():
    out = filter_patches({"name": "клиент"}, "звонящий клиент", bot_name="Ксения")
    assert "name" not in out


def test_keeps_real_name_after_blacklist_filter():
    """Regression check: the blacklist must not reject real names."""
    out = filter_patches({"name": "Никита"}, "я Никита", bot_name="Ксения")
    assert out.get("name") == "Никита"


def test_blacklist_case_insensitive():
    """Classifier might emit any casing."""
    out = filter_patches({"name": "Не Указано"}, "...", bot_name="Ксения")
    assert "name" not in out


def test_ipeshnik_collapsed_to_yur_litso():
    # Fix 41a: ИП / ипэшник / самозанятый / микробизнес all collapse to
    # "Юридическое лицо" so readback says "юр.лицо" consistent with the
    # API mapping (Mikro Leasing accepts only Физ/Юр).
    out = filter_patches({"client_type": "ИП"}, "я ипэшник", bot_name="Ксения")
    assert out.get("client_type") == "Юридическое лицо"


def test_currency_bare_rubli_in_belarus_is_byn():
    out = filter_patches({"currency": "RUB"}, "ну давай в рублях")
    assert out.get("currency") == "BYN"


def test_currency_russian_rubles_is_rub():
    out = filter_patches({"currency": "RUB"}, "в российских рублях")
    assert out.get("currency") == "RUB"


def test_prepaid_out_of_range_is_forwarded():
    # Fix 39: hygiene no longer silently drops OOR values. The calculator's
    # validate_calc_inputs raises on them so the bot can tell the client the
    # exact allowed range.
    out = filter_patches({"prepaid_pct": 50}, "аванс пятьдесят процентов")
    assert out.get("prepaid_pct") == 50


def test_prepaid_in_range_is_kept():
    out = filter_patches({"prepaid_pct": 20}, "аванс двадцать процентов")
    assert out.get("prepaid_pct") == 20


def test_term_out_of_range_is_forwarded():
    # Fix 39: term=300 passes through to the calculator where it is rejected
    # as param_out_of_range.
    out = filter_patches({"term_months": 300}, "триста месяцев")
    assert out.get("term_months") == 300


def test_term_non_numeric_is_dropped():
    out = filter_patches({"term_months": "много"}, "много месяцев")
    assert "term_months" not in out


def test_prepaid_non_numeric_is_dropped():
    out = filter_patches({"prepaid_pct": "чуть-чуть"}, "чуть-чуть")
    assert "prepaid_pct" not in out


def test_single_word_annuity_passes():
    patches = {"type_schedule": 0}
    result = filter_patches(patches, "Аннуитет")
    assert result == {"type_schedule": 0}


def test_single_word_linear_passes():
    patches = {"type_schedule": 1}
    result = filter_patches(patches, "Линейный")
    assert result == {"type_schedule": 1}


def test_single_word_physical_passes():
    patches = {"client_type": "Физическое лицо"}
    result = filter_patches(patches, "Физлицо")
    assert result == {"client_type": "Физическое лицо"}


def test_single_word_ip_collapses_to_yur_litso():
    # Fix 41a: single-word "ИП" still passes the noise filter (enum slot-fill
    # whitelist) but the normalized result is "Юридическое лицо".
    patches = {"client_type": "ИП"}
    result = filter_patches(patches, "ИП")
    assert result == {"client_type": "Юридическое лицо"}


def test_single_word_new_passes():
    patches = {"condition_new": 1}
    result = filter_patches(patches, "Новый")
    assert result == {"condition_new": 1}


def test_single_word_noise_still_dropped():
    # "э" — not in enum whitelist — existing behavior preserved
    patches = {"name": "Никита"}
    result = filter_patches(patches, "э")
    assert result == {}


def test_single_word_agree_still_dropped():
    # "ага" — 1 token, not enum, must still drop
    patches = {"client_type": "Физическое лицо"}
    result = filter_patches(patches, "ага")
    assert result == {}


def test_multi_word_unchanged():
    # Existing multi-word utterances must still pass through
    patches = {"prepaid_pct": 20}
    result = filter_patches(patches, "аванс двадцать процентов")
    assert result == {"prepaid_pct": 20}


def test_numeric_cost_with_currency_passes():
    # "49 500 рублей" — 1 non-digit token, classifier extracts cost+currency.
    # Post Fix 27: `subject` is dropped if the utterance has no subject cue
    # ("49 500 рублей" has neither машина/авто/грузовой/etc, so any subject
    # patch was inferred, not grounded). cost + currency are unaffected —
    # they're numeric-answer keys and ride through the noise filter.
    patches = {"cost": 49500, "currency": "BYN", "subject": "Легковой автомобиль"}
    result = filter_patches(patches, "49 500 рублей.")
    assert result.get("cost") == 49500
    assert result.get("currency") == "BYN"
    assert "subject" not in result, result  # Fix 27 regression guard


def test_numeric_term_with_unit_passes():
    # "36 месяцев" — 1 non-digit token, classifier extracts term_months
    patches = {"term_months": 36}
    result = filter_patches(patches, "36 месяцев")
    assert result.get("term_months") == 36


# Bug S (live call 4e522fb5 2026-04-26): years-as-term grounding.
# Both word-order ("три года срок" / "срок три года") and word-numeral
# ("на пять лет") forms must ground term_months. These exercise
# has_field_signal directly (the orchestrator's grounding gate);
# filter_patches doesn't run grounding for term_months.

def test_term_years_word_form_suffix_срок_grounds():
    # Live regression: "Ну давай где-то три года срок и аванс 38 процентов."
    assert has_field_signal(
        "term_months", 36,
        "ну давай где-то три года срок и аванс 38 процентов",
    ) is True


def test_term_years_word_form_prefix_на_grounds():
    assert has_field_signal("term_months", 60, "на пять лет") is True


def test_term_years_digit_form_suffix_срок_grounds():
    assert has_field_signal("term_months", 24, "2 года срок") is True


def test_term_years_digit_form_prefix_срок_grounds():
    # Existing prefix path stays green.
    assert has_field_signal("term_months", 84, "срок 7 лет") is True


def test_term_bare_word_years_without_cue_dropped():
    # Bug Q regression guard: bare "Два года" (age answer) must NOT
    # ground term_months even if the classifier mistakenly emits it.
    assert has_field_signal("term_months", 24, "два года") is False


def test_term_bare_digit_years_without_cue_dropped():
    assert has_field_signal("term_months", 24, "2 года") is False


def test_numeric_prepaid_percent_passes():
    # "10 процентов" — 1 non-digit token
    patches = {"prepaid_pct": 10}
    result = filter_patches(patches, "10 процентов.")
    assert result.get("prepaid_pct") == 10


def test_numeric_amount_passes():
    # "14 тысяч рублей" — 2 non-digit tokens, already fine; but verify
    patches = {"prepaid_amount": 14000, "currency": "BYN"}
    result = filter_patches(patches, "14 тысяч рублей")
    assert result.get("prepaid_amount") == 14000


def test_numeric_cost_standalone_number_passes():
    # Just "100000" — 0 non-digit tokens, but classifier extracted cost
    patches = {"cost": 100000}
    result = filter_patches(patches, "100000")
    assert result.get("cost") == 100000


def test_noise_utterance_still_dropped_despite_name():
    # "э" with patch {name: ...} — no enum match, no numeric field — drop
    patches = {"name": "Э"}
    result = filter_patches(patches, "э")
    assert result == {}


def test_numeric_out_of_range_now_forwarded():
    # Fix 39: hygiene forwards OOR numeric values. The calculator layer
    # (validate_calc_inputs) is responsible for rejecting them with a
    # user-facing range message.
    patches = {"prepaid_pct": 99}
    result = filter_patches(patches, "99 процентов")
    assert result.get("prepaid_pct") == 99
