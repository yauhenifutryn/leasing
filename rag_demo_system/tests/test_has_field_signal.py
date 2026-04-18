"""Fix 31: `has_field_signal` — stricter check that a classifier-extracted
hint value actually appears (category cue OR literal digits) in the user's
utterance. Used by the orchestrator to reject derived / echoed values that
would otherwise land in the multi-field change-confirm prompt.

Scenario (production 2026-04-18, session ce186779):
  User: "Ой, давай всё-таки ещё срок на 48 месяцев."
  Bot:  "Меняю срок на 48 и сумму аванса на 16000, остальное оставляю."
                               ^^^^^^^^^^^^^^^^^^^^  ← not in utterance
"""

from __future__ import annotations

from backend.profile_hygiene import has_field_signal


# ----- Numeric fields require literal digits -----

def test_numeric_term_digits_in_utterance():
    assert has_field_signal("term_months", 48, "срок на 48 месяцев") is True
    assert has_field_signal("term_months", 48, "на сорок восемь месяцев") is False  # digits absent


def test_numeric_cost_digits_with_spaces():
    assert has_field_signal("cost", 150000, "150 000 рублей") is True
    # Post Fix 34: "80 тысяч" is now recognised as 80000 via the Russian
    # multiplier heuristic. This test used to assert False on the strict
    # literal-digits rule; Fix 34 deliberately relaxes that for cost /
    # prepaid_amount so live speech ("восемьдесят тысяч" style) works.
    assert has_field_signal("cost", 80000, "за 80 тысяч") is True


def test_numeric_prepaid_amount_rejected_when_not_in_utterance():
    """THE Fix 31 case: user says "срок на 48", classifier emits
    prepaid_amount=16000 (echoed from prior calc result). Must be rejected.
    """
    utterance = "Ой, давай всё-таки ещё срок на 48 месяцев."
    assert has_field_signal("prepaid_amount", 16000, utterance) is False


def test_numeric_prepaid_pct_accepts_when_digit_present():
    assert has_field_signal("prepaid_pct", 20, "аванс 20 процентов") is True


def test_numeric_rejects_none_and_empty():
    assert has_field_signal("cost", None, "anything") is False
    assert has_field_signal("cost", "", "anything") is False


# ----- Enum fields use cue check -----

def test_client_type_uses_cue_check():
    assert has_field_signal("client_type", "Юридическое лицо", "мы ООО") is True
    assert has_field_signal("client_type", "Физическое лицо", "хочу машину") is False


def test_subject_uses_cue_check():
    assert has_field_signal("subject", "Легковой автомобиль", "легковой новый") is True
    assert has_field_signal("subject", "Грузовой автомобиль", "80 000 рублей") is False


def test_currency_cue():
    assert has_field_signal("currency", "BYN", "в рублях") is True
    assert has_field_signal("currency", "USD", "в долларах") is True
    assert has_field_signal("currency", "BYN", "80 000") is False


def test_condition_new_cue():
    assert has_field_signal("condition_new", 1, "новый автомобиль") is True
    assert has_field_signal("condition_new", 0, "б/у") is True
    assert has_field_signal("condition_new", 1, "за 80 тысяч") is False


def test_type_schedule_cue():
    assert has_field_signal("type_schedule", "0", "аннуитетный график") is True
    assert has_field_signal("type_schedule", "1", "линейный график") is True
    assert has_field_signal("type_schedule", "0", "срок 48") is False
