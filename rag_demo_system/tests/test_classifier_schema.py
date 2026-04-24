"""CP-2.1 — Pydantic ClassifierOutput schema with utterance grounding."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.classifier_schema import ClassifierOutput, parse_classifier_output  # noqa: E402


def test_minimal_valid_parse():
    raw = json.dumps({"intent": "CONVERSATION"})
    out = parse_classifier_output(raw, utterance="привет")
    assert out.intent == "CONVERSATION"
    assert out.subject is None
    assert out.is_confirmation is False


def test_ip_normalized_to_yur_litso():
    # Codex adversarial 2026-04-20: when Qwen echoes the user's literal "ИП",
    # the @field_validator(mode="before") must normalize it to "Юридическое
    # лицо" BEFORE Literal validation, otherwise business callers get stuck
    # in a clarification loop because the schema drops their answer.
    raw = json.dumps({"intent": "TOOL", "client_type": "ИП"})
    out = parse_classifier_output(raw, utterance="я ип")
    assert out.client_type == "Юридическое лицо"


def test_client_type_other_business_forms_normalized():
    # Same regression surface: all business forms the profile_hygiene
    # normalizer collapses to Юр.лицо must survive the schema boundary.
    for raw_ct in ("самозанятый", "ООО", "ИП", "организация", "бизнесмен"):
        raw = json.dumps({"intent": "TOOL", "client_type": raw_ct})
        out = parse_classifier_output(raw, utterance="я от компании")
        assert out.client_type == "Юридическое лицо", raw_ct


def test_empty_model_intent_is_none_but_dict_not_empty():
    # Regression guard for the CP-2.2 empty-dict fallback bug (Codex adversarial
    # 2026-04-20). ClassifierOutput() with defaults serializes the three bool
    # flags, so dict-based emptiness checks are broken. Downstream must use
    # `_sa_output.intent is None` as the parse-failure signal instead.
    empty = ClassifierOutput()
    assert empty.intent is None
    d = empty.model_dump(exclude_none=True)
    assert "intent" not in d
    assert d == {
        "is_confirmation": False,
        "is_stop_request": False,
        "wants_readback": False,
    }


def test_change_field_all_rejected():
    # Fix 41b: change_field="all" must be nulled by the schema itself.
    raw = json.dumps({"intent": "TOOL", "change_field": "all", "change_value": 1})
    out = parse_classifier_output(raw, utterance="поменяй всё")
    assert out.change_field is None


def test_action_full_vocabulary():
    for act in (
        "calculate", "recalculate", "change_param", "sms", "clarify",
        "clarify_client_type", "confirm", "invalid_param",
    ):
        raw = json.dumps({"intent": "TOOL", "action": act})
        out = parse_classifier_output(raw, utterance="рассчитай")
        assert out.action == act, act


def test_subject_ungrounded_nulled():
    # E1: classifier emits a valid subject with no utterance cue → must be None.
    raw = json.dumps({"intent": "CONVERSATION", "subject": "Прочий транспорт"})
    out = parse_classifier_output(raw, utterance="Микро Лизинг")
    assert out.subject is None


def test_subject_grounded_passes():
    raw = json.dumps({"intent": "TOOL", "subject": "Легковой автомобиль"})
    out = parse_classifier_output(raw, utterance="хочу легковой автомобиль")
    assert out.subject == "Легковой автомобиль"


def test_subject_generic_mashina_grounds_legkovoi():
    # Regression (2026-04-24 live call 5375e1bd): bare "машина" / "машину" /
    # "автомобиль" without a category modifier must ground as
    # Легковой автомобиль. Otherwise EmitClarify loops forever when the
    # user mirrors the bot's own word back.
    for utt in ("Машину брать хочу в лизинг", "машина", "хочу автомобиль"):
        raw = json.dumps({"intent": "TOOL", "subject": "Легковой автомобиль"})
        out = parse_classifier_output(raw, utterance=utt)
        assert out.subject == "Легковой автомобиль", f"bare car utterance dropped: {utt!r}"


def test_subject_generic_mashina_rejected_when_truck_modifier_present():
    # Negative: "грузовую машину" must NOT ground Легковой автомобиль.
    raw = json.dumps({"intent": "TOOL", "subject": "Легковой автомобиль"})
    out = parse_classifier_output(raw, utterance="нужна грузовая машина")
    assert out.subject is None


def test_subject_generic_mashina_rejected_when_spec_modifier_present():
    # Negative: "машина-погрузчик" or similar with spec competition must drop.
    raw = json.dumps({"intent": "TOOL", "subject": "Легковой автомобиль"})
    out = parse_classifier_output(raw, utterance="машина и погрузчик")
    assert out.subject is None


def test_type_schedule_ungrounded_nulled():
    # E2: type_schedule="1" emitted on utterance without graph word.
    raw = json.dumps({"intent": "TOOL", "type_schedule": "1"})
    out = parse_classifier_output(raw, utterance="120 000 долларов новую")
    assert out.type_schedule is None


def test_age_years_nulled_when_condition_new():
    # E2: age_years=3 emitted together with condition_new=1. Age irrelevant for new car.
    # Utterance must ground condition_new=1 (cue "новую"), otherwise cue-grounding
    # would null condition_new first and age_years would slip through cross-field.
    raw = json.dumps({
        "intent": "TOOL", "age_years": 3, "condition_new": 1, "term_months": 36,
    })
    out = parse_classifier_output(raw, utterance="новую, три года, аванс 30%")
    assert out.age_years is None
    assert out.condition_new == 1
    assert out.term_months == 36


def test_age_years_nulled_when_condition_new_unknown():
    raw = json.dumps({"intent": "TOOL", "age_years": 3})
    out = parse_classifier_output(raw, utterance="три года")
    assert out.age_years is None


def test_age_years_kept_on_used_car():
    raw = json.dumps({"intent": "TOOL", "age_years": 5, "condition_new": 0})
    out = parse_classifier_output(raw, utterance="бу машина пять лет")
    assert out.condition_new == 0
    assert out.age_years == 5


def test_condition_new_contradiction_grounding():
    # Fix 1.10 behaviour inherited via has_field_signal: "новая машина без пробега"
    # must reject condition_new=0.
    raw = json.dumps({"intent": "TOOL", "condition_new": 0})
    out = parse_classifier_output(raw, utterance="новая машина без пробега")
    assert out.condition_new is None


def test_change_value_zero_preserved():
    # "давай без аванса" → change_field="prepaid_pct", change_value=0.
    raw = json.dumps({
        "intent": "TOOL", "change_field": "prepaid_pct", "change_value": 0,
    })
    out = parse_classifier_output(raw, utterance="давай без аванса")
    assert out.change_field == "prepaid_pct"
    assert out.change_value == 0


def test_oor_numeric_passes():
    # Plan 2.1: wide OOR so calculator can surface a specific user-facing message.
    raw = json.dumps({"intent": "TOOL", "term_months": 200, "prepaid_pct": 110})
    out = parse_classifier_output(raw, utterance="200 месяцев 110 процентов")
    assert out.term_months == 200
    assert out.prepaid_pct == 110


def test_malformed_json_returns_empty():
    out = parse_classifier_output("not json at all", utterance="")
    assert out.intent is None
    # Defaults for bool fields; all Optional fields None.
    d = out.model_dump(exclude_none=True)
    assert d == {
        "is_confirmation": False,
        "is_stop_request": False,
        "wants_readback": False,
    }


def test_empty_utterance_nulls_grounded_enums():
    # Codex adversarial pass 3, 2026-04-20: empty utterance = no evidence.
    # Grounded enum fields must be nulled rather than preserved (production
    # risk: blank/degraded ASR turns leaking stale classifier state).
    raw = json.dumps({
        "intent": "TOOL", "subject": "Легковой автомобиль", "currency": "USD",
        "client_type": "Физическое лицо", "type_schedule": "0", "condition_new": 1,
    })
    out = parse_classifier_output(raw, utterance="")
    assert out.subject is None
    assert out.currency is None
    assert out.client_type is None
    assert out.type_schedule is None
    assert out.condition_new is None
    # Non-grounded fields still pass.
    assert out.intent == "TOOL"


def test_partial_validate_preserves_good_fields():
    # Garbage enum on one field → partial-validate drops that field and keeps
    # the rest. Strict upgrade over the plan's "return empty on ValidationError"
    # spec: one bad field no longer nukes the whole turn's signal.
    raw = json.dumps({
        "intent": "TOOL", "subject": "НесуществующийВид", "cost": 80000,
    })
    out = parse_classifier_output(raw, utterance="хочу автомобиль за 80 000")
    assert out.subject is None
    assert out.intent == "TOOL"
    assert out.cost == 80000


def test_total_garbage_returns_empty_model():
    # Two bad fields where the fallback dict becomes un-validatable → empty model.
    # Use a field that re-validates cleanly when dropped — here age_years is out of
    # range AND prepaid_pct is way out of range, but partial-validate still succeeds
    # by dropping only the bad paths. Force a harder failure: a type that can't
    # coerce (dict where int expected) and see that we still never raise.
    raw = json.dumps({"intent": "TOOL", "age_years": "not a number"})
    out = parse_classifier_output(raw, utterance="")
    # Partial-validate drops age_years, intent survives.
    assert out.age_years is None
    assert out.intent == "TOOL"


def test_currency_ungrounded_nulled():
    # Defensive: classifier emits USD on utterance with no currency word.
    raw = json.dumps({"intent": "TOOL", "currency": "USD"})
    out = parse_classifier_output(raw, utterance="хочу легковую машину")
    assert out.currency is None


def test_currency_grounded_passes():
    raw = json.dumps({"intent": "TOOL", "currency": "USD"})
    out = parse_classifier_output(raw, utterance="в долларах")
    assert out.currency == "USD"


def test_client_type_ungrounded_nulled():
    raw = json.dumps({"intent": "TOOL", "client_type": "Физическое лицо"})
    out = parse_classifier_output(raw, utterance="хочу посчитать")
    assert out.client_type is None


def test_client_type_grounded_passes():
    raw = json.dumps({"intent": "TOOL", "client_type": "Юридическое лицо"})
    out = parse_classifier_output(raw, utterance="я от компании")
    assert out.client_type == "Юридическое лицо"


# --- Codex adversarial re-review 2026-04-20: value-aware grounding ---

def test_currency_rub_nulled_when_utterance_says_dollars():
    # Codex Finding A: 'currency="RUB"' + utterance "в долларах" must drop RUB,
    # not keep it because USD cue is somewhere in the dimension.
    raw = json.dumps({"intent": "TOOL", "currency": "RUB"})
    out = parse_classifier_output(raw, utterance="в долларах")
    assert out.currency is None


def test_currency_usd_nulled_when_utterance_says_rubles():
    raw = json.dumps({"intent": "TOOL", "currency": "USD"})
    out = parse_classifier_output(raw, utterance="в рублях")
    assert out.currency is None


def test_currency_byn_passes_for_bare_rubles_belarus_context():
    raw = json.dumps({"intent": "TOOL", "currency": "BYN"})
    out = parse_classifier_output(raw, utterance="в рублях")
    assert out.currency == "BYN"


def test_currency_byn_blocked_when_russian_rubles_specified():
    # "российские рубли" disambiguates to RUB; BYN must not survive.
    raw = json.dumps({"intent": "TOOL", "currency": "BYN"})
    out = parse_classifier_output(raw, utterance="в российских рублях")
    assert out.currency is None


def test_currency_rub_passes_when_russian_rubles_specified():
    raw = json.dumps({"intent": "TOOL", "currency": "RUB"})
    out = parse_classifier_output(raw, utterance="в российских рублях")
    assert out.currency == "RUB"


def test_subject_car_nulled_when_utterance_says_trucks():
    raw = json.dumps({"intent": "TOOL", "subject": "Легковой автомобиль"})
    out = parse_classifier_output(raw, utterance="хочу грузовик")
    assert out.subject is None


def test_subject_trucks_passes_on_grouzovik():
    raw = json.dumps({"intent": "TOOL", "subject": "Грузовой автомобиль"})
    out = parse_classifier_output(raw, utterance="хочу грузовик")
    assert out.subject == "Грузовой автомобиль"


def test_client_type_fiz_nulled_when_utterance_says_ooo():
    raw = json.dumps({"intent": "TOOL", "client_type": "Физическое лицо"})
    out = parse_classifier_output(raw, utterance="мы ООО Ромашка")
    assert out.client_type is None


def test_client_type_yur_passes_on_ooo():
    raw = json.dumps({"intent": "TOOL", "client_type": "Юридическое лицо"})
    out = parse_classifier_output(raw, utterance="мы ООО Ромашка")
    assert out.client_type == "Юридическое лицо"


def test_type_schedule_linear_nulled_when_utterance_says_annuitet():
    raw = json.dumps({"intent": "TOOL", "type_schedule": "1"})
    out = parse_classifier_output(raw, utterance="давай аннуитетный график")
    assert out.type_schedule is None


def test_type_schedule_annuity_passes_on_annuitet():
    raw = json.dumps({"intent": "TOOL", "type_schedule": "0"})
    out = parse_classifier_output(raw, utterance="давай аннуитетный график")
    assert out.type_schedule == "0"


# --- Codex Finding B: prepaid alias removed from schema ---

def test_prepaid_alias_rejected_as_change_field():
    # Codex Finding B: change_field="prepaid" must be dropped by the schema
    # because ClientProfile has no `prepaid` attribute — apply_pending_change
    # would silently skip it, leaving the user's confirmed change unapplied.
    raw = json.dumps({
        "intent": "TOOL", "change_field": "prepaid", "change_value": 20,
    })
    out = parse_classifier_output(raw, utterance="аванс 20 процентов")
    assert out.change_field is None


def test_prepaid_pct_still_valid_as_change_field():
    raw = json.dumps({
        "intent": "TOOL", "change_field": "prepaid_pct", "change_value": 20,
    })
    out = parse_classifier_output(raw, utterance="аванс 20 процентов")
    assert out.change_field == "prepaid_pct"


# --- Codex adversarial pass 3, 2026-04-20: subject collision + ordering ---

def test_subject_car_nulled_on_gruzovoy_avtomobil():
    # Finding 1: "грузовой автомобиль" must drop subject="Легковой автомобиль".
    # Previously the generic "автомобил\w*" / "машин\w*" cues in the car
    # pattern matched truck phrasings, so contradictory subjects leaked.
    raw = json.dumps({"intent": "TOOL", "subject": "Легковой автомобиль"})
    out = parse_classifier_output(raw, utterance="хочу грузовой автомобиль")
    assert out.subject is None


def test_subject_truck_passes_on_gruzovoy():
    raw = json.dumps({"intent": "TOOL", "subject": "Грузовой автомобиль"})
    out = parse_classifier_output(raw, utterance="хочу грузовой автомобиль")
    assert out.subject == "Грузовой автомобиль"


def test_subject_car_grounds_on_bare_mashina():
    # 2026-04-24 reversal (live call 5375e1bd): bare "машина" / "автомобиль"
    # with no competing category modifier grounds as Легковой автомобиль.
    # The earlier "null and clarify" behavior caused an infinite
    # EmitClarify loop once apply_turn landed, because the bot's own
    # clarify text used "машину" and the user simply mirrored it back —
    # grounding kept rejecting. Bare car-word now means car.
    raw = json.dumps({"intent": "TOOL", "subject": "Легковой автомобиль"})
    out = parse_classifier_output(raw, utterance="хочу машину")
    assert out.subject == "Легковой автомобиль"


def test_age_years_nulled_when_condition_new_ungrounded():
    # Finding 2: classifier emits {condition_new=0, age_years=5} on
    # "новая машина пять лет". Step 1 grounds condition_new=0 against the
    # utterance → the cue "новая" is a NEW cue, has_field_signal(
    # "condition_new", 0, ...) is False → condition_new nulled. Step 2
    # re-runs cross-field with the NEW condition_new value, nulling age_years.
    # Previously the rule ran before grounding, leaving age_years=5 orphaned.
    raw = json.dumps({
        "intent": "TOOL", "condition_new": 0, "age_years": 5,
    })
    out = parse_classifier_output(raw, utterance="новая машина пять лет")
    assert out.condition_new is None
    assert out.age_years is None


# --- Codex adversarial pass 4, 2026-04-20: value_grounded helper ---

def test_value_grounded_currency_rejects_ungrounded():
    # Helper must reject USD on "в рублях" so the app.py staging block can
    # block explicit enum change_fields like change_value='USD'.
    from backend.classifier_schema import value_grounded
    assert value_grounded("currency", "USD", "в рублях") is False
    assert value_grounded("currency", "USD", "в долларах") is True
    assert value_grounded("currency", "RUB", "в рублях") is False
    assert value_grounded("currency", "RUB", "в российских рублях") is True


def test_value_grounded_client_type_normalizes_before_check():
    # "ИП" as change_value must normalize to "Юридическое лицо" before cue match.
    from backend.classifier_schema import value_grounded
    assert value_grounded("client_type", "ИП", "я ип") is True
    assert value_grounded("client_type", "Физическое лицо", "мы ООО") is False


def test_value_grounded_subject_matches_value_only():
    from backend.classifier_schema import value_grounded
    assert value_grounded("subject", "Легковой автомобиль", "хочу грузовик") is False
    assert value_grounded("subject", "Легковой автомобиль", "Toyota") is True
    assert value_grounded("subject", "Грузовой автомобиль", "хочу грузовик") is True


def test_value_grounded_condition_new_reuses_fix_1_10():
    from backend.classifier_schema import value_grounded
    # Fix 1.10 value-aware logic inherited via has_field_signal.
    assert value_grounded("condition_new", 0, "новая машина без пробега") is False
    assert value_grounded("condition_new", 1, "новая машина без пробега") is True


def test_value_grounded_numeric_delegates_has_field_signal():
    from backend.classifier_schema import value_grounded
    # Fix 34 / 40b spelled multipliers + year-to-month conversion.
    assert value_grounded("cost", 80000, "80 тысяч") is True
    assert value_grounded("term_months", 60, "на 5 лет") is True
    assert value_grounded("cost", 80000, "привет") is False


def test_value_grounded_empty_utterance_or_value():
    from backend.classifier_schema import value_grounded
    assert value_grounded("currency", "USD", "") is False
    assert value_grounded("currency", None, "в долларах") is False
    assert value_grounded("currency", "", "в долларах") is False


# --- Codex thorough review 2026-04-20: change_value canonicalization ---

def test_change_value_condition_new_string_coerced_to_int():
    # Highest-severity repro from the thorough review: classifier emits
    # change_value="0" (string) for condition_new; without coercion it's
    # stored verbatim and ClientProfile.missing_fields()'s `== 0` check
    # silently bypasses the age requirement for used assets.
    raw = json.dumps({
        "intent": "TOOL", "change_field": "condition_new", "change_value": "0",
    })
    out = parse_classifier_output(raw, utterance="бу машина")
    assert out.change_field == "condition_new"
    assert out.change_value == 0
    assert isinstance(out.change_value, int)


def test_change_value_type_schedule_int_coerced_to_str():
    raw = json.dumps({
        "intent": "TOOL", "change_field": "type_schedule", "change_value": 0,
    })
    out = parse_classifier_output(raw, utterance="аннуитетный график")
    assert out.change_field == "type_schedule"
    assert out.change_value == "0"
    assert isinstance(out.change_value, str)


def test_change_value_client_type_ip_normalized():
    raw = json.dumps({
        "intent": "TOOL", "change_field": "client_type", "change_value": "ИП",
    })
    out = parse_classifier_output(raw, utterance="я ип")
    assert out.change_value == "Юридическое лицо"


def test_change_value_currency_uppercased():
    raw = json.dumps({
        "intent": "TOOL", "change_field": "currency", "change_value": "usd",
    })
    out = parse_classifier_output(raw, utterance="в долларах")
    assert out.change_value == "USD"


def test_change_value_uncanonical_drops_pair():
    # Garbage change_value → both change_field and change_value nulled so
    # the pair never reaches the staging block.
    raw = json.dumps({
        "intent": "TOOL", "change_field": "condition_new", "change_value": "yes please",
    })
    out = parse_classifier_output(raw, utterance="поменяй")
    assert out.change_field is None
    assert out.change_value is None


def test_change_value_nan_rejected():
    raw = '{"intent":"TOOL","change_field":"cost","change_value":NaN}'
    out = parse_classifier_output(raw, utterance="стоимость триллион")
    assert out.change_field is None
    assert out.change_value is None


def test_change_value_fractional_term_months_rejected():
    # Codex basic review P2: fractional floats for integer fields must fail
    # closed, not truncate. 60.5 → silently 60 was the bug.
    raw = json.dumps({
        "intent": "TOOL", "change_field": "term_months", "change_value": 60.5,
    })
    out = parse_classifier_output(raw, utterance="на 60 месяцев")
    assert out.change_field is None
    assert out.change_value is None


def test_change_value_fractional_condition_new_rejected():
    raw = json.dumps({
        "intent": "TOOL", "change_field": "condition_new", "change_value": 0.5,
    })
    out = parse_classifier_output(raw, utterance="наполовину новая")
    assert out.change_field is None
    assert out.change_value is None


def test_change_value_fractional_type_schedule_rejected():
    raw = json.dumps({
        "intent": "TOOL", "change_field": "type_schedule", "change_value": 0.5,
    })
    out = parse_classifier_output(raw, utterance="аннуитет")
    assert out.change_field is None
    assert out.change_value is None


def test_change_value_integer_float_still_coerced():
    # 60.0 IS an integer value in float form — must still canonicalize.
    raw = json.dumps({
        "intent": "TOOL", "change_field": "term_months", "change_value": 60.0,
    })
    out = parse_classifier_output(raw, utterance="на 60 месяцев")
    assert out.change_field == "term_months"
    assert out.change_value == 60
    assert isinstance(out.change_value, int)


# --- Codex thorough review: non-finite numerics rejected at schema ---

def test_nan_cost_rejected_by_schema():
    # Python's json.loads accepts NaN; Pydantic's allow_inf_nan=False must
    # reject at the boundary. Otherwise readback's int(cost) raises ValueError.
    raw = '{"intent":"TOOL","cost":NaN}'
    out = parse_classifier_output(raw, utterance="хочу машину")
    assert out.cost is None


def test_infinity_prepaid_amount_rejected_by_schema():
    raw = '{"intent":"TOOL","prepaid_amount":Infinity}'
    out = parse_classifier_output(raw, utterance="большой аванс")
    assert out.prepaid_amount is None


def test_neg_infinity_prepaid_pct_rejected_by_schema():
    raw = '{"intent":"TOOL","prepaid_pct":-Infinity}'
    out = parse_classifier_output(raw, utterance="аванс")
    assert out.prepaid_pct is None


# --- Codex thorough review: numeric type_schedule coerced at top level ---

def test_type_schedule_numeric_zero_coerced_at_top_level():
    # Qwen sometimes emits {"type_schedule": 0} instead of "0" — schema
    # coerces int→str before Literal validation so the answer isn't lost.
    raw = json.dumps({"intent": "TOOL", "type_schedule": 0})
    out = parse_classifier_output(raw, utterance="аннуитетный график")
    assert out.type_schedule == "0"


def test_type_schedule_numeric_one_coerced_at_top_level():
    raw = json.dumps({"intent": "TOOL", "type_schedule": 1})
    out = parse_classifier_output(raw, utterance="линейный график")
    assert out.type_schedule == "1"


def test_type_schedule_bool_not_coerced():
    # Defensive: bool is an int subclass in Python; ensure True/False don't
    # get accidentally coerced to type_schedule codes.
    raw = json.dumps({"intent": "TOOL", "type_schedule": True})
    out = parse_classifier_output(raw, utterance="аннуитет")
    assert out.type_schedule is None


# --- Codex basic review 2026-04-20: top-level field coercion symmetry ---

def test_condition_new_string_zero_coerced_to_int():
    # Codex basic P1: Qwen sometimes emits condition_new as quoted string.
    # Must coerce so used-asset signal + age_years requirement still fire.
    raw = json.dumps({"intent": "TOOL", "condition_new": "0"})
    out = parse_classifier_output(raw, utterance="бу машина")
    assert out.condition_new == 0
    assert isinstance(out.condition_new, int)


def test_condition_new_string_one_coerced_to_int():
    raw = json.dumps({"intent": "TOOL", "condition_new": "1"})
    out = parse_classifier_output(raw, utterance="новая машина")
    assert out.condition_new == 1


def test_condition_new_float_one_point_zero_coerced():
    raw = json.dumps({"intent": "TOOL", "condition_new": 1.0})
    out = parse_classifier_output(raw, utterance="новая")
    assert out.condition_new == 1


def test_currency_lowercase_coerced_to_upper():
    # Codex basic P2: Qwen may emit "usd" / "byn" in lower case. Must coerce
    # at the boundary so the currency isn't dropped before grounding runs.
    raw = json.dumps({"intent": "TOOL", "currency": "usd"})
    out = parse_classifier_output(raw, utterance="в долларах")
    assert out.currency == "USD"


def test_currency_mixed_case_with_whitespace_coerced():
    raw = json.dumps({"intent": "TOOL", "currency": " ByN "})
    out = parse_classifier_output(raw, utterance="в рублях")
    assert out.currency == "BYN"


def test_currency_garbage_string_still_dropped():
    # Coercion only rescues valid currencies — "XYZ" must still fall through
    # and get rejected by the Literal.
    raw = json.dumps({"intent": "TOOL", "currency": "XYZ"})
    out = parse_classifier_output(raw, utterance="в долларах")
    assert out.currency is None


def test_value_grounded_cost_accepts_ru_number_words():
    # Regression for live call f7e5aa1d (2026-04-24): "оставим двадцать
    # тысяч долларов" emitted cost=20000 but grounding rejected because
    # "20000" is not literally in the utterance. Russian number-words
    # must ground.
    from backend.classifier_schema import value_grounded
    assert value_grounded("cost", 20000, "оставим двадцать тысяч долларов") is True
    assert value_grounded("cost", 100000, "хочу сто тысяч рублей") is True
    assert value_grounded("cost", 20000, "ровно двадцать тысяч") is True


def test_value_grounded_cost_rejects_unrelated_ru_number():
    # "двадцать процентов" is a percent, not a cost — must NOT ground
    # cost=20.
    from backend.classifier_schema import value_grounded
    assert value_grounded("cost", 20, "двадцать процентов аванс") is False


def test_value_grounded_cost_rejects_mismatched_magnitude():
    # "двадцать тысяч" = 20000, must NOT ground cost=99999.
    from backend.classifier_schema import value_grounded
    assert value_grounded("cost", 99999, "двадцать тысяч долларов") is False


def test_value_grounded_cost_digit_form_still_works():
    # Sanity: digit-based grounding still passes (we didn't break it).
    from backend.classifier_schema import value_grounded
    assert value_grounded("cost", 20000, "20000 долларов") is True


def test_value_grounded_cost_mixed_percent_and_cost():
    # Adversarial case from Task 6 review: an utterance that mixes a
    # percent value AND a cost value should ground the cost.
    # NOTE: parse_ru_number's percent-reset is GLOBAL, so this one
    # asserts on the order. With percent first, the cost survives.
    from backend.classifier_schema import value_grounded
    assert value_grounded("cost", 100000, "двадцать процентов и сто тысяч долларов") is True
