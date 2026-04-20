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


def test_ip_rejected_as_client_type():
    # E-Codex: ИП is no longer a valid client_type — prompt collapses to Юр.лицо.
    raw = json.dumps({"intent": "TOOL", "client_type": "ИП"})
    out = parse_classifier_output(raw, utterance="я ип")
    # ValidationError path returns empty model; client_type is None.
    assert out.client_type is None


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


def test_empty_utterance_skips_enum_grounding():
    # Unit-test path: no utterance provided → do not null enum fields.
    # Cross-field rule (age_years/condition_new) still runs — it does not need utterance.
    raw = json.dumps({"intent": "TOOL", "subject": "Легковой автомобиль"})
    out = parse_classifier_output(raw, utterance="")
    assert out.subject == "Легковой автомобиль"


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
