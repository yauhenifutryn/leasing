from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_prompts import build_change_confirm_text


def test_change_confirm_type_schedule_annuity():
    text = build_change_confirm_text({"field": "type_schedule", "new_value": "0"})
    assert "аннуитетный" in text
    assert " 0," not in text  # must NOT contain raw code


def test_change_confirm_type_schedule_linear():
    text = build_change_confirm_text({"field": "type_schedule", "new_value": "1"})
    assert "линейный" in text


def test_change_confirm_type_schedule_int_code():
    # Classifier may emit int vs str — both must translate.
    text = build_change_confirm_text({"field": "type_schedule", "new_value": 0})
    assert "аннуитетный" in text


def test_change_confirm_client_type_ip():
    text = build_change_confirm_text({"field": "client_type", "new_value": "ИП"})
    assert "индивидуальный предприниматель" in text


def test_change_confirm_currency_usd():
    text = build_change_confirm_text({"field": "currency", "new_value": "USD"})
    assert "доллары США" in text


def test_change_confirm_condition_new():
    text = build_change_confirm_text({"field": "condition_new", "new_value": 1})
    assert "новый" in text
    assert " 1," not in text


def test_change_confirm_numeric_field_passthrough():
    # For non-enum fields, raw numeric value is fine (it's the actual number).
    text = build_change_confirm_text({"field": "term_months", "new_value": 48})
    assert "48" in text


def test_change_confirm_empty_returns_prompt():
    text = build_change_confirm_text({})
    assert "Уточните" in text or text.startswith("Уточните")


def test_change_confirm_none_returns_prompt():
    text = build_change_confirm_text(None)
    assert "Уточните" in text or text.startswith("Уточните")
