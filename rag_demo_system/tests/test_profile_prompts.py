from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_prompts import (
    build_change_confirm_text,
    build_clarification_prompt,
    build_readback_text,
    render_calc_result,
)


# Bug 24: every place the bot SPEAKS the schedule type uses lay phrasing.
# "аннуитет"/"линейный" are banking jargon — clients ask "что такое
# аннуитет?" routinely (Stanislav 15:08:23, Valery 15:29:01).

def test_change_confirm_type_schedule_annuity():
    text = build_change_confirm_text({"field": "type_schedule", "new_value": "0"})
    assert "равными платежами" in text
    assert "аннуитет" not in text.lower()
    assert " 0," not in text  # must NOT contain raw code


def test_change_confirm_type_schedule_linear():
    text = build_change_confirm_text({"field": "type_schedule", "new_value": "1"})
    assert "с уменьшением суммы к концу срока" in text
    assert "линейн" not in text.lower()


def test_change_confirm_type_schedule_int_code():
    # Classifier may emit int vs str — both must translate.
    text = build_change_confirm_text({"field": "type_schedule", "new_value": 0})
    assert "равными платежами" in text


def test_clarification_prompt_type_schedule_uses_lay_phrasing():
    text = build_clarification_prompt({"type_schedule"}, profile=SimpleNamespace())
    assert "равными платежами" in text
    assert "с уменьшением" in text
    assert "аннуитет" not in text.lower()
    assert "линейн" not in text.lower()


def test_readback_speaks_schedule_in_lay_phrasing_annuity():
    profile = SimpleNamespace(
        subject="Легковой автомобиль",
        condition_new=1,
        age_years=None,
        prepaid_pct=20,
        prepaid_amount=None,
        currency="BYN",
        cost=50000,
        original_cost=None,
        original_currency=None,
        client_type="Физическое лицо",
        term_months=36,
        type_schedule="0",
    )
    text = build_readback_text(profile)
    assert "равные платежи" in text
    assert "аннуитет" not in text.lower()


def test_readback_speaks_schedule_in_lay_phrasing_linear():
    profile = SimpleNamespace(
        subject="Легковой автомобиль",
        condition_new=1,
        age_years=None,
        prepaid_pct=20,
        prepaid_amount=None,
        currency="BYN",
        cost=50000,
        original_cost=None,
        original_currency=None,
        client_type="Физическое лицо",
        term_months=36,
        type_schedule="1",
    )
    text = build_readback_text(profile)
    assert "с уменьшением суммы к концу срока" in text
    assert "линейн" not in text.lower()


def test_render_calc_linear_schedule_narrates_average_not_single_payment():
    # Live transcript 2026-05-08 client feedback: убывающий график was
    # spoken as "Ежемесячный платёж — X" (single number, copied from
    # annuity render). Misleading because payments decrease over time.
    # Expected: name the AVERAGE payment + tell the caller payments
    # start higher and end lower. Mirrors Just AI's narration.
    result = {
        "params": {
            "cost": 250000, "currency": "USD", "prepaid": 20, "term": 60,
            "type_schedule": "1",
        },
        "num_payments": 60,
        "payment_min": 3000,   # last payment (smallest)
        "payment_max": 5000,   # first payment (largest)
        "buyout_sum": 2500,
        "total": 361127,
        "increase_percent": 9,
    }
    text = render_calc_result(result, detailed=False)
    # Average is the right anchor for убывающий — exact for a linear schedule.
    assert "средний" in text.lower(), text
    assert "4000" in text, text  # (5000 + 3000) / 2
    # Must explicitly tell the caller payments are not flat.
    assert ("в начале" in text.lower()
            or "первые платежи" in text.lower()
            or "к концу срока" in text.lower()), text
    # Must NOT use the annuity phrasing for a non-flat schedule.
    assert "Ежемесячный платёж — 3000" not in text
    assert "Ежемесячный платёж — 5000" not in text


def test_render_calc_annuity_keeps_single_payment_phrasing():
    # Annuity: payment_min == payment_max. Current "Ежемесячный платёж — X"
    # phrasing is correct; do NOT regress it.
    result = {
        "params": {
            "cost": 100000, "currency": "BYN", "prepaid": 30, "term": 36,
            "type_schedule": "0",
        },
        "num_payments": 36,
        "payment_min": 1990,
        "payment_max": 1990,
        "buyout_sum": 100,
        "total": 71640,
        "increase_percent": 5,
    }
    text = render_calc_result(result, detailed=False)
    assert "Ежемесячный платёж — 1990 BYN" in text
    assert "средний" not in text.lower()


def test_render_calc_defaults_note_uses_lay_schedule():
    result = {
        "params": {"cost": 50000, "currency": "BYN", "prepaid": 30, "term": 36},
        "num_payments": 36,
        "payment_min": 1500,
        "buyout_sum": 100,
        "total": 60000,
        "increase_percent": 20,
        "defaulted": ["type_schedule"],
    }
    text = render_calc_result(result, detailed=False)
    assert "график равными платежами" in text
    assert "аннуитетный график" not in text


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
