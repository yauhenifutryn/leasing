"""Fix 39 — range validation for calculator params.

Catches out-of-range inputs before they reach the Mikro Leasing API and
surfaces a specific marker per violation so the orchestrator can emit a
user-facing Russian message ('срок должен быть от 12 до 84 месяцев').
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools.calculator import (  # noqa: E402
    CalculatorTool,
    IncompleteProfileError,
    validate_calc_inputs,
)


def _base_params() -> dict:
    return {
        "subject": "Легковой автомобиль",
        "cost": 50000,
        "client_type": "Физическое лицо",
        "currency": "BYN",
        "condition_new": 1,
        "term": 36,
        "type_schedule": "0",
        "prepaid": 20,
    }


# ── validate_calc_inputs direct unit tests ─────────────────────────────

def test_valid_params_return_no_violations() -> None:
    assert validate_calc_inputs(_base_params()) == []


def test_term_too_low_flagged() -> None:
    p = _base_params()
    p["term"] = 5
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:term=5") for m in out)


def test_term_too_high_flagged() -> None:
    p = _base_params()
    p["term"] = 200
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:term=200") for m in out)


def test_term_zero_flagged() -> None:
    p = _base_params()
    p["term"] = 0
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:term=0") for m in out)


def test_term_negative_flagged() -> None:
    p = _base_params()
    p["term"] = -12
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:term=-12") for m in out)


def test_prepaid_pct_over_40_flagged() -> None:
    p = _base_params()
    p["prepaid"] = 110
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:prepaid_pct=110") for m in out)


def test_prepaid_pct_negative_flagged() -> None:
    p = _base_params()
    p["prepaid"] = -5
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:prepaid_pct=-5") for m in out)


def test_cost_zero_flagged() -> None:
    p = _base_params()
    p["cost"] = 0
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:cost=0") for m in out)


def test_cost_negative_flagged() -> None:
    p = _base_params()
    p["cost"] = -5000
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:cost=-5000") for m in out)


def test_cost_absurdly_large_flagged() -> None:
    p = _base_params()
    p["cost"] = 999_999_999_999  # 1 trillion BYN
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:cost=") for m in out)


def test_prepaid_amount_exceeds_cost_flagged() -> None:
    p = _base_params()
    p.pop("prepaid")
    p["prepaid_amount"] = 80000  # cost is 50000
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:prepaid_amount=80000") for m in out)


def test_prepaid_amount_zero_flagged() -> None:
    p = _base_params()
    p.pop("prepaid")
    p["prepaid_amount"] = 0
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:prepaid_amount=0") for m in out)


def test_age_on_used_over_range_flagged() -> None:
    p = _base_params()
    p["condition_new"] = 0
    p["age"] = 50
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:age=50") for m in out)


def test_age_on_new_is_ignored() -> None:
    p = _base_params()
    p["condition_new"] = 1
    p["age"] = 50  # absurd age but ignored when new
    out = validate_calc_inputs(p)
    assert not any(m.startswith("param_out_of_range:age") for m in out)


def test_cost_non_numeric_flagged_as_bad_type() -> None:
    p = _base_params()
    p["cost"] = "not a number"
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_bad_type:cost=") for m in out)


def test_cost_nan_flagged_as_bad_type() -> None:
    p = _base_params()
    p["cost"] = float("nan")
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_bad_type:cost=") for m in out)


def test_cost_inf_flagged_as_bad_type() -> None:
    p = _base_params()
    p["cost"] = float("inf")
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_bad_type:cost=") for m in out)


def test_multiple_violations_all_reported() -> None:
    p = _base_params()
    p["term"] = 5
    p["prepaid"] = 110
    p["cost"] = -1
    out = validate_calc_inputs(p)
    assert any(m.startswith("param_out_of_range:term=5") for m in out)
    assert any(m.startswith("param_out_of_range:prepaid_pct=110") for m in out)
    assert any(m.startswith("param_out_of_range:cost=-1") for m in out)


# ── execute() integration: OOR raises IncompleteProfileError ───────────

def _make_tool() -> CalculatorTool:
    return CalculatorTool(base_url="http://example.invalid", token="x")


def test_execute_raises_on_term_out_of_range() -> None:
    tool = _make_tool()
    p = _base_params()
    p["term"] = 5
    with pytest.raises(IncompleteProfileError) as exc:
        tool.execute(p, {})
    assert any("param_out_of_range:term=5" in m for m in exc.value.missing)


def test_execute_raises_on_prepaid_out_of_range() -> None:
    tool = _make_tool()
    p = _base_params()
    p["prepaid"] = 110
    with pytest.raises(IncompleteProfileError) as exc:
        tool.execute(p, {})
    assert any("param_out_of_range:prepaid_pct=110" in m for m in exc.value.missing)


def test_execute_raises_on_negative_cost() -> None:
    tool = _make_tool()
    p = _base_params()
    p["cost"] = -1000
    with pytest.raises(IncompleteProfileError) as exc:
        tool.execute(p, {})
    assert any("param_out_of_range:cost=-1000" in m for m in exc.value.missing)


def test_execute_raises_on_prepaid_amount_gt_cost() -> None:
    tool = _make_tool()
    p = _base_params()
    p.pop("prepaid")
    p["prepaid_amount"] = 80000
    with pytest.raises(IncompleteProfileError) as exc:
        tool.execute(p, {})
    assert any("param_out_of_range:prepaid_amount=80000" in m for m in exc.value.missing)
