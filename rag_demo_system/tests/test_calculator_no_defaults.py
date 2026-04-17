"""Calculator no longer applies hidden defaults; missing fields raise IncompleteProfileError."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools.calculator import (  # noqa: E402
    CalculatorTool,
    IncompleteProfileError,
)


def _make_tool() -> CalculatorTool:
    return CalculatorTool(base_url="http://example.invalid", token="x")


def test_defaults_returns_empty_dict() -> None:
    assert _make_tool().defaults() == {}


def test_execute_raises_on_empty_params() -> None:
    tool = _make_tool()
    with pytest.raises(IncompleteProfileError) as exc:
        tool.execute({}, {})
    missing = exc.value.missing
    for required in ("subject", "cost", "client_type", "currency",
                     "condition_new", "term", "type_schedule", "prepaid"):
        assert required in missing


def test_execute_raises_when_prepaid_missing() -> None:
    tool = _make_tool()
    params = dict(
        subject="Легковой автомобиль",
        cost=70000,
        client_type="Физическое лицо",
        currency="BYN",
        condition_new=1,
        term=60,
        type_schedule="0",
    )
    with pytest.raises(IncompleteProfileError) as exc:
        tool.execute(params, {})
    assert "prepaid" in exc.value.missing


def test_execute_accepts_prepaid_amount_derives_pct() -> None:
    tool = _make_tool()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "0": {"URL": "http://x", "id": "1", "sum": 14000.0, "increase_percent": 17.0, "increase_factor": 0.17},
        "1": {"sum": 1500, "number": 1},
        "2": {"sum": 1500, "number": 2},
        "999": {"sum": 0},
    }
    params = dict(
        subject="Легковой автомобиль",
        cost=70000,
        client_type="Физическое лицо",
        currency="BYN",
        condition_new=1,
        prepaid_amount=14000,
        term=60,
        type_schedule="0",
    )
    with patch("backend.tools.calculator.httpx.get", return_value=fake_resp) as mock_get:
        result = tool.execute(params, {})
    assert result["ok"] is True
    # prepaid_amount=14000 / cost=70000 -> 20.0%
    call_params = mock_get.call_args.kwargs["params"]
    assert call_params["prepaid"] == 20.0


def test_execute_prepaid_pct_forwarded_directly() -> None:
    tool = _make_tool()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "0": {"URL": "http://x", "id": "1", "sum": 21000.0, "increase_percent": 12.0, "increase_factor": 0.12},
        "1": {"sum": 1500, "number": 1},
        "999": {"sum": 0},
    }
    params = dict(
        subject="Легковой автомобиль",
        cost=70000,
        client_type="Физическое лицо",
        currency="BYN",
        condition_new=1,
        prepaid_pct=30,
        term=36,
        type_schedule="0",
    )
    with patch("backend.tools.calculator.httpx.get", return_value=fake_resp) as mock_get:
        result = tool.execute(params, {})
    assert result["ok"] is True
    assert mock_get.call_args.kwargs["params"]["prepaid"] == 30.0


def test_execute_raises_on_out_of_range_prepaid() -> None:
    tool = _make_tool()
    params = dict(
        subject="Легковой автомобиль",
        cost=70000,
        client_type="Физическое лицо",
        currency="BYN",
        condition_new=1,
        prepaid=50,  # over 40% max
        term=60,
        type_schedule="0",
    )
    with pytest.raises(IncompleteProfileError) as exc:
        tool.execute(params, {})
    assert any("prepaid_pct_out_of_range" in m for m in exc.value.missing)


def test_execute_forwards_type_schedule_linear() -> None:
    tool = _make_tool()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "0": {"URL": "http://x", "id": "1", "sum": 14000.0, "increase_percent": 10.0, "increase_factor": 0.10},
        "1": {"sum": 1500, "number": 1},
        "999": {"sum": 0},
    }
    params = dict(
        subject="Легковой автомобиль",
        cost=70000,
        client_type="Физическое лицо",
        currency="BYN",
        condition_new=1,
        prepaid_pct=20,
        term=84,
        type_schedule="1",  # linear
    )
    with patch("backend.tools.calculator.httpx.get", return_value=fake_resp) as mock_get:
        tool.execute(params, {})
    assert mock_get.call_args.kwargs["params"]["type_schedule"] == "1"
