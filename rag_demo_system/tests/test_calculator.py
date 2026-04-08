"""Tests for the payment calculator tool."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools.calculator import CalculatorTool  # noqa: E402

SAMPLE_RESPONSE = {
    "0": {
        "URL": "https://mikro-leasing.by/graphic/?57030126",
        "id": "57030126",
        "increase_factor": 0.12914956,
        "increase_percent": 13.0,
        "number": 0,
        "sum": 9000.0,
    },
    "1": {"number": 1, "sum": 897.87},
    "2": {"number": 2, "sum": 897.87},
    "3": {"number": 3, "sum": 897.87},
    "999": {"number": 999, "sum": 300.0},
}


@pytest.fixture()
def tool() -> CalculatorTool:
    return CalculatorTool(base_url="https://api.example.com", token="test-token")


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------


def test_schema_has_required_fields(tool: CalculatorTool) -> None:
    s = tool.schema()
    assert s["type"] == "function"
    fn = s["function"]
    assert fn["name"] == "calculator"
    assert "subject" in fn["parameters"]["properties"]
    assert "cost" in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == ["subject", "cost"]


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------


def test_defaults(tool: CalculatorTool) -> None:
    d = tool.defaults()
    assert d["client_type"] == "Физическое лицо"
    assert d["condition_new"] == 1
    assert d["currency"] == "BYN"
    assert d["prepaid"] == 30
    assert d["term"] == 36
    assert d["type_schedule"] == "0"


def test_fill_defaults_marks_defaulted(tool: CalculatorTool) -> None:
    params = {"subject": "Легковой автомобиль", "cost": 30000}
    filled, defaulted = tool.fill_defaults(params)
    assert "client_type" in defaulted
    assert "currency" in defaulted
    assert "prepaid" in defaulted
    assert "term" in defaulted
    assert "type_schedule" in defaulted
    assert "condition_new" in defaulted
    # User-provided values must NOT be in defaulted
    assert "subject" not in defaulted
    assert "cost" not in defaulted


# ------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------


@patch("backend.tools.calculator.httpx.get")
def test_execute_calls_api(mock_get: MagicMock, tool: CalculatorTool) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SAMPLE_RESPONSE
    mock_get.return_value = mock_resp

    result = tool.execute({"subject": "Легковой автомобиль", "cost": 30000}, {})

    assert result["ok"] is True
    assert result["url"] == "https://mikro-leasing.by/graphic/?57030126"
    assert result["calculation_id"] == "57030126"
    assert result["advance_sum"] == 9000.0
    assert result["buyout_sum"] == 300.0
    assert result["increase_percent"] == 13.0
    assert len(result["payments"]) == 3
    assert result["payments"][0]["sum"] == 897.87

    # Verify API was called with correct URL and auth
    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args
    assert "Bearer test-token" in call_kwargs.kwargs.get("headers", {}).get("Authorization", "")


@patch("backend.tools.calculator.httpx.get")
def test_execute_handles_404(mock_get: MagicMock, tool: CalculatorTool) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp

    result = tool.execute({"subject": "Легковой автомобиль", "cost": 30000}, {})

    assert result["ok"] is False
    assert "не найдены" in result["error"]


def test_execute_used_without_age(tool: CalculatorTool) -> None:
    result = tool.execute({"subject": "Легковой автомобиль", "cost": 20000, "condition_new": 0}, {})

    assert result["ok"] is False
    assert "возраст" in result["error"].lower() or "age" in result["error"].lower()


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------


@patch("backend.tools.calculator.httpx.get")
def test_voice_summary_contains_key_numbers(mock_get: MagicMock, tool: CalculatorTool) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SAMPLE_RESPONSE
    mock_get.return_value = mock_resp

    result = tool.execute({"subject": "Легковой автомобиль", "cost": 30000}, {})
    summary = tool.format_voice_summary(result)

    assert "9000.0" in summary  # advance
    assert "897.87" in summary  # monthly payment
    assert "300.0" in summary  # buyout
    assert "13.0" in summary  # increase percent
    assert "*" in summary  # defaulted markers


@patch("backend.tools.calculator.httpx.get")
def test_format_sms_body_contains_link(mock_get: MagicMock, tool: CalculatorTool) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SAMPLE_RESPONSE
    mock_get.return_value = mock_resp

    result = tool.execute({"subject": "Легковой автомобиль", "cost": 30000}, {})
    sms = tool.format_sms_body(result)

    assert sms is not None
    assert "https://mikro-leasing.by/graphic/?57030126" in sms
    assert "+375 17 322 77 00" in sms
    assert "Микро Лизинг" in sms
