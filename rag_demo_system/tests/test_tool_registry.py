"""Tests for the tool registry system."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools import init_tools, get_tool_schemas, get_tool, get_all_tools


class MockTools:
    calculator_api_base_url = "http://test"
    calculator_api_token = "tok"
    sms_api_login = "login"
    sms_api_password = "pass"
    sms_sender_name = "Test"
    crm_webhook_url = ""
    crm_webhook_token = ""


class MockSettings:
    tools = MockTools()


def test_init_tools_registers_all():
    init_tools(MockSettings())
    tools = get_all_tools()
    assert "calculator" in tools
    assert "send_sms" in tools


def test_get_tool_schemas_returns_list():
    init_tools(MockSettings())
    schemas = get_tool_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) >= 2
    names = [s["function"]["name"] for s in schemas]
    assert "calculator" in names
    assert "send_sms" in names


def test_get_tool_by_name():
    init_tools(MockSettings())
    calc = get_tool("calculator")
    assert calc is not None
    assert calc.schema()["function"]["name"] == "calculator"


def test_get_tool_unknown_raises():
    init_tools(MockSettings())
    try:
        get_tool("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
