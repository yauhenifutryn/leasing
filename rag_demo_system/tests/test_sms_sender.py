from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools.sms_sender import SmsSenderTool


def test_schema():
    tool = SmsSenderTool(login="test", password="test", sender="MikroLizing")
    schema = tool.schema()
    func = schema["function"]
    assert func["name"] == "send_sms"
    assert "phone" in func["parameters"]["properties"]
    assert "message" in func["parameters"]["properties"]
    assert func["parameters"]["required"] == ["phone", "message"]


def test_execute_success():
    tool = SmsSenderTool(login="test", password="test", sender="MikroLizing")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "1454982446"
    with patch("backend.tools.sms_sender.httpx.get", return_value=mock_resp):
        result = tool.execute({"phone": "375291224557", "message": "Test message"}, {})
    assert result["ok"] is True
    assert result["message_id"] == "1454982446"


def test_execute_auth_failure():
    tool = SmsSenderTool(login="test", password="wrong", sender="MikroLizing")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "-2"
    with patch("backend.tools.sms_sender.httpx.get", return_value=mock_resp):
        result = tool.execute({"phone": "375291224557", "message": "Test"}, {})
    assert result["ok"] is False
    assert "error" in result


def test_phone_validation():
    tool = SmsSenderTool(login="test", password="test", sender="MikroLizing")
    result = tool.execute({"phone": "123", "message": "Test"}, {})
    assert result["ok"] is False


def test_empty_message():
    tool = SmsSenderTool(login="test", password="test", sender="MikroLizing")
    result = tool.execute({"phone": "375291224557", "message": ""}, {})
    assert result["ok"] is False


class TestSmsSenderSessionPhone:
    def test_schema_uses_session_phone_when_provided(self):
        tool = SmsSenderTool(login="test", password="test", sender="Test")
        schema = tool.schema(session_phone="375291234567")
        desc = schema["function"]["description"]
        assert "375291234567" in desc
        assert "375291224557" not in desc

    def test_schema_uses_hardcoded_phone_when_no_session(self):
        tool = SmsSenderTool(login="test", password="test", sender="Test")
        schema = tool.schema()
        desc = schema["function"]["description"]
        assert "375291224557" in desc
