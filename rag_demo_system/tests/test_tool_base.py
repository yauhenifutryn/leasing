from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools.base import ToolDefinition


class DummyTool(ToolDefinition):
    def schema(self) -> dict:
        return {"type": "function", "function": {"name": "dummy", "parameters": {}}}

    def defaults(self) -> dict:
        return {"color": "red", "size": 10}

    def execute(self, params: dict, session_context: dict) -> dict:
        return {"ok": True}

    def format_voice_summary(self, result: dict) -> str:
        return "done"


def test_fill_defaults_merges_missing():
    tool = DummyTool()
    filled, defaulted = tool.fill_defaults({"color": "blue"})
    assert filled == {"color": "blue", "size": 10}
    assert defaulted == ["size"]


def test_fill_defaults_no_defaults_needed():
    tool = DummyTool()
    filled, defaulted = tool.fill_defaults({"color": "blue", "size": 5})
    assert filled == {"color": "blue", "size": 5}
    assert defaulted == []


def test_fill_defaults_all_defaulted():
    tool = DummyTool()
    filled, defaulted = tool.fill_defaults({})
    assert filled == {"color": "red", "size": 10}
    assert sorted(defaulted) == ["color", "size"]
