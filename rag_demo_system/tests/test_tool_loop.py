from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.llm_stream import parse_tool_calls_from_events


def test_parse_tool_call_from_stream():
    """Simulate vLLM streaming a tool_call response."""
    events = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_123", "function": {"name": "calculator", "arguments": ""}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"subject": "Лег'}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ковой автомобиль", "cost": 30000}'}}]}, "finish_reason": "tool_calls"}]},
    ]
    result = parse_tool_calls_from_events(events)
    assert len(result) == 1
    assert result[0]["id"] == "call_123"
    assert result[0]["function"]["name"] == "calculator"
    args = json.loads(result[0]["function"]["arguments"])
    assert args["subject"] == "Легковой автомобиль"
    assert args["cost"] == 30000


def test_parse_content_no_tool_calls():
    """Regular content response should return empty list."""
    events = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
    ]
    result = parse_tool_calls_from_events(events)
    assert result == []


def test_parse_multiple_tool_calls():
    """Multiple tool calls in one response."""
    events = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "calculator", "arguments": '{"cost": 1000}'}},
            {"index": 1, "id": "call_2", "function": {"name": "send_sms", "arguments": '{"phone": "375291224557"}'}},
        ]}, "finish_reason": "tool_calls"}]},
    ]
    result = parse_tool_calls_from_events(events)
    assert len(result) == 2
    assert result[0]["function"]["name"] == "calculator"
    assert result[1]["function"]["name"] == "send_sms"
