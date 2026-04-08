from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any


def iter_openai_stream_text(lines: Iterable[str]) -> Iterator[str]:
    for raw in lines:
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        text = delta.get("content") or ""
        if text:
            yield text


def iter_openai_stream_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for raw in lines:
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        yield data


def parse_tool_calls_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accumulate streamed tool_call deltas into complete tool_call objects.

    Returns a list of tool call dicts: [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
    Returns empty list if no tool calls in the events.
    """
    calls: dict[int, dict[str, Any]] = {}
    for event in events:
        choice = (event.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        tool_calls = delta.get("tool_calls")
        if not tool_calls:
            continue
        for tc in tool_calls:
            idx = tc.get("index", 0)
            if idx not in calls:
                calls[idx] = {
                    "id": tc.get("id", ""),
                    "function": {"name": "", "arguments": ""},
                }
            if tc.get("id"):
                calls[idx]["id"] = tc["id"]
            func = tc.get("function") or {}
            if func.get("name"):
                calls[idx]["function"]["name"] = func["name"]
            if func.get("arguments"):
                calls[idx]["function"]["arguments"] += func["arguments"]
    return list(calls.values())
