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
