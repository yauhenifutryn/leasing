from __future__ import annotations

from typing import Any


def build_memory_block(transcript: list[dict[str, Any]], max_turns: int) -> str:
    if max_turns <= 0 or not transcript:
        return ""

    cleaned: list[dict[str, Any]] = []
    for item in transcript:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        cleaned.append({"role": item.get("role"), "text": text})

    if not cleaned:
        return ""

    limit = max_turns * 2
    items = cleaned[-limit:] if limit > 0 else cleaned
    lines: list[str] = []
    for item in items:
        role = item.get("role") or ""
        label = "Клиент" if role == "user" else "Агент"
        lines.append(f"{label}: {item['text']}")

    if not lines:
        return ""

    return (
        "Контекст диалога (используй для продолжения темы и уточняющих вопросов):\n"
        + "\n".join(lines)
        + "\n\n"
    )
