from __future__ import annotations

from typing import Any


_INTERRUPT_MARKER = "[прервано клиентом]"

_INTERRUPT_INSTRUCTION = (
    "ВАЖНО: последний ответ был прерван клиентом — он услышал его только частично. На этом ходе:\n"
    "1. Сначала ответь на новую реплику клиента.\n"
    "2. Если клиент уточнил параметры или поправил тебя — учти это и переформулируй/пересчитай.\n"
    "3. Если клиент просит продолжить и нет нового ввода — кратко продолжи прерванную мысль с того места, где остановился.\n"
    "4. Если клиент задал другой вопрос — ответь на него и не возвращайся к прерванной мысли."
)


def _last_assistant_was_interrupted(items: list[dict[str, Any]]) -> bool:
    for item in reversed(items):
        if item.get("role") == "assistant":
            text = str(item.get("text") or "")
            return _INTERRUPT_MARKER in text
    return False


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

    body = (
        "ПОЛНАЯ ИСТОРИЯ ТЕКУЩЕГО РАЗГОВОРА (это ваш разговор с этим клиентом, "
        "вы помните ВСЁ что обсуждалось ниже и можете напомнить клиенту):\n"
        + "\n".join(lines)
    )

    if _last_assistant_was_interrupted(items):
        body += "\n\n" + _INTERRUPT_INSTRUCTION

    return body + "\n\n"
