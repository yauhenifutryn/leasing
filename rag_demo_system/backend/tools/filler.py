from __future__ import annotations

import random

_FILLERS: dict[str, list[str]] = {
    "calculator": [
        "Секундочку, рассчитываю.",
        "Один момент, считаю для вас.",
        "Сейчас посчитаю.",
    ],
    "send_sms": [
        "Отправляю сообщение.",
        "Секунду, отправляю СМС.",
    ],
    "escalate_to_human": [
        "Передаю информацию специалисту.",
        "Секунду, связываю со специалистом.",
    ],
}

_DEFAULT_FILLER = "Один момент."


def get_filler(tool_name: str) -> str:
    """Return a random filler phrase for the given tool."""
    phrases = _FILLERS.get(tool_name, [_DEFAULT_FILLER])
    return random.choice(phrases)
