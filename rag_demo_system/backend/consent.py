from __future__ import annotations

import re


CONSENT_POSITIVE = re.compile(
    r"\b(да|согласен|согласна|да,\s*согласен|да,\s*согласна|подтверждаю|подтверждаю согласие|даю согласие|согласие даю)\b",
    re.I,
)
CONSENT_NEGATIVE = re.compile(r"\b(не\s*согласен|не\s*согласна|отказываюсь|нет,\s*не\s*согласен|нет,\s*не\s*согласна)\b", re.I)


def detect_consent(message: str) -> str:
    text = message.strip().lower()
    if not text:
        return "unknown"
    if CONSENT_NEGATIVE.search(text):
        return "denied"
    if CONSENT_POSITIVE.search(text):
        return "granted"
    return "unknown"


def consent_request() -> str:
    return (
        "Перед началом разговора нужно ваше согласие на обработку и трансграничную передачу "
        "персональных данных. Подтвердите, пожалуйста, согласие."
    )


def consent_denied_response() -> str:
    return (
        "Понимаю. Без вашего согласия продолжить консультацию нельзя. "
        "Спасибо за обращение."
    )


def consent_granted_response() -> str:
    return "Спасибо за согласие. Чем могу помочь?"
