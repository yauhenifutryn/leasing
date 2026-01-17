from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .llm import call_openai_compatible


@dataclass
class RouterDecision:
    kind: str
    response: str


def _parse_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _classify_with_llm(message: str, base_url: str, model: str) -> str | None:
    system_prompt = (
        "Ты классификатор сообщений в чате поддержки. "
        "Верни строго JSON с полем intent из списка: "
        "greeting, meta, off_topic, identity, question, unclear. "
        "Никаких пояснений."
    )
    user_prompt = f"Сообщение клиента: {message}\nВерни JSON."
    resp = call_openai_compatible(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=64,
        timeout_sec=10,
    )
    payload = _parse_json(resp.text)
    if not payload:
        return None
    intent = str(payload.get("intent") or "").strip().lower()
    if intent in {"greeting", "meta", "off_topic", "identity", "question", "unclear"}:
        return intent
    return None


def _heuristic_intent(message: str) -> str | None:
    text = message.strip().lower()
    if not text:
        return None
    if len(text) <= 32:
        if text.startswith(("привет", "здрав", "добрый", "доброе", "добрая")):
            return "greeting"
    if "как вас зовут" in text or "как тебя зовут" in text or "кто вы" in text:
        return "identity"
    return None


def route_non_rag(message: str, base_url: str | None = None, model: str | None = None) -> RouterDecision | None:
    text = message.strip()
    if not text:
        return None

    if not base_url or not model:
        return None

    try:
        heuristic = _heuristic_intent(message)
        if heuristic == "greeting":
            return RouterDecision(kind="greeting", response="Здравствуйте. Чем могу помочь?")
        if heuristic == "identity":
            return RouterDecision(
                kind="identity",
                response="Я голосовой помощник компании «Микро Лизинг». Чем могу помочь?",
            )
        intent = _classify_with_llm(message, base_url, model)
        if intent == "greeting":
            return RouterDecision(kind="greeting", response="Здравствуйте. Чем могу помочь?")
        if intent == "meta":
            return RouterDecision(
                kind="meta",
                response=(
                    "Я консультирую по услугам компании «Микро Лизинг». "
                    "Задайте, пожалуйста, конкретный вопрос, я помогу."
                ),
            )
        if intent == "off_topic":
            return RouterDecision(
                kind="offtopic",
                response=(
                    "Я могу консультировать только по услугам компании «Микро Лизинг». "
                    "Если у вас есть вопрос по лизингу, условиям, документам или оплате, я помогу."
                ),
            )
        if intent == "identity":
            return RouterDecision(
                kind="identity",
                response="Я голосовой помощник компании «Микро Лизинг». Чем могу помочь?",
            )
        if intent == "unclear":
            return RouterDecision(
                kind="unclear",
                response="Подскажите, пожалуйста, в чем именно ваш вопрос по лизингу?",
            )
        return None
    except Exception:
        return None
