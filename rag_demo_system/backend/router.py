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
        "Ты классификатор сообщений в чате поддержки компании «Микро Лизинг». "
        "Верни строго JSON с полем intent из списка: "
        "greeting, meta, off_topic, identity, question, unclear. "
        "identity = клиент спрашивает кто ТЫ (бот/помощник). "
        "question = любой вопрос о компании, её услугах, руководстве, владельце, офисах, условиях. "
        "Если сомневаешься между identity и question, выбирай question. "
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


def _looks_like_business_question(text: str, check_qmark: bool = True) -> bool:
    if check_qmark and "?" in text:
        return True
    markers = (
        "лизинг",
        "услов",
        "треб",
        "ставк",
        "аванс",
        "срок",
        "документ",
        "платеж",
        "калькуля",
        "авто",
        "машин",
        "транспорт",
        "груз",
        "оборуд",
        "недвиж",
        "ип",
        "юрлиц",
    )
    return any(m in text for m in markers)


def _heuristic_intent(message: str) -> str | None:
    text = message.strip().lower()
    if not text:
        return None
    if len(text) <= 32:
        if text.startswith(("привет", "здрав", "добрый", "доброе", "добрая")):
            if not _looks_like_business_question(text, check_qmark=True):
                return "greeting"
    if "как вас зовут" in text or "как тебя зовут" in text or "кто вы" in text or "кто ты" in text:
        if not _looks_like_business_question(text, check_qmark=False):
            return "identity"
    return None


def _fallback_response(intent: str) -> str:
    if intent == "greeting":
        return "Здравствуйте. Чем могу помочь?"
    if intent == "identity":
        return "Я виртуальный помощник компании «Микро Лизинг». Чем могу помочь?"
    if intent == "meta":
        return "Я консультирую по услугам компании «Микро Лизинг». Чем могу помочь?"
    if intent == "off_topic":
        return (
            "Я могу консультировать только по услугам компании «Микро Лизинг». "
            "Если у вас есть вопрос по лизингу, я помогу."
        )
    if intent == "unclear":
        return "Подскажите, пожалуйста, в чем именно ваш вопрос по лизингу?"
    return "Чем могу помочь?"


def _llm_router_response(intent: str, message: str, base_url: str, model: str) -> str:
    system_prompt = (
        "Ты Ксения, голосовая помощница компании «Микро Лизинг». "
        "Тон тёплый, уверенный, с лёгким деловым юмором. Не роботизированный. "
        "Не запрашивай согласие и не упоминай базу знаний. "
        "Используй женские формы (рада, готова, смогла). "
        "Если intent:\n"
        "- greeting: короткое приветствие + вопрос, чем помочь.\n"
        "- identity: коротко представься как Ксения из «Микро Лизинг» и спроси, чем помочь.\n"
        "- meta: объясни кратко, что консультируешь по лизингу компании.\n"
        "- off_topic: одно лёгкое замечание или мини-шутка в стиле «Анекдоты не моя сильная сторона, но условия лизинга — пожалуйста» или «Где живёт сервер не подскажу, а где офисы — с радостью», потом мягко верни к лизингу. Не используй формулировку «Я консультирую только по вопросам лизинга».\n"
        "- unclear: попроси уточнить вопрос по лизингу.\n"
        "Ответ 1-2 короткими предложениями."
    )
    user_prompt = f"intent: {intent}\nсообщение клиента: {message}\nОтветь кратко."
    resp = call_openai_compatible(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=120,
        timeout_sec=8,
    )
    return resp.text.strip()


def route_non_rag(message: str, base_url: str | None = None, model: str | None = None) -> RouterDecision | None:
    text = message.strip()
    if not text:
        return None

    try:
        heuristic = _heuristic_intent(message)
        intent = heuristic
        if not intent:
            if _looks_like_business_question(text.lower(), check_qmark=True):
                return None
            if not base_url or not model:
                return None
            intent = _classify_with_llm(message, base_url, model)
        if intent == "greeting":
            response = _fallback_response("greeting")
            if base_url and model:
                try:
                    response = _llm_router_response("greeting", message, base_url, model)
                except Exception:
                    response = _fallback_response("greeting")
            return RouterDecision(kind="greeting", response=response)
        if intent == "meta":
            response = _fallback_response("meta")
            if base_url and model:
                try:
                    response = _llm_router_response("meta", message, base_url, model)
                except Exception:
                    response = _fallback_response("meta")
            return RouterDecision(kind="meta", response=response)
        if intent == "off_topic":
            response = _fallback_response("off_topic")
            if base_url and model:
                try:
                    response = _llm_router_response("off_topic", message, base_url, model)
                except Exception:
                    response = _fallback_response("off_topic")
            return RouterDecision(kind="offtopic", response=response)
        if intent == "identity":
            response = _fallback_response("identity")
            if base_url and model:
                try:
                    response = _llm_router_response("identity", message, base_url, model)
                except Exception:
                    response = _fallback_response("identity")
            return RouterDecision(kind="identity", response=response)
        if intent == "unclear":
            response = _fallback_response("unclear")
            if base_url and model:
                try:
                    response = _llm_router_response("unclear", message, base_url, model)
                except Exception:
                    response = _fallback_response("unclear")
            return RouterDecision(kind="unclear", response=response)
        return None
    except Exception:
        return None
