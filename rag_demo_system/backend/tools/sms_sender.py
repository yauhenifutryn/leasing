"""SMS sender tool that calls the sms-assistent.by API."""

from __future__ import annotations

import re
from typing import Any

import httpx

from .base import ToolDefinition

_SMS_API_URL = "https://userarea.sms-assistent.by/api/v1/send_sms/plain"

_ERROR_CODES = {
    "-1": "Недостаточно средств на балансе SMS-сервиса",
    "-2": "Ошибка авторизации SMS-сервиса",
    "-10": "SMS API не активирован",
    "-13": "Трафик заблокирован",
}

_PHONE_RE = re.compile(r"^375\d{9}$")


class SmsSenderTool(ToolDefinition):
    def __init__(self, login: str, password: str, sender: str) -> None:
        self._login = login
        self._password = password
        self._sender = sender

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "send_sms",
                "description": (
                    "Отправить СМС клиенту. ВЫЗЫВАЙ когда клиент согласился получить СМС. "
                    "Используй номер 375291224557 и текст из результата расчёта."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone": {
                            "type": "string",
                            "description": "Номер телефона клиента в формате 375XXXXXXXXX (12 цифр без +)",
                        },
                        "message": {
                            "type": "string",
                            "description": "Текст СМС сообщения",
                        },
                    },
                    "required": ["phone", "message"],
                },
            },
        }

    def defaults(self) -> dict[str, Any]:
        return {}

    def execute(self, params: dict[str, Any], session_context: dict[str, Any]) -> dict[str, Any]:
        phone = re.sub(r"[^\d]", "", params.get("phone", ""))
        if phone.startswith("+"):
            phone = phone[1:]
        message = params.get("message", "")

        if not _PHONE_RE.match(phone):
            return {"ok": False, "error": f"Некорректный номер телефона: {phone}. Ожидается формат 375XXXXXXXXX."}
        if not message:
            return {"ok": False, "error": "Текст сообщения пустой."}

        try:
            resp = httpx.get(
                _SMS_API_URL,
                params={
                    "user": self._login,
                    "password": self._password,
                    "recipient": phone,
                    "message": message,
                    "sender": self._sender,
                },
                timeout=10.0,
            )
        except httpx.TimeoutException:
            return {"ok": False, "error": "SMS-сервис временно недоступен."}
        except httpx.RequestError as exc:
            return {"ok": False, "error": f"Ошибка подключения к SMS-сервису: {exc}"}

        body = resp.text.strip()
        if body.lstrip("-").isdigit() and int(body) > 0:
            return {"ok": True, "message_id": body}

        error_msg = _ERROR_CODES.get(body, f"Ошибка SMS-сервиса (код {body})")
        return {"ok": False, "error": error_msg}

    def format_voice_summary(self, result: dict[str, Any]) -> str:
        if result.get("ok"):
            return (
                "СМС успешно отправлено. "
                "Скажи клиенту КОРОТКО: 'Отправила! Чем ещё могу помочь?' "
                "НЕ повторяй результаты расчёта. НЕ прощайся."
            )
        return f"Не удалось отправить СМС: {result.get('error', 'неизвестная ошибка')}. Сообщи клиенту об ошибке."
