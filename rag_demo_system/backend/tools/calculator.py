"""Payment calculator tool that calls the Mikro Leasing calculator API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import ToolDefinition

logger = logging.getLogger(__name__)


class CalculatorTool(ToolDefinition):
    """Calculates leasing payments via the Mikro Leasing public API."""

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._token = token

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": (
                    "Рассчитать график лизинговых платежей. "
                    "Обязательные параметры: subject (предмет лизинга) и cost (стоимость). "
                    "Остальные параметры имеют значения по умолчанию."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Предмет лизинга, например 'Легковой автомобиль'.",
                        },
                        "cost": {
                            "type": "number",
                            "description": "Стоимость предмета лизинга в валюте.",
                        },
                        "client_type": {
                            "type": "string",
                            "description": "Тип клиента: 'Физическое лицо' или 'Юридическое лицо'.",
                        },
                        "condition_new": {
                            "type": "integer",
                            "description": "1 = новый, 0 = б/у.",
                        },
                        "age": {
                            "type": "integer",
                            "description": "Возраст предмета (лет). Обязателен при condition_new=0.",
                        },
                        "currency": {
                            "type": "string",
                            "description": "Валюта: BYN, USD, EUR.",
                        },
                        "prepaid": {
                            "type": "number",
                            "description": "Аванс в процентах (например, 30).",
                        },
                        "term": {
                            "type": "integer",
                            "description": "Срок лизинга в месяцах.",
                        },
                        "type_schedule": {
                            "type": "string",
                            "description": "Тип графика: '0' = аннуитетный, '1' = убывающий.",
                        },
                    },
                    "required": ["subject", "cost"],
                },
            },
        }

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def defaults(self) -> dict[str, Any]:
        return {
            "client_type": "Физическое лицо",
            "condition_new": 1,
            "currency": "BYN",
            "prepaid": 30,
            "term": 36,
            "type_schedule": "0",
        }

    # ------------------------------------------------------------------
    # Execute (synchronous, called via asyncio.to_thread)
    # ------------------------------------------------------------------

    def execute(self, params: dict[str, Any], session_context: dict[str, Any]) -> dict[str, Any]:
        filled, defaulted = self.fill_defaults(params)

        # Validate: used item requires age
        if filled.get("condition_new") == 0 and "age" not in params:
            return {
                "ok": False,
                "error": "Для б/у предмета необходимо указать возраст (age).",
                "params": filled,
                "defaulted": defaulted,
            }

        api_params = {
            "client_type": filled["client_type"],
            "subject": filled["subject"],
            "condition_new": filled["condition_new"],
            "currency": filled["currency"],
            "cost": filled["cost"],
            "prepaid": filled["prepaid"],
            "term": filled["term"],
            "type_schedule": filled["type_schedule"],
        }
        if "age" in filled:
            api_params["age"] = filled["age"]

        headers = {"Authorization": f"Bearer {self._token}"}
        url = f"{self._base_url}/1.0/calculate/"

        try:
            resp = httpx.get(url, params=api_params, headers=headers, timeout=15.0)
        except httpx.HTTPError as exc:
            logger.error("Calculator API request failed: %s", exc)
            return {
                "ok": False,
                "error": f"Ошибка при обращении к API калькулятора: {exc}",
                "params": filled,
                "defaulted": defaulted,
            }

        if resp.status_code == 404:
            return {
                "ok": False,
                "error": "По заданным параметрам условия лизинга не найдены.",
                "params": filled,
                "defaulted": defaulted,
            }

        if resp.status_code != 200:
            logger.error("Calculator API returned %s: %s", resp.status_code, resp.text)
            return {
                "ok": False,
                "error": f"API вернул код {resp.status_code}.",
                "params": filled,
                "defaulted": defaulted,
            }

        data = resp.json()
        return self._parse_response(data, filled, defaulted)

    # ------------------------------------------------------------------
    # Parse API response
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        data: dict[str, Any],
        params: dict[str, Any],
        defaulted: list[str],
    ) -> dict[str, Any]:
        advance = data.get("0", {})
        buyout = data.get("999", {})

        # Collect monthly payments (keys "1" .. "N", excluding "0" and "999")
        payments: list[dict[str, Any]] = []
        idx = 1
        while str(idx) in data:
            payments.append(data[str(idx)])
            idx += 1

        payment_sums = [p["sum"] for p in payments]

        return {
            "ok": True,
            "url": advance.get("URL", ""),
            "calculation_id": advance.get("id", ""),
            "advance_sum": advance.get("sum", 0),
            "increase_factor": advance.get("increase_factor", 0),
            "increase_percent": advance.get("increase_percent", 0),
            "buyout_sum": buyout.get("sum", 0),
            "payments": payments,
            "payment_min": min(payment_sums) if payment_sums else 0,
            "payment_max": max(payment_sums) if payment_sums else 0,
            "num_payments": len(payments),
            "total": advance.get("sum", 0) + sum(payment_sums) + buyout.get("sum", 0),
            "params": params,
            "defaulted": defaulted,
        }

    # ------------------------------------------------------------------
    # Format for voice
    # ------------------------------------------------------------------

    def format_voice_summary(self, result: dict[str, Any]) -> str:
        if not result.get("ok"):
            return result.get("error", "Ошибка расчёта.")

        p = result["params"]
        d = set(result.get("defaulted", []))

        def _mark(key: str, value: Any) -> str:
            return f"{value}*" if key in d else str(value)

        lines = [
            "Результат расчёта лизинга:",
            f"  Предмет: {p.get('subject', '?')}",
            f"  Стоимость: {p.get('cost', '?')} {_mark('currency', p.get('currency', 'BYN'))}",
            f"  Тип клиента: {_mark('client_type', p.get('client_type', '?'))}",
            f"  Состояние: {'новый' if p.get('condition_new', 1) == 1 else 'б/у'}"
            + (f" ({p.get('age')} лет)" if p.get("age") is not None else ""),
            f"  Аванс: {_mark('prepaid', p.get('prepaid', '?'))}%: {result['advance_sum']} {p.get('currency', 'BYN')}",
            f"  Срок: {_mark('term', p.get('term', '?'))} мес.",
        ]

        if result["payment_min"] == result["payment_max"]:
            lines.append(f"  Ежемесячный платёж: {result['payment_min']} {p.get('currency', 'BYN')}")
        else:
            lines.append(
                f"  Ежемесячный платёж: {result['payment_max']}...{result['payment_min']} {p.get('currency', 'BYN')}"
            )

        lines.extend([
            f"  Выкупной платёж: {result['buyout_sum']} {p.get('currency', 'BYN')}",
            f"  Итого: {result['total']:.2f} {p.get('currency', 'BYN')}",
            f"  Удорожание: {result['increase_percent']}%",
            f"  График платежей: {result['url']}",
            "",
            "* = значение по умолчанию",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Format for SMS
    # ------------------------------------------------------------------

    def format_sms_body(self, result: dict[str, Any]) -> str | None:
        if not result.get("ok"):
            return None

        p = result["params"]
        return (
            f"Микро Лизинг: расчёт лизинга\n"
            f"{p.get('subject', '?')}, {p.get('cost', '?')} {p.get('currency', 'BYN')}\n"
            f"Аванс {p.get('prepaid', '?')}%: {result['advance_sum']} {p.get('currency', 'BYN')}\n"
            f"Срок: {result['num_payments']} мес.\n"
            f"Удорожание: {result['increase_percent']}%\n"
            f"График платежей: {result['url']}\n"
            f"+375 17 322 77 00"
        )
