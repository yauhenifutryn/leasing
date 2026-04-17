"""Payment calculator tool that calls the Mikro Leasing calculator API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import ToolDefinition

logger = logging.getLogger(__name__)


class IncompleteProfileError(Exception):
    """Raised when calculator is invoked without all required profile fields."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        super().__init__(
            f"Калькулятор требует обязательные поля: {', '.join(self.missing)}"
        )


class UnsupportedCurrencyError(Exception):
    """Raised when a currency is not allowed for a client type (MVP: EUR/RUB for Физ лицо)."""

    def __init__(self, currency: str, client_type: str) -> None:
        self.currency = currency
        self.client_type = client_type
        super().__init__(
            f"Валюта {currency} не поддерживается для клиента '{client_type}'. "
            "Поддерживаются: BYN, USD (физлицо с конвертацией по курсу)."
        )


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
                    "Рассчитать график лизинговых платежей по предмету лизинга и стоимости. "
                    "ВЫЗЫВАЙ НЕМЕДЛЕННО когда клиент просит расчёт. Достаточно subject и cost. "
                    "Остальные параметры имеют умолчания: физлицо, BYN, аванс 30%, 36 мес., аннуитет. "
                    "После расчёта озвучь результат, перечисли умолчания, предложи изменить параметры и отправить график по СМС."
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
        """No defaults. All parameters must come from the confirmed ClientProfile.

        Returning {} instead of raising to preserve the old interface for any
        callers that still invoke defaults() defensively. The real guard is in
        execute() which raises IncompleteProfileError if any required field
        is missing.
        """
        return {}

    # ------------------------------------------------------------------
    # Execute (synchronous, called via asyncio.to_thread)
    # ------------------------------------------------------------------

    # Canonical subject names (API requires exact casing)
    _SUBJECT_MAP = {
        "легковой автомобиль": "Легковой автомобиль",
        "грузовой автомобиль": "Грузовой автомобиль",
        "спецтехника": "Спецтехника",
        "оборудование": "Оборудование",
        "недвижимость": "Недвижимость",
        "прочий транспорт": "Прочий транспорт",
    }

    # API only accepts these two client types; ИП maps to Юридическое лицо
    _CLIENT_TYPE_MAP = {
        "ип": "Юридическое лицо",
        "индивидуальный предприниматель": "Юридическое лицо",
        "физическое лицо": "Физическое лицо",
        "физлицо": "Физическое лицо",
        "юридическое лицо": "Юридическое лицо",
        "юрлицо": "Юридическое лицо",
    }

    def execute(self, params: dict[str, Any], session_context: dict[str, Any]) -> dict[str, Any]:
        # ── Validation: every required field must come from ClientProfile. ──
        REQUIRED = [
            "subject", "cost", "client_type", "currency",
            "condition_new", "term", "type_schedule",
        ]
        missing = [k for k in REQUIRED if params.get(k) in (None, "")]
        has_pct = params.get("prepaid") not in (None, "")  # legacy: 'prepaid' key
        has_pct = has_pct or params.get("prepaid_pct") not in (None, "")
        has_amount = params.get("prepaid_amount") not in (None, "")
        if not has_pct and not has_amount:
            missing.append("prepaid")
        if params.get("condition_new") == 0 and params.get("age") in (None, "") \
                and params.get("age_years") in (None, ""):
            missing.append("age")
        if missing:
            raise IncompleteProfileError(missing=missing)

        # ── Normalization ──
        _subj = params.get("subject", "")
        params["subject"] = self._SUBJECT_MAP.get(_subj.lower().strip(), _subj)
        _ct = params.get("client_type", "")
        if _ct:
            params["client_type"] = self._CLIENT_TYPE_MAP.get(_ct.lower().strip(), _ct)

        # ── Prepaid: accept pct or amount, derive pct for API. ──
        if params.get("prepaid") in (None, ""):
            if params.get("prepaid_pct") not in (None, ""):
                params["prepaid"] = float(params["prepaid_pct"])
            elif params.get("prepaid_amount") not in (None, ""):
                cost = float(params["cost"])
                amount = float(params["prepaid_amount"])
                if cost <= 0:
                    raise IncompleteProfileError(missing=["cost"])
                params["prepaid"] = round((amount / cost) * 100.0, 2)
        prepaid_pct = float(params["prepaid"])
        if prepaid_pct < 0 or prepaid_pct > 40:
            raise IncompleteProfileError(
                missing=[f"prepaid_pct_out_of_range:{prepaid_pct}"]
            )

        # Age may arrive as 'age' (legacy) or 'age_years' (new profile key)
        if params.get("age") in (None, "") and params.get("age_years") not in (None, ""):
            params["age"] = params["age_years"]

        filled = {
            "client_type": params["client_type"],
            "subject": params["subject"],
            "condition_new": params["condition_new"],
            "currency": params["currency"],
            "cost": params["cost"],
            "prepaid": prepaid_pct,
            "term": params["term"],
            "type_schedule": params["type_schedule"],
        }
        if "age" in params and params["age"] not in (None, ""):
            filled["age"] = params["age"]
        defaulted: list[str] = []  # no defaults ever

        api_params = dict(filled)

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

        advance_sum = advance.get("sum", 0)
        prepaid_pct_out = params.get("prepaid")
        return {
            "ok": True,
            "url": advance.get("URL", ""),
            "calculation_id": advance.get("id", ""),
            "advance_sum": advance_sum,
            "increase_factor": advance.get("increase_factor", 0),
            "increase_percent": advance.get("increase_percent", 0),
            "buyout_sum": buyout.get("sum", 0),
            "payments": payments,
            "payment_min": min(payment_sums) if payment_sums else 0,
            "payment_max": max(payment_sums) if payment_sums else 0,
            "num_payments": len(payments),
            "total": advance_sum + sum(payment_sums) + buyout.get("sum", 0),
            "prepaid_pct": prepaid_pct_out,
            "prepaid_amount": advance_sum,
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
