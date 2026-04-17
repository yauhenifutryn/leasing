"""Prompt builders for the ClientProfile state-machine gate.

Used by the orchestrator (app.py) to generate clarification questions,
readback confirmations, and change-confirm prompts. Keeps the prompt text
out of the orchestrator logic so it can be tuned without touching control flow.
"""

from __future__ import annotations

from typing import Any


def build_clarification_prompt(fields: set[str], profile: Any) -> str:
    """Ask for missing fields in natural grouped chunks."""
    if fields & {"client_type", "subject"}:
        if "client_type" in fields and "subject" in fields:
            return (
                "Подскажите, пожалуйста, тип клиента, физическое лицо, ИП или юридическое, "
                "и что именно хотите в лизинг: машину, оборудование или что-то ещё?"
            )
        if "client_type" in fields:
            return "Вы физическое лицо, ИП или юридическое лицо?"
        return "Что планируете брать в лизинг, машину, оборудование, спецтехнику или что-то другое?"

    if fields & {"cost", "currency", "condition_new"}:
        parts = []
        if "cost" in fields:
            parts.append("стоимость")
        if "currency" in fields:
            parts.append("валюта (BYN или USD)")
        if "condition_new" in fields:
            parts.append("новый или б/у")
        return "Уточните, пожалуйста, " + ", ".join(parts) + "."

    if fields & {"term_months", "prepaid", "type_schedule"}:
        parts = []
        if "term_months" in fields:
            parts.append("срок (от 12 до 84 месяцев)")
        if "prepaid" in fields:
            parts.append("аванс (от 0 до 40 процентов)")
        if "type_schedule" in fields:
            parts.append("тип графика (аннуитет или линейный)")
        return "Подскажите " + ", ".join(parts) + "."

    return "Уточните параметры расчёта, пожалуйста."


def build_readback_text(profile: Any) -> str:
    """Produce the readback confirmation string listing all calculator params."""
    subj = profile.subject or "предмет лизинга"
    cond = (
        "новый" if profile.condition_new == 1
        else "б/у" if profile.condition_new == 0
        else "—"
    )
    if profile.prepaid_pct is not None:
        prepaid = f"{int(profile.prepaid_pct)}%"
    elif profile.prepaid_amount is not None:
        prepaid = f"{int(profile.prepaid_amount)} {profile.currency or ''}".strip()
    else:
        prepaid = "—"
    sched = (
        "аннуитет" if profile.type_schedule == "0"
        else "линейный" if profile.type_schedule == "1"
        else "—"
    )
    cost_str = f"{int(profile.cost)}" if profile.cost else "—"
    currency_str = profile.currency or ""
    client_type_str = profile.client_type or ""
    term_str = str(profile.term_months) if profile.term_months is not None else "—"
    return (
        f"Давайте подтвердим параметры: {subj}, {cond}, стоимость {cost_str} {currency_str}, "
        f"{client_type_str}, срок {term_str} месяцев, аванс {prepaid}, "
        f"график {sched}. Всё верно?"
    )


_FIELD_RU = {
    "term_months": "срок",
    "prepaid_pct": "аванс",
    "prepaid_amount": "сумма аванса",
    "type_schedule": "тип графика",
    "currency": "валюта",
    "cost": "стоимость",
    "condition_new": "состояние (новый/б/у)",
    "subject": "предмет лизинга",
}


def build_change_confirm_text(pending_change: dict[str, Any] | None) -> str:
    """Produce the single-field change-confirm prompt."""
    if not pending_change:
        return "Уточните, пожалуйста, что именно нужно изменить."
    field_name = pending_change.get("field", "")
    field_ru = _FIELD_RU.get(field_name, field_name)
    new_value = pending_change.get("new_value")
    return f"Меняю {field_ru} на {new_value}, остальное оставляю. Всё верно?"
