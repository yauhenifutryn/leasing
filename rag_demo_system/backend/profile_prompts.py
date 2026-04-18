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
                "Подскажите, пожалуйста, тип клиента: физическое или юридическое лицо, "
                "и что именно хотите в лизинг: машину, оборудование или что-то ещё?"
            )
        if "client_type" in fields:
            return "Вы физическое или юридическое лицо?"
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


def _format_cost_phrase(profile: Any) -> str:
    """Render the cost portion of the readback.

    Fix 1.2 (2026-04-19): when the profile carries `original_currency="USD"`
    (set by the direct-call USD -> BYN conversion path), speak both amounts
    so the client can reconcile what they said with what the calculator saw.
    TTS will convert the digits to Russian words via voice_adapters, e.g.
    "20000 долларов" -> "двадцать тысяч долларов".
    """
    orig_cur = getattr(profile, "original_currency", None)
    orig_cost = getattr(profile, "original_cost", None)
    cost_str = f"{int(profile.cost)}" if profile.cost else "—"
    if orig_cur == "USD" and orig_cost is not None and profile.cost:
        rate = int(round(profile.cost / orig_cost)) if orig_cost else 3
        return (
            f"стоимость {int(orig_cost)} долларов "
            f"(это {cost_str} белорусских рублей по курсу {rate} к 1)"
        )
    currency_str = profile.currency or ""
    return f"стоимость {cost_str} {currency_str}".rstrip()


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
    cost_phrase = _format_cost_phrase(profile)
    client_type_str = profile.client_type or ""
    term_str = str(profile.term_months) if profile.term_months is not None else "—"
    return (
        f"Давайте подтвердим параметры: {subj}, {cond}, {cost_phrase}, "
        f"{client_type_str}, срок {term_str} месяцев, аванс {prepaid}, "
        f"график {sched}. Всё верно?"
    )


def render_calc_result(result: dict[str, Any]) -> str:
    """Render the post-calculator voice summary deterministically.

    Fix 1.1 (2026-04-19) — extracted from the inline f-string that lived in
    app.py's direct-call presentation block. All monetary numbers come
    straight from the calculator result; the LLM downstream is asked only
    to paraphrase tone, never to synthesise figures. A companion
    `[deterministic_readback]` log marker is emitted by the caller so
    session_analyzer can confirm this path drove the spoken result.

    When the direct-call path carried a USD->BYN conversion (Fix 1.2), the
    result dict contains `currency_conversion` and the summary is prefixed
    with "Стоимость N долларов (это M белорусских рублей по курсу X к 1)."
    so the client reconciles their quoted USD against the converted BYN.
    """
    params = result.get("params", {}) or {}
    currency = params.get("currency", "BYN")

    # Defaults note (for fields the calculator stamped with stand-ins).
    defaulted = set(result.get("defaulted", []) or [])
    defaults_note = ""
    if defaulted:
        parts: list[str] = []
        if "prepaid" in defaulted:
            parts.append(f"аванс {params.get('prepaid', 30)}% (по умолчанию)")
        if "term" in defaulted:
            parts.append(f"срок {params.get('term', 36)} мес. (по умолчанию)")
        if "client_type" in defaulted:
            parts.append(f"тип клиента: {params.get('client_type', '?')} (по умолчанию)")
        if "type_schedule" in defaulted:
            parts.append("аннуитетный график (по умолчанию)")
        if parts:
            defaults_note = f" Параметры по умолчанию: {', '.join(parts)}."

    # Fix 1.2 prefix — disclose original USD amount alongside BYN equivalent.
    conv_prefix = ""
    conv = result.get("currency_conversion") or {}
    if conv.get("from") == "USD" and conv.get("amount_from") is not None:
        rate_disp = int(round(conv.get("rate") or 3))
        conv_prefix = (
            f"Стоимость {int(conv['amount_from'])} долларов "
            f"(это {int(params.get('cost', 0))} белорусских рублей "
            f"по курсу {rate_disp} к 1). "
        )

    return (
        f"{conv_prefix}"
        f"Аванс {params.get('prepaid', 30)}%: {result.get('advance_sum', '?')} {currency}. "
        f"Ежемесячный платёж: {result.get('payment_min', '?')} {currency}. "
        f"Выкупной: {result.get('buyout_sum', '?')} {currency}. "
        f"Общая сумма: {result.get('total', '?')} {currency}. "
        f"Удорожание: {result.get('increase_percent', '?')}%. "
        f"Срок: {result.get('num_payments', '?')} мес."
        f"{defaults_note}"
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


# Human-readable translations for enum values in change-confirm prompts.
# Calculator API uses internal codes ("0"/"1" for type_schedule, etc.); we
# never say those codes to the caller.
_VALUE_RU: dict[str, dict[Any, str]] = {
    "type_schedule": {
        "0": "аннуитетный",
        0: "аннуитетный",
        "1": "линейный",
        1: "линейный",
    },
    "condition_new": {
        "0": "б/у",
        0: "б/у",
        "1": "новый",
        1: "новый",
    },
    "client_type": {
        "Физическое лицо": "физическое лицо",
        "ИП": "индивидуальный предприниматель",
        "Юридическое лицо": "юридическое лицо",
    },
    "currency": {
        "BYN": "белорусские рубли",
        "USD": "доллары США",
        "EUR": "евро",
        "RUB": "российские рубли",
    },
}


def _value_ru(field_name: str, new_value: Any) -> str:
    """Return a human-readable label for a calculator field value.

    For mapped enum fields, returns the Russian label; for unmapped fields
    (cost, term_months, prepaid_pct), returns the raw value cast to str.
    """
    mapped = _VALUE_RU.get(field_name, {}).get(new_value)
    if mapped is not None:
        return mapped
    if new_value is None or new_value == "":
        return ""
    return str(new_value)


def build_change_confirm_text(pending_change: dict[str, Any] | None) -> str:
    """Produce the change-confirm prompt.

    Supports both shapes:
      * Single-field legacy: {"field": str, "new_value": Any}
      * Multi-field (Fix 28): {"changes": {field_name: {"old": Any, "new": Any}, ...}}

    For multi-field, the prompt lists every change ("Меняю X на A и Y на B,
    остальное оставляю. Всё верно?") so the caller hears exactly what will
    be applied and doesn't see the bot silently tack on extra edits.
    """
    if not pending_change:
        return "Уточните, пожалуйста, что именно нужно изменить."
    # Explicit multi-field shape: empty `changes` dict means "the classifier
    # flagged a change intent but gave us no new value" — prompt clarification.
    if "changes" in pending_change and not pending_change["changes"]:
        return "Уточните, пожалуйста, что именно нужно изменить."
    _changes = pending_change.get("changes")
    if isinstance(_changes, dict) and _changes:
        parts: list[str] = []
        for field_name, vals in _changes.items():
            field_ru = _FIELD_RU.get(field_name, field_name)
            new_value = vals.get("new") if isinstance(vals, dict) else vals
            new_value_ru = _value_ru(field_name, new_value)
            parts.append(f"{field_ru} на {new_value_ru}")
        if len(parts) == 1:
            return f"Меняю {parts[0]}, остальное оставляю. Всё верно?"
        # Join with "," except for the last which gets " и "
        head = ", ".join(parts[:-1])
        body = f"{head} и {parts[-1]}"
        return f"Меняю {body}, остальное оставляю. Всё верно?"
    # Legacy single-field.
    field_name = pending_change.get("field", "")
    field_ru = _FIELD_RU.get(field_name, field_name)
    new_value = pending_change.get("new_value")
    new_value_ru = _value_ru(field_name, new_value)
    return f"Меняю {field_ru} на {new_value_ru}, остальное оставляю. Всё верно?"
