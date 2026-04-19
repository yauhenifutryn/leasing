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

    # Fix 1.13 (2026-04-19) — age_years must be asked before term/prepaid.
    # Live call 743c1a0e exposed that when a б/у client reached a state
    # with {age_years, term_months, prepaid, type_schedule} all missing,
    # the clarify asked for term/prepaid/graph and skipped age. The client
    # then said "Два года" meaning a 2-year lease, and the classifier
    # assigned that both to term_months=24 AND age_years=2 — the exact
    # "Два года" ambiguity Section 2's cross-field validator will close.
    # Asking age first kills the collision because age gets captured on
    # its own turn, then term/prepaid/graph are asked cleanly afterward.
    # Fix 1.5 (2026-04-19) — the age_years branch originally lived below
    # term/prepaid; this priority bump supersedes 1.5's placement.
    if "age_years" in fields:
        return (
            "Сколько лет вашему транспорту? Для б/у техники это обязательный параметр."
        )

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


# Fallback USD->BYN rate if settings can't be loaded (unit tests, isolated
# imports). Production reads the real rate from settings.tools.usd_byn_rate
# via _get_usd_byn_rate() so readback matches whatever rate the DirectTool
# USD->BYN conversion actually applies (Codex review flagged the drift
# risk of hardcoding in this module: readback and calc could disagree
# if the setting ever changes).
_USD_BYN_RATE_FALLBACK = 3.0
_USD_BYN_RATE_CACHE: float | None = None


def _get_usd_byn_rate() -> float:
    """Return the live USD->BYN rate from settings, cached after first call.

    Lazy import so this module stays importable in unit tests without a
    loaded settings file. First call hits settings; subsequent calls are
    effectively free. The rate does not change during a process lifetime
    — it is read from env / config at startup.
    """
    global _USD_BYN_RATE_CACHE
    if _USD_BYN_RATE_CACHE is not None:
        return _USD_BYN_RATE_CACHE
    try:
        from .settings import load_settings  # lazy
        _USD_BYN_RATE_CACHE = float(load_settings().tools.usd_byn_rate)
    except Exception:  # noqa: BLE001 — settings may be absent in tests
        _USD_BYN_RATE_CACHE = _USD_BYN_RATE_FALLBACK
    return _USD_BYN_RATE_CACHE


def _format_cost_phrase(profile: Any) -> str:
    """Render the cost portion of the readback.

    Dual-disclosure rules for USD cost (Физическое лицо flow):

      1. Post-conversion (the DirectTool USD->BYN branch already ran and
         populated `profile.original_cost` + `profile.original_currency`,
         AND profile.currency is now BYN — the "BYN" guard added after
         Codex review prevents a misread on hybrid paths where the
         conversion hasn't actually advanced the profile):
         speak both amounts using the actual applied rate.
      2. Pre-conversion (classifier just captured cost=N, currency='USD'
         but the profile is not yet at CONFIRMED so DirectTool has not
         fired): speak both amounts using the rate from settings so the
         readback itself discloses the conversion the client is about to
         confirm. Without this, readback said bare "120000 USD" and the
         caller did not realise BYN figures would be used for the
         calculation (live call 2026-04-19, session 205add5a).

    TTS will convert the digits to Russian words via voice_adapters, e.g.
    "20000 долларов" -> "двадцать тысяч долларов".
    """
    orig_cur = getattr(profile, "original_currency", None)
    orig_cost = getattr(profile, "original_cost", None)

    # 1) Post-conversion readback. Cost has been converted to BYN; originals
    # on the profile. Defensive `profile.currency == "BYN"` guard: if for
    # any reason the conversion metadata is on the profile but cost is
    # still USD, skip this branch and let the pre-conversion or legacy
    # path handle rendering correctly.
    if (
        orig_cur == "USD"
        and orig_cost is not None
        and profile.cost
        and (profile.currency or "") == "BYN"
    ):
        # Fix 1.9 (Codex review) — compute rate from the actual applied
        # conversion (authoritative) and narrate it losslessly. Before,
        # int(round(rate)) for both math and narration caused fractional
        # rates (e.g. USD_BYN_RATE=3.25) to readback a BYN amount derived
        # from a 3x rate while saying "по курсу 3 к 1" — diverged from
        # the actual calculator conversion at 3.25x.
        rate_exact = (profile.cost / orig_cost) if orig_cost else _get_usd_byn_rate()
        return (
            f"стоимость {int(orig_cost)} долларов "
            f"(это {int(profile.cost)} белорусских рублей "
            f"по курсу {_fmt_rate(rate_exact)} к 1)"
        )

    # 2) Pre-conversion readback (Физ лицо + USD, DirectTool hasn't run yet).
    is_phys = (profile.client_type or "") == "Физическое лицо"
    if is_phys and (profile.currency or "") == "USD" and profile.cost:
        # Fix 1.9 — carry the exact settings rate through the arithmetic.
        # int(round(rate)) for math turned 3.25 into 3 and reported a
        # 360000 BYN equivalent for a 120000 USD quote while the calc
        # actually applied 390000. Use the exact float for BYN math and
        # format the rate losslessly for narration.
        rate_exact = _get_usd_byn_rate()
        usd = int(profile.cost)
        byn = int(round(usd * rate_exact))
        return (
            f"стоимость {usd} долларов "
            f"(это {byn} белорусских рублей "
            f"по курсу {_fmt_rate(rate_exact)} к 1)"
        )

    # Legacy single-currency path (BYN-only, or юрлицо with any currency).
    cost_str = f"{int(profile.cost)}" if profile.cost else "—"
    currency_str = profile.currency or ""
    return f"стоимость {cost_str} {currency_str}".rstrip()


def _age_noun(n: int) -> str:
    """Russian numeric agreement for "год": 1 год / 2-4 года / 5+ лет.

    Handles teens correctly (11-14 use "лет", not "год"/"года")."""
    last_two = n % 100
    last = n % 10
    if 10 <= last_two <= 20:
        return "лет"
    if last == 1:
        return "год"
    if 2 <= last <= 4:
        return "года"
    return "лет"


def build_readback_text(profile: Any) -> str:
    """Produce the readback confirmation string listing all calculator params."""
    subj = profile.subject or "предмет лизинга"
    cond = (
        "новый" if profile.condition_new == 1
        else "б/у" if profile.condition_new == 0
        else "—"
    )
    # Fix 1.11 (2026-04-19) — when condition_new=0, age_years is a required
    # calculator input. Live call 22028754 exposed that the readback
    # omitted it: client confirmed "Верно" on a parameter set they never
    # heard (age=5 was in the profile, never spoken). Audit-critical
    # because "no silent inputs to the calculator before confirmation".
    cond_phrase = cond
    if profile.condition_new == 0 and profile.age_years is not None:
        age_n = int(profile.age_years)
        cond_phrase = f"{cond}, возраст {age_n} {_age_noun(age_n)}"
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
        f"Давайте подтвердим параметры: {subj}, {cond_phrase}, {cost_phrase}, "
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
    # Fix 1.9 — narrate the rate losslessly instead of rounding to int; the
    # BYN amount shown (`params['cost']`) is already the actual calculator
    # input so fractional rates stay consistent across paths.
    conv_prefix = ""
    conv = result.get("currency_conversion") or {}
    if conv.get("from") == "USD" and conv.get("amount_from") is not None:
        rate_exact = conv.get("rate") or _get_usd_byn_rate()
        conv_prefix = (
            f"Стоимость {int(conv['amount_from'])} долларов "
            f"(это {int(params.get('cost', 0))} белорусских рублей "
            f"по курсу {_fmt_rate(rate_exact)} к 1). "
        )

    # Fix 1.8 (2026-04-19) — TTS mispronounces decimal monetary amounts.
    # Live call 674e3957: "Ежемесячный платёж: 536.55 USD" spoken as
    # "пятьсот тридцать шесть запятая пятьдесят пять" — awkward and
    # unclear. Round all monetary fields to integers at the renderer
    # boundary. Sub-unit precision is preserved in SMS (it goes through
    # calculator.format_sms_body and the client can read exact figures).
    # Percentages render as int when whole, else one decimal place.
    return (
        f"{conv_prefix}"
        f"Аванс {_fmt_pct(params.get('prepaid', 30))}%: "
        f"{_fmt_money(result.get('advance_sum'))} {currency}. "
        f"Ежемесячный платёж: {_fmt_money(result.get('payment_min'))} {currency}. "
        f"Выкупной: {_fmt_money(result.get('buyout_sum'))} {currency}. "
        f"Общая сумма: {_fmt_money(result.get('total'))} {currency}. "
        f"Удорожание: {_fmt_pct(result.get('increase_percent'))}%. "
        f"Срок: {result.get('num_payments', '?')} мес."
        f"{defaults_note}"
    )


def _fmt_money(v: Any) -> str:
    """Round a monetary value to nearest integer for TTS-friendly output.

    Numeric or numeric-string inputs become e.g. "536" from "536.55".
    Non-numeric / None inputs fall back to '?' so the renderer never
    emits a NoneType-stringified field into the spoken summary.
    """
    if v in (None, ""):
        return "?"
    try:
        return f"{int(round(float(v)))}"
    except (TypeError, ValueError):
        return "?"


def _fmt_pct(v: Any) -> str:
    """Format a percentage: integer form when whole, else one decimal."""
    if v in (None, ""):
        return "?"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "?"
    return f"{int(f)}" if f == int(f) else f"{f:.1f}"


def _fmt_rate(v: Any) -> str:
    """Format an FX rate losslessly for the spoken "по курсу X к 1" phrase.

    Whole rate -> "3"; fractional -> "3.25" / "3.5"; bad input -> "3" to
    match the legacy fallback. Fix 1.9 — before this helper, rates were
    int(round())-ed for narration while math sometimes used the exact
    float, so readback promised "3 к 1" while calc applied 3.25.
    """
    if v in (None, ""):
        return "3"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "3"
    if f == int(f):
        return f"{int(f)}"
    # Strip trailing zeros on decimals: 3.50 -> "3.5", 3.2500 -> "3.25".
    return f"{f:g}"


_FIELD_RU = {
    "term_months": "срок",
    "prepaid_pct": "аванс",
    "prepaid_amount": "сумма аванса",
    "type_schedule": "тип графика",
    "currency": "валюта",
    "cost": "стоимость",
    "condition_new": "состояние (новый/б/у)",
    "subject": "предмет лизинга",
    # Fix 1.12 (2026-04-19) — live call 743c1a0e: change-confirm read out
    # "Меняю age_years на 5" because age_years was missing from this map
    # and _value_ru fell back to the raw key name. Caller heard a Python
    # field name spoken out loud. Ship the Russian label.
    "age_years": "возраст",
    "age": "возраст",
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
