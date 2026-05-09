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
                "и что именно хотите в лизинг: легковой автомобиль, грузовой, "
                "оборудование или что-то ещё?"
            )
        if "client_type" in fields:
            return "Вы физическое или юридическое лицо?"
        return (
            "Что планируете брать в лизинг: легковой автомобиль, грузовой, "
            "спецтехнику, оборудование или что-то другое?"
        )

    if fields & {"cost", "currency", "condition_new"}:
        parts = []
        if "cost" in fields:
            parts.append("стоимость")
        if "currency" in fields:
            parts.append("валюта (BYN или USD)")
        if "condition_new" in fields:
            # Issue 3 (2026-04-25): client prefers "подержанный" wording over
            # "б/у". Both spellings still ground via _CONDITION_USED_CUE_RE,
            # so callers who say "б/у" are still understood; the bot just
            # phrases the question with the more natural word.
            parts.append("новый или подержанный")
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
        # B5 fix: pick the dative-case noun + matching possessive that
        # agree with the subject category's gender. Without this:
        #   - real-estate / equipment leases got "вашему транспорту"
        #     (wrong noun)
        #   - спецтехника got "вашему технике" (wrong possessive — техника
        #     is feminine, needs "вашей")
        # Tuple = (possessive, dative_noun).
        _subject_age_phrase = {
            "Легковой автомобиль":  ("вашему", "транспорту"),
            "Грузовой автомобиль":  ("вашему", "транспорту"),
            "Прочий транспорт":     ("вашему", "транспорту"),
            "Спецтехника":          ("вашей",  "технике"),
            "Оборудование":         ("вашему", "оборудованию"),
            "Недвижимость":         ("вашему", "объекту"),
        }
        _subject = getattr(profile, "subject", None)
        _poss, _noun = _subject_age_phrase.get(
            _subject or "", ("вашему", "транспорту")
        )
        return (
            f"Сколько лет {_poss} {_noun}? "
            f"Для подержанной техники это обязательный параметр."
        )

    if fields & {"term_months", "prepaid", "type_schedule"}:
        parts = []
        if "term_months" in fields:
            parts.append("срок (от 12 до 84 месяцев)")
        if "prepaid" in fields:
            parts.append("аванс (от 0 до 40 процентов)")
        if "type_schedule" in fields:
            # Bug 24: lay phrasing instead of banking jargon. Clients ask
            # "что такое аннуитет?" routinely — see Stanislav 15:08:23 and
            # Valery 15:29:01 in the 2026-04-29 call set. Input grounding
            # still accepts аннуитет/линейный; output never speaks them.
            parts.append(
                "график удобнее равными платежами или с уменьшением суммы "
                "к концу срока"
            )
        return "Подскажите " + ", ".join(parts) + "."

    return "Уточните параметры расчёта, пожалуйста."


# Foreign-currency → BYN rate sourcing — live from National Bank of
# Belarus public API with TTL cache + graceful fallback to a static
# value. NBRB publishes the official daily rate at:
#   https://api.nbrb.by/exrates/rates/{CURRENCY}?parammode=2
# Supported codes here: USD, EUR, RUB. BYN is identity (1.0). Other
# codes are accepted by NBRB (CNY, GBP, PLN, etc.) but the dispatcher
# layer drifts those to BYN before they reach calc, so per-currency
# lookups stay scoped to the four-currency calculator surface.
# Hard 1.5s timeout keeps cold-path classifier latency safe; subsequent
# turns within the hour-long TTL hit the cache and pay nothing.
_USD_BYN_RATE_FALLBACK = 3.0
_USD_BYN_RATE_TTL_SECONDS = 3600.0  # 1 hour
_NBRB_TIMEOUT = 1.5
_NBRB_BASE_URL = "https://api.nbrb.by/exrates/rates"

# Per-currency cache: code → (rate, monotonic_ts). EUR's fetch must not
# poison the USD cache and vice versa, so we key by currency code.
_NBRB_RATE_CACHE: dict[str, tuple[float, float]] = {}

# Legacy single-float cache, retained for backward-compat with tests
# that reset state directly on `pp._USD_BYN_RATE_CACHE`. Kept as a
# mirror of `_NBRB_RATE_CACHE["USD"]` so old call sites keep working.
_USD_BYN_RATE_CACHE: float | None = None
_USD_BYN_RATE_CACHE_TS: float | None = None


def _fetch_nbrb_rate(currency: str) -> float | None:
    """Pull the live `{currency}/BYN` rate from NBRB. Return None on any
    failure (network, parse, missing field) so the caller can fall back.
    The 1.5s timeout caps cold-start cost; in practice NBRB responds in
    50-200ms.
    """
    try:
        import json as _json
        import urllib.request as _urlreq
        url = f"{_NBRB_BASE_URL}/{currency}?parammode=2"
        req = _urlreq.Request(
            url,
            headers={"User-Agent": "leasing-voice-bot/1.0"},
        )
        with _urlreq.urlopen(req, timeout=_NBRB_TIMEOUT) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        rate = data.get("Cur_OfficialRate")
        scale = data.get("Cur_Scale", 1) or 1
        if rate is None:
            return None
        return float(rate) / float(scale)
    except Exception:  # noqa: BLE001 — any failure → fallback
        return None


def _fetch_nbrb_usd_byn_rate() -> float | None:
    """Backward-compat wrapper. New code should call `_fetch_nbrb_rate`."""
    return _fetch_nbrb_rate("USD")


def _get_nbrb_rate(currency: str = "USD") -> float:
    """Return the current `{currency}/BYN` rate.

    Order of preference:
      1. BYN → 1.0 (identity, no API call).
      2. Live NBRB rate (cached for _USD_BYN_RATE_TTL_SECONDS).
      3. Last successfully cached value, even if stale.
      4. For USD: static settings.tools.usd_byn_rate (config / env).
      5. _USD_BYN_RATE_FALLBACK constant.

    The same helper drives BOTH the calc-time conversion in
    turn_dispatcher and the spoken "по курсу X к 1" disclosure in
    render_calc_result, keeping them in lockstep.
    """
    global _USD_BYN_RATE_CACHE, _USD_BYN_RATE_CACHE_TS
    import time as _time

    if currency == "BYN":
        return 1.0

    now = _time.monotonic()
    # Legacy backdoor: tests (and a few production paths) monkeypatch
    # `_USD_BYN_RATE_CACHE` directly to inject a USD rate. Honor that
    # override before consulting the per-currency dict cache.
    if currency == "USD" and _USD_BYN_RATE_CACHE is not None:
        return _USD_BYN_RATE_CACHE
    cached = _NBRB_RATE_CACHE.get(currency)
    if cached is not None and (now - cached[1]) < _USD_BYN_RATE_TTL_SECONDS:
        return cached[0]

    fresh = _fetch_nbrb_rate(currency)
    if fresh is not None:
        _NBRB_RATE_CACHE[currency] = (fresh, now)
        if currency == "USD":
            _USD_BYN_RATE_CACHE = fresh
            _USD_BYN_RATE_CACHE_TS = now
        return fresh

    # NBRB failed. Prefer the previous (stale) cached value over the
    # static fallback — last-known-good is closer to truth than 3.0.
    if cached is not None:
        return cached[0]

    if currency == "USD":
        try:
            from .settings import load_settings  # lazy
            fallback = float(load_settings().tools.usd_byn_rate)
        except Exception:  # noqa: BLE001 — settings may be absent in tests
            fallback = _USD_BYN_RATE_FALLBACK
        _NBRB_RATE_CACHE["USD"] = (fallback, now)
        _USD_BYN_RATE_CACHE = fallback
        _USD_BYN_RATE_CACHE_TS = now
        return fallback

    # No settings fallback for non-USD; use the static constant.
    _NBRB_RATE_CACHE[currency] = (_USD_BYN_RATE_FALLBACK, now)
    return _USD_BYN_RATE_FALLBACK


def _get_usd_byn_rate() -> float:
    """Backward-compat wrapper. New code should call `_get_nbrb_rate`."""
    return _get_nbrb_rate("USD")


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
        else "подержанный" if profile.condition_new == 0
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
    # Bug 24: lay phrasing in the spoken readback. The label is dropped
    # from "график X" to a self-contained noun phrase so it slots into
    # "график равные платежи" / "график с уменьшением суммы к концу срока"
    # without sounding awkward when concatenated below.
    sched = (
        "равные платежи" if profile.type_schedule == "0"
        else "с уменьшением суммы к концу срока" if profile.type_schedule == "1"
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


def render_calc_result(result: dict[str, Any], detailed: bool = False) -> str:
    """Render the post-calculator voice summary deterministically.

    Fix 1.1 (2026-04-19) — extracted from the inline f-string that lived in
    app.py's direct-call presentation block. All monetary numbers come
    straight from the calculator result; the LLM downstream is asked only
    to paraphrase tone, never to synthesise figures. A companion
    `[deterministic_readback]` log marker is emitted by the caller so
    session_analyzer can confirm this path drove the spoken result.

    Bug 25 (ANALYSIS.md §8) — default form is terse (4 values: cost,
    term, prepaid pct, monthly payment) plus a follow-up offer. Bug 8
    fix (2026-05-04) split the offer: terse form asks only about detail,
    detailed form asks only about SMS. The advanced breakdown (выкупной
    / общая сумма / удорожание) renders only when `detailed=True`,
    which apply_turn signals via EmitCalcDetail when the caller asks
    "подробнее" / "полный расчёт" / "удорожание". This mirrors Just AI's
    terse readback pattern (clip 6:03-6:25): the bot leads with the
    headline figure and the caller can drill in on demand.

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
            # Bug 24: lay phrasing.
            parts.append("график равными платежами (по умолчанию)")
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

    cost_str = _fmt_money(params.get("cost"))
    term_str = f"{result.get('num_payments', '?')}"
    prepaid_pct_str = _fmt_pct(params.get("prepaid", 30))

    # Live transcript 2026-05-08 client feedback: убывающий schedule
    # rendered as "Ежемесячный платёж — X" (a single number) is misleading
    # — payments decrease month-over-month. For linear schedules we
    # narrate the AVERAGE payment + an explicit shape hint, mirroring
    # Just AI's pattern. payment_min == payment_max is the precise
    # annuity signal (calculator returns equal values for аннуитет).
    pmt_min = result.get("payment_min")
    pmt_max = result.get("payment_max", pmt_min)
    is_linear = (
        pmt_min is not None
        and pmt_max is not None
        and pmt_min != pmt_max
    )
    if is_linear:
        avg_str = _fmt_money((float(pmt_max) + float(pmt_min)) / 2.0)
        payment_phrase = (
            f"Средний платёж — {avg_str} {currency}. "
            f"В начале платежи больше, к концу срока — меньше."
        )
    else:
        monthly_str = _fmt_money(pmt_min)
        payment_phrase = f"Ежемесячный платёж — {monthly_str} {currency}."

    # Terse 4-value headline (Bug 25). When `conv_prefix` already
    # narrated the cost in USD, omit it here to avoid a stutter
    # ("Стоимость 20000 долларов ... Стоимость 60000 BYN").
    if conv_prefix:
        head = (
            f"Срок {term_str} месяцев, аванс {prepaid_pct_str} процентов. "
            f"{payment_phrase}"
        )
    else:
        head = (
            f"Стоимость {cost_str} {currency}, "
            f"срок {term_str} месяцев, "
            f"аванс {prepaid_pct_str} процентов. "
            f"{payment_phrase}"
        )

    detail_block = ""
    if detailed:
        # Fix 1.8 — round monetary fields to integers; percentages stay
        # decimal when needed. Issue #5 — spell out "месяцев". Issue #3
        # — the deterministic offer line lives at the end of the terse
        # path; on the detail path the caller has already heard the
        # offer, so we close with a softer "что-нибудь ещё?".
        detail_block = (
            f" Выкупной: {_fmt_money(result.get('buyout_sum'))} {currency}. "
            f"Общая сумма: {_fmt_money(result.get('total'))} {currency}. "
            f"Удорожание: {_fmt_pct(result.get('increase_percent'))}%."
        )

    # Bug 8 fix (2026-05-04) — sequential offers, never combined.
    # Pre-fix: a single combined "Хотите подробный расчёт ИЛИ отправить
    # график по СМС?" question made bare "давай" ambiguous in chat — the
    # classifier returned generic CONFIRM, the LLM fallback generated
    # "отправим..." text but the dispatcher never fired send_sms.
    # Post-fix: ask one thing at a time, matching the natural voice flow.
    # First post-calc turn → detail offer; once detail is delivered → SMS
    # offer. Either offer accepts a bare "давай" unambiguously: detail
    # routes through EmitCalcDetail; SMS routes through detect_sms_intent
    # affirmation-after-calc path (sms_intent.py).
    if detailed:
        offer = " Отправить график платежей по СМС?"
    else:
        offer = " Хотите услышать подробный расчёт?"

    return f"{conv_prefix}{head}{detail_block}{defaults_note}{offer}"


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
    "condition_new": "состояние (новый/подержанный)",
    "subject": "предмет лизинга",
    # Fix 1.12 (2026-04-19) — live call 743c1a0e: change-confirm read out
    # "Меняю age_years на 5" because age_years was missing from this map
    # and _value_ru fell back to the raw key name. Caller heard a Python
    # field name spoken out loud. Ship the Russian label.
    "age_years": "возраст",
    "age": "возраст",
}


# B3 fix: accusative case (винительный падеж) for "Меняю ___ на …".
# Only feminine nominatives ending in -а need a switch to -у; masculine
# inanimate (срок/аванс/тип/предмет/возраст) and feminine -ь
# (стоимость) already coincide with nominative.
_FIELD_RU_ACCUSATIVE = {
    "currency": "валюту",
    "prepaid_amount": "сумму аванса",
}


# Human-readable translations for enum values in change-confirm prompts.
# Calculator API uses internal codes ("0"/"1" for type_schedule, etc.); we
# never say those codes to the caller.
_VALUE_RU: dict[str, dict[Any, str]] = {
    # Bug 24: change-confirm output uses lay phrasing. Input grounding
    # still accepts the legacy banking terms (аннуитет / линейный); the
    # output side speaks "равными платежами" / "с уменьшением суммы к
    # концу срока" because clients consistently asked "что такое
    # аннуитет?" on live calls (Stanislav 15:08:23, Valery 15:29:01).
    "type_schedule": {
        "0": "равными платежами",
        0: "равными платежами",
        "1": "с уменьшением суммы к концу срока",
        1: "с уменьшением суммы к концу срока",
    },
    "condition_new": {
        "0": "подержанный",
        0: "подержанный",
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
            new_value = vals.get("new") if isinstance(vals, dict) else vals
            # B2 fix: skip nulled fields (e.g. age_years cleared as a
            # side-effect of subject flip). Otherwise the renderer
            # produces "и возраст на ," — empty after "на".
            if new_value in (None, ""):
                continue
            # B3: prefer accusative form for fields after "Меняю".
            field_ru = _FIELD_RU_ACCUSATIVE.get(
                field_name, _FIELD_RU.get(field_name, field_name)
            )
            new_value_ru = _value_ru(field_name, new_value)
            if not new_value_ru:
                continue
            parts.append(f"{field_ru} на {new_value_ru}")
        if not parts:
            # Every staged change had a null new value — degenerate.
            # Don't emit "Меняю , остальное оставляю"; ask for clarity.
            return "Уточните, пожалуйста, что именно нужно изменить."
        if len(parts) == 1:
            return f"Меняю {parts[0]}, остальное оставляю. Всё верно?"
        # Join with "," except for the last which gets " и "
        head = ", ".join(parts[:-1])
        body = f"{head} и {parts[-1]}"
        return f"Меняю {body}, остальное оставляю. Всё верно?"
    # Legacy single-field.
    field_name = pending_change.get("field", "")
    field_ru = _FIELD_RU_ACCUSATIVE.get(
        field_name, _FIELD_RU.get(field_name, field_name)
    )
    new_value = pending_change.get("new_value")
    new_value_ru = _value_ru(field_name, new_value)
    return f"Меняю {field_ru} на {new_value_ru}, остальное оставляю. Всё верно?"
