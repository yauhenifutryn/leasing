"""Utterance-level fallback grounding.

Extracts slot values directly from the user's utterance text when the
classifier omits them. Runs as a deterministic safety net for the
small Qwen3-4B classifier, which sometimes returns intent=RAG /
CONVERSATION with no slot extraction even though the utterance
clearly named a category.

Issue 7 (live call 77cfa127, 2026-04-25): user said "Я думаю взять
себе машину." — Qwen3-4B returned `intent: RAG` with no `subject`
field. classifier_schema's `_subject_value_grounded` operates on the
classifier's emitted VALUE, so it had nothing to ground. Profile
stayed `subj=-` and the orchestrator legitimately asked for subject
on the next turn (annoying re-ask).

Reuses the same regex tables from classifier_schema.py so the
fallback obeys identical category cues. Conservative by design:
returns None on ambiguous utterances.
"""
from __future__ import annotations

import re
from typing import Optional

# Mirror classifier_schema._SUBJECT_VALUE_CUES priority order: most-specific
# categories first so "грузовая машина" doesn't get caught by the bare-car
# fallback below.
_SPECIFIC_SUBJECT_CUES: list[tuple[str, re.Pattern[str]]] = [
    (
        "Грузовой автомобиль",
        re.compile(
            r"\b(грузов\w*|грузовик\w*|фур\w+|тягач\w*|самосвал\w*|"
            r"микроавтобус\w*|камаз|уаз)",
            re.IGNORECASE,
        ),
    ),
    (
        "Спецтехника",
        re.compile(
            r"\b(спецтехник\w*|погрузчик\w*|экскаватор\w*|бульдозер\w*|"
            r"кран\w*|каток\w*|трактор\w*|комбайн\w*)",
            re.IGNORECASE,
        ),
    ),
    (
        "Оборудование",
        re.compile(
            r"\b(оборудовани\w*|станк\w+|установк\w+)|\bлиния\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Недвижимость",
        re.compile(
            r"\b(недвижимост\w*|квартир\w+|здани\w+|помещени\w+|"
            r"склад\w*|офис\w*)",
            re.IGNORECASE,
        ),
    ),
    (
        "Прочий транспорт",
        re.compile(
            r"\b(автобус\w*|прицеп\w*|мотоцикл\w*|скутер\w*)",
            re.IGNORECASE,
        ),
    ),
    # "Легковой" specific cues run before the bare-car fallback.
    (
        "Легковой автомобиль",
        re.compile(
            r"\b("
            r"легков\w*|седан\w*|внедорожник\w*|кроссовер\w*|"
            r"bmw|mercedes|mercedes-benz|toyota|kia|hyundai|audi|volkswagen|vw|"
            r"lexus|mazda|renault|peugeot|ford|lada|skoda|fiat|chevrolet|"
            r"nissan|honda|"
            r"мерседес|тойот\w+|киа|хендай|ауди|фольксваген|лексус|мазд\w+|"
            r"фольцваген|рено|пежо|форд|лад\w+|шкод\w+|ниссан|хонд\w+"
            r")",
            re.IGNORECASE,
        ),
    ),
]

_GENERIC_CAR_RE = re.compile(r"\b(машин\w*|автомобил\w*|авто)\b", re.IGNORECASE)


# Issue 1 (live call 3d3e17b9, 2026-04-25): on terse single-slot replies
# like "4 года" / "5 лет" the small Qwen3-4B classifier inconsistently
# omits `age_years`. The orchestrator's deterministic clarify gate then
# skips (no patches changed) and the LLM falls back, hallucinates "учтён"
# without actually mutating the profile, and the bot loops asking for
# age again on the next turn.
#
# This regex matches the canonical "N лет / N года / N годов" answer
# shapes plus a few colloquial dash/abbreviation variants. Capture
# group 1 is the integer year count. The valid range mirrors
# `ClassifierOutput.age_years` (0-50): any larger number is rejected so
# stray numerals from term answers ("60 месяцев", "84") never grab a
# spurious age.
_AGE_YEARS_RE = re.compile(
    r"\b(\d{1,2})\s*(?:лет|года|год|годов|г\.?)\b",
    re.IGNORECASE,
)


def extract_age_years_from_utterance(utterance: str) -> Optional[int]:
    """Return the integer age-in-years from a 'N лет / N года' utterance, or None.

    Conservative: returns None when the utterance is empty, doesn't
    contain a year-shaped pattern, or the integer is out of the calc-
    eligible 0-50 range. Caller is expected to gate this on
    `profile.condition_new == 0 and profile.age_years is None` so a
    user answering a term question ("60 месяцев") doesn't accidentally
    fill age via an unrelated numeric.
    """
    if not utterance:
        return None
    m = _AGE_YEARS_RE.search(utterance)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except (TypeError, ValueError):
        return None
    if n < 0 or n > 50:
        return None
    return n


def extract_subject_from_utterance(utterance: str) -> Optional[str]:
    """Return the most likely subject value from the utterance, or None.

    Order:
      1. Specific category cues (Грузовой / Спецтехника / Оборудование /
         Недвижимость / Прочий транспорт / Легковой brand+category words).
         First match wins — they're mutually exclusive in practice.
      2. Bare-car fallback ("машина" / "автомобиль" / "авто") → Легковой,
         only when no specific cue fired.

    Returns None when nothing matches or the utterance is empty.
    """
    if not utterance:
        return None
    utt = utterance
    for value, pattern in _SPECIFIC_SUBJECT_CUES:
        if pattern.search(utt):
            return value
    if _GENERIC_CAR_RE.search(utt):
        return "Легковой автомобиль"
    return None


# ============================================================ client_type
# "Физическое лицо" / "Юридическое лицо" with the same canonical mapping
# `profile_hygiene._normalize_client_type` already enforces. The orchestrator
# wires this in alongside the other fallbacks; only fires when the
# classifier omitted client_type and profile.client_type is currently None.
_CLIENT_TYPE_PHYS_RE = re.compile(
    r"\b(физлиц\w*|физик\w*|физическ\w+|"
    r"я\s+физ\w*|как\s+физлиц\w*|"
    r"частн\w+\s+(?:лиц\w+|клиент\w*))\b",
    re.IGNORECASE,
)
_CLIENT_TYPE_LEGAL_RE = re.compile(
    r"\b(юрлиц\w*|юридическ\w+|"
    r"ооо|оао|зао|"
    r"организаци\w+|компани\w+|предприяти\w+|фирм\w+|"
    r"ип\b|ипэшник\w*|самозанят\w+|индивидуальн\w+\s+предпринимател\w+|"
    r"бизнес\w*|предпринимат\w+|"
    r"микробизнес\w*|малый\s+бизнес)\b",
    re.IGNORECASE,
)


def extract_client_type_from_utterance(utterance: str) -> Optional[str]:
    """Return canonical client_type ("Физическое лицо" / "Юридическое лицо")
    from the utterance, or None if neither cue fires or both fire."""
    if not utterance:
        return None
    has_phys = bool(_CLIENT_TYPE_PHYS_RE.search(utterance))
    has_legal = bool(_CLIENT_TYPE_LEGAL_RE.search(utterance))
    if has_phys and not has_legal:
        return "Физическое лицо"
    if has_legal and not has_phys:
        return "Юридическое лицо"
    return None


# ============================================================ condition_new
# Single-word slot-fill answers like "новый" / "подержанный" — when the
# classifier omits condition_new entirely (terse one-word replies are the
# regression-prone case for the small Qwen3-4B). Mirrors the cue tables in
# profile_hygiene._CONDITION_*_RE, including the double-д Whisper variant.
_CONDITION_USED_RE = re.compile(
    r"\b(подержан\w+|поддержан\w+|бывш\w+|"
    r"б/у|б-у|бу|бэу|"
    r"пробег\w*|"
    r"не\s+нов\w+|"
    r"стар(?:ый|ая|ое|ые))\b",
    re.IGNORECASE,
)
_CONDITION_NEW_RE = re.compile(r"\bнов\w+\b", re.IGNORECASE)
_CONDITION_NEW_NEG_RE = re.compile(
    r"\b(без\s+пробег\w*|нулев\w+\s+пробег\w*|без\s+износ\w*)\b",
    re.IGNORECASE,
)
_CONDITION_NOT_NEW_RE = re.compile(r"\bне\s+нов\w+\b", re.IGNORECASE)


def extract_condition_new_from_utterance(utterance: str) -> Optional[int]:
    """Return 0 (used) or 1 (new) — value-aware, mirrors the contradiction
    handling in profile_hygiene.has_field_signal('condition_new', ...).
    Returns None on ambiguous / contradictory utterances."""
    if not utterance:
        return None
    has_neg = bool(_CONDITION_NEW_NEG_RE.search(utterance))
    has_used_raw = bool(_CONDITION_USED_RE.search(utterance))
    has_used = has_used_raw and not has_neg
    has_new_raw = bool(_CONDITION_NEW_RE.search(utterance))
    has_not_new = bool(_CONDITION_NOT_NEW_RE.search(utterance))
    has_new = (has_new_raw and not has_not_new) or has_neg
    if has_used and not has_new:
        return 0
    if has_new and not has_used:
        return 1
    return None


# ============================================================ term_months
# "60 месяцев" → 60. "на 5 лет" / "пять лет" → 60 (years×12). Conservative
# range gate (12-84 months mirrors `ClientProfile` calc-eligible range)
# rejects stray numerics. The "N лет" branch overlaps with age extraction
# above — caller MUST gate on `profile.condition_new == 1 OR
# profile.age_years is not None` so age and term don't fight over the
# same regex match.
_TERM_MONTHS_RE = re.compile(
    r"\b(\d{1,3})\s*(?:месяц\w*|мес\.?)\b",
    re.IGNORECASE,
)
_TERM_YEARS_RE = re.compile(
    r"\b(\d{1,2})\s*(?:лет|года|год|годов)\b",
    re.IGNORECASE,
)


def extract_term_months_from_utterance(utterance: str) -> Optional[int]:
    """Return term in months from "N месяцев" / "N лет" / etc. Out-of-range
    values are rejected so a stray cost number doesn't poison term."""
    if not utterance:
        return None
    m = _TERM_MONTHS_RE.search(utterance)
    if m:
        try:
            n = int(m.group(1))
        except (TypeError, ValueError):
            return None
        if 12 <= n <= 84:
            return n
        return None
    y = _TERM_YEARS_RE.search(utterance)
    if y:
        try:
            yr = int(y.group(1))
        except (TypeError, ValueError):
            return None
        if 1 <= yr <= 7:
            return yr * 12
    return None


# ============================================================ prepaid
# "20 процентов" / "20 %" → prepaid_pct. "без аванса" / "ноль" / "нулевой
# аванс" → 0%. Whole-amount form (e.g. "20 тысяч аванса") is more
# ambiguous and intentionally NOT covered here — the calc API supports
# either pct or absolute amount, and falling back on absolute requires
# context (currency, cost) the orchestrator already has via cost capture.
_PREPAID_PCT_RE = re.compile(
    r"\b(\d{1,3})\s*(?:%|процент\w*|проц\.?)",
    re.IGNORECASE,
)
_PREPAID_ZERO_RE = re.compile(
    r"\b(без\s+аванс\w*|нулев\w+\s+аванс\w*|"
    r"без\s+первоначальн\w+|"
    r"ноль\s+аванс\w*)\b",
    re.IGNORECASE,
)


def extract_prepaid_pct_from_utterance(utterance: str) -> Optional[float]:
    """Return prepaid percentage 0-100 from the utterance, or None.
    Range filter mirrors classifier_schema's allow band so a stray number
    ("100 тысяч") doesn't grab prepaid."""
    if not utterance:
        return None
    if _PREPAID_ZERO_RE.search(utterance):
        return 0.0
    m = _PREPAID_PCT_RE.search(utterance)
    if not m:
        return None
    try:
        pct = float(m.group(1))
    except (TypeError, ValueError):
        return None
    if 0 <= pct <= 100:
        return pct
    return None


# ============================================================ type_schedule
# "Аннуитет" → 0 (annuity). "Линейный" / "дифференцированный" / "убывающий"
# → 1 (linear). Stored as a string in the profile (Literal["0", "1"]) so
# return matches the canonical enum the calculator and classifier use.
_SCHEDULE_ANNUITY_RE = re.compile(r"\b(аннуитет\w*|равн\w+\s+платеж\w*)\b", re.IGNORECASE)
_SCHEDULE_LINEAR_RE = re.compile(
    r"\b(линейн\w+|дифференциров\w+|убыва\w+|уменьшающ\w+)\b",
    re.IGNORECASE,
)


def extract_type_schedule_from_utterance(utterance: str) -> Optional[str]:
    """Return "0" (annuity) or "1" (linear) from schedule cues, or None."""
    if not utterance:
        return None
    has_annuity = bool(_SCHEDULE_ANNUITY_RE.search(utterance))
    has_linear = bool(_SCHEDULE_LINEAR_RE.search(utterance))
    if has_annuity and not has_linear:
        return "0"
    if has_linear and not has_annuity:
        return "1"
    return None


# ============================================================ currency
# "рубли" / "BYN" → BYN. "долларов" / "USD" → USD. Belarus context: bare
# "рубли" maps to BYN (matches profile_hygiene._normalize_currency).
_CUR_BYN_RE = re.compile(
    r"\b(byn|blr|белорусск\w+\s+рубл\w*|бел\.?\s*руб\w*|"
    r"в\s+рубл\w+|рубл\w+(?!\s+росси))\b",
    re.IGNORECASE,
)
_CUR_USD_RE = re.compile(r"\b(usd|доллар\w*|бакс\w*|зелен\w+)\b", re.IGNORECASE)
_CUR_EUR_RE = re.compile(r"\b(eur|евро)\b", re.IGNORECASE)
_CUR_RUB_RE = re.compile(
    r"\b(российск\w+\s+рубл\w*|росс\.?\s*руб\w*|rub|русск\w+\s+рубл\w*)\b",
    re.IGNORECASE,
)


def extract_currency_from_utterance(utterance: str) -> Optional[str]:
    """Return canonical currency from the utterance. Multiple-cue
    utterances (bilingual price like "100 рублей или долларов?") return
    None — let the orchestrator clarify.

    RUB precedence: when the utterance explicitly names "российск..."
    + "рубл..." we must NOT also count the bare-rubles BYN cue. The
    BYN regex includes a negative lookahead for "росси" trailing
    "рубл", but the qualifier may also precede ("российских рублей"),
    so we explicitly suppress BYN when RUB matched.
    """
    if not utterance:
        return None
    has_rub = bool(_CUR_RUB_RE.search(utterance))
    has_byn = bool(_CUR_BYN_RE.search(utterance)) and not has_rub
    has_usd = bool(_CUR_USD_RE.search(utterance))
    has_eur = bool(_CUR_EUR_RE.search(utterance))
    hits = sum([has_byn, has_usd, has_eur, has_rub])
    if hits != 1:
        return None
    if has_byn:
        return "BYN"
    if has_usd:
        return "USD"
    if has_eur:
        return "EUR"
    if has_rub:
        return "RUB"
    return None
