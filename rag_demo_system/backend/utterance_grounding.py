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

from .numeric_words_ru import parse_ru_number

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
        # Issue 1 (live call d5174335 2026-04-27): "офис\w*" was removed.
        # Users asking about company office addresses ("адреса офисов",
        # "где офис", "приехать в офис") were having profile.subject
        # poisoned to Недвижимость. Real-estate-leasing customers say
        # "помещение", "склад", "недвижимость" — never bare "офис".
        re.compile(
            r"\b(недвижимост\w*|квартир\w+|здани\w+|помещени\w+|склад\w*)",
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
    # Forward form: "5 лет" / "3 года" / "1 год".
    # Inverted form (Polish E 2026-04-27, live call 56c0e2f9): "лет на 5"
    # — STT often surfaces this word order when the user thinks aloud
    # ("на, например, лет на 5"). Both forms match the same numeric.
    r"\b(?:(\d{1,2})\s*(?:лет|года|год|годов|г\.?)|"
    r"(?:лет|года|год|годов)\s+на\s+(\d{1,2}))\b",
    re.IGNORECASE,
)

# Bug 17 (live call 9ec121bc, 2026-04-25): user said "Два года" — STT
# transcribes the spoken word verbatim, no digit. Without word-numeral
# support the regex misses, the patch never lands, the gate never
# clarifies, and the LLM hallucinates re-asking already-captured fields.
# Range 0-15 covers realistic vehicle ages (older numbers usually appear
# as digits anyway) and avoids parsing things like "сто" → 100 which is
# already filtered by the 0-50 range cap below.
_RUSSIAN_NUMERAL_WORDS: dict[str, int] = {
    "ноль": 0,
    "один": 1, "одна": 1, "одного": 1,
    "два": 2, "две": 2, "двух": 2,
    "три": 3, "трех": 3, "трёх": 3,
    "четыре": 4, "четырех": 4, "четырёх": 4,
    "пять": 5, "пяти": 5,
    "шесть": 6, "шести": 6,
    "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8,
    "девять": 9, "девяти": 9,
    "десять": 10, "десяти": 10,
    "одиннадцать": 11, "одиннадцати": 11,
    "двенадцать": 12, "двенадцати": 12,
    "тринадцать": 13, "тринадцати": 13,
    "четырнадцать": 14, "четырнадцати": 14,
    "пятнадцать": 15, "пятнадцати": 15,
}
# Match a numeral word followed (within ≤2 tokens) by a year-unit so
# bare "два" doesn't ground (could be ИП-status answer "два", part of
# a phone number, etc.). The year-unit anchor mirrors the digit regex.
_AGE_WORD_NUMERAL_RE = re.compile(
    r"\b("
    r"ноль|один(?:ого)?|одна|два|две|двух|три|тр[её]х|"
    r"четыре|четыр[её]х|пят[ьи]|шест[ьи]|сем[ьи]|"
    r"восем[ь]|восьми|девят[ьи]|десят[ьи]|"
    r"одиннадцат[ьи]|двенадцат[ьи]|тринадцат[ьи]|"
    r"четырнадцат[ьи]|пятнадцат[ьи]"
    r")\b"
    r"(?:\s+\S+){0,2}?"
    r"\s*(?:лет|года|год|годов|г\.?)\b",
    re.IGNORECASE,
)


def extract_age_years_from_utterance(utterance: str) -> Optional[int]:
    """Return the integer age-in-years from a 'N лет / N года' utterance, or None.

    Accepts both digit forms ("3 года") and Russian numeral words
    ("Два года", "Пять лет") — the latter added 2026-04-25 after live
    call 9ec121bc showed STT transcribing word-for-word for slow voice
    answers.

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
    if m:
        # Group 1 = forward form ("5 лет"); group 2 = inverted ("лет на 5").
        digits = m.group(1) or m.group(2)
        try:
            n = int(digits)
        except (TypeError, ValueError):
            return None
        if n < 0 or n > 50:
            return None
        return n
    # Fall back to word numerals when no digit was found.
    wm = _AGE_WORD_NUMERAL_RE.search(utterance)
    if wm:
        word = wm.group(1).lower()
        # Normalize ё/е variants for the dictionary lookup.
        word = word.replace("ё", "е")
        # Look up the canonical form (also normalize dictionary keys).
        for key, val in _RUSSIAN_NUMERAL_WORDS.items():
            if key.replace("ё", "е") == word:
                return val
    return None


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
    from the utterance, or None if neither cue fires or both fire.

    Live regression 5e6f4c48 (2026-04-26): the юр-cue list contains
    ambiguous nouns (организация / компания / фирма / предприятие /
    бизнес) that the user can use to refer to the BOT's company while
    RAG-asking ("вашей компании"). To prevent the silent юр capture
    that chains into Bug R's _has_any_core_field gate, delegate the
    юр decision to ``classifier_schema._client_type_value_grounded``
    so the same self/other-reference rules apply uniformly across the
    classifier-validate path AND the utterance-fallback path.
    """
    if not utterance:
        return None
    from .classifier_schema import _client_type_value_grounded  # lazy

    has_phys = bool(_CLIENT_TYPE_PHYS_RE.search(utterance))
    has_legal = _client_type_value_grounded("Юридическое лицо", utterance)
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


# ============================================================ cost
# Issue 1 (live call 5fa0bb3d, 2026-04-26): user said "Сто десять тысяч
# долларов и поддержанный". The Qwen3-4B classifier captured
# currency=USD and condition_new=0 but silently dropped cost. The bot
# then re-asked "Уточните, пожалуйста, стоимость." User repeated with
# digits ("110 тысяч долларов") and it worked — so this is exclusively
# a classifier reliability gap on fully-spelled-out RU numerals.
#
# Fix: utterance-level fallback that reuses parse_ru_number from
# numeric_words_ru (the same parser profile_hygiene.has_field_signal
# uses for cost grounding) and a digit/scale-word path mirroring the
# Fix-34 multiplier rule. Range gate (10_000 ≤ cost ≤ 100_000_000)
# rejects stray small numerics ("5 лет", "60 месяцев") that would
# otherwise leak through. parse_ru_number itself drops percent contexts
# ("двадцать процентов") so the prepaid-pct path is safe.
_COST_MIN = 10_000
_COST_MAX = 100_000_000
# Digit + thousand/million word ("80 тысяч", "3 миллиона", "150 000").
_COST_DIGIT_THOUSAND_RE = re.compile(
    r"\b(\d{1,4})\s*(тысяч\w*|тыс\b|k\b|к\b)",
    re.IGNORECASE,
)
_COST_DIGIT_MILLION_RE = re.compile(
    r"\b(\d{1,3})\s*(миллион\w*|млн\b)",
    re.IGNORECASE,
)
# Bare digit form, allowing space/comma grouping. Caller flattens
# whitespace before applying. Out-of-range values are dropped, so a
# stray "60 месяцев" won't ground as cost=60.
_COST_BARE_DIGIT_RE = re.compile(r"(?<!\d)(\d{4,9})(?!\d)")
# Year/month-unit suffix on the numeric — reject so age/term answers
# do not surface as cost.
_NON_COST_NUMERIC_RE = re.compile(
    r"\b\d+\s*(?:лет|года|год|годов|г\.?|месяц\w*|мес\.?|"
    r"процент\w*|проц\.?|%)",
    re.IGNORECASE,
)


def extract_cost_from_utterance(utterance: str) -> Optional[int]:
    """Return integer cost from the utterance, or None.

    Order of attempts:
      1. Word-form via parse_ru_number ("сто десять тысяч долларов").
      2. Digit + scale-word ("80 тысяч", "3 миллиона").
      3. Bare digits with grouping ("150 000").

    Conservative range gate (10_000-100_000_000) rejects small numerics
    that are almost always age/term/prepaid leakage. The
    `_NON_COST_NUMERIC_RE` guard suppresses the bare-digit path when the
    utterance carries a year/month/percent suffix on the same number,
    so "5 лет" / "60 месяцев" / "20 процентов" never ground as cost.
    """
    if not utterance:
        return None

    # Word-form (the primary regression case).
    parsed = parse_ru_number(utterance)
    if parsed is not None and _COST_MIN <= parsed <= _COST_MAX:
        return parsed

    # Digit + scale-word.
    m = _COST_DIGIT_THOUSAND_RE.search(utterance)
    if m:
        try:
            n = int(m.group(1)) * 1000
        except (TypeError, ValueError):
            n = 0
        if _COST_MIN <= n <= _COST_MAX:
            return n
    m = _COST_DIGIT_MILLION_RE.search(utterance)
    if m:
        try:
            n = int(m.group(1)) * 1_000_000
        except (TypeError, ValueError):
            n = 0
        if _COST_MIN <= n <= _COST_MAX:
            return n

    # Bare digit. Reject when the number carries a non-cost unit suffix
    # (years/months/percent) — those answers belong to other slots.
    if _NON_COST_NUMERIC_RE.search(utterance):
        return None
    flat = re.sub(r"[\s,_]+", "", utterance)
    bm = _COST_BARE_DIGIT_RE.search(flat)
    if bm:
        try:
            n = int(bm.group(1))
        except (TypeError, ValueError):
            return None
        if _COST_MIN <= n <= _COST_MAX:
            return n

    return None


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
