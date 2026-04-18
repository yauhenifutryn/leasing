"""Guard rails on classifier profile_patches before merging into ClientProfile.

Drops or normalizes patches that would corrupt the profile from noise
utterances, bot-name echoes, malformed enum values, or out-of-MVP-range numbers.
"""

from __future__ import annotations

import re
from typing import Any

MVP_PREPAID_RANGE = (0.0, 40.0)
MVP_TERM_RANGE = (12, 84)

# Enum-field slot-fill answers — single-word utterances that match these are
# valid replies to clarification questions (e.g. "Аннуитет" → type_schedule).
# We let such patches bypass the <2-token noise filter.
_ENUM_SLOT_FILL_WORDS: frozenset[str] = frozenset({
    # type_schedule
    "аннуитет", "аннуитетный", "аннуитетная", "аннуитетное",
    "линейный", "линейная", "линейное", "равный",
    # client_type (raw single-word forms — _normalize_client_type handles full phrases)
    "физлицо", "физик", "физическое",
    "юрлицо", "юридическое", "ооо", "оао", "зао", "организация", "компания",
    "ип", "ипэшник", "самозанятый",
    # Fix 40d + hotfix: "бизнес" variants as single-word slot-fill answers to
    # "физическое, ИП или юр.лицо?". Without this, "Микробизнес." was dropped
    # by the <2-token noise filter (session a685ce41, 2026-04-18).
    "бизнес", "бизнесмен", "микробизнес", "предприниматель",
    # condition_new
    "новый", "новая", "новое",
    "бу", "б/у", "подержанный", "подержанная", "подержанное",
    # subject (single-word slot-fill replies to "легковой или грузовой?")
    "легковой", "грузовой", "спецтехника", "оборудование",
    "недвижимость", "машина", "автомобиль", "авто",
})


def _normalize_client_type(v: Any) -> str | None:
    if not isinstance(v, str):
        return None
    s = v.strip().lower()
    if s in {"физлицо", "физик", "физ. лицо", "физическое лицо", "физическое"}:
        return "Физическое лицо"
    if s in {"ип", "ипэшник", "индивидуальный предприниматель", "самозанятый",
             "предприниматель", "микробизнес"}:
        return "ИП"
    if s in {"юрлицо", "юридическое лицо", "ооо", "оао", "зао", "организация",
             "компания", "юридическое", "бизнес", "бизнесмен", "малый бизнес"}:
        return "Юридическое лицо"
    return None


# Fix 26 — explicit-cue keywords for client_type. A classifier `client_type`
# patch is accepted ONLY if the utterance contains one of these words. This
# catches the "assumes individual without asking" failure mode where the
# classifier infers `Физическое лицо` from context (e.g. "хочу машину")
# because prompt examples lean that way on early turns.
#
# The orchestrator falls back to its clarification path when no client_type
# is captured, so the client is explicitly asked (vs being silently labelled).
_CLIENT_TYPE_CUE_RE = re.compile(
    r"(?:"
    # Main alternatives require a leading word boundary.
    r"\b(?:"
    r"физлиц\w*|физик\w*|физическ\w+|"
    r"юрлиц\w*|юридическ\w+|"
    r"ооо|оао|зао|"
    r"организаци\w+|компани\w+|предприяти\w+|фирм\w+|"
    r"ип\b|ипэшник\w*|самозанят\w+|индивидуальн\w+|"
    # Fix 40d: "бизнес" / "бизнесмен" map to юр.лицо in Belarus context.
    r"бизнес\w*|предпринимат\w+"
    r")"
    # Also match "бизнес" inside compound words without word boundary
    # (e.g. "Микробизнес", "малый бизнес", "бизнесмен"). Session a685ce41
    # dropped "Микробизнес." because of the leading \b.
    r"|бизнес"
    r")",
    re.IGNORECASE,
)


def utterance_has_client_type_cue(utterance: str) -> bool:
    """Return True if the utterance explicitly names a client-type category.

    Used by `filter_patches` to reject classifier `client_type` patches that
    aren't grounded in the user's words. Prevents "assumes individual"
    hallucinations at the start of the call.
    """
    if not utterance:
        return False
    return bool(_CLIENT_TYPE_CUE_RE.search(utterance))


# Fix 27 — explicit-cue keywords for subject (vehicle / equipment category).
# Same pattern as client_type: classifier was extracting "Легковой автомобиль"
# even when user clearly said "Грузовой" (observed 2026-04-18). Reject subject
# patches without a cue in the utterance. Orchestrator will clarify.
_SUBJECT_CUE_RE = re.compile(
    r"\b("
    # Cars (легковой автомобиль)
    r"легков\w*|седан\w*|машин\w*|автомобил\w*|авто|внедорожник\w*|кроссовер\w*|"
    # Trucks (грузовой автомобиль)
    r"грузов\w*|грузовик\w*|фур\w+|тягач\w*|самосвал\w*|микроавтобус\w*|"
    # Special equipment (спецтехника)
    r"спецтехник\w*|погрузчик\w*|экскаватор\w*|бульдозер\w*|кран\w*|каток\w*|"
    r"трактор\w*|комбайн\w*|"
    # Equipment (оборудование)
    r"оборудовани\w*|станк\w+|линия|установк\w+|"
    # Real estate (недвижимость)
    r"недвижимост\w*|квартир\w+|дом\b|здани\w+|помещени\w+|склад\w*|офис\w*|"
    # Other transport
    r"транспорт\w*|автобус\w*|прицеп\w*|мотоцикл\w*|скутер\w*|"
    # Brand names also imply a car (classifier already maps these)
    r"bmw|mercedes|mercedes-benz|toyota|kia|hyundai|audi|volkswagen|vw|lexus|"
    r"mazda|renault|peugeot|ford|lada|skoda|fiat|chevrolet|nissan|honda|"
    r"мерседес|тойот\w+|киа|хендай|ауди|фольксваген|лексус|мазд\w+|фольцваген|"
    r"рено|пежо|форд|лад\w+|уаз|камаз|шкод\w+|ниссан|хонд\w+"
    r")",
    re.IGNORECASE,
)


def utterance_has_subject_cue(utterance: str) -> bool:
    """Return True if the utterance explicitly names a leasing subject category.

    Used by `filter_patches` to reject classifier `subject` patches that
    aren't grounded in the user's words. Prevents "truck -> car" category
    hallucinations observed in voice transcripts.
    """
    if not utterance:
        return False
    return bool(_SUBJECT_CUE_RE.search(utterance))


# Fix 31 — currency and enum cues for use by `has_field_signal`.
_CURRENCY_CUE_RE = re.compile(
    r"\b(рубл\w*|руб\b|byn|blr|доллар\w*|usd|евро|eur|российск\w+|rub)\b",
    re.IGNORECASE,
)
_CONDITION_NEW_CUE_RE = re.compile(
    r"\b(нов\w+|подержан\w+|б/у|бу|бывш\w+)\b",
    re.IGNORECASE,
)
_TYPE_SCHEDULE_CUE_RE = re.compile(
    r"\b(аннуитет\w*|линейн\w+|дифференциров\w+|убыва\w+|равн\w+)\b",
    re.IGNORECASE,
)


def has_field_signal(field: str, value: Any, utterance: str) -> bool:
    """Return True if `utterance` carries an explicit signal for `field=value`.

    Used by the orchestrator when deciding whether a classifier-extracted
    hint is a real user intent or a leak from history / calc context.
    For numeric fields we require the digits of `value` to appear in the
    utterance; for enum fields we require a category cue.

    This is a stricter check than the hygiene cue guards (which only ask
    whether a category was named). It prevents Fix 28's multi-field change-
    confirm from listing derived fields the user never mentioned (e.g.
    `prepaid_amount=16000` echoed from a prior calc result).
    """
    if utterance is None:
        return False
    if value is None or value == "":
        return False
    if field == "subject":
        return utterance_has_subject_cue(utterance)
    if field == "client_type":
        return utterance_has_client_type_cue(utterance)
    if field == "currency":
        return bool(_CURRENCY_CUE_RE.search(utterance))
    if field == "condition_new":
        return bool(_CONDITION_NEW_CUE_RE.search(utterance))
    if field == "type_schedule":
        return bool(_TYPE_SCHEDULE_CUE_RE.search(utterance))
    if field in ("cost", "term_months", "prepaid_pct", "prepaid_amount", "age_years"):
        # Numeric: the digits of the value must appear literally in the
        # utterance. Handles integer and float-with-trailing-zero cases.
        try:
            _int = int(value)
        except (TypeError, ValueError):
            return False
        v_str = str(_int)
        # Permit digit grouping like "150 000" — strip spaces/commas from
        # utterance before checking.
        _flat = re.sub(r"[\s,_]+", "", utterance)
        if v_str in _flat:
            return True
        # Also check raw utterance in case the number fits a compact span
        # (rare; covered by the flat check too).
        if v_str in utterance:
            return True
        # Fix 34: Russian spelled multipliers. "80 тысяч" → 80000, "3 миллиона"
        # → 3000000. Only applies to cost / prepaid_amount (big numbers).
        # Term / prepaid_pct / age are small enough that callers say the raw
        # digits ("36 месяцев", "20 процентов", "22 года").
        if field in ("cost", "prepaid_amount"):
            # N тысяч / N тыс
            if _int % 1000 == 0:
                _thou = _int // 1000
                if _thou > 0 and re.search(
                    rf"\b{_thou}\s*(?:тысяч\w*|тыс\b|k\b|к\b)",
                    utterance,
                    re.IGNORECASE,
                ):
                    return True
            # N миллион / N млн
            if _int % 1000000 == 0:
                _mil = _int // 1000000
                if _mil > 0 and re.search(
                    rf"\b{_mil}\s*(?:миллион\w*|млн\b)",
                    utterance,
                    re.IGNORECASE,
                ):
                    return True
        # Fix 40b: years-to-months conversion for term_months.
        # User says "на 7 лет" → classifier emits term_months=84.
        # The digits "84" never appear in the utterance, so require the
        # whole-year equivalent to match.
        if field == "term_months":
            if _int > 0 and _int % 12 == 0:
                _years = _int // 12
                if re.search(
                    rf"\b{_years}\s*(?:лет\b|год\w*|года\b)",
                    utterance,
                    re.IGNORECASE,
                ):
                    return True
            # Half-year ("полтора года" → 18, "полгода" → 6)
            if _int == 18 and re.search(r"полтора\s*года", utterance, re.IGNORECASE):
                return True
            if _int == 6 and re.search(r"\bполгода\b", utterance, re.IGNORECASE):
                return True
            # Single-word "год" without number → 12
            if _int == 12 and re.search(r"\b(один\s+)?год\b", utterance, re.IGNORECASE):
                return True
        return False
    # Unknown field — conservative: require explicit value string
    return str(value) in (utterance or "")


def _normalize_currency(patch_value: Any, utterance: str) -> str | None:
    if not isinstance(patch_value, str):
        return None
    s = patch_value.strip().upper()
    if s in {"BYN", "BLR"}:
        return "BYN"
    if s == "USD":
        return "USD"
    if s == "EUR":
        return "EUR"
    if s == "RUB":
        low = (utterance or "").lower()
        if "росси" in low or "российск" in low:
            return "RUB"
        # Bare "рубли" in Belarus context -> BYN
        return "BYN"
    return None


def filter_patches(
    patches: dict[str, Any],
    utterance: str,
    bot_name: str = "Ксения",
) -> dict[str, Any]:
    """Run classifier-emitted patches through hygiene checks. Returns filtered dict."""
    if not patches:
        return {}

    # Noise filter: drop everything when the utterance is too short to carry info.
    # EXCEPTIONS (value-carrying answers that pass even with <2 non-digit tokens):
    #   1. Enum slot-fill: "Аннуитет", "Физлицо", "Новый" etc.
    #   2. Numeric-field answer: "49 500 рублей", "сто тысяч", "36 месяцев" etc.
    #      Classifier reliably extracts cost/term/prepaid numbers from these
    #      utterances even though the only non-digit token is the unit word.
    tokens = [t for t in (utterance or "").strip().split() if not t.isdigit()]
    _NUMERIC_FIELD_KEYS = frozenset({
        "cost", "term_months", "prepaid_pct", "prepaid_amount", "age_years",
    })
    if len(tokens) < 2:
        _stripped = (utterance or "").strip().lower().rstrip(".!,?;:")
        _is_enum_fill = _stripped in _ENUM_SLOT_FILL_WORDS
        _has_numeric_answer = any(k in patches for k in _NUMERIC_FIELD_KEYS)
        if not _is_enum_fill and not _has_numeric_answer:
            return {}
        if _is_enum_fill:
            print(f"[Profile] single-word enum patch accepted: '{_stripped}' patches={list(patches.keys())}", flush=True)
        elif _has_numeric_answer:
            _numeric_keys = [k for k in _NUMERIC_FIELD_KEYS if k in patches]
            print(f"[Profile] numeric-answer patch accepted: '{_stripped[:40]}' numeric_fields={_numeric_keys}", flush=True)

    out: dict[str, Any] = dict(patches)

    # Drop bot name (or bot name + patronymic) echoed as user name.
    # Classifier sometimes captures "Ксения Николаевна" from formal user
    # address — we reject anything whose first token matches bot_name.
    _raw_name = out.get("name")
    if isinstance(_raw_name, str):
        _name_tokens = _raw_name.strip().lower().split()
        if _name_tokens and _name_tokens[0] == bot_name.lower():
            out.pop("name")

    # Normalize / validate client_type.
    if "client_type" in out:
        ct = _normalize_client_type(out["client_type"])
        if ct is None:
            out.pop("client_type")
        elif not utterance_has_client_type_cue(utterance or ""):
            # Fix 26: reject classifier-inferred client_type that has no
            # explicit keyword in the user utterance. Orchestrator will
            # ask the client directly instead of silently labelling them.
            print(
                f"[Profile] client_type cue missing — dropping inferred "
                f"'{ct}' utterance='{(utterance or '')[:60]}'",
                flush=True,
            )
            out.pop("client_type")
        else:
            out["client_type"] = ct

    # Fix 27 — reject subject patches without an explicit cue.
    # Classifier was observed extracting "Легковой автомобиль" when the user
    # clearly said "Грузовой". Orchestrator will ask or re-read.
    if "subject" in out:
        _subj = out.get("subject")
        if isinstance(_subj, str) and _subj.strip():
            if not utterance_has_subject_cue(utterance or ""):
                print(
                    f"[Profile] subject cue missing — dropping inferred "
                    f"'{_subj}' utterance='{(utterance or '')[:60]}'",
                    flush=True,
                )
                out.pop("subject")

    # Normalize currency.
    if "currency" in out:
        cur = _normalize_currency(out["currency"], utterance)
        if cur is None:
            out.pop("currency")
        else:
            out["currency"] = cur

    # Prepaid type check only.
    # Fix 39: do NOT silently drop out-of-range values. Forward them so the
    # calculator's validate_calc_inputs can surface a specific user-facing
    # message ("аванс должен быть от 0 до 40 процентов") instead of hygiene
    # eating them and the bot improvising.
    if "prepaid_pct" in out:
        try:
            float(out["prepaid_pct"])
        except (TypeError, ValueError):
            out.pop("prepaid_pct")

    # Term type check only (range validation moved to calculator).
    if "term_months" in out:
        try:
            int(out["term_months"])
        except (TypeError, ValueError):
            out.pop("term_months")

    dropped_keys = [k for k in patches.keys() if k not in out]
    if dropped_keys:
        print(f"[Profile] normalized-dropped patch keys={dropped_keys} utterance='{(utterance or '')[:60]}'", flush=True)

    return out
