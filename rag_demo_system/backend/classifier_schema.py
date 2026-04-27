"""Pydantic schema for Qwen classifier output (CP-2.1).

Two layers of protection:

1. Pydantic Literal + Field constraints — kills malformed enum values and the
   `change_field="all"` failure mode (retires the Fix 41b runtime whitelist).
2. ``@model_validator(mode="after")`` + ``ValidationInfo.context['utterance']``
   — nulls enum fields whose cue is not present in the user utterance, and
   nulls ``age_years`` when ``condition_new`` is not 0. Kills the E1/E2
   hallucinated-yet-schema-valid leak documented in
   ``.planning/master_plan_2026_04_18/02_structured_classifier.md``.

Cue regex authorities stay in ``profile_hygiene.py``; this module imports them
so there is exactly one source of truth for "what counts as a cue".
"""
from __future__ import annotations

import json as _json
import math
import re
from typing import Literal, Optional, Union

# Sentinel: canonicalize_change_value returns this when the value cannot be
# coerced to the canonical type for its field; callers null both change_field
# and change_value so the pair never reaches staging.
_DROP_CHANGE_VALUE = object()

from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator, model_validator

from .profile_hygiene import (
    _normalize_client_type,
    has_field_signal,
)

# Value-specific utterance cue sets (Codex adversarial 2026-04-20 fix for
# Finding A). profile_hygiene's ``utterance_has_*_cue`` helpers answer "is any
# cue for this DIMENSION present", which lets contradictory-but-plausible
# values through — e.g. classifier emits ``currency="RUB"`` on "в долларах"
# and the USD cue satisfies the dimension check. These per-value patterns
# demand the cue MATCH the emitted value, not just the dimension.
_SUBJECT_VALUE_CUES: dict[str, re.Pattern[str]] = {
    "Легковой автомобиль": re.compile(
        r"\b("
        # Specific car-category words only — NOT generic машин\w*/автомобил\w*
        # which also match "грузовую машину" / "грузовой автомобиль"
        # (Codex adversarial pass 3, 2026-04-20).
        r"легков\w*|седан\w*|внедорожник\w*|кроссовер\w*|"
        # brand names implying car:
        r"bmw|mercedes|mercedes-benz|toyota|kia|hyundai|audi|volkswagen|vw|lexus|"
        r"mazda|renault|peugeot|ford|lada|skoda|fiat|chevrolet|nissan|honda|"
        r"мерседес|тойот\w+|киа|хендай|ауди|фольксваген|лексус|мазд\w+|фольцваген|"
        r"рено|пежо|форд|лад\w+|шкод\w+|ниссан|хонд\w+"
        r")",
        re.IGNORECASE,
    ),
    "Грузовой автомобиль": re.compile(
        r"\b(грузов\w*|грузовик\w*|фур\w+|тягач\w*|самосвал\w*|микроавтобус\w*|камаз|уаз)",
        re.IGNORECASE,
    ),
    "Спецтехника": re.compile(
        r"\b(спецтехник\w*|погрузчик\w*|экскаватор\w*|бульдозер\w*|кран\w*|каток\w*|"
        r"трактор\w*|комбайн\w*)",
        re.IGNORECASE,
    ),
    "Оборудование": re.compile(
        r"\b(оборудовани\w*|станк\w+|установк\w+)|\bлиния\b",
        re.IGNORECASE,
    ),
    "Недвижимость": re.compile(
        r"\b(недвижимост\w*|квартир\w+|дом\b|здани\w+|помещени\w+|склад\w*|офис\w*)",
        re.IGNORECASE,
    ),
    "Прочий транспорт": re.compile(
        r"\b(транспорт\w*|автобус\w*|прицеп\w*|мотоцикл\w*|скутер\w*)",
        re.IGNORECASE,
    ),
}

# Generic "car" fallback for bare "машина" / "автомобиль" with no modifier.
# When the user says just "машину" we treat it as Легковой автомобиль unless
# a competing subject category (truck / spec / equipment / real estate / other
# transport) also appears in the utterance. Fixes the EmitClarify loop where
# the user mirrors the bot's own word and grounding rejects it.
_SUBJECT_GENERIC_CAR_RE = re.compile(
    r"\b(машин\w*|автомобил\w*|авто)\b",
    re.IGNORECASE,
)
_SUBJECT_COMPETING_RE = re.compile(
    r"\b("
    r"грузов\w*|грузовик\w*|фур\w+|тягач\w*|самосвал\w*|микроавтобус\w*|камаз|уаз|"
    r"спецтехник\w*|погрузчик\w*|экскаватор\w*|бульдозер\w*|кран\w*|каток\w*|"
    r"трактор\w*|комбайн\w*|"
    r"оборудовани\w*|станк\w+|установк\w+|"
    r"недвижимост\w*|квартир\w+|здани\w+|помещени\w+|склад\w*|офис\w*|"
    r"автобус\w*|прицеп\w*|мотоцикл\w*|скутер\w*"
    r")",
    re.IGNORECASE,
)


def _subject_value_grounded(value: str, utterance: str) -> bool:
    """Return True when `utterance` grounds `value` under the value-aware
    subject-cue rules, including the bare-"машина" fallback for
    Легковой автомобиль.
    """
    if not utterance:
        return False
    cue_re = _SUBJECT_VALUE_CUES.get(value)
    if cue_re and cue_re.search(utterance):
        return True
    if value == "Легковой автомобиль":
        if _SUBJECT_GENERIC_CAR_RE.search(utterance) and not _SUBJECT_COMPETING_RE.search(utterance):
            return True
    return False


_CLIENT_TYPE_VALUE_CUES: dict[str, re.Pattern[str]] = {
    "Физическое лицо": re.compile(
        r"\b(физлиц\w*|физик\w*|физическ\w+)",
        re.IGNORECASE,
    ),
    "Юридическое лицо": re.compile(
        r"(\b(ип\b|ипэшник\w*|самозанят\w+|индивидуальн\w+|"
        r"юрлиц\w*|юридическ\w+|ооо|оао|зао|"
        r"организаци\w+|компани\w+|предприяти\w+|фирм\w+|"
        r"предпринимат\w+)|бизнес\w*)",
        re.IGNORECASE,
    ),
}


# Live regression 5e6f4c48 (2026-04-26): self/other reference disambiguation
# for юридическое лицо. The full _CLIENT_TYPE_VALUE_CUES["Юридическое лицо"]
# regex matches ambiguous nouns like "компании / организации / фирмы /
# предприятия / бизнеса" that the user can use to refer to the BOT's
# company while RAG-asking ("вашей компании", "о вашей фирме"). Without
# this distinction, every such RAG turn silently captures
# client_type=Юридическое лицо and chains into Bug R's _has_any_core_field
# gate, looping step 5b on every subsequent turn.
#
# Strong cues are linguistically self-status terms — users essentially
# never ask the bot "вы ИП?" or "вы юрлицо?", so these ground
# unconditionally. Ambiguous cues require an explicit self-reference
# marker AND the absence of an other-reference marker; other-reference
# wins over self-reference because "я хочу узнать про вашу компанию"
# is a question about the bot, not a self-status statement.
_CLIENT_TYPE_LEGAL_STRONG_RE = re.compile(
    r"\b("
    r"ип\b|ипэшник\w*|самозанят\w+|индивидуальн\w+|"
    r"юрлиц\w*|юридическ\w+|"
    r"ооо|оао|зао|"
    r"микробизнес\w*|малый\s+бизнес|"
    r"предпринимат\w+"
    r")",
    re.IGNORECASE,
)
_CLIENT_TYPE_LEGAL_AMBIGUOUS_RE = re.compile(
    r"\b("
    r"организаци\w+|компани\w+|предприяти\w+|фирм\w+|бизнес\w*"
    r")\b",
    re.IGNORECASE,
)
_SELF_REFERENCE_RE = re.compile(
    r"\b("
    r"я\b|мы\b|меня\b|мне\b|нас\b|нам\b|нами\b|"
    r"наш\w*|"          # наш, наша, наше, наши, нашей, нашу, нашего, ...
    r"мо[йяёе]\w*|"      # мой, моя, моё, моё, моего, моей, ...
    r"у\s+меня|у\s+нас|"
    r"от\s+(?:меня|нас)|для\s+(?:меня|нас)"
    r")",
    re.IGNORECASE,
)
_OTHER_REFERENCE_RE = re.compile(
    r"\b("
    r"вы\b|вас\b|вам\b|вами\b|"
    r"ваш\w*|"           # ваш, ваша, ваше, ваши, вашей, вашу, вашего, ...
    r"у\s+вас|от\s+вас|для\s+вас"
    r")",
    re.IGNORECASE,
)


def _client_type_value_grounded(value: str, utterance: str) -> bool:
    """Self/other-reference-aware grounding for client_type.

    Replaces the bare ``_CLIENT_TYPE_VALUE_CUES`` regex match for
    Юридическое лицо. Strong self-status cues ground unconditionally;
    ambiguous nouns ("компания / организация / фирма / предприятие /
    бизнес") require an explicit self-reference marker and the
    absence of an other-reference marker. Физическое лицо bypasses
    the gate because its cue set ("физлицо / физик / физическ") is
    already restricted to self-status terms.
    """
    if not utterance:
        return False
    cue_re = _CLIENT_TYPE_VALUE_CUES.get(value)
    if not cue_re or not cue_re.search(utterance):
        return False
    if value == "Физическое лицо":
        return True
    if _CLIENT_TYPE_LEGAL_STRONG_RE.search(utterance):
        return True
    if _CLIENT_TYPE_LEGAL_AMBIGUOUS_RE.search(utterance):
        if _OTHER_REFERENCE_RE.search(utterance):
            return False
        return bool(_SELF_REFERENCE_RE.search(utterance))
    return False

_RUSSIAN_RUBLE_RE = re.compile(r"\bросси\w*\s*рубл\w*|\brub\b", re.IGNORECASE)
_BELARUS_RUBLE_RE = re.compile(r"\bрубл\w*|\bруб\b|\bbyn\b|\bblr\b|\bбелорусск\w+\s*рубл\w*", re.IGNORECASE)


def _currency_cue_match(value: str, utterance: str) -> bool:
    """Currency is value-aware because 'рубли' in Belarus context means BYN,
    and Russian rubles only ground if 'российск' modifier appears. Order
    matters: check RUB modifier before BYN match.
    """
    if value == "USD":
        return bool(re.search(r"\bдоллар\w*|\busd\b", utterance, re.IGNORECASE))
    if value == "EUR":
        return bool(re.search(r"\bевро\b|\beur\b", utterance, re.IGNORECASE))
    if value == "RUB":
        return bool(_RUSSIAN_RUBLE_RE.search(utterance))
    if value == "BYN":
        # Any рубл\w* mention that's NOT explicitly "российский рубль".
        if _RUSSIAN_RUBLE_RE.search(utterance):
            return False
        return bool(_BELARUS_RUBLE_RE.search(utterance))
    return False


_TYPE_SCHEDULE_VALUE_CUES: dict[str, re.Pattern[str]] = {
    # "0" = аннуитет (equal monthly payments). Direct word OR semantic
    # synonyms describing payment behavior (Polish C 2026-04-27): user
    # often says "равные / одинаковые / фиксированные платежи" instead
    # of naming the schedule type. Note: "равн\w+" was previously
    # mis-listed under "1" (linear) — moved here because semantically
    # "равные платежи" = annuitet, not linear.
    "0": re.compile(
        r"\bаннуитет\w*|\bравн[оыа]\w*|\bодинаков\w+|\bфиксиров\w+|"
        r"\bстабильн\w+|\bпостоянн\w+",
        re.IGNORECASE,
    ),
    # "1" = линейный / дифференцированный (decreasing payments, principal
    # paid faster). Direct words OR semantic synonyms ("уменьшающиеся /
    # убывающие / падающие платежи", "первый платёж больше").
    "1": re.compile(
        r"\bлинейн\w+|\bдифференциров\w+|\bубыва\w+|\bуменьш\w+|"
        r"\bпадающ\w+|\bпервый\s+(?:платеж|платёж|больше)",
        re.IGNORECASE,
    ),
}


_VALID_SUBJECTS = frozenset({
    "Легковой автомобиль", "Грузовой автомобиль", "Спецтехника",
    "Оборудование", "Недвижимость", "Прочий транспорт",
})
_VALID_CURRENCIES = frozenset({"BYN", "USD", "EUR", "RUB"})


def canonicalize_change_value(field: str, value):
    """Coerce a classifier-emitted ``change_value`` to the canonical runtime
    type for ``change_field``. Returns ``_DROP_CHANGE_VALUE`` when the value
    cannot be canonicalized (caller must null both change_field and
    change_value so the bad pair never reaches staging).

    Qwen sometimes mirrors the raw JSON representation from the prompt —
    strings where numbers are expected, ints where schedule codes are
    strings, user-phrased ``"ИП"`` for client_type. Without coercion the
    staging block stores these raw, ClientProfile's string-based checks
    silently bypass invariants (e.g. ``missing_fields()`` sees
    ``condition_new="0"`` as set but ``== 0`` fails, so the used-asset age
    requirement is skipped). (Codex thorough review 2026-04-20.)
    """
    if value is None:
        return None
    # Reject bools masquerading as ints (True/False coerce to 1/0 silently).
    if isinstance(value, bool):
        return _DROP_CHANGE_VALUE

    if field == "condition_new":
        # Reject fractional floats — 0.5 must not silently truncate to 0.
        # (Codex basic review 2026-04-20 P2: fail closed on malformed input
        # rather than mutating the confirmed profile to the wrong value.)
        if isinstance(value, float) and not value.is_integer():
            return _DROP_CHANGE_VALUE
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return _DROP_CHANGE_VALUE
        return iv if iv in (0, 1) else _DROP_CHANGE_VALUE

    if field == "type_schedule":
        if isinstance(value, bool):
            return _DROP_CHANGE_VALUE
        if isinstance(value, float):
            # Fractional floats must not truncate to a schedule code.
            if not value.is_integer():
                return _DROP_CHANGE_VALUE
            iv = int(value)
            return str(iv) if iv in (0, 1) else _DROP_CHANGE_VALUE
        if isinstance(value, int):
            return str(value) if value in (0, 1) else _DROP_CHANGE_VALUE
        if isinstance(value, str):
            s = value.strip()
            if s in ("0", "1"):
                return s
            # Bug G (live call 504eace0 2026-04-26): Qwen sometimes emits
            # the human form ("линейный" / "аннуитетный") in change_value
            # instead of the canonical "0" / "1" code. Without this lookup,
            # canonicalize drops the pair and apply_turn never sees the
            # delta — the schedule change request silently falls back to
            # FireLLMFallback. Map via the same cue patterns that
            # value_grounded uses for top-level type_schedule consistency.
            for code, cue_re in _TYPE_SCHEDULE_VALUE_CUES.items():
                if cue_re.search(s):
                    return code
            return _DROP_CHANGE_VALUE
        return _DROP_CHANGE_VALUE

    if field == "currency":
        if not isinstance(value, str):
            return _DROP_CHANGE_VALUE
        u = value.strip().upper()
        return u if u in _VALID_CURRENCIES else _DROP_CHANGE_VALUE

    if field == "client_type":
        if not isinstance(value, str):
            return _DROP_CHANGE_VALUE
        normalized = _normalize_client_type(value)
        return normalized if normalized is not None else _DROP_CHANGE_VALUE

    if field == "subject":
        if not isinstance(value, str):
            return _DROP_CHANGE_VALUE
        return value if value in _VALID_SUBJECTS else _DROP_CHANGE_VALUE

    if field in ("cost", "prepaid_pct", "prepaid_amount"):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return _DROP_CHANGE_VALUE
        if not math.isfinite(f):
            return _DROP_CHANGE_VALUE
        return f

    if field in ("term_months", "age_years"):
        # Reject fractional floats — 60.5 term_months must not silently
        # truncate to 60 and confirm the wrong change. (Codex basic P2.)
        try:
            f = float(value)
        except (TypeError, ValueError):
            return _DROP_CHANGE_VALUE
        if not math.isfinite(f):
            return _DROP_CHANGE_VALUE
        if not f.is_integer():
            return _DROP_CHANGE_VALUE
        return int(f)

    # Unknown field (shouldn't happen — change_field is a Literal). Pass through.
    return value


def value_grounded(field: str, value, utterance: str) -> bool:
    """Return True iff `value` is grounded by a matching cue in `utterance`.

    Mirrors the value-aware logic inside
    :meth:`ClassifierOutput._ground_against_utterance` and is exposed so
    callers that inspect ``change_field`` / ``change_value`` pairs can apply
    the same check (Codex adversarial pass 4, 2026-04-20: explicit enum
    changes otherwise bypass grounding and can mutate confirmed state on a
    hallucinated value).

    Numeric fields delegate to :func:`profile_hygiene.has_field_signal` so
    the Fix 34 / 40b / 1.10 multiplier and year-to-month logic is reused.
    """
    if value is None or value == "" or not utterance:
        return False
    if field == "subject" and isinstance(value, str):
        return _subject_value_grounded(value, utterance)
    if field == "client_type" and isinstance(value, str):
        normalized = _normalize_client_type(value) or value
        return _client_type_value_grounded(normalized, utterance)
    if field == "currency" and isinstance(value, str):
        return _currency_cue_match(value, utterance)
    if field == "type_schedule":
        cue_re = _TYPE_SCHEDULE_VALUE_CUES.get(str(value))
        return bool(cue_re and cue_re.search(utterance))
    if field == "condition_new":
        try:
            return has_field_signal("condition_new", int(value), utterance)
        except (TypeError, ValueError):
            return False
    return has_field_signal(field, value, utterance)

_SUBJECT_VALUES = Literal[
    "Легковой автомобиль",
    "Грузовой автомобиль",
    "Спецтехника",
    "Оборудование",
    "Недвижимость",
    "Прочий транспорт",
]

# Stored client_type is one of two values (matches calculator API payload).
# Qwen is told to collapse all business forms to "Юридическое лицо", but it
# sometimes mirrors user phrasing (e.g. emits "ИП" when the user said "я ип").
# A @field_validator(mode="before") normalizes those echoes using the shared
# profile_hygiene._normalize_client_type map BEFORE Literal validation runs,
# so valid answers never get dropped at the schema boundary.
# (Codex adversarial review 2026-04-20: previously these answers were rejected
#  by the schema and the downstream normalizer never saw them.)
_CLIENT_TYPE_VALUES = Literal["Физическое лицо", "Юридическое лицо"]

_CURRENCY_VALUES = Literal["BYN", "USD", "EUR", "RUB"]
_SCHEDULE_VALUES = Literal["0", "1"]

# "prepaid" alias intentionally dropped (Codex adversarial 2026-04-20,
# Finding B). ClientProfile has `prepaid_pct` and `prepaid_amount` fields
# but no `prepaid` attribute, so pending_change with field="prepaid" was
# silently skipped by apply_pending_change's hasattr gate, and the user's
# confirmed аванс change never reached the calculator. Schema now forces
# classifier to disambiguate to pct or amount.
_CHANGE_FIELDS = Literal[
    "subject", "cost", "currency", "client_type", "condition_new",
    "age_years", "term_months", "type_schedule",
    "prepaid_pct", "prepaid_amount",
]

# Full downstream action vocabulary (E-Codex). app.py reads these values at
# 1593 (TOOL override), 2008/2068 (change_param path), 2321 (clarify gate):
#   "calculate", "recalculate", "change_param", "sms", "clarify",
#   "clarify_client_type", "confirm", "invalid_param"
# The schema must list every value the orchestrator can dispatch on, otherwise
# Qwen can only emit the five advertised by the prompt and the other branches
# become unreachable.
_ACTION_VALUES = Literal[
    "calculate", "recalculate", "change_param",
    "sms", "clarify", "clarify_client_type",
    "confirm", "invalid_param",
]


class ClassifierOutput(BaseModel):
    """Validated shape of the Qwen SessionAgent classifier response.

    Every field is Optional (or bool with default False). Parse failure or
    ValidationError returns an empty model — this schema never raises up
    the orchestrator stack.
    """

    intent: Optional[Literal["TOOL", "RAG", "CONVERSATION"]] = None
    subject: Optional[_SUBJECT_VALUES] = None
    # allow_inf_nan=False (Codex thorough review 2026-04-20): Python json
    # accepts NaN/Infinity literals, and Pydantic passes them through by
    # default. Non-finite numerics then crash readback at int() conversion.
    cost: Optional[float] = Field(None, allow_inf_nan=False)
    currency: Optional[_CURRENCY_VALUES] = None
    client_type: Optional[_CLIENT_TYPE_VALUES] = None
    condition_new: Optional[Literal[0, 1]] = None
    age_years: Optional[int] = Field(None, ge=0, le=50)
    # Wide OOR ranges (-100..500). Calculator's validate_calc_inputs still
    # catches OOR at execution time (Fix 39 behaviour preserved); Pydantic's
    # job here is to catch NaN/Inf/non-numeric, not range-clamp.
    prepaid_pct: Optional[float] = Field(None, ge=-100, le=500, allow_inf_nan=False)
    prepaid_amount: Optional[float] = Field(None, allow_inf_nan=False)
    term_months: Optional[int] = Field(None, ge=-100, le=500)
    type_schedule: Optional[_SCHEDULE_VALUES] = None
    name: Optional[str] = None

    is_confirmation: bool = False
    is_stop_request: bool = False
    wants_readback: bool = False

    change_field: Optional[_CHANGE_FIELDS] = None
    change_value: Optional[Union[str, int, float]] = None
    action: Optional[_ACTION_VALUES] = None

    # Populated by the post-validator when it nulls fields. Inspectable by
    # `parse_classifier_output` for one-line diagnostic logging. Leading
    # underscore keeps it out of `model_dump` output.
    _grounding_drops: list[str] = []

    @field_validator("client_type", mode="before")
    @classmethod
    def _coerce_client_type(cls, v):
        """Normalize Qwen echoes ("ИП", "ООО", "самозанятый", ...) to the two
        canonical values BEFORE Literal validation. Delegates to
        profile_hygiene._normalize_client_type — single source of truth.
        Returns raw value on miss so Literal validation can reject it.
        (Codex adversarial review 2026-04-20.)
        """
        if isinstance(v, str):
            canonical = _normalize_client_type(v)
            if canonical is not None:
                return canonical
        return v

    @field_validator("type_schedule", mode="before")
    @classmethod
    def _coerce_type_schedule(cls, v):
        """Qwen sometimes mirrors numeric codes (0 / 1) when the prompt shows
        them as strings. Coerce int/float → str before Literal validation so
        a legitimate schedule answer isn't dropped at the schema boundary.
        (Codex thorough review 2026-04-20.)
        """
        if isinstance(v, bool):  # bool is a subclass of int — guard first
            return v
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("condition_new", mode="before")
    @classmethod
    def _coerce_condition_new(cls, v):
        """Symmetric to _coerce_type_schedule: Qwen may emit the code as a
        quoted string ("0" / "1") instead of an int. Coerce before Literal
        validation so a used-asset signal isn't dropped at the schema
        boundary. (Codex basic review 2026-04-20 P1.)
        """
        if isinstance(v, bool):
            return v  # let Literal reject
        if isinstance(v, str):
            s = v.strip()
            if s in ("0", "1"):
                return int(s)
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    @field_validator("currency", mode="before")
    @classmethod
    def _coerce_currency(cls, v):
        """Uppercase + strip currency strings before Literal validation.
        Qwen sometimes emits lowercase ("usd" / "byn") which the old
        _normalize_currency path handled downstream; the Literal is stricter,
        so normalize here at the boundary. (Codex basic review 2026-04-20 P2.)
        """
        if isinstance(v, str):
            u = v.strip().upper()
            if u in ("BYN", "USD", "EUR", "RUB"):
                return u
        return v

    @model_validator(mode="after")
    def _ground_against_utterance(self, info: ValidationInfo) -> "ClassifierOutput":
        """Null schema-valid-but-ungrounded enum fields and enforce cross-field
        rules. Order (post-Codex pass 3, 2026-04-20):

        1. Cue-ground every enum against the utterance (value-aware).
        2. Apply cross-field rules AFTER grounding, so they see the final
           grounded state (otherwise age_years could survive when its
           sibling condition_new got nulled by step 1).

        Empty utterance: treated as "no evidence" in production — all
        cue-grounded enums are nulled rather than passed through. Unit tests
        that want to inspect raw schema behaviour must pass a non-empty
        utterance that carries the cues they expect.
        """
        drops: list[str] = []
        ctx = info.context if isinstance(info.context, dict) else {}
        utterance = ctx.get("utterance", "") if ctx else ""

        # --- Step 1: value-aware cue grounding ---
        if self.subject is not None:
            if not _subject_value_grounded(self.subject, utterance or ""):
                drops.append(f"subject={self.subject!r}")
                self.subject = None
        if self.client_type is not None:
            if not _client_type_value_grounded(self.client_type, utterance or ""):
                drops.append(f"client_type={self.client_type!r}")
                self.client_type = None
        if self.currency is not None and not (
            utterance and _currency_cue_match(self.currency, utterance)
        ):
            drops.append(f"currency={self.currency!r}")
            self.currency = None
        if self.type_schedule is not None:
            cue_re = _TYPE_SCHEDULE_VALUE_CUES.get(self.type_schedule)
            if cue_re is None or not (utterance and cue_re.search(utterance)):
                drops.append(f"type_schedule={self.type_schedule!r}")
                self.type_schedule = None
        if self.condition_new is not None and not (
            utterance and has_field_signal("condition_new", self.condition_new, utterance)
        ):
            drops.append(f"condition_new={self.condition_new}")
            self.condition_new = None

        # --- Step 2: cross-field rules (run AFTER grounding) ---
        # age_years is only meaningful for used equipment.
        #
        # Bug 17 (live call 9ec121bc 2026-04-25): the previous rule dropped
        # age_years whenever `condition_new != 0` — including the common
        # case where the small classifier omits condition_new on the age-
        # answer turn (it was captured a turn earlier). The validator
        # cannot see profile state, so it nulled valid age values, the
        # downstream gate stayed silent because no patch landed, and the
        # LLM hallucinated re-asking already-captured fields.
        #
        # Fixed rule: drop age_years only when this turn carries
        # contradictory or invalid signal:
        #   (a) classifier emitted condition_new=1 on the same turn
        #       ("новая машина пять лет" with condition_new=1), OR
        #   (b) Step 1 grounding nulled condition_new because the
        #       utterance contradicted the value (e.g. classifier said
        #       condition_new=0 but utterance is "новая машина" — the
        #       drops list will carry condition_new=...).
        # When condition_new is simply omitted (None, no drop), trust
        # the downstream apply path:
        #   - utterance-fallback gate (app.py:1499) requires
        #     profile.condition_new == 0 before adopting,
        #   - sticky-block first-capture path applies it directly only
        #     in COLLECTING state.
        # A hallucinated age_years on a NEW-car profile therefore still
        # can't poison state, while a real age-answer turn now flows
        # through correctly.
        if self.age_years is not None:
            explicit_new = (self.condition_new == 1)
            grounding_dropped_condition = any(
                d.startswith("condition_new=") for d in drops
            )
            if explicit_new or grounding_dropped_condition:
                drops.append(
                    f"age_years={self.age_years} "
                    f"(condition_new={self.condition_new}, "
                    f"grounding_dropped={grounding_dropped_condition})"
                )
                self.age_years = None

        # --- Step 3: canonicalize change_value by change_field ---
        # Codex thorough review 2026-04-20: classifier often mirrors raw JSON
        # ("0" as string, 0 as int) so downstream state stores non-canonical
        # types and string-based invariant checks get bypassed silently.
        # Coerce here so the pair that reaches staging is type-clean OR None.
        if self.change_field is not None and self.change_value is not None:
            canonical = canonicalize_change_value(self.change_field, self.change_value)
            if canonical is _DROP_CHANGE_VALUE:
                drops.append(
                    f"change_value={self.change_value!r} uncanonical for "
                    f"change_field={self.change_field!r}"
                )
                self.change_field = None
                self.change_value = None
            elif canonical is not None:
                self.change_value = canonical

        if drops:
            object.__setattr__(self, "_grounding_drops", drops)
        return self


def parse_classifier_output(raw_text: str, utterance: str) -> ClassifierOutput:
    """Parse a raw Qwen classifier response into a validated ClassifierOutput.

    Never raises. Returns an empty ``ClassifierOutput()`` on:
      - missing or malformed JSON body,
      - ``json.loads`` error,
      - ``ValidationError`` (e.g. classifier emits ``client_type="ИП"``,
        ``change_field="all"``, or an unknown action).

    Utterance is passed through ``model_validate`` context so the post-validator
    can null ungrounded enum fields.
    """
    text = (raw_text or "").strip()
    js_start = text.find("{")
    js_end = text.rfind("}") + 1
    if js_start < 0 or js_end <= js_start:
        return ClassifierOutput()
    try:
        data = _json.loads(text[js_start:js_end])
    except Exception as exc:  # noqa: BLE001
        print(f"[ClassifierSchema] json parse failure: {exc}", flush=True)
        return ClassifierOutput()
    if not isinstance(data, dict):
        return ClassifierOutput()
    try:
        out = ClassifierOutput.model_validate(data, context={"utterance": utterance})
    except ValidationError as exc:
        # Try again with non-strict fields nulled so partial output survives
        # when only one field is bad (e.g. garbage client_type killing the
        # whole turn). Build a clean dict by dropping only the error paths.
        bad_paths = {err["loc"][0] for err in exc.errors() if err.get("loc")}
        clean = {k: v for k, v in data.items() if k not in bad_paths}
        try:
            out = ClassifierOutput.model_validate(clean, context={"utterance": utterance})
            print(
                f"[ClassifierSchema] partial-validate: dropped fields={sorted(bad_paths)} "
                f"kept keys={sorted(clean.keys())}",
                flush=True,
            )
        except ValidationError as exc2:
            print(
                f"[ClassifierSchema] validation failed: {exc2.errors()[:3]} "
                f"raw_keys={sorted(data.keys())}",
                flush=True,
            )
            return ClassifierOutput()
    drops = getattr(out, "_grounding_drops", None)
    if drops:
        print(f"[ClassifierSchema] grounding dropped: {drops}", flush=True)
    return out
