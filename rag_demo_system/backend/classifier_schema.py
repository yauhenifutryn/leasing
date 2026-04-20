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
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator, model_validator

from .profile_hygiene import (
    _CURRENCY_CUE_RE,
    _TYPE_SCHEDULE_CUE_RE,
    _normalize_client_type,
    has_field_signal,
    utterance_has_client_type_cue,
    utterance_has_subject_cue,
)

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

_CHANGE_FIELDS = Literal[
    "subject", "cost", "currency", "client_type", "condition_new",
    "age_years", "term_months", "type_schedule",
    "prepaid_pct", "prepaid_amount", "prepaid",
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
    cost: Optional[float] = None
    currency: Optional[_CURRENCY_VALUES] = None
    client_type: Optional[_CLIENT_TYPE_VALUES] = None
    condition_new: Optional[Literal[0, 1]] = None
    age_years: Optional[int] = Field(None, ge=0, le=50)
    # Wide OOR ranges (-100..500). Calculator's validate_calc_inputs still
    # catches OOR at execution time (Fix 39 behaviour preserved); Pydantic's
    # job here is to catch NaN/Inf/non-numeric, not range-clamp.
    prepaid_pct: Optional[float] = Field(None, ge=-100, le=500)
    prepaid_amount: Optional[float] = None
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

    @model_validator(mode="after")
    def _ground_against_utterance(self, info: ValidationInfo) -> "ClassifierOutput":
        drops: list[str] = []

        # Cross-field rule (no utterance needed): age_years is only meaningful
        # for used equipment. Classifier sometimes emits age_years=3 when the
        # user said "Три года" meaning term_months=36. Null it when
        # condition_new is 1 or unknown.
        if self.age_years is not None and self.condition_new != 0:
            drops.append(f"age_years={self.age_years} (condition_new={self.condition_new})")
            self.age_years = None

        ctx = info.context if isinstance(info.context, dict) else {}
        utterance = ctx.get("utterance", "") if ctx else ""

        if not utterance:
            # Empty utterance path: skip cue-based grounding (keeps unit tests
            # that do not pass utterance working). Cross-field rule above still
            # ran — that's intentional.
            if drops:
                object.__setattr__(self, "_grounding_drops", drops)
            return self

        if self.subject is not None and not utterance_has_subject_cue(utterance):
            drops.append(f"subject={self.subject!r}")
            self.subject = None
        if self.client_type is not None and not utterance_has_client_type_cue(utterance):
            drops.append(f"client_type={self.client_type!r}")
            self.client_type = None
        if self.currency is not None and not _CURRENCY_CUE_RE.search(utterance):
            drops.append(f"currency={self.currency!r}")
            self.currency = None
        if self.type_schedule is not None and not _TYPE_SCHEDULE_CUE_RE.search(utterance):
            drops.append(f"type_schedule={self.type_schedule!r}")
            self.type_schedule = None
        if self.condition_new is not None and not has_field_signal(
            "condition_new", self.condition_new, utterance
        ):
            drops.append(f"condition_new={self.condition_new}")
            self.condition_new = None

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
