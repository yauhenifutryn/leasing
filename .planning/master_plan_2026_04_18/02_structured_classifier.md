# Section 2 — Structured classifier output

**Status**: pending
**Prereqs**: none (can run parallel to Section 1)
**Estimated effort**: ~1 day
**Base commit**: `mvp-2026-04-18` (`a20aa4f`)

## Goal

Replace the free-form JSON classifier prompt with schema-validated structured output (Pydantic). Kills three recurring failure modes:
1. `change_value: 0` emitted when user meant "no value" (causes term=0 ghost)
2. `change_field: "all"` or other non-field values (leaked into readback as "Меняю all на...")
3. `is_confirmation: true` falsely set on messages containing "давай"

This is a **self-contained, invisible-to-user change** that makes Section 3 (architecture refactor) much easier to reason about.

## Required memories (retrieve before starting)

- `project_mvp_complete_2026_04_18.md` — current state
- `project_orchestrator_refactor_pending.md` — why we're doing this
- `feedback_universal_fixes.md` — systemic > special-case
- `feedback_pin_all_versions.md` — any new dependency must be version-pinned

## Primary skill

`superpowers:writing-plans` first (this is a contained implementation — write the plan, get approval, execute). Then `superpowers:executing-plans` to work through the plan. `claude-api` skill for Anthropic SDK best-practices if we use function-calling format. `superpowers:test-driven-development` for each classifier output shape.

## Research needed first

- Does Qwen3-4B support OpenAI function-calling format reliably via vLLM? Confirm at current deployment.
- Alternative: use `response_format: {"type": "json_object"}` with Pydantic validation post-parse.
- Look up vLLM structured outputs guidance (they support guided decoding / guided JSON).

Run these as part of Research phase before writing code:
```bash
# Confirm vLLM structured output support
grep -r "response_format\|guided_json\|guided_regex" /ephemeral/leasing/rag_demo_system/scripts/ 2>/dev/null
# Check current classifier call site
grep -n "classify_resp\|classifier" backend/app.py | head -20
```

## Evidence from live calls 2026-04-19 (Section 1 testing)

Two live calls (SIP sessions `c422f14b`, `205add5a`) exposed failure modes this section must close. Raw logs quoted from `.state/backend.log`:

**E1 — Classifier emits a valid enum value from an utterance with no matching signal.**

Session `205add5a`, turn 11 — after a successful readback, Whisper decoded the user saying our own brand name as "Микро Лизинг". Classifier response:

```
[SessionAgent] raw={"intent": "CONVERSATION", "subject": "Прочий транспорт", ...}
```

`profile_hygiene.py:126 utterance_has_subject_cue()` correctly dropped the patch (`[Profile] stale subject patch ignored: 'Прочий транспорт'`), but the downstream implicit-change staging path still picked it up and emitted a "Меняю предмет лизинга на Прочий транспорт" readback. **Pydantic alone will not prevent this** — "Прочий транспорт" is a valid enum value. What's missing is an *utterance-grounding* check inside the schema itself (or immediately after model_validate).

**E2 — Same failure on two other enum fields in the same session.**

- `type_schedule: "1"` emitted from utterance "120 000 долларов новую" (no graph word). Profile log:
  ```
  [Profile] patches_post_filter={'condition_new': 1, 'cost': 120000, 'type_schedule': '1', 'currency': 'USD'}
  ```
- `age_years: 3` emitted from "Три года, аванс 30%, и всё-таки давай аннуитетный график" — client meant `term_months=36`, classifier ALSO assigned 3 to age_years. Profile log:
  ```
  [Profile] patches_post_filter={'age_years': 3, 'prepaid_pct': 30, 'term_months': 36, 'type_schedule': '0'}
  ```

Note `condition_new=1` (new car) — age is not even relevant to the calculator for new equipment. A Pydantic cross-field validator could flag this: "age_years only meaningful when condition_new==0".

**Implication for this section's scope:** Pydantic schema as specced in 2.1 constrains enum sets and numeric ranges but does NOT prevent hallucinated-yet-schema-valid extraction. A second layer is needed.

**E-Codex — additional prompt-contract drift, surfaced by independent review 2026-04-19.**

- The classifier prompt's `client_type` schema at `rag_demo_system/backend/app.py:1146-1147` still advertises `"ИП"` as a valid value, while the extraction rule at lines 1181-1189 tells the model to collapse all business forms (ИП, самозанятый, etc.) to `"Юридическое лицо"`. `ClientProfile` accepts `"ИП"` as a `ClientType` literal (`session.py:15`). That's three sources of truth, all disagreeing. Pydantic will reify whichever one Claude encodes into `ClassifierOutput`, but the prompt the model reads is self-contradictory. **2.1 must pick one answer** — recommended: drop ИП from both the schema and the ClientType literal, keep only "Физическое лицо" and "Юридическое лицо" (matches calculator payload).
- The classifier prompt's `action` enum at `app.py:1159` lists only `"calculate"|"recalculate"|"sms"|"clarify"|"confirm"`, but downstream orchestrator branches at `app.py:1592-1594` and `2321-2339` require `"change_param"`, `"clarify_client_type"`, and `"invalid_param"`. Model cannot emit what it isn't told about. **2.1 must list the full downstream vocabulary** or **2.1's design must consciously fold the missing actions back into the 5 advertised ones** and document the fold.

**E-Codex-2 — READBACK_PENDING deny-with-correction leaks ungrounded hints.**

The staging path at `rag_demo_system/backend/app.py:1830-1843` takes raw classifier hints that differ from profile and pushes them into `pending_change` without running through `has_field_signal(...)`. This is a second ingress for hallucinated fields, **independent from** the implicit-change bug Claude called out (E1/E4). If Section 2 ships with schema-grounded output but this path still consumes raw hints, it's still vulnerable. **Requirement:** every state gate must consume only schema-grounded output post-2.3b — no raw `_sa_parsed.get(...)` reads remain in app.py's state-gate regions.

## Scope

### 2.1 — Define `ClassifierOutput` Pydantic model

New file `backend/classifier_schema.py`:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional

class ClassifierOutput(BaseModel):
    intent: Literal["TOOL", "RAG", "CONVERSATION"]
    subject: Optional[Literal["Легковой автомобиль", "Грузовой автомобиль",
                              "Спецтехника", "Оборудование", "Недвижимость",
                              "Прочий транспорт"]] = None
    cost: Optional[float] = None
    currency: Optional[Literal["BYN", "USD", "EUR", "RUB"]] = None
    client_type: Optional[Literal["Физическое лицо", "Юридическое лицо"]] = None
    condition_new: Optional[Literal[0, 1]] = None
    age_years: Optional[int] = Field(None, ge=0, le=50)
    prepaid_pct: Optional[float] = Field(None, ge=-100, le=500)  # OOR allowed
    prepaid_amount: Optional[float] = None
    term_months: Optional[int] = Field(None, ge=-100, le=500)    # OOR allowed
    type_schedule: Optional[Literal["0", "1"]] = None
    name: Optional[str] = None

    is_confirmation: bool = False
    is_stop_request: bool = False
    wants_readback: bool = False

    change_field: Optional[Literal[
        "subject", "cost", "currency", "client_type", "condition_new",
        "age_years", "term_months", "type_schedule",
        "prepaid_pct", "prepaid_amount", "prepaid"
    ]] = None
    change_value: Optional[str | int | float] = None

    action: Optional[Literal[
        "calculate", "recalculate", "sms", "clarify",
        "clarify_client_type", "confirm", "change_param"
    ]] = None
```

Note: OOR numeric ranges kept wide (not 0-40 for prepaid) so classifier can emit raw user input; calculator's `validate_calc_inputs` still catches OOR at execution time (Fix 39 behavior preserved).

### 2.2 — Wrap the classifier call

Replace the free-form prompt call in `app.py:1226` with a structured call:
- Option A: Pass Pydantic schema to vLLM via `response_format` / `guided_json`
- Option B: Keep current prompt, parse output through `ClassifierOutput.model_validate(...)`, fall back to empty on validation failure

Start with Option B (safer, easier to revert). Measure validation failure rate in prod. If low (<2%), Option B is sufficient. If high, move to Option A.

### 2.3 — Update hint extraction

Lines 1240-1275 in `app.py` become:

```python
parsed = ClassifierOutput.model_validate(_sa_parsed) if _sa_parsed else None
if parsed:
    if parsed.subject: _extracted_hints["subject"] = parsed.subject
    # ... etc, but now Pydantic guarantees types
```

Remove the type-guard prints (`if _sa_parsed.get("prepaid_pct") is not None` etc) since Pydantic handled it.

### 2.3b — Add utterance-grounding layer (NEW — added 2026-04-19 from live-call evidence)

Pydantic constrains the enum *set* but not whether the classifier's extraction matched something the user actually said. Today's call `205add5a` emitted three schema-valid yet utterance-ungrounded fields in a single session (E1/E2 above).

Two options for where this layer lives:

**Option A — Pydantic post-validate step inside `classifier_schema.py`.** Pass `utterance` as a context dict to `model_validate(data, context={"utterance": msg})`. Use `@model_validator(mode="after")` on `ClassifierOutput` to null out enum fields whose cue isn't present in the utterance. Reuses the regex authorities already in `profile_hygiene.py` (`utterance_has_subject_cue`, `utterance_has_client_type_cue`, `_CONDITION_NEW_CUE_RE`, `_TYPE_SCHEDULE_CUE_RE`, `_CURRENCY_CUE_RE`). Pros: one place for all classifier validation, ugly extractions never leave the schema. Cons: couples schema file to regex module.

**Option B — Keep `profile_hygiene.filter_patches` as the authority and run it on the Pydantic-validated dict.** This is closer to current flow. Pros: less churn. Cons: schema and grounding split across two files; staging path (`app.py` around `[Profile] extras: dropping ...`) still needs its own duplicate grounding check — the source of the "Прочий транспорт" implicit-change leak in session `205add5a`.

**Recommendation:** Option A. Move the regex authorities to the schema module (or have the schema import them). Then the staging path can `ClassifierOutput.model_validate(raw, context={"utterance": msg})` and trust that ungrounded fields are already `None`. Kills the "dropped by hygiene but picked up by staging" divergence.

Also add a `@model_validator` for the cross-field rule surfaced by E2: `age_years` must be `None` when `condition_new in (None, 1)`. Today's classifier emits `age_years=3` from "Три года" intended as term.

### 2.4 — Remove the `change_field` whitelist guard (Fix 41b)

Our Fix 41b ad-hoc whitelist becomes redundant — Pydantic Literal enforces the same rule. Delete those lines.

### 2.5 — Remove `param_out_of_range` numeric fallback handling

Partially: Pydantic won't catch OOR (we want it wide), but catches NaN/Inf/string-where-number. So `param_bad_type` from Fix 39 validator can rely on inputs already being numeric. Simplify `validate_calc_inputs`.

### 2.6 — Tests

- `test_classifier_schema.py` — every field validates correctly
- `test_classifier_schema.py::test_invalid_change_field_rejected` — "all" raises ValidationError
- `test_classifier_schema.py::test_change_value_zero_requires_signal` — still handled by Fix 40e (post-Pydantic)
- Integration test: feed 10 known classifier outputs (happy path + malformed) through the new parsing

## Checkpoints

- CP-2.1: `backend/classifier_schema.py` created with ClassifierOutput model. Import test passes.
- CP-2.2: Hint extraction in `app.py` uses `ClassifierOutput.model_validate`. Validation failure falls back to empty parse (never raises up the stack).
- CP-2.3: Fix 41b `_VALID_CHANGE_FIELDS` whitelist removed (redundant with Pydantic Literal).
- CP-2.4: All existing tests pass (`pytest tests/ -q` matches baseline).
- CP-2.5: Live call produces valid parse on every turn. Log `grep -c "ClassifierOutput validation failed" .state/backend.log` returns <5 over a 10-turn session.
- CP-2.6: Section closed — commit tagged `structured-classifier-v1`.

## Rollback

```bash
git revert <commits-in-this-section> --no-edit
git push
```

Or hard reset to `mvp-2026-04-18` if Section 1 is also on the same branch.

## Handoff to Section 3

Section 3 (architecture refactor) assumes `ClassifierOutput` exists and is used everywhere. Verify before starting 3:
```bash
grep -n "ClassifierOutput" backend/app.py | wc -l   # should be > 0
grep -n "_sa_parsed\.get" backend/app.py | wc -l    # should be 0 after refactor
```
