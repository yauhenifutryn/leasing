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
