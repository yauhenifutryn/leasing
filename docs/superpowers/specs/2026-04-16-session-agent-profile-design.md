# Spec 2: SessionAgent and ClientProfile

**Cluster:** B — Session state
**Depends on:** Spec 3 recommended (faster SessionAgent), but can ship with current 35B model
**Blocks:** Spec 1 (calculator consumes profile), Spec 4 (stop detection uses SessionAgent)

## Context

Transcript 779154b4 shows the bot asking *"вы физическое лицо, ИП или юридическое лицо?"* seven times across one call despite Sergey answering once. Same question for cost twice, for subject three times. Current classifier (`app.py:668-798`) reads the last 14 messages of raw transcript and re-decides context per call. No structured memory.

## Problem

1. No `ClientProfile` data structure. Every turn is stateless for business data.
2. Classifier is narrow: it emits `{intent, subject, cost, currency, client_type, prepaid, term, action}` but downstream code uses these as throwaway hints, not as profile updates.
3. No confirmation gate. Calculator can fire before client has confirmed what we heard.
4. No semantic detection of client confirmations ("да, всё верно"), stop requests ("стоп"), or readback asks ("повтори параметры").
5. Classifier re-asks client_type because it doesn't see its own prior extractions.

## Goals

- Introduce `ClientProfile` dataclass, attached to `ChatSession`, mutated incrementally.
- Rename classifier conceptually to `SessionAgent`. Same LLM call, richer JSON schema.
- SessionAgent emits profile patches, semantic intents, completeness flag, missing-field list.
- Collection flow supports batch (one utterance, many fields) and sequential (ask-one-at-a-time).
- Read-back gate: enumerate all fields before first calculator call; on client confirmation, mark `confirmed_at`.
- Change gate: any post-confirmation change triggers single-field read-back before recalc.
- Profile survives the entire session (in-memory).

## Non-goals

- Cross-session persistence (future spec — already designed in `project_session_persistence_design.md`)
- CRM mirror / external sync (separate spec)
- Multiple concurrent profiles per session (single caller = single profile)

## Design

### Data structure

**`backend/session.py` (new file or additions):**

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

ClientType = Literal["Физическое лицо", "Юридическое лицо"]
ScheduleType = Literal["0", "1"]  # 0=annuity, 1=linear

@dataclass
class ClientProfile:
    name: Optional[str] = None
    client_type: Optional[ClientType] = None      # ФЛ / ЮЛ (ИП maps to ЮЛ)
    subject: Optional[str] = None                  # canonical subject name
    cost: Optional[float] = None                   # in stated currency
    currency: Optional[str] = None                 # BYN / USD
    condition_new: Optional[int] = None            # 1 / 0
    age_years: Optional[int] = None                # required when condition_new=0
    prepaid_pct: Optional[float] = None
    prepaid_amount: Optional[float] = None
    term_months: Optional[int] = None
    type_schedule: Optional[ScheduleType] = None

    confirmed_at: Optional[float] = None           # unix ts of last client confirmation
    last_change_pending: Optional[str] = None      # field name awaiting re-confirm
    locked_fields: set[str] = field(default_factory=set)  # fields client said don't change

    REQUIRED_FOR_CALC = {
        "client_type", "subject", "cost", "currency",
        "condition_new", "term_months", "type_schedule",
        "prepaid_pct_or_amount",  # special: either pct or amount
    }

    def missing_fields(self) -> set[str]:
        missing = set()
        for f in self.REQUIRED_FOR_CALC:
            if f == "prepaid_pct_or_amount":
                if self.prepaid_pct is None and self.prepaid_amount is None:
                    missing.add(f)
                continue
            if getattr(self, f) is None:
                missing.add(f)
        if self.condition_new == 0 and self.age_years is None:
            missing.add("age_years")
        return missing

    def is_complete_for_calc(self) -> bool:
        return not self.missing_fields()
```

Attached to existing `ChatSession`:

```python
@dataclass
class ChatSession:
    # ... existing fields
    client_profile: ClientProfile = field(default_factory=ClientProfile)
```

### SessionAgent JSON schema

SessionAgent replaces the classifier. Same LLM call, expanded output:

```json
{
  "intent": "TOOL" | "RAG" | "CONVERSATION",
  "profile_patches": {
    "subject": "Легковой автомобиль" | null,
    "cost": 70000 | null,
    "currency": "BYN" | null,
    "client_type": "Физическое лицо" | null,
    "condition_new": 1 | null,
    "age_years": 3 | null,
    "prepaid_pct": 20 | null,
    "prepaid_amount": null,
    "term_months": 84 | null,
    "type_schedule": "1" | null,
    "name": "Сергей" | null
  },
  "is_confirmation": true | false,
  "is_stop_request": true | false,
  "wants_readback": true | false,
  "change_field": "term_months" | null,
  "change_value": 48 | null,
  "action": "calculate" | "recalculate" | "sms" | "clarify" | "conversation"
}
```

Semantic extraction rules (in the system prompt):
- `is_confirmation=true` when the client's utterance means "yes, go ahead" in context (not just the word "да" alone — context matters).
- `is_stop_request=true` when the client wants the bot silent: "стоп", "подожди", "помолчи", "хватит", "не продолжай", "замолчи".
- `wants_readback=true` when the client asks to review collected parameters ("какие параметры?", "повтори что мы насчитали").
- `change_field/value` pair when the client expresses a change intent: "поменяй срок на 48", "давай без аванса", "нет, в долларах".

### Collection flow

**State machine:**

```
COLLECTING → [profile complete] → READBACK_PENDING → [is_confirmation=true] → CONFIRMED
                                                   → [change_field set] → update field → READBACK_PENDING
CONFIRMED → [change_field set] → CHANGE_CONFIRM_PENDING → [is_confirmation=true] → update → CONFIRMED → recalc
          → [is_stop_request] → LISTEN_MODE (Spec 4)
```

**Priority order for sequential collection (asking next missing field):**

1. `subject` — "Что планируете в лизинг?"
2. `client_type` — "Вы физлицо, ИП или юрлицо?" (but only if not inferable)
3. `cost + currency` — "Какая стоимость и в какой валюте?"
4. `condition_new` — "Новый или с пробегом?"
5. `age_years` (only if used) — "Какого года?"
6. `term_months` — "На какой срок?"
7. `prepaid_pct_or_amount` — "Какой аванс?"
8. `type_schedule` — "График аннуитетный или линейный?"

**Batch collection:** if SessionAgent returns `profile_patches` with ≥ 4 new fields in one turn, bot skips sequential and goes straight to read-back.

### Read-back gate

On transition `COLLECTING → READBACK_PENDING`:

```
Bot: "Проверим параметры: легковой автомобиль, новый, 70 тысяч рублей, физлицо,
      аванс 20 процентов, срок 84 месяца, аннуитетный график. Всё верно?"
```

Then waits for SessionAgent's next output. `is_confirmation=true` → set `confirmed_at`, fire calculator. Any `change_field/value` → update, ask single-field re-confirmation.

### Change gate (post-confirm)

```
Client: "поменяй срок на 48"
SessionAgent: {change_field: "term_months", change_value: 48, is_confirmation: false}
Bot: "Меняю срок на 48 месяцев. Всё верно?"
Client: "да"
SessionAgent: {is_confirmation: true}
Bot: update profile.term_months=48, fire calculator.
```

### Profile merge rules

- On `profile_patches` from SessionAgent: non-null values overwrite profile
  fields IF field is not in `locked_fields`.
- `locked_fields` is populated when client says "не меняй X" or after the third
  attempt to change the same field in a turn (stability heuristic).
- If SessionAgent returns a value conflicting with an existing confirmed value
  without `change_field` set, DO NOT auto-overwrite. Log warning. Trust existing.

## Files to change

- `rag_demo_system/backend/session.py` — add `ClientProfile` dataclass
- `rag_demo_system/backend/app.py` — rewrite classifier → SessionAgent, read-back state machine, profile merge
- `rag_demo_system/config/system_prompt_ru_v2.txt` — new collection protocol
- `rag_demo_system/backend/prompts.py` (or inline in app.py) — SessionAgent prompt template
- Tests: `rag_demo_system/tests/test_session_agent.py`, `test_client_profile.py`

## Testing

**Unit — `test_client_profile.py`**
1. `ClientProfile()` → `missing_fields()` includes all 8 required.
2. Populate all required → `is_complete_for_calc() is True`.
3. `condition_new=0, age_years=None` → `age_years` in missing.
4. Populate pct only → `prepaid_pct_or_amount` not missing.

**Unit — `test_session_agent.py`**
1. Mock LLM: return `{is_stop_request: true}`. Assert state transitions to listen_mode (cross-spec; mock Spec 4 listener).
2. Mock LLM: profile_patches with 5 fields. Assert profile updates all 5.
3. Mock LLM: `change_field="term_months", change_value=48`. Assert `last_change_pending="term_months"`, profile NOT yet updated.
4. Mock LLM: `is_confirmation=true` after `last_change_pending` set. Assert profile.term_months=48, `last_change_pending=None`.

**Integration — transcript replay**
1. Feed transcript 779154b4 turn-by-turn.
2. Assert: after turn at 18:02:16 ("меня зовут Сергей"), `profile.name="Сергей"`.
3. Assert: at turn 18:03:37 ("Отечественные") subject classifier stops re-asking client_type (it's already in profile if previously stated; if not, ask once and remember).
4. Assert: no calculator call before `confirmed_at` is set.
5. Count calculator calls in replay → should drop from current ~10 to ≤ 4.

**Metrics**
- Log profile state at every turn boundary (structured JSON to `.state/profile_snapshots/{session_id}.jsonl`).
- Count duplicate question emissions (bot asks same field twice in same session) — target: zero.

## Risks

| Risk | Mitigation |
|---|---|
| SessionAgent miscategorizes a statement as confirmation | Read-back gate still requires explicit "да"/"всё верно"; ambiguous → bot says "не совсем поняла, всё верно?" |
| Profile gets corrupt value | Change gate requires confirmation before updating post-confirm |
| `locked_fields` heuristic over-locks | Client can say "нет, поменяй X" with explicit change intent → unlock |
| SessionAgent prompt too long, exceeds Qwen context | Use Spec 3's smaller model; prompt fits in 1500 tokens easily |

## Rollback

Feature-flag via env: `SESSION_AGENT_ENABLED=1`. If disabled, fall back to existing classifier-only path. Rollback is env change + restart.
