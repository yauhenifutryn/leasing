# Section 3 — Architecture refactor (apply_turn transaction model)

**Status**: pending
**Prereqs**: Section 2 complete (ClassifierOutput in use)
**Estimated effort**: ~2-3 days
**Base commit**: end of Section 2

## Goal

Extract the orchestrator's 7+ mutation gates into a single transaction function that takes `(profile, classifier_output, utterance)` and returns `(new_profile, emit_action)`. Orchestrator becomes dumb plumbing. Removes `_just_confirmed_this_turn`-style state-leak hacks.

This is the **high-risk, high-reward** section. Expect a full day of bug-hunting after the initial extraction. Mitigate with thorough TDD and the superpowers subagent pattern for parallel verification.

## Required memories (retrieve before starting)

- `project_orchestrator_refactor_pending.md` — the full plan this section executes
- `project_mvp_complete_2026_04_18.md` — baseline to regression-test against
- `feedback_universal_fixes.md` — systemic > special-case
- `feedback_benchmark_order.md` — the user wants the refactor to preserve all 42 fixes' behaviors
- `feedback_propose_docs_update.md` — update README + architecture docs after

## Primary skill

`superpowers:brainstorming` first (validate the design one more time against fresh eyes). Then `superpowers:writing-plans` to lay out the extraction. Then `superpowers:executing-plans` or `superpowers:subagent-driven-development` for the work itself. `superpowers:test-driven-development` is non-negotiable here. `superpowers:requesting-code-review` before merging.

Borrow `gsd-forensics` if something breaks badly mid-refactor — it's designed for post-mortem of a failed GSD workflow.

## Design summary

From `project_orchestrator_refactor_pending.md`:

```python
def apply_turn(
    profile: ClientProfile,
    classifier_output: ClassifierOutput,  # Section 2 output
    utterance: str,
) -> TurnResult:
    """Returns (new_profile, action) where action is one of:
       - EmitClarify(missing_fields)
       - EmitReadback()
       - EmitChangeConfirm(changes)
       - FireCalc(params)
       - FireLLMFallback(reason)
       - Noop
    """
```

- Profile mutations happen in one place (apply_turn) not 5
- State machine drives the action, not side effects of gate ordering
- Orchestrator becomes: call classifier → call apply_turn → execute action → log

## Evidence from live calls 2026-04-19 (Section 1 testing)

Session `205add5a` exposed two structural failures this refactor must close, both unfixable with more gate-patches and both outside Section 1's scope:

**E3 — LLM paraphrase strips deterministic content.**

Fix 1.1 (commit `454dac8`) made the post-calc voice summary deterministic via `render_calc_result` in `profile_prompts.py`. Log confirms the renderer emitted the full string with USD prefix:

```
[deterministic_readback] source=render_calc_result session=205add5a chars=226
[DirectTool] presenting: Стоимость 120000 долларов (это 360000 белорусских рублей по курсу 3 к 1). Аванс 30.0%: 108000.0 BYN.
```

Yet what the client actually heard on the call (TTS of LLM paraphrase of that string):

> "Аванс составляет 108 тысяч рублей, ежемесячный платёж 10 635 рублей, а общая сумма сделки 494 459 рублей."

The USD disclosure ("Стоимость 120000 долларов …") was dropped by the LLM. The call site at `rag_demo_system/backend/app.py:~2384` still hands `_result_summary` to the LLM with the instruction "Назови аванс, ежемесячный платёж и общую сумму" — the LLM obeys that narrower instruction and discards the leading sentence. Fix 1.1 only made the *input* deterministic; the *output* is still LLM-authored.

**Refactor requirement:** post-calc TTS must be driven directly from `render_calc_result` output, not from an LLM paraphrase of it. Either:
- Route the calc-result summary straight to TTS (skip LLM for this turn), OR
- Keep the LLM in the loop but strictly for tone/empathy markers, structured as a prompt that *appends to* rather than replaces the deterministic body.

This should live inside `apply_turn`'s `FireCalc` action handling — that action returns a deterministic text payload and the orchestrator's `execute_action` sends it verbatim to TTS.

**E4 — Implicit-change auto-apply from a hallucinated classifier turn.**

Same session, `205add5a`, turn 11 after a successful readback confirm:

```
[SessionAgent] raw={"intent": "CONVERSATION", "subject": "Прочий транспорт", ...}
[Profile] stale subject patch ignored: 'Прочий транспорт' ...
[Profile] snapshot: state=READBACK_PENDING ... → CHANGE_PENDING
```

The hygiene gate correctly dropped the hallucinated `subject` patch, but the implicit-change staging block (`[Profile] CHANGE_PENDING (implicit)` emitted later) still captured it and produced a "Меняю предмет лизинга на Прочий транспорт" change-confirm. The next classifier turn had `is_confirmation: true` (from the caller saying "Микро Лизинг") and the change applied — calc subsequently ran on `subject="Прочий транспорт"` despite the caller having confirmed `"Легковой автомобиль"` one turn earlier.

**Refactor requirement:** the implicit-change pathway and the sticky-patch pathway must go through the SAME grounded-patch set. Today they divide responsibility: hygiene owns the primary patches, staging has its own `has_field_signal` check, and the two can disagree. In `apply_turn`, there is one `grounded_patches` set computed up-front (Section 2.3b's Option A), and every downstream decision — primary apply, staging, implicit-change, change-confirm — reads from the same set.

This supersedes Fix 40b's `multi-field unlock` logic at `app.py` (currently spread across ~4 call sites) and the `[Profile] CHANGE_PENDING (implicit)` path (around app.py emitting that log line).

## Evidence from live call 2026-04-20 (post-Section-2 SIP test)

Session `cc7fc318` (38.80.122.98, 2026-04-20 17:55-17:58) was the CP-2.5 live validation for Section 2. The schema layer passed all metrics (0 validation failures, 0 parse failures, 0 state-loss guards, 4 healthy grounding drops). But the SAME CALL surfaced four orchestrator bugs that Section 2 cannot fix — they belong to Section 3 by design. These are the concrete acceptance tests for `apply_turn`:

**E5 — Readback gate requires intent=TOOL, skips on intent=CONVERSATION (app.py:2022).**

At 17:57:07 user said "Аннуитетный график" after all other params were captured. Classifier labeled the turn `intent=CONVERSATION` (Qwen's legitimate read: "user is confirming a param choice, not asking to calculate"). Profile transitioned to `missing=[]` on this turn.

```
[Orchestrator] COLLECTING clarify: patched=['type_schedule'] still_missing=[] intent=CONVERSATION
```

Gate 2 (readback emit) at `app.py:2022-2090` is wrapped in `if needs_tool:` — since `needs_tool = (_intent == "TOOL")` and intent was CONVERSATION, the gate was skipped entirely. Bot went to the LLM clarify path instead, which freewheeled chat ("Аннуитетный график означает равные ежемесячные платежи...") and never spoke the deterministic readback.

User eventually said "Ну, рассчитай" three turns later, forcing intent=TOOL, and only then did calc fire — but with no readback-confirm step, so the contract was broken.

Section 1's 5 SIP test calls didn't hit this because those callers ended the last field-fill turn with an explicit calc trigger word. Qwen's intent labelling for plain-answer turns is legitimately ambiguous; the orchestrator must not depend on it.

**Refactor requirement:** `apply_turn` returns `EmitReadback()` whenever `profile.is_complete_for_calc() and state == COLLECTING and confirmed_at is None and not is_confirmation`. Classifier intent label is IRRELEVANT to this transition. Verify: write a unit test where `classifier_output.intent == "CONVERSATION"` and profile just became complete — assert the returned action is `EmitReadback`.

**E6 — Change-confirm bypassed; direct-call fires on user's first change utterance (app.py:2047-2059 + 2106-2168).**

Session `cc7fc318` turn at 17:58:47. User said "А давай всё-таки поменяем срок на 60". At 17:58:48 — 1 second later — calculator was called with `term=60`. No "Меняю срок на 60 месяцев, всё верно?" step. No CHANGE_PENDING transition.

Root cause in the current orchestrator: `_is_param_change_for_gate` (line 2047) bypasses Gate 1/2 when `action in ("change_param", "recalculate")` AND there's a prior calc in history AND there's a fresh param hint. Then the direct-call path at line 2106 applies the change and fires calc in one turn. The CHANGE_PENDING staging block at the always-on state-gate region runs BEFORE this, but only fires when `state in (CONFIRMED, CHANGE_PENDING)` — and today's state was stuck in COLLECTING (see E5) so CHANGE_PENDING was never entered.

**Refactor requirement:** `apply_turn` returns `EmitChangeConfirm(changes)` on any turn where the classifier emitted a change_field/change_value pair that differs from the current profile AND the profile is complete, REGARDLESS of current state. Direct-apply is never allowed for user-initiated changes — only confirmed changes flow through. Verify: unit test with profile complete + classifier emits `change_field=term_months, change_value=60` — assert action is `EmitChangeConfirm({"term_months": {"old": 36, "new": 60}})`, NOT `FireCalc`.

**E7 — Profile state lost across implicit subject flips (same session).**

At 17:57:49 user said "Да, я всё-таки хочу грузовой автомобиль". Profile had: subject=Легковой, client_type=Физическое, cost=80000, currency=USD, term=36, prepaid=20, type_schedule=0. Bot response at 17:57:50: "Грузовые автомобили доступны только для юридических лиц. Вы ИП или организация?" — correct pivot. Then at 17:58:00 after user said "Я организация", bot responded "подтвердите стоимость, год выпуска и размер аванса" — asking for data it already had.

Log confirms profile state was intact the whole time:
```
[Profile] snapshot: state=COLLECTING ... subj=Грузовой автомобиль cost=80000.0 USD
client_type=Юридическое лицо cond_new=1 term=36 prepaid=20.0% graph=0 missing=[]
```

`missing=[]` yet bot asked clarifying questions about captured params. The LLM clarify path was re-entered because state never transitioned to CONFIRMED (see E5), and the LLM-facing prompt didn't include the current profile snapshot as constraint — it improvised.

**Refactor requirement:** when `apply_turn` decides to emit clarification text (`EmitClarify` or a re-prompt variant), the rendered prompt MUST contain the current profile's captured values inline as anti-hallucination anchors. This was partially done in Fix 1.1's deterministic `render_calc_result`; `apply_turn` extends the same discipline to clarification turns. Verify: feed a complete-profile + subject-change turn through `apply_turn` twice in a row, assert the LLM prompt on the second call contains "cost=80000" verbatim.

**E8 — USD disclosure regression is E3, same bug, fresh repro.**

Session `cc7fc318` at 17:57:32 heard: "Аванс составит 48 000 рублей, ежемесячный платёж 8109 рублей, а общая сумма сделки 342 317 рублей." No USD context, even though user said $80 000 and the deterministic renderer emitted it. Same pattern as Section 1 call `205add5a` (E3 above). Two independent repros confirm E3 is structural, not a flake. `apply_turn`'s `FireCalc` handler must ship the deterministic body to TTS unparaphrased.

## Updated acceptance criteria for CP-3.5 (live regression sweep)

CP-3.5 fails if any of these replay on a fresh SIP call after the refactor:

1. Bot skips readback when the last-field-fill utterance is ambiguous intent (E5).
2. Bot applies a change_field and fires calc in the same turn without a confirmation beat (E6).
3. Bot asks for data already captured after an implicit profile pivot (E7).
4. Post-calc narration drops USD disclosure when `original_currency == "USD"` (E8/E3).

Each must have a unit test at `apply_turn` level BEFORE the live sweep.

## Scope (phased execution)

### Phase 3.A — Extract pure functions first

Move these to `backend/profile_state.py` (pure, no side effects, no I/O):
- Sticky-patch logic (current: inline in app.py)
- `has_field_signal` (current: profile_hygiene.py) — move
- Staging block (current: inline)
- State-gate decisions (current: gates 3/4 inline)
- Completeness checks (current: `is_complete_for_calc`) — already on ClientProfile, keep

**Target block (Codex review 2026-04-19):** the always-on state-gate body at `rag_demo_system/backend/app.py:1785-1917` is the specific code region `apply_turn` subsumes. It owns READBACK denial, CHANGE_PENDING confirmation, and the `_just_confirmed_this_turn` / `_change_staged_this_turn` bookkeeping flags that the original orchestrator-refactor memo flagged as code smell. The existing Section 3 scope says "state machine drives the action, not side effects of gate ordering" but did not name the block — naming it here removes ambiguity during the refactor and makes it trivial to verify on completion (`grep -n "_just_confirmed_this_turn" backend/app.py` must return 0, `grep -n "apply_turn" backend/app.py` must return 1).

Each function gets unit tests that lock in current behavior. Run full regression before moving to 3.B.

### Phase 3.B — Define `TurnResult` action ADT

```python
from dataclasses import dataclass
from typing import Union

@dataclass
class EmitClarify: missing: list[str]
@dataclass
class EmitReadback: pass
@dataclass
class EmitChangeConfirm: changes: dict
@dataclass
class FireCalc: params: dict
@dataclass
class FireLLMFallback: reason: str
@dataclass
class FireOORMessage: message: str
@dataclass
class Noop: pass

TurnAction = Union[EmitClarify, EmitReadback, EmitChangeConfirm,
                   FireCalc, FireLLMFallback, FireOORMessage, Noop]
```

### Phase 3.C — Write `apply_turn` as a dispatch

- Input: profile (immutable view), classifier_output, utterance
- Output: (new_profile, TurnAction)
- Uses only pure functions from 3.A
- 100% unit test coverage before calling from orchestrator

### Phase 3.D — Wire into orchestrator

Replace the big block in `app.py` (roughly lines 1290-2100) with:

```python
new_profile, action = apply_turn(session.client_profile, classifier_output, message)
session.client_profile = new_profile
await execute_action(action, websocket, session, session_id, backend, message)
```

`execute_action` is thin — just reads the TurnAction variant and does the IO.

### Phase 3.E — Delete dead code

After 3.D, many legacy variables become dead: `_change_staged_this_turn`, `_just_confirmed_this_turn`, `_has_live_signal`, `_allow_direct_apply`, `_collect_profile_42b`, etc. Delete them.

### Phase 3.F — Regression sweep

Run the full unified scenario from `project_mvp_complete_2026_04_18.md` live. Verify every checkpoint passes. Compare LATENCY log means before/after — expect slight improvement from less branching.

## Checkpoints

- CP-3.1: `backend/profile_state.py` exists with all pure functions + unit tests. `pytest tests/test_profile_state.py` green.
- CP-3.2: `TurnAction` ADT defined. `apply_turn` implemented. 100% unit test coverage on `apply_turn`.
- CP-3.3: Orchestrator (`_stream_voice_response`) uses `apply_turn`. All existing tests pass.
- CP-3.4: Dead code removed. Line count of `app.py` down by ~800 lines.
- CP-3.5: Live regression sweep — full unified scenario passes end-to-end.
- CP-3.6: Code review invoked (`superpowers:requesting-code-review`). Issues triaged.
- CP-3.7: Section closed — commit tagged `refactor-v1`. Docs updated (`README.md` architecture section).

## Rollback

```bash
# Rollback to pre-refactor (end of Section 2):
git reset --hard <section-2-end-sha>
git push --force-with-lease origin feature/voice-pipeline
```

Full MVP fallback:
```bash
git reset --hard mvp-2026-04-18
git push --force-with-lease origin feature/voice-pipeline
```

## Risk notes

- **The 42 fixes are behavior-locked**. Unit tests exist for most but not all behaviors. Before deleting a code path, write a test that captures its current behavior, then delete + verify test still passes under the new implementation.
- **Snapshot format changes will break logs**. Keep the `[Profile] snapshot:` format identical so log greps from operators still work.
- **Jambonz/WebSocket integration points stay stable**. Only internal orchestrator changes.

## Handoff to Section 4

Section 4 (natural turn-taking) depends on a clean state machine to add adaptive endpointing cleanly. Verify before starting 4:
```bash
grep -n "_just_confirmed_this_turn\|_change_staged_this_turn" backend/app.py   # should be 0
grep -n "apply_turn" backend/app.py   # should be 1
```
