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

## Scope (phased execution)

### Phase 3.A — Extract pure functions first

Move these to `backend/profile_state.py` (pure, no side effects, no I/O):
- Sticky-patch logic (current: inline in app.py)
- `has_field_signal` (current: profile_hygiene.py) — move
- Staging block (current: inline)
- State-gate decisions (current: gates 3/4 inline)
- Completeness checks (current: `is_complete_for_calc`) — already on ClientProfile, keep

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
