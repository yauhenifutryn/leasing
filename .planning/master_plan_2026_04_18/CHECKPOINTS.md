# Master Plan Checkpoints

**Convention**: checkpoints are `[ ]` pending, `[x]` completed, `[~]` in progress, `[!]` blocked.
Update this file atomically as work progresses. Include commit SHA next to completed items.

## Section 1 — Pre-refactor stability

- [ ] CP-1.1 — Fix 1.1 (deterministic numeric rendering) shipped + live-verified
- [ ] CP-1.2 — Fix 1.2 (USD disclosure) shipped + live-verified
- [ ] CP-1.3 — Fix 1.3 (б/у robustness) shipped + live-verified
- [ ] CP-1.4 — Fix 1.4 (VAD bump 500→900) shipped + live-verified
- [ ] CP-1.5 — Section closed, `stability-v1` tag pushed

## Section 2 — Structured classifier

- [ ] CP-2.1 — `backend/classifier_schema.py` with `ClassifierOutput` model
- [ ] CP-2.2 — Hint extraction uses `ClassifierOutput.model_validate`
- [ ] CP-2.3 — Fix 41b `_VALID_CHANGE_FIELDS` whitelist removed
- [ ] CP-2.4 — All existing tests pass
- [ ] CP-2.5 — Live validation failure rate <2%
- [ ] CP-2.6 — Section closed, `structured-classifier-v1` tag pushed

## Section 3 — Architecture refactor

- [ ] CP-3.1 — `backend/profile_state.py` with pure functions + unit tests
- [ ] CP-3.2 — `TurnAction` ADT + `apply_turn` with 100% coverage
- [ ] CP-3.3 — Orchestrator wired to `apply_turn`
- [ ] CP-3.4 — Dead code removed; `app.py` down ~800 lines
- [ ] CP-3.5 — Live regression sweep green
- [ ] CP-3.6 — Code review resolved
- [ ] CP-3.7 — Section closed, `refactor-v1` tag pushed, docs updated

## Section 4 — Natural turn-taking

- [ ] CP-4.0 — Research findings documented
- [ ] CP-4.1 — Baseline perceived-naturalness captured
- [ ] CP-4.2 — Adaptive silence per state
- [ ] CP-4.3 — Filler tolerance
- [ ] CP-4.4 — Prosody endpoint (if feasible)
- [ ] CP-4.5 — Pre-speech delay
- [ ] CP-4.6 — Section closed, `turn-taking-v1` tag pushed

## Sections 5 & 6

Deferred — not tracked until explicitly picked up.

---

## Commit log per checkpoint

| CP | Commit SHA | Date | Notes |
|---|---|---|---|
| baseline | a20aa4f | 2026-04-18 | MVP stable; tag `mvp-2026-04-18` |

(Append rows as checkpoints complete.)
