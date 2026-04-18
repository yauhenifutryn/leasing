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

## Section 5 — Speaker mode adaptive VAD

- [ ] CP-5.0 — Multipliers validated on pilot handset+speaker call
- [ ] CP-5.1 — `derive_rms_floor` + `SPEAKER_RMS_HEURISTIC_ENABLED` knob with unit tests
- [ ] CP-5.2 — Ambient RMS measurement wired; per-session log line present
- [ ] CP-5.3 — Test A (handset regression) passes — ≥10 BARGE-IN, zero ignored speech
- [ ] CP-5.4 — Test B (speaker mode) passes — reliable barge-in, no echo false positives
- [ ] CP-5.5 — Section closed, `speaker-mode-v1` tag pushed

## Section 6 — Final: verification, docs, self-improvement

- [ ] CP-6.0 — Verification gate: VERIFICATION.md committed, all rows PASS
- [ ] CP-6.1 — Full UAT scenario passes live; `uat_<date>.log` archived
- [ ] CP-6.2 — README.md architecture section updated
- [ ] CP-6.3 — PROJECT_LOG.md appended with section entries
- [ ] CP-6.4 — Deploy runbook updated (if needed)
- [ ] CP-6.5 — session_analyzer.py extended with new fix signals
- [ ] CP-6.6 — kb_gap_report extended
- [ ] CP-6.7 — Obsolete memories archived; `project_v1_complete_*.md` written
- [ ] CP-6.8 — Final tag pushed; PR opened to main

---

## Commit log per checkpoint

| CP | Commit SHA | Date | Notes |
|---|---|---|---|
| baseline | a20aa4f | 2026-04-18 | MVP stable; tag `mvp-2026-04-18` |

(Append rows as checkpoints complete.)
