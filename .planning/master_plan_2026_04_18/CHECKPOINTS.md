# Master Plan Checkpoints

**Convention**: checkpoints are `[ ]` pending, `[x]` completed, `[~]` in progress, `[!]` blocked.
Update this file atomically as work progresses. Include commit SHA next to completed items.

## Section 1 — Pre-refactor stability

- [x] CP-1.1 — Fix 1.1 (454dac8), [deterministic_readback] marker confirmed on calls 205add5a, 674e3957, 1e30bfee, 22028754, 743c1a0e, 9ec50128
- [x] CP-1.2 — Fix 1.2 (4efa723), SMS body USD disclosure verified on call 1e30bfee; post-calc summary verified (with expected E3 LLM-paraphrase caveat)
- [x] CP-1.3 — Fix 1.3 (a8f2cf3), "БУ мотоцикл" / "бэу-машину" → condition_new=0 first turn, calls c422f14b + 9ec50128
- [x] CP-1.4 — Fix 1.4 (f4041ac), VAD 900ms live (Section 4 is the adaptive successor)
- [x] CP-1.5b — Fix 1.5 + 1.11 + 1.12 + 1.13 (15536d2, c630fa8, 0ed4b2e), all live-verified on call 9ec50128 (age clarify fires, readback includes age, no raw field names)
- [x] CP-1.5 — Section closed, `stability-v1` tag pushed 2026-04-19 at commit 0ed4b2e

## Section 2 — Structured classifier

- [x] CP-2.1 — `backend/classifier_schema.py` (72b91cb): ClassifierOutput + @model_validator grounding + parse_classifier_output
- [x] CP-2.2 — `parse_classifier_output` wired into app.py (484dab7)
- [x] CP-2.2b — ИП dropped from ClientType + prompt; action enum expanded (41aa50c)
- [x] CP-2.3 — Fix 41b `_VALID_CHANGE_FIELDS` whitelist retired (277ea78)
- [x] CP-2.4 — Test baseline: 504 pass + 8 pre-existing env failures; 68 new schema tests green
- [x] CP-2.4b — Codex review loop (5 adversarial + 2 basic passes) resolved all 13 findings:
    - a100e74 — empty-dict fallback dead, ИП dropped at schema (pass 2)
    - 7528262 — cue-presence vs value-aware grounding, prepaid alias state-loss (pass 3)
    - a668482 — subject regex collision, validator ordering, empty-utterance bypass (pass 4)
    - 6559724 — change_field enum bypass, apply_pending_change silent-success (pass 5)
    - 623a4e2 — change_value canonicalization, NaN/inf, type_schedule numeric (thorough)
    - 945aaea — E-Codex-2 READBACK deny grounding
    - 9ea4f7b — condition_new / currency top-level coercers (basic P1/P2)
    - 25cd065 — fractional change_value fail-closed (basic P2)
- [x] CP-2.5 — Live SIP call 2026-04-20 (session cc7fc318 on 38.80.122.98): 0 validation failures, 0 json parse failures, 0 state-loss guards, 4 grounding drops (healthy). Well below <5 threshold.
- [x] CP-2.6 — `structured-classifier-v1` tag pushed 2026-04-20. Section CLOSED. Orchestrator bugs observed in cc7fc318 (readback skip, change-confirm bypass, profile-memory loss, USD disclosure drop) are NOT Section 2 scope — documented in Section 3 plan as E5/E6/E7/E8 and are acceptance criteria for `apply_turn`.

## Section 3 — Architecture refactor

- [x] CP-3.1 — `backend/profile_state.py` with pure functions + unit tests (c0de06d..f7ed575, 2026-04-23). 23 tests, 100% line coverage. build_snapshot + partition_patches + derive_implied_flips + build_calc_params.
- [x] CP-3.2 — `TurnAction` ADT + `apply_turn` with 100% coverage (1a77245..[task-14-commit], 2026-04-23). 69 tests (29 apply_turn + 17 turn_action + 23 profile_state). 100% line coverage on turn_action.py, profile_state.py, turn_dispatcher.py (162 stmts). All 8 dispatch steps wired with RED→GREEN per step. 580 total tests pass, 8 pre-existing env failures unchanged.
- [x] CP-3.3 — Orchestrator wired to `apply_turn` (Tasks 16-21, 2026-04-24). `execute_action` handlers (FireCalc + EmitReadback + EmitClarify + EmitChangeConfirm + FireOORMessage + Noop + FireLLMFallback) complete with calc circuit breaker + tool-call history. Four production adapters (`LLMStreamBackend`, `TtsSink`, `CalcAdapter`, `RagFuture`) bridge execute_action to `_stream_voice_response`. `APPLY_TURN_ENABLED` flag gates the new path (default "0" — legacy still live until CP-3.5 passes). 99 Phase 3 tests + 11 adapter tests green on both flag values.
- [x] CP-3.4 — Dead code removed (a1e53f4, 2026-04-26). `_stream_voice_response` shrank ~2629→850 lines (1779 net deletion). Removed legacy 5-gate block, sticky-patch staging, change-staging extras, `apply_patches` invocation, helpers `_sticky_calc_ready` + `_format_invalid_params_msg`, `APPLY_TURN_ENABLED` env flag, `detect_sms_intent` import. 5 obsolete contract tests deleted. Test baseline 808 passed / 8 pre-existing failures preserved.
- [x] CP-3.5 — Live regression sweep green. Tag `demo-mvp-flag1-2026-04-26-validated` at 67201ec on `feature/section-3-apply-turn`. Call e180107b on 2026-04-26 cleared every edge case from the Bug Q+S playbook: age-phase ("Два года" → age_years=2), term-phase mixed-field ("Давай три года, 35 процентов и линейный" all captured in one turn), Bug S inside change cycle ("на 7 лет" → срок 84 месяца change-confirm), recalc cycle (linear→annuity), SMS one-shot, RAG owner post-calc. Bugs A-S all closed. Year-form grounding is fully semantic (state-aware, no keyword regex). 324 relevant unit tests pass.
- [x] CP-3.6 — Codex adversarial review + standard review both clean. 1 HIGH (apply_pending_change return value, `73ccd9a`) + 2 P1/P2 (utterance fallbacks dead, memory_block dropped — both `32b3324`) factual findings auto-fixed with 5 new regression tests. Final test count: 814 passed / 8 pre-existing failures unchanged.
- [x] CP-3.7 — Section closed, `refactor-v1` tag pushed, README + PROJECT_LOG updated. Live SIP test on call `bf7a95a8` (2026-04-26 12:07) cleared the full happy path + USD→BYN dual disclosure + change-confirm 36→24 months recalc + SMS direct-fire + post-calc RAG (Codex P2 memory_block fix verified — bot retained context across calc). Two open observations from the live call deferred to memory: subject default-to-car (open product question) and clarify-meta-question regression (Section-3.5/4 candidate, ~30 lines).

## Section 3.5 — Calculator funnel API integration

**Spec**: `docs/superpowers/specs/2026-04-24-calculator-funnel-design.md`
**Prereq**: Section 3 fully closed (CP-3.4..CP-3.7 all `[x]`), `refactor-v1` tag pushed, merged to `feature/voice-pipeline`.

- [ ] CP-3.5.1 — `backend/calc_limits.py` with `subjects` / `currencies` / `ranges` fetchers + 24h process cache + 20+ unit tests green
- [ ] CP-3.5.2 — `terms` fetcher + `disagreement` logic + conflict-suggestion generation + unit tests green
- [ ] CP-3.5.3 — `EmitConstraintConflict` TurnAction variant + `CalcLimitsAdapter` + execute_action EmitReadback handler upgraded + `CALCULATOR_FUNNEL_ENABLED` flag gated (default `0`) + all existing tests green on both flag values
- [ ] CP-3.5.4 — LLM-side rule deletion per spec §7 (COMMERCIAL_SUBJECTS, range constants, _SUBJECT_MAP, _VALID_SUBJECTS/CURRENCIES); `_CLIENT_TYPE_MAP` ИП→Юр and 3:1 USD hack preserved (deferred to post-section)
- [ ] CP-3.5.5 — Live SIP regression green on `CALCULATOR_FUNNEL_ENABLED=1`: happy path + combo-conflict scenario both pass on `38.80.122.98`
- [ ] CP-3.5.6 — Flag + legacy branches removed; `calc-funnel-v1` tag pushed; section closed

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
| CP-1.3 | a8f2cf3 | 2026-04-19 | б/у variant recognition — regex + classifier prompt + Whisper vocab |
| CP-1.4 | f4041ac | 2026-04-19 | VAD silence 500→900 across all 7 sites (kills 500/700 inconsistency) |
| CP-1.2 | 4efa723 | 2026-04-19 | USD dual-disclosure in readback / SMS / calc summary |
| CP-1.1 | 454dac8 | 2026-04-19 | render_calc_result extracted; [deterministic_readback] marker; lint guard |
| CP-1.5b | 15536d2 | 2026-04-19 | Fix 1.5 — age_years clarify branch (live-call regression from Fix 1.3) |
| stability-v1 | 0ed4b2e | 2026-04-19 | Section 1 close (13 fixes, 5 SIP calls) |
| CP-2.1 | 72b91cb | 2026-04-20 | ClassifierOutput Pydantic schema + utterance grounding + 21 unit tests |
| CP-2.2 | 484dab7 | 2026-04-20 | parse_classifier_output wired into app.py (legacy dict shape preserved) |
| CP-2.2b | 41aa50c | 2026-04-20 | Drop ИП from ClientType + prompt; expand action enum to 8 values (E-Codex) |
| CP-2.3 | 277ea78 | 2026-04-20 | Fix 41b `_VALID_CHANGE_FIELDS` whitelist retired (Literal covers it) |
| CP-2.4b (codex rev) | 25cd065 | 2026-04-20 | End of 8-pass Codex review loop (13 findings resolved) |
| structured-classifier-v1 | 25cd065 | 2026-04-20 | Section 2 close; CP-2.5 SIP-validated on session cc7fc318 |
| CP-3.1 | f7ed575 | 2026-04-23 | profile_state.py pure functions + 23 unit tests |
| CP-3.2 | (Phase 3.C) | 2026-04-23 | TurnAction ADT + apply_turn dispatch (29 + 17 + 23 = 69 tests) |
| CP-3.3 | 32d782e | 2026-04-24 | execute_action handlers + 4 production adapters; APPLY_TURN_ENABLED flag |
| CP-3.5 | 67201ec | 2026-04-26 | Live regression sweep green on call e180107b. Bugs A-S all closed. Tag `demo-mvp-flag1-2026-04-26-validated`. |
| CP-3.4 | a1e53f4 | 2026-04-26 | Legacy `apply_patches` path deleted (~1779 lines net). 5 obsolete contract tests removed. 808 pass / 8 pre-existing fail. |
| CP-3.6 (high #1) | 73ccd9a | 2026-04-26 | Codex HIGH: `apply_pending_change()` return value now honoured — locked/unknown payloads re-emit EmitChangeConfirm instead of advancing to FireCalc on stale terms. |
| CP-3.6 (P1+P2) | 32b3324 | 2026-04-26 | Codex P1: utterance-fallback extractors re-wired into apply_turn pre-compute. Codex P2: `memory_block` threaded through FireLLMFallback prompt via `session.memory_block`. 5 regression tests added. |
| (provisioner) | f5f54bb | 2026-04-26 | vLLM 0.19 cudagraph accounting fix: `gpu_memory_utilization` tiers bumped ~+0.10 (H100 80GB: 0.60→0.70). Fixes "Available KV cache memory: -0.3 GiB" engine-refused-to-start on fresh VMs. |
| refactor-v1 | (set after this commit) | 2026-04-26 | Section 3 close. Live SIP test on call bf7a95a8 GREEN: full happy path + USD→BYN + change-confirm recalc + SMS + post-calc RAG (Codex P2 fix verified). |

(Append rows as checkpoints complete.)
