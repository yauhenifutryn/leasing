# Section 6 — Final: verification, docs, self-improvement refresh

**Status**: pending (run last)
**Prereqs**: Sections 1, 2, 3, 4 all closed + live-verified by client
**Estimated effort**: ~1 day
**Base commit**: end of Section 4

## Goal

Close out the whole master plan. Before touching docs:
1. **Verify every pending client issue is actually fixed** (from prior feedback lists, not just code tests)
2. **Update all user-facing documentation** (README, architecture sections, deploy runbook)
3. **Refresh self-improvement tooling** (session_analyzer + kb_gap_report extensions for new fix signals)
4. **Append PROJECT_LOG.md** with the major decisions from sections 1-4

This is the section where we also archive obsolete memories and tag the final `v1` release.

## Required memories (retrieve before starting)

- `master_plan_pointer.md` — confirm plan structure
- `project_mvp_complete_2026_04_18.md` — baseline to compare against
- `project_client_feedback_fixes.md` — every tracked client complaint
- `project_self_improvement_handover.md` — analyzer context
- `project_self_improvement_extensions.md` — prior analyzer extensions
- `feedback_docs_sync.md` — docs must stay in sync with code
- `feedback_propose_docs_update.md` — proactively update docs after big changes
- `project_contradiction_check_future.md` — future KB validation script
- `project_client_kb_doc_pending.md` — client KB doc pending items

## Primary skill

`superpowers:verification-before-completion` for Phase 0 (verification gate).
Then `gsd-docs-update` for Phase 2 (main docs refresh) — this skill is purpose-built for verified-against-codebase documentation.
Then `superpowers:requesting-code-review` before the final PR.

## Phase 0 — Verification gate (DO THIS FIRST)

Before writing any docs, confirm every pending issue is actually fixed. The client has complained about:

| Complaint | Expected fix source | Verification command |
|---|---|---|
| Tool sometimes not called | Section 3 (refactor) | Live scenario replay from `project_mvp_complete_2026_04_18.md` — every step fires tool |
| LLM fabricates numbers | Section 1.1 | `grep -rE '\{(cost\|prepaid\|term\|payment)' backend/*.py` outside `profile_prompts.py` returns empty |
| USD stored as BYN silently | Section 1.2 | `test_usd_disclosure_in_readback` passes; live call shows both amounts |
| "б/у" not recognized | Section 1.3 | `test_condition_new_variants.py` covers 8+ variants |
| Barge-in too aggressive | Section 1.4 + 4 | VAD_SILENCE_MS >= 900; adaptive if Section 4 shipped |
| Currency omitted in readback | Section 1.1 + 1.2 | All readback paths go through `build_readback_text` or `render_profile_summary` |
| "давай" false-confirm | Section 2 (Pydantic) + Section 3 (state machine) | Live test: user says "давай сменим" — bot asks for confirmation, doesn't auto-apply |
| Change-confirm drops fields | Section 3 | Live test: multi-field change lists every field |
| `change_field: "all"` leaked | Section 2 (Pydantic Literal) | Schema rejects; test `test_invalid_change_field_rejected` passes |
| `change_value: 0` ghost | Fix 40e (already in MVP) + Section 3 | Test already exists; run to confirm |

Produce a short report `VERIFICATION.md` in this directory with PASS/FAIL per row and evidence. If any row is FAIL, **STOP and create a patch section** (e.g. 6.1.1) to close the gap before docs.

## Phase 1 — Run the full UAT scenario one more time

Use `gsd-verify-work` skill or manual:

1. Call the bot, run every step in the unified scenario from `project_mvp_complete_2026_04_18.md`
2. Run the 14-row trickery checklist from Fix 39 (OOR, negative, absurd values)
3. Capture `.state/backend.log` for the session, save as `uat_<date>.log`
4. Confirm no unexpected ERROR-level logs

If any step fails, the issue becomes a patch section. Don't proceed to docs until clean.

## Phase 2 — Docs refresh

### 2.1 README.md

Update the architecture section to reflect the refactor (Section 3):
- Remove references to the 7-gate orchestrator
- Add architecture diagram for `apply_turn` + TurnAction dispatch
- Update deploy runbook if any changes to service startup
- Update troubleshooting section with any new log patterns from Section 3

### 2.2 PROJECT_LOG.md (append-only)

Add one entry per section:
```
## 2026-MM-DD — Section 1: pre-refactor stability
Why: client-reported numeric hallucinations + USD confusion + б/у misses
Changes: deterministic rendering helper; USD/BYN dual disclosure; expanded condition_new cues; VAD bump
Commits: <hashes>
Impact: perceived consistency jump on standard flows

## 2026-MM-DD — Section 2: structured classifier
...

## 2026-MM-DD — Section 3: architecture refactor
...

## 2026-MM-DD — Section 4: natural turn-taking
...
```

### 2.3 Architecture docs (if any separate from README)

If the repo has a `docs/` directory, update diagrams, state machine descriptions, and API references.

### 2.4 Deploy runbook

Check `rag_demo_system/scripts/provision_server.sh` and `restart_all.sh` for any new steps introduced by sections 1-4. Update README deploy section.

## Phase 3 — Self-improvement tooling

### 3.1 backend/session_analyzer.py

Add detection for new fix signals from sections 1-4:
- `OOR_range_response` — bot said "от X до Y месяцев" style range message
- `multi_field_staging` — `CHANGE_PENDING (implicit): fields=[...]` with ≥2 fields
- `prepaid_counterpart_cleared` — `clearing prepaid_pct` or `clearing prepaid_amount` log line
- `deterministic_readback` — bot output came from `build_readback_text` (will need a log marker added in Section 1.1)
- `adaptive_silence_fired` — if Section 4 shipped, log marker for per-state threshold

Each signal becomes a counter + sample utterance captured to JSON for later analysis.

### 3.2 KB gap report extensions

Update `kb_gap_report` scripts (check current location) to surface:
- Questions the bot handled via LLM fallback instead of KB retrieval
- Questions where bot said "обратитесь к специалисту" or similar deflection
- Target rates for fix coverage (e.g. "OOR responses within expected ranges" > 95%)

### 3.3 Contradiction-detection script (optional stretch)

From memory `project_contradiction_check_future.md` — pre-ingestion contradiction detection for KB updates. If time permits, spike this in ~2 hours. Otherwise defer.

### 3.4 RAG eval harness (path A from 2026-04-23 brainstorm)

Measurement tooling for future RAG tuning. Purely additive, zero production risk.

- **Fixture set**: `rag_demo_system/tests/fixtures/eval_queries_ru.jsonl` — 30-50 hand-crafted Russian Q→expected-chunk pairs covering the realistic query distribution (terms, rates, condition, contact, OOR edge cases).
- **Script**: `rag_demo_system/scripts/eval_rag.py` — runs fixture queries against current Qdrant+BM25+rerank pipeline (respecting voice preset), prints recall@K and MRR, writes results to `results/eval_rag_<date>.json` for diffing between runs.
- **Use**: run once before any RAG parameter change; re-run after; diff the two reports. Becomes the reference for Path B tuning work in the post-v1 milestone.
- **Does NOT** touch production config, Qdrant state, or retrieval code. Read-only measurement.
- **Defer trigger for tuning work itself**: ~200 client feedback events in `kb_viz_feedback.jsonl` (see `project_kb_viz_dedup_plan.md`). This section builds the harness; the sweep is post-v1.

## Phase 4 — Archive obsolete memories

Memories that become obsolete after the refactor:
- `project_orchestrator_refactor_pending.md` — refactor is done; archive into `memory/archive/`
- `project_handoff_2026_04_18.md` — older handoff snapshot, superseded

New memory to create:
- `project_v1_complete_<date>.md` — summary of final state, stable commit, links to UAT log + verification report

Update `MEMORY.md` index accordingly.

## Phase 5 — Final tag + PR

```bash
git tag -a v1-2026-MM-DD <final-commit-sha> -m "V1 release — master plan 2026-04-18 complete"
git push origin v1-2026-MM-DD
```

Open a PR from `feature/voice-pipeline` to `main` via `gsd-ship` skill. Include:
- Link to VERIFICATION.md
- Link to UAT log
- Short changelog of sections 1-4

## Checkpoints

- [ ] CP-6.0 — Verification gate PASSES all rows in the table above. `VERIFICATION.md` committed.
- [ ] CP-6.1 — Full UAT scenario passes live. `uat_<date>.log` archived.
- [ ] CP-6.2 — README.md architecture section updated.
- [ ] CP-6.3 — PROJECT_LOG.md appended with section entries.
- [ ] CP-6.4 — Deploy runbook updated (if changes needed).
- [ ] CP-6.5 — session_analyzer.py extended with new fix signals.
- [ ] CP-6.6 — kb_gap_report extended.
- [ ] CP-6.6a — RAG eval harness shipped: `scripts/eval_rag.py` + `tests/fixtures/eval_queries_ru.jsonl`, baseline report archived.
- [ ] CP-6.7 — Obsolete memories archived; new `project_v1_complete_*.md` memory written.
- [ ] CP-6.8 — Final tag pushed; PR opened to main.

## Rollback

Docs changes are low-risk; revert individual commits if needed.
Session_analyzer changes affect local diagnostics only; safe to revert.
For catastrophic regressions: `git reset --hard mvp-2026-04-18`.

## What NOT to do in this section

- No new feature work. If client feedback surfaces new issues during UAT, defer to Section 7 or a future milestone.
- No architecture changes. Section 3 is done and frozen.
- No skill installs unless specifically needed for `gsd-docs-update` workflow.

## Done criterion

All checkpoints green + PR merged + user confirms client UAT passed. Master plan 2026-04-18 closed; next milestone planning starts from fresh state.
