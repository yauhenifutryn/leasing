# Section 6 — Self-improvement + docs refresh (DEFERRED)

**Status**: deferred
**Prereqs**: none (independent)
**Estimated effort**: ~0.5 day

## Why deferred

Internal tooling — updates the session analyzer + transcripts aggregator for new fix signals. Important for ongoing quality tracking but doesn't ship client-visible improvements.

## Scope summary (when picked up)

- Update `backend/session_analyzer.py` to detect new fix signals (OOR responses, multi-field staging, deterministic readback usage)
- Update `kb_gap_report` extensions
- Refresh README architecture section post-refactor
- Append to `PROJECT_LOG.md` with the major decisions from Sections 1-4

## Required memories (when picked up)

- `project_self_improvement_handover.md`
- `project_self_improvement_extensions.md`
- `feedback_docs_sync.md`
- `feedback_propose_docs_update.md`

## Do not start without explicit user instruction.

## Exception

This section CAN be woven into Sections 1-4 as they close out (each section updates the relevant analyzer signal). If done inline, mark the per-section CP as covering this section's overlap.
