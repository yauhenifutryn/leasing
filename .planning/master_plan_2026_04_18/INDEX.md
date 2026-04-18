# Master Plan — 2026-04-18 onward

**Stable base**: git tag `mvp-2026-04-18` at commit `a20aa4f` on `feature/voice-pipeline`.

## Purpose

This plan consolidates all post-MVP work into discrete sections, each resumable in its own Claude Code session. A fresh Claude opens `INDEX.md` first, picks a section based on prerequisites and user instruction, reads the section file + referenced memories, and resumes from the last incomplete checkpoint in `CHECKPOINTS.md`.

## How to use this plan in a fresh session

Open a new Claude Code chat, paste:

> Read `.planning/master_plan_2026_04_18/INDEX.md` then `.planning/master_plan_2026_04_18/CHECKPOINTS.md`. Resume section N (description). Use the memories and skills listed in that section file.

Where N is one of: 1, 2, 3, 4, 5, 6. Claude should:
1. Read INDEX + CHECKPOINTS
2. Read the specific section file (`0N_*.md`)
3. Retrieve every memory listed in the section's **Required memories** block
4. Invoke the primary skill listed in **Primary skill**
5. Resume at the first checkpoint with `status: pending`
6. Update `CHECKPOINTS.md` atomically as checkpoints complete

## Sections

| # | File | Status | Prereqs | Effort | Priority |
|---|---|---|---|---|---|
| 1 | [01_pre_refactor_stability.md](01_pre_refactor_stability.md) | pending | none | ~1 day | HIGH — client-facing wins |
| 2 | [02_structured_classifier.md](02_structured_classifier.md) | pending | none (parallel to 1 OK) | ~1 day | MEDIUM — enables 3 |
| 3 | [03_architecture_refactor.md](03_architecture_refactor.md) | pending | 2 done | ~2-3 days | MEDIUM — structural |
| 4 | [04_natural_turn_taking.md](04_natural_turn_taking.md) | pending | ideally after 3 | ~2 days | HIGH — UX polish |
| 5 | [05_speaker_mode_adaptive_vad.md](05_speaker_mode_adaptive_vad.md) | pending | after 4 | ~1 day | MEDIUM — experimental |
| 6 | [06_final_docs_and_self_improvement.md](06_final_docs_and_self_improvement.md) | pending | all above + client UAT | ~1 day | FINAL — close out plan |

**Recommended execution order**: 1 → 2 → 3 → 4 → 5 → 6.

- Section 1 ships visible client wins fast while 2-3 cook the deeper fix.
- Section 5 (speaker mode) is experimental — isolated session recommended, easy-revert via env var.
- Section 6 is the verification + docs + self-improvement close-out. It has a **verification gate** as its first phase that must pass before any docs work.

## Rollback

If any section introduces a regression:

```bash
git reset --hard mvp-2026-04-18
git push --force-with-lease origin feature/voice-pipeline
```

Every section documents its own scoped rollback procedure at the bottom of its file.

## Skill framework choice

The user decided (see memory `feedback_skill_framework_superpowers.md`):
- **Main loop: superpowers** (brainstorming, writing-plans, executing-plans, TDD, systematic-debugging, verification-before-completion)
- **Borrow from GSD at the right moments**: `gsd-thread`, `gsd-ship`, `gsd-map-codebase`, `gsd-forensics`, `gsd-verify-work`, `gsd-docs-update` (in Section 6)

Each section specifies which skills to invoke and when.

## Client feedback tracking

All sections reference `project_client_feedback_fixes.md` and ongoing client complaints. Section 6's Phase 0 verification gate runs through the full complaint list against implemented fixes before docs are touched.

## Done criterion for the whole plan

- Sections 1-5 all `completed` in CHECKPOINTS.md
- Section 6 Phase 0 verification gate: all rows PASS
- Section 6 full UAT scenario passes live (client-tested, not just Claude)
- Docs refreshed (README, PROJECT_LOG, session_analyzer)
- PR merged to main
- Final tag pushed
- Memory `project_mvp_complete_2026_04_18.md` archived; `project_v1_complete_<date>.md` written
