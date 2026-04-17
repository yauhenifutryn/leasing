# Client Feedback Round — Index

**Date:** 2026-04-16
**Trigger:** Client test call transcript (call_id 779154b4, 18:00-18:25) + user-raised issues.
**Scope:** 7 specs covering MVP fixes, 1 production deliverable.

## Architectural North Star

**Bot = translator. Calculator API = source of truth for business logic.**

No defaults in the bot. Every calculator field is collected into a `ClientProfile`
before the API is called. Profile is confirmed by the client through a read-back
gate. Changes after confirmation re-use single-field confirmation.

## Spec Inventory

| # | File | Cluster | Owner agent | Blocks | Blocked by |
|---|------|---------|-------------|--------|------------|
| 1 | [calc-mvp-relaxation-design.md](2026-04-16-calc-mvp-relaxation-design.md) | A — Calculator MVP | `calc-mvp` | — | 2 |
| 2 | [session-agent-profile-design.md](2026-04-16-session-agent-profile-design.md) | B — Session state | `session-state` | 1, 4 | — |
| 3 | [classifier-latency-design.md](2026-04-16-classifier-latency-design.md) | C — Performance | `perf-classifier` | — | — |
| 4 | [turn-taking-control-design.md](2026-04-16-turn-taking-control-design.md) | D+G — Turn-taking + stop | `turn-taking` | — | 2 |
| 5 | [whisper-prompt-design.md](2026-04-16-whisper-prompt-design.md) | E — STT | `whisper-fix` | — | — |
| 6 | [kb-audit-dedup-design.md](2026-04-16-kb-audit-dedup-design.md) | F — KB | `kb-audit` | — | user input |
| 7 | [../calculator-api-production-spec-ru.md](../../calculator-api-production-spec-ru.md) | A-production | (external team) | — | — |

## Execution Order

**Wave 1 (parallel, no inter-dependencies):** 3, 5
**Wave 2 (depends on Wave 1):** 2 — uses small-model URL from 3
**Wave 3 (depends on 2):** 1, 4 — consume SessionAgent + ClientProfile
**Wave 4 (user-gated):** 6 — requires transcript questions from user

Spec 7 is a deliverable handed to the calculator team; not a code change in this repo.

## Cross-cutting invariants

1. **No defaults in `CalculatorTool.defaults()`.** Method returns `{}` or is removed.
   Profile gate raises `IncompleteProfileError` if invoked without all 8 fields.
2. **`SessionAgent` JSON schema is shared.** Specs 1, 2, 4 all consume the same
   fields: `intent`, `extracted_fields`, `profile_patches`, `is_confirmation`,
   `is_stop_request`, `wants_readback`, `is_complete_for_calc`, `missing_fields`.
3. **`ClientProfile` dataclass lives in `backend/session.py`.** Specs 1 and 4 read it;
   Spec 2 owns it.
4. **Env vars are the tuning surface.** `VAD_SILENCE_MS`, `PRE_RESPONSE_HOLD_MS`,
   `SESSIONAGENT_BASE_URL`, `SESSIONAGENT_MODEL`, `USD_BYN_RATE` (MVP only).
5. **No post-STT correction dictionary.** Whisper biasing is the only STT lever.

## Out of Scope

- Excel rule matrix on bot side (production, Spec 7)
- NBRB live currency rate (production, Spec 7)
- Cross-session client profile persistence (future spec)
- STT engine swap / Whisper fine-tuning (`project_stt_v2_roadmap.md` memory)
- Speculative LLM start, streaming classifier first-token routing

## Deliverables Manifest

- 6 new spec markdown files in this directory
- 1 index doc (this file)
- 1 Russian production spec (`docs/calculator-api-production-spec-ru.md`, committed)
- 2 memory files updated (`project_calculator_production_backlog.md`, `project_stt_v2_roadmap.md`)
- 1 PROJECT_LOG.md entry
