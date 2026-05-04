# Deferred bugs from ANALYSIS.md (2026-04-29 client review)

ANALYSIS.md cataloged 28 bugs surfaced by live calls + the client review round. Batches 1-4 shipped 23 of them on `feature/voice-pipeline` (tags `analysis-batch-1-shipped` through `analysis-batch-4-shipped`). The remaining 5 are deferred to specific master-plan sections.

This file is the registry: which deferred bug lives where, and why.

## Registry

| # | Symptom | Target section | Notes |
|---|---|---|---|
| 1 | Classifier hard to call (multi-field drop on 7-field utterances) | [05c_classifier_model_upgrade_eval.md](05c_classifier_model_upgrade_eval.md) | 4B model recall ceiling. Fix is the 4B → 8B/7B eval. Gated on post-Section-3 residual-failure ≥ 2% (current rate ~1-2% after Batches 1-3). |
| 2 | Classifier latency floor ~1000 ms | [05c_classifier_model_upgrade_eval.md](05c_classifier_model_upgrade_eval.md) | Same dependency as Bug 1 — bigger model would be slower; needs benchmark vs recall trade-off. Could also fall to Section 8 capacity work (parallel batch instead of bigger model). |
| 3 | TTS too fast (Maria 14:20:07) | post-MVP TTS engine swap | Five DSP algorithms rejected (memory `feedback_tts_speedup_dead_end_2026_04_29`). No clean post-synthesis time-stretch on Silero v5_4_ru. Real fix = engine swap (different TTS provider). Not yet in any master-plan section; create a new section when the client commits to the swap budget. |
| 5 | TTS truncates long phone "+" | opportunistic, ~30 min | TTS chunker boundary — the leading "+" is dropped on a stream chunk break. One-line fix in `backend/text_utils.py:split_for_tts_streaming` to keep the "+" attached. Bundle into the next dispatcher PR or do as a one-off when convenient. |
| 15 | Multi-question drops (user asks 2 things, bot answers 1) | [04_natural_turn_taking.md](04_natural_turn_taking.md) | Surfaces in the natural-turn-taking section. Fix surface is parallel-agent dispatching (memory `feedback_voice_parallel_agents`) — let the bot recognise "X? И ещё Y?" and answer both via independent retrieval calls before replying. Section 4 is the natural home; lock in there. |
| 30 | RESOLVED 2026-05-04 — topical KB (`kb_topics_ru.md`) was missing company-identity content despite "nothing dropped" claim in Phase B decision doc | Was: KB sync — now CLOSED on `feature/chat-widget` | Original wrong analysis: I said `kb_faq_ru_v2.md` was missing owner data — actually v2 has it (lines 17151-17212). The real gap was in `kb_topics_ru.md` (the active topical KB after KB_LAYOUT=topical swap): the Phase C refactor consolidated 350 v2 sections to 40 topical sections at ~80% coverage and silently dropped owner/director/leadership/supervisory-board/UNP/email/founding-date/legal-address facts. Voice's RAG worked because Qdrant still held stale legacy chunks. Fix shipped on `feature/chat-widget`: added `section_id: company-identity-and-leadership` to `kb_topics_ru.md` + fixed `offices-addresses-and-hours` (4 of 6 addresses said "уточняется"; hours dropped Минск weekend exception). Closing requires Qdrant full rebuild on next deploy. Remaining audit gaps deferred — see `project_chat_widget_refinement_backlog_2026_05_04.md` for the lease-types-overview / advantages / financial-disclosure / achievements list. |
| 31 | Chat refinements from 2026-05-03 live test | `feature/chat-widget` before tag/merge | See memory `project_chat_widget_refinement_backlog_2026_05_04` for full list and fix order. Real bugs: profile.name from intake (#5), bot kickoff message (#2), consent micro-copy (#1), TTS-to-chat render (#3), SMS-on-combined-offer (#8), panel fixed-height (#9). Polish: send/receive sound (#6). RAG-ranking #4 affects both modalities — not chat-specific. #7 reclassified as input-shape variance — not a bug. |

## Pointers

Memory entries with full context:
- `project_voice_pipeline_baseline_2026_04_29_final.md` — current baseline state.
- `feedback_tts_speedup_dead_end_2026_04_29.md` — why Bug 3 stays parked.
- `project_multiparam_recall_signal_2026_04_29.md` — Bug 1/2 measurement signals.
- `feedback_voice_parallel_agents.md` — Bug 15 fix shape.
- `project_classifier_model_upgrade_plan.md` — Bug 1+2 surface.

## Closing criterion

Each deferred bug is closed when the targeted master-plan section ships its fix AND a live call validates the symptom is gone. Section 6's Phase 0 verification gate should run through this registry before docs/UAT.
