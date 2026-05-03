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

## Pointers

Memory entries with full context:
- `project_voice_pipeline_baseline_2026_04_29_final.md` — current baseline state.
- `feedback_tts_speedup_dead_end_2026_04_29.md` — why Bug 3 stays parked.
- `project_multiparam_recall_signal_2026_04_29.md` — Bug 1/2 measurement signals.
- `feedback_voice_parallel_agents.md` — Bug 15 fix shape.
- `project_classifier_model_upgrade_plan.md` — Bug 1+2 surface.

## Closing criterion

Each deferred bug is closed when the targeted master-plan section ships its fix AND a live call validates the symptom is gone. Section 6's Phase 0 verification gate should run through this registry before docs/UAT.
