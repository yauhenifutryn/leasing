---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: Ready to execute
stopped_at: "Completed 01-03-PLAN.md: UI selectors and session.update wiring"
last_updated: "2026-03-25T21:00:42.377Z"
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 3
  completed_plans: 2
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** Accurate, low-latency Russian voice answers grounded in the company knowledge base, with full on-premises control over every pipeline component
**Current focus:** Phase 01 — instrumentation-and-ui-switching

## Current Position

Phase: 01 (instrumentation-and-ui-switching) — EXECUTING
Plan: 3 of 3

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: --
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: --
- Trend: --

*Updated after each plan completion*
| Phase 01-instrumentation-and-ui-switching P01 | 2 | 2 tasks | 3 files |
| Phase 01-instrumentation-and-ui-switching P03 | 10 | 2 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap v2]: Restructured from 7 phases to 5 phases on 2026-03-25; all code built locally in Phases 1-4, server work isolated to Phase 5
- [Roadmap v2]: Phase 3 (Brain + Benchmark Framework) depends only on Phase 1, not Phase 2; it can proceed in parallel with Phase 2 (voice adapters)
- [Roadmap v2]: Phase 4 (Omni) depends on Phase 2 and Phase 3; cannot start until both complete
- [Roadmap v2]: Phase 5 benchmark execution order is fixed: RAG comparison first, brain second, STT/TTS third, Omni vs. split pipeline last; this order isolates one variable per run
- [Roadmap v2]: DEPLOY-01 (env profiles) moved from Phase 1 to Phase 3 (Benchmark Framework) where it logically belongs; DEPLOY-02 and DEPLOY-03 remain in Phase 5
- [Phase 01-instrumentation-and-ui-switching]: stack_id is a @property not a dataclass field: avoids stale value bugs when backend or provider fields are mutated at runtime
- [Phase 01-instrumentation-and-ui-switching]: primary_kpi_ms formula is (playback_started - speech_stopped) * 1000: speech_stopped is end of user utterance, playback_started is first audio reaching the browser
- [Phase 01-instrumentation-and-ui-switching]: voiceProviderSelect added alongside the 3 new selectors because buildSessionUpdate() payload requires voice_provider field that was absent from existing code
- [Phase 01-instrumentation-and-ui-switching]: buildSessionUpdate() helper centralises all 5 selector fields in one place, eliminating drift across session.update call sites

### Pending Todos

None yet.

### Blockers/Concerns

- [Pre-Phase 2]: Qwen3-TTS exact HuggingFace repo ID and native sample rate are LOW confidence; confirm before writing the sidecar server
- [Pre-Phase 2]: Qwen3-ASR faster-whisper vs. raw transformers path unconfirmed; determines sidecar implementation approach
- [Pre-Phase 2]: Voxtral self-host availability unknown; may become a cloud-API thin adapter instead of local sidecar
- [Pre-Phase 4]: Qwen3-Omni vLLM audio input support version unconfirmed; may require vLLM upgrade
- [Phase 5]: GPU OOM risk when loading Qwen3.5-35B-A3B (~70 GB bfloat16) and Qwen3-Omni-30B (~60 GB) — never load both simultaneously; swap via supervisorctl and confirm nvidia-smi headroom before each model load

### Prior Context

- Branch: `claude/qwen-voice-next` based on `codex/split-voice-providers` (commit 9ef1b3d)
- Playbook at `docs/voice_ai_playbook_2026-03-25.md` is the authoritative planning document
- 4 voice providers already implemented: `local`, `yandex_speechkit`, `oss_russian`, `yandex_realtime`
- Existing contract test suite at `rag_demo_system/tests/test_voice_adapters_official.py`
- Critical version constraint: never upgrade the shared `rag_demo_system` venv `transformers==4.37.2` pin; all new model services must use per-service venvs

## Session Continuity

Last session: 2026-03-25T21:00:42.374Z
Stopped at: Completed 01-03-PLAN.md: UI selectors and session.update wiring
Resume file: None
