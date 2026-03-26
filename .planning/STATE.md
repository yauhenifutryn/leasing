---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: Phase complete — ready for verification
stopped_at: Completed 03-03-PLAN.md
last_updated: "2026-03-26T12:02:03.585Z"
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 8
  completed_plans: 8
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** Accurate, low-latency Russian voice answers grounded in the company knowledge base, with full on-premises control over every pipeline component
**Current focus:** Phase 03 — brain-upgrade-and-benchmark-framework

## Current Position

Phase: 03 (brain-upgrade-and-benchmark-framework) — EXECUTING
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
| Phase 01-instrumentation-and-ui-switching P02 | 8 | 2 tasks | 3 files |
| Phase 02-voice-provider-adapters P01 | 2 | 3 tasks | 3 files |
| Phase 02 P02 | 2 | 3 tasks | 6 files |
| Phase 03 P02 | 4 | 2 tasks | 9 files |
| Phase 03-brain-upgrade-and-benchmark-framework P01 | 4 | 2 tasks | 2 files |
| Phase 03 P03 | 5 | 2 tasks | 5 files |

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
- [Phase 01-instrumentation-and-ui-switching]: time.time() chosen over perf_counter for voice turn timestamps — absolute epoch values required for cross-process log correlation in benchmarks
- [Phase 01-instrumentation-and-ui-switching]: t_llm_first_token = t_retrieval_done is conservative approximation for non-streaming chat() — TODO to extract real first token in Phase 3 streaming
- [Phase 01-instrumentation-and-ui-switching]: brain_model validated against allowlist (Qwen3-30B-A3B, Qwen3.5-35B-A3B); invalid values fall back to default silently to prevent misconfiguration
- [Phase 02-voice-provider-adapters]: _HARD_FAIL_STT frozenset at module level: new STT providers hard-fail on missing BASE_URL; no silent fallback to sensevoice/whisper
- [Phase 02-voice-provider-adapters]: qwen3_tts hard-fail mirrors STT pattern: all three new providers raise RuntimeError when unconfigured rather than degrading silently
- [Phase 02]: Qwen3-TTS language='Russian' hardcoded in generate_voice_clone call: qwen-tts API uses full language names, not ISO codes
- [Phase 02]: Voxtral batch/offline API only (processor->generate->batch_decode); streaming API excluded per Pitfall 6 to avoid instability
- [Phase 02]: Voxtral _target_sr from processor.feature_extractor.sampling_rate (dynamic, not hardcoded) for forward-compatibility
- [Phase 03]: JSONL fixture format with question_id/category/text_ru/expected_keywords; category prefix encoding sf/lf/kb/amb/oos
- [Phase 03]: .env.bench overlay pattern: profiles contain only overriding variables; runner loads base .env then applies profile with load_dotenv(override=True)
- [Phase 03]: Sidecar BASE_URLs embedded in voice provider profiles to prevent hard-fail RuntimeError at benchmark runtime
- [Phase 03-brain-upgrade-and-benchmark-framework]: Per-request brain_model override via ChatRequest field, not settings mutation: avoids race conditions on module-level singleton
- [Phase 03-brain-upgrade-and-benchmark-framework]: _voice_chat_streaming_sync is synchronous (requests-based streaming); wrapped in asyncio.to_thread() to avoid blocking the FastAPI event loop during voice turns
- [Phase 03-brain-upgrade-and-benchmark-framework]: dify_rag backend falls back to non-streaming with t_llm_first_token = t_retrieval_done — Dify manages its own API, streaming extraction not possible
- [Phase Phase 03]: POST /api/tts as REST proxy for benchmark runner: runner must not need direct sidecar access; backend routes TTS via synthesize_audio_with_provider
- [Phase Phase 03]: Fresh WS connection per benchmark question: prevents session brain_model/provider state from contaminating next question result
- [Phase Phase 03]: statistics.quantiles(n=100, method='inclusive') for p50/p95: returns interpolated values not element values; p50 of [1..100] = 50.5 with this method

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

Last session: 2026-03-26T12:02:03.582Z
Stopped at: Completed 03-03-PLAN.md
Resume file: None
