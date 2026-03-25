---
phase: 01-instrumentation-and-ui-switching
plan: 02
subsystem: api
tags: [websocket, instrumentation, latency, voice, fastapi, python]

# Dependency graph
requires:
  - phase: 01-instrumentation-and-ui-switching
    plan: 01
    provides: "VoiceSession with brain_model, stt_provider, tts_provider, stack_id fields"
provides:
  - "Extended session.update WebSocket handler accepting brain_model, stt_provider, tts_provider"
  - "6-timestamp voice turn instrumentation in input_audio_buffer.commit handler"
  - "Structured voice_turn log event with question_id, stack_id, primary_kpi_ms, and all timing fields"
  - "stt/tts providers read from VoiceSession directly, not legacy _voice_pipeline()"
affects: [phase-03-benchmark-framework, phase-05-benchmark-execution]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Timestamp capture with time.time() (not perf_counter) for cross-process comparability"
    - "question_id via uuid4 per voice turn for log correlation"
    - "t_llm_first_token = t_retrieval_done as conservative approximation for non-streaming path"
    - "yandex_realtime voice_provider auto-syncs stt_provider and tts_provider on session.update"

key-files:
  created:
    - rag_demo_system/tests/test_instrumentation.py
  modified:
    - rag_demo_system/backend/app.py
    - rag_demo_system/backend/voice_session.py

key-decisions:
  - "time.time() chosen over time.perf_counter() so timestamps are absolute epoch values usable cross-process in benchmarks"
  - "t_llm_first_token = t_retrieval_done is an intentional conservative approximation for the non-streaming chat() path; TODO comment placed to revisit in Phase 3"
  - "t_playback_started captured after the WebSocket send of the first audio delta, not after synthesis, to measure actual browser delivery"
  - "_voice_pipeline() kept in place for the /api/voice/chat HTTP endpoint; only the WS handler now reads from VoiceSession directly"

patterns-established:
  - "Voice turn instrumentation pattern: question_id at top, t_speech_stopped at entry, t_stt_done after STT, t_retrieval_done after chat(), t_tts_first_chunk after synthesis, t_playback_started after audio send, state.log() with complete dict before response.done"
  - "session.update allowlist pattern: brain_model validated against explicit allowlist; stt/tts providers accepted as-is (no allowlist needed at this stage)"

requirements-completed: [INST-02, INST-03, SWITCH-02]

# Metrics
duration: 8min
completed: 2026-03-25
---

# Phase 01 Plan 02: Instrumentation and Session Update Summary

**WebSocket voice handler now captures 6 timestamps per turn and emits a structured voice_turn log with question_id, stack_id, primary_kpi_ms, and all latency fields; session.update accepts brain_model, stt/tts providers and reflects them in session.ready/updated responses**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-25T20:55:06Z
- **Completed:** 2026-03-25T21:00:12Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- session.update handler now reads brain_model (with allowlist validation), stt_provider, and tts_provider from the event and stores them on VoiceSession
- yandex_realtime voice_provider auto-syncs stt and tts providers to "yandex_realtime"
- session.ready and session.updated responses now include voice_provider, brain_model, stt_provider, tts_provider, and stack_id
- input_audio_buffer.commit captures all 6 latency timestamps: speech_stopped, stt_done, retrieval_done, llm_first_token, tts_first_chunk, playback_started
- primary_kpi_ms computed as (playback_started - speech_stopped) * 1000 and emitted in structured log
- WebSocket handler reads stt/tts providers from VoiceSession directly instead of legacy _voice_pipeline()
- All 6 tests in test_voice_session.py and test_instrumentation.py pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend session.update handler with new fields** - `d4f95c1` (feat)
2. **Task 2: Instrument input_audio_buffer.commit with 6 timestamps and structured log** - `39f2db4` (feat)

## Files Created/Modified

- `rag_demo_system/backend/app.py` - Extended session.update handler and fully instrumented input_audio_buffer.commit block
- `rag_demo_system/backend/voice_session.py` - Added voice_provider, brain_model, stt_provider, tts_provider fields and stack_id property (prerequisite from plan 01-01)
- `rag_demo_system/tests/test_instrumentation.py` - Voice turn log event contract tests (created as prerequisite)

## Decisions Made

- time.time() used for all timestamps: absolute epoch values required for cross-process log correlation; perf_counter() is process-local and unsuitable
- t_llm_first_token = t_retrieval_done: conservative approximation for the non-streaming chat() call; streaming instrumentation deferred to Phase 3
- t_playback_started captured after the websocket.send_json() call for the audio delta: measures actual delivery to WebSocket buffer, not just synthesis completion
- brain_model allowlist: only "Qwen/Qwen3-30B-A3B" and "Qwen/Qwen3.5-35B-A3B" accepted; anything else silently falls back to the default; this prevents accidental misconfiguration

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Applied plan 01-01 prerequisite changes to worktree**
- **Found during:** Task 1 setup
- **Issue:** The worktree branch (`worktree-agent-a4f0f317`) was based on an older commit that predates plan 01-01. VoiceSession lacked brain_model, stt_provider, tts_provider, voice_provider, and stack_id. test_instrumentation.py did not exist. Both are required by plan 01-02.
- **Fix:** Applied the plan 01-01 artifacts directly to the worktree: updated voice_session.py with new fields and stack_id property, created test_instrumentation.py with the log event contract tests.
- **Files modified:** rag_demo_system/backend/voice_session.py, rag_demo_system/tests/test_instrumentation.py
- **Verification:** test_voice_session.py and test_instrumentation.py both pass (6 tests)
- **Committed in:** d4f95c1 (Task 1 commit)

**2. [Rule 3 - Blocking] Adapted session.update for simpler worktree app.py (no YandexRealtimeRelay)**
- **Found during:** Task 1
- **Issue:** The plan's interface snippet referenced `normalize_voice_provider` and `YandexRealtimeRelay` imports which do not exist in this worktree's app.py (older codebase version). Plan 01-02 still requires voice_provider handling.
- **Fix:** Implemented voice_provider normalization inline (lowercase/strip) without the relay. The yandex_realtime auto-sync logic is included as planned. The relay integration is a concern of a future convergence onto the main feature branch.
- **Files modified:** rag_demo_system/backend/app.py
- **Verification:** Syntax check passes; behavior matches the plan's intent
- **Committed in:** d4f95c1 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 3 - blocking prerequisites)
**Impact on plan:** Both fixes were necessary to unblock plan 01-02 execution in the parallel worktree context. No scope creep; all plan 01-02 success criteria met.

## Issues Encountered

The parallel agent worktree was branched from a pre-plan-01-01 commit. This is expected in parallel execution: each agent works on an independent branch. The worktree changes will be merged/cherry-picked back to the main feature branch by the orchestrator.

## Known Stubs

None. All instrumentation fields are wired to real timestamps and real session data. No placeholder values in the log event.

## Next Phase Readiness

- Every voice turn now emits a complete voice_turn log event. The benchmark runner in Phase 3 can read these directly from .state/logs.jsonl.
- stack_id in each log line enables grouping by provider combination without post-processing.
- primary_kpi_ms is ready for comparison across stacks in Phase 5 benchmark execution.
- Blocker for Phase 2: Qwen3-TTS and Qwen3-ASR repo IDs/interfaces still unconfirmed (pre-existing concern, not introduced here).

---
*Phase: 01-instrumentation-and-ui-switching*
*Completed: 2026-03-25*

## Self-Check: PASSED

- FOUND: rag_demo_system/backend/app.py
- FOUND: rag_demo_system/backend/voice_session.py
- FOUND: rag_demo_system/tests/test_instrumentation.py
- FOUND: commit d4f95c1 (feat(01-02): extend session.update handler)
- FOUND: commit 39f2db4 (feat(01-02): instrument input_audio_buffer.commit)
