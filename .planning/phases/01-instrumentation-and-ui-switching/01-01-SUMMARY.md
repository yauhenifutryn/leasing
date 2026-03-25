---
phase: 01-instrumentation-and-ui-switching
plan: 01
subsystem: testing

tags: [python, dataclass, voice-session, pytest, instrumentation]

requires: []

provides:
  - VoiceSession dataclass extended with brain_model, stt_provider, tts_provider fields and stack_id computed property
  - test_voice_session.py extended with 5 new unit tests covering all new fields and stack_id behavior
  - test_instrumentation.py scaffold defining the 17-field log event contract for Plan 02

affects:
  - 01-02 (instrumentation): test_instrumentation.py defines the log event REQUIRED_LOG_FIELDS contract that Plan 02 must satisfy
  - 01-03 (ui-switching): VoiceSession.brain_model, stt_provider, tts_provider fields are consumed by UI selector bindings
  - 02 (voice-adapters): stack_id format documents the adapter identity contract

tech-stack:
  added: []
  patterns:
    - "stack_id is a @property (not a stored field) so it always reflects current field state without manual sync"
    - "brain_model strips the HuggingFace org prefix via split('/')[-1] to keep stack_id readable"
    - "Test scaffold pattern: define REQUIRED_LOG_FIELDS as a module-level set, then test .issubset(event.keys())"

key-files:
  created:
    - rag_demo_system/tests/test_instrumentation.py
  modified:
    - rag_demo_system/backend/voice_session.py
    - rag_demo_system/tests/test_voice_session.py

key-decisions:
  - "stack_id is a @property not a dataclass field: avoids stale value bugs when backend or provider fields are mutated at runtime"
  - "brain prefix stripping via split('/')[-1]: keeps stack_id short and human-readable in log files and benchmark CSVs"
  - "primary_kpi_ms formula is (playback_started - speech_stopped) * 1000: speech_stopped is the end of the user's utterance, playback_started is the moment audio reaches the browser speaker"

patterns-established:
  - "TDD with explicit RED/GREEN separation: tests added and confirmed failing before any implementation"
  - "Log event contract tested by structure (set.issubset) not by runtime execution"

requirements-completed: [INST-01, SWITCH-03]

duration: 2min
completed: 2026-03-25
---

# Phase 01 Plan 01: VoiceSession Fields and Log Event Contract Summary

**VoiceSession dataclass extended with brain_model/stt_provider/tts_provider fields and computed stack_id property; 12 tests confirm all fields, defaults, and log event contract**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-25T20:52:24Z
- **Completed:** 2026-03-25T20:54:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `brain_model`, `stt_provider`, `tts_provider` fields to VoiceSession with correct defaults (`Qwen/Qwen3-30B-A3B`, `sensevoice`, `cosyvoice`)
- Added `stack_id` as a `@property` returning the composite string `backend__brain__stt__tts` (org prefix stripped from brain_model)
- Created `test_instrumentation.py` with `REQUIRED_LOG_FIELDS` set (17 fields) and 3 tests that define the log event contract for Plan 02
- All 12 tests pass; 4 original tests remain green

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend VoiceSession with new fields and stack_id property** - `8e35727` (feat/test via TDD)
2. **Task 2: Create test_instrumentation.py scaffold** - `8ea1f9b` (feat)

**Plan metadata:** committed after SUMMARY creation (docs)

## Files Created/Modified

- `rag_demo_system/backend/voice_session.py` - Added brain_model, stt_provider, tts_provider fields and stack_id @property
- `rag_demo_system/tests/test_voice_session.py` - Added 5 new tests: test_default_brain_model, test_default_stt_provider, test_default_tts_provider, test_stack_id_composition, test_stack_id_updates_on_field_change
- `rag_demo_system/tests/test_instrumentation.py` - New file: REQUIRED_LOG_FIELDS set, 3 contract tests for Plan 02

## Decisions Made

- `stack_id` implemented as `@property` (not a stored field) so any runtime mutation of `backend`, `brain_model`, `stt_provider`, or `tts_provider` is reflected immediately without a sync step
- HuggingFace org prefix stripped via `split("/")[-1]` to keep log lines and benchmark CSV columns readable
- `primary_kpi_ms = (playback_started - speech_stopped) * 1000`: measures total user-perceived latency from end of speech to first audio playback

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02 (instrumentation) can immediately import `REQUIRED_LOG_FIELDS` from `test_instrumentation.py` and build the `build_voice_turn_event()` helper against that contract
- Plan 03 (UI switching) can bind `brain_model`, `stt_provider`, `tts_provider` from VoiceSession to UI selectors
- No blockers for Phase 01 continuation

---
*Phase: 01-instrumentation-and-ui-switching*
*Completed: 2026-03-25*
