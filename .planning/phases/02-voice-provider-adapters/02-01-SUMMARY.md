---
phase: 02-voice-provider-adapters
plan: "01"
subsystem: api
tags: [voice, stt, tts, qwen3-asr, qwen3-tts, voxtral, adapters, pytest, python]

requires:
  - phase: 01-instrumentation-and-ui-switching
    provides: "UI selectors (sttProviderSelect, ttsProviderSelect) and session.update wiring that routes provider strings to backend dispatch"

provides:
  - "Adapter dispatch branches for qwen3_asr, qwen3_tts, voxtral in voice_adapters.py"
  - "Hard-fail guards for new STT providers (qwen3_asr, voxtral) and new TTS provider (qwen3_tts)"
  - "Health status entries for all 3 new providers in build_voice_statuses()"
  - "Frontend dropdown options exposing qwen3_asr, voxtral (STT) and qwen3_tts (TTS)"
  - "12 contract tests (6 existing + 6 new) proving routing and hard-fail behavior"

affects: [02-02-voice-provider-adapters, 05-benchmark-execution]

tech-stack:
  added: []
  patterns:
    - "_HARD_FAIL_STT frozenset: new STT providers hard-fail when BASE_URL unset instead of silently falling back"
    - "Early-return pattern in transcribe_audio(): hard-fail providers bypass the fallback loop entirely"
    - "Consistent sidecar contract: STT POST /transcribe with audio_b64+session_id+language+sample_rate_hz; TTS POST /speak with text+session_id+language"

key-files:
  created: []
  modified:
    - rag_demo_system/tests/test_voice_adapters_official.py
    - rag_demo_system/backend/voice_adapters.py
    - rag_demo_system/frontend/index.html

key-decisions:
  - "_HARD_FAIL_STT frozenset placed at module level: avoids checking provider name in multiple places and makes it trivial to add future hard-fail providers"
  - "qwen3_asr and voxtral hard-fail on missing BASE_URL rather than silently degrading: ensures misconfiguration surfaces immediately instead of ghost-routing to fallback providers"
  - "qwen3_tts hard-fail mirrors the STT pattern for consistency: all three new providers raise RuntimeError when unconfigured"

patterns-established:
  - "Hard-fail provider pattern: add name to _HARD_FAIL_STT; early-return block in transcribe_audio() handles routing; no changes to fallback loop needed"
  - "New TTS provider pattern: add if-branch before vosk_tts block; check env var; raise RuntimeError if unset; POST to /speak with standard payload"

requirements-completed: [VPROV-01, VPROV-02, VPROV-03, VPROV-04, VPROV-05]

duration: 2min
completed: "2026-03-25"
---

# Phase 02 Plan 01: Voice Provider Adapters Summary

**HTTP dispatch branches for qwen3_asr, qwen3_tts, and voxtral wired into voice_adapters.py with hard-fail guards, health status entries, frontend dropdowns, and 12 passing contract tests**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-25T22:05:41Z
- **Completed:** 2026-03-25T22:07:42Z
- **Tasks:** 3 (Task 1: RED tests, Task 2: GREEN implementation, Task 3: frontend)
- **Files modified:** 3

## Accomplishments

- Added `_HARD_FAIL_STT` frozenset and early-return guard in `transcribe_audio()` that routes `qwen3_asr` and `voxtral` directly to their sidecar endpoints with hard-fail on missing `BASE_URL`
- Added `qwen3_tts` branch in `synthesize_audio_with_provider()` with hard-fail on missing `QWEN3_TTS_BASE_URL`, POST to `/speak` with standard `{text, session_id, language}` payload
- Added health status entries for all 3 providers in `build_voice_statuses()` using `_service_status()` against their respective env vars
- Extended `sttProviderSelect` to 6 options and `ttsProviderSelect` to 4 options in the frontend
- 12/12 contract tests pass (6 existing regression-free, 6 new covering routing and hard-fail paths)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write contract tests for 3 new providers + 3 hard-fail tests** - `af9ccd6` (test)
2. **Task 2: Add adapter branches + hard-fail + health status to voice_adapters.py** - `9c40587` (feat)
3. **Task 3: Add new provider options to frontend dropdowns** - `1b8c7ce` (feat)

_Note: Task 1 follows TDD RED phase; 4 of 6 new tests failed as expected before Task 2._

## Files Created/Modified

- `rag_demo_system/tests/test_voice_adapters_official.py` - 6 new contract tests appended (3 routing + 3 hard-fail)
- `rag_demo_system/backend/voice_adapters.py` - `_HARD_FAIL_STT` constant, hard-fail guard in `transcribe_audio()`, `qwen3_tts` branch in `synthesize_audio_with_provider()`, 3 new health status entries
- `rag_demo_system/frontend/index.html` - `qwen3_asr` and `voxtral` options in `sttProviderSelect`; `qwen3_tts` option in `ttsProviderSelect`

## Decisions Made

- `_HARD_FAIL_STT` placed as module-level `frozenset` so future hard-fail providers can be added by appending to the set without touching the dispatch logic
- All three new providers raise `RuntimeError` on missing env var: ensures misconfiguration is immediately visible rather than silently routing to a fallback provider
- `qwen3_tts` TTS branch placed before the `vosk_tts` branch to follow the natural ordering of newer-first provider checks

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

During the RED phase, `test_transcribe_audio_supports_qwen3_asr_service` and `test_transcribe_audio_supports_voxtral_service` passed unexpectedly. This is because the existing fallback loop in `transcribe_audio()` already handles arbitrary provider names via `os.getenv(f"{name.upper()}_BASE_URL")`. The RED state still held for the 4 remaining tests (qwen3_tts dispatch branch absent, hard-fail logic absent). After Task 2 added the `_HARD_FAIL_STT` guard, all 12 tests pass consistently.

## User Setup Required

None - no external service configuration required. The new providers activate automatically when their `BASE_URL` env vars are set at runtime.

## Next Phase Readiness

- Plan 02-02 (sidecar server implementations) can now start: the adapter layer is in place and the contract tests define the exact HTTP interface the sidecars must implement
- All 5 VPROV requirements satisfied
- No blockers for Phase 02 completion

## Self-Check: PASSED

- FOUND: rag_demo_system/tests/test_voice_adapters_official.py
- FOUND: rag_demo_system/backend/voice_adapters.py
- FOUND: rag_demo_system/frontend/index.html
- FOUND: .planning/phases/02-voice-provider-adapters/02-01-SUMMARY.md
- Commits af9ccd6, 9c40587, 1b8c7ce all present in git log

---
*Phase: 02-voice-provider-adapters*
*Completed: 2026-03-25*
