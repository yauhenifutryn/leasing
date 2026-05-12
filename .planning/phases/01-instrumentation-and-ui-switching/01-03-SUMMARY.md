---
phase: 01-instrumentation-and-ui-switching
plan: 03
subsystem: ui
tags: [vanilla-js, websocket, localstorage, html-select, session-update]

# Dependency graph
requires:
  - phase: 01-instrumentation-and-ui-switching/01-01
    provides: VoiceSession fields (brain_model, stt_provider, tts_provider, stack_id) that the UI now sends
provides:
  - voiceProviderSelect HTML element with local/yandex_speechkit/oss_russian/yandex_realtime options
  - brainModelSelect HTML element with Qwen3-30B-A3B and Qwen3.5-35B-A3B options
  - sttProviderSelect HTML element with sensevoice/whisper/vosk/yandex_speechkit options
  - ttsProviderSelect HTML element with cosyvoice/vosk_tts/yandex_speechkit options
  - buildSessionUpdate() helper that bundles all 5 selector values into session.update payload
  - localStorage persistence under rag_voice_provider, rag_brain_model, rag_stt_provider, rag_tts_provider
  - session.updated handler shows stack_id in voice status display
affects: [01-02, phase-03-benchmark, phase-05-deploy]

# Tech tracking
tech-stack:
  added: []
  patterns: [buildSessionUpdate() centralises multi-field session.update payload to avoid field drift across call sites]

key-files:
  created: []
  modified:
    - rag_demo_system/frontend/index.html
    - rag_demo_system/frontend/app.js

key-decisions:
  - "voiceProviderSelect added alongside the 3 new selectors because plan's buildSessionUpdate() payload requires voice_provider field and the variable was absent from existing code"
  - "All 4 new selectors placed in the voice section of index.html for logical grouping"
  - "buildSessionUpdate() helper centralises payload construction to ensure all 5 fields are always sent together"

patterns-established:
  - "buildSessionUpdate() pattern: all session.update sends go through a single helper; adding new fields only requires editing one function"
  - "localStorage key naming: rag_{field_name} (rag_brain_model, rag_stt_provider, rag_tts_provider, rag_voice_provider)"

requirements-completed: [SWITCH-01, SWITCH-02]

# Metrics
duration: 10min
completed: 2026-03-25
---

# Phase 01 Plan 03: UI Selectors and Session Update Wiring Summary

**Four operator-facing HTML selectors (voice provider, brain model, STT, TTS) wired to WebSocket session.update via a centralised buildSessionUpdate() helper with full localStorage persistence**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-25T20:58:00Z
- **Completed:** 2026-03-25T20:59:26Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `voiceProviderSelect`, `brainModelSelect`, `sttProviderSelect`, `ttsProviderSelect` elements to index.html in the voice section
- Introduced `buildSessionUpdate()` in app.js consolidating all 5 selector values into every session.update send
- Wired change listeners for all 4 new selectors; each updates its variable, writes localStorage, and sends session.update when the socket is open
- `initSession()` now restores all 4 new selector values from localStorage and syncs them back into the DOM elements
- `connectVoice()` onopen sends the full 5-field session.update via `buildSessionUpdate()` instead of a partial payload
- `session.updated` handler now shows `stack_id` in the voice status badge (falls back to `voice_provider` then "local")

## Task Commits

1. **Task 1: Add 3 new select elements to index.html** - `347fd6f` (feat)
2. **Task 2: Wire JS variables, change listeners, localStorage, and session.update** - `88d63f4` (feat)

## Files Created/Modified

- `rag_demo_system/frontend/index.html` - Added voiceProviderSelect + 3 new selector toggle-rows in the voice section
- `rag_demo_system/frontend/app.js` - Added 4 new variables, buildSessionUpdate(), updated initSession(), connectVoice(), backendSelect listener, plus 4 new change listeners; updated session.updated handler

## Decisions Made

- `voiceProviderSelect` added as a Rule 2 auto-fix: plan requires `voice_provider` in `buildSessionUpdate()` payload but the variable and DOM element were absent from the codebase. Adding it is required for correctness.
- All new selectors placed in the voice card section (below transcript/status) rather than the chat card, since they are voice pipeline controls.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added voiceProviderSelect HTML element and selectedVoiceProvider JS variable**
- **Found during:** Task 1 (Add selectors to index.html) and Task 2 (Wire JS)
- **Issue:** Plan's `buildSessionUpdate()` payload includes `voice_provider: selectedVoiceProvider`, but neither the HTML selector nor the JS variable existed in the codebase. The session.update payload would have been malformed (undefined field) without them.
- **Fix:** Added `<select id="voiceProviderSelect">` with 4 option values in the voice section of index.html; added `let selectedVoiceProvider = "local"` declaration, localStorage restore/sync in `initSession()`, and a `voiceProviderSelect` change listener in app.js.
- **Files modified:** rag_demo_system/frontend/index.html, rag_demo_system/frontend/app.js
- **Verification:** `grep voiceProviderSelect index.html` confirms element present; `grep selectedVoiceProvider app.js` confirms variable and listener present.
- **Committed in:** 347fd6f (Task 1), 88d63f4 (Task 2)

---

**Total deviations:** 1 auto-fixed (Rule 2 - missing critical functionality)
**Impact on plan:** Auto-fix necessary for session.update payload correctness. No scope creep — the variable and selector were implied by the plan's payload spec but absent from the existing code.

## Issues Encountered

None beyond the voiceProviderSelect gap documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- UI is ready to drive Plan 02 (backend session.update handler): the frontend sends all 5 fields on every change and on connect
- Plan 02 can read `brain_model`, `stt_provider`, `tts_provider` from the WebSocket session.update message and update `VoiceSession` accordingly
- `stack_id` display in voice status will show meaningful values once Plan 02 wires the backend response

---
*Phase: 01-instrumentation-and-ui-switching*
*Completed: 2026-03-25*

## Self-Check: PASSED

- FOUND: rag_demo_system/frontend/index.html
- FOUND: rag_demo_system/frontend/app.js
- FOUND: .planning/phases/01-instrumentation-and-ui-switching/01-03-SUMMARY.md
- FOUND commit: 347fd6f (Task 1)
- FOUND commit: 88d63f4 (Task 2)
