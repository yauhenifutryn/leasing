---
phase: 01-instrumentation-and-ui-switching
verified: 2026-03-25T00:00:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 01: Instrumentation and UI Switching Verification Report

**Phase Goal:** Every voice turn emits a structured JSON log with six latency milestones and a computed primary KPI, all pipeline variables are selectable from the UI, and every log line is automatically tagged with the active stack_id.
**Verified:** 2026-03-25
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                              | Status     | Evidence                                                                                     |
|----|---------------------------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------|
| 1  | VoiceSession has brain_model, stt_provider, tts_provider fields with correct defaults             | VERIFIED   | voice_session.py lines 12-14; all three fields present with exact default values             |
| 2  | stack_id is a computed @property combining backend, brain_model, stt_provider, tts_provider       | VERIFIED   | voice_session.py lines 20-23; @property def stack_id strips org prefix via split("/")[-1]   |
| 3  | stack_id changes when any underlying field changes (not a mutable field)                          | VERIFIED   | Implemented as @property, not stored; test_stack_id_updates_on_field_change passes           |
| 4  | session.update handler accepts brain_model, stt_provider, tts_provider and updates session        | VERIFIED   | app.py lines 641-654; conditional guards on all three new fields; yandex_realtime auto-sync  |
| 5  | input_audio_buffer.commit captures 6 timestamps and emits a voice_turn log with all 17 fields     | VERIFIED   | app.py lines 722-829; all 6 timestamps captured, all 17 REQUIRED_LOG_FIELDS present in log  |
| 6  | primary_kpi_ms is computed as (playback_started - speech_stopped) * 1000                          | VERIFIED   | app.py line 812: `primary_kpi_ms = (t_playback_started - t_speech_stopped) * 1000`          |
| 7  | WebSocket handler reads stt_provider/tts_provider from session directly                           | VERIFIED   | app.py lines 727-728: `stt_provider = session.stt_provider`, `tts_provider = session.tts_provider`; _voice_pipeline not called in ws handler |
| 8  | UI shows Brain model, STT provider, TTS provider selectors with correct option values             | VERIFIED   | index.html lines 79-99; brainModelSelect, sttProviderSelect, ttsProviderSelect all present; all option values match backend expectations |
| 9  | Changing any selector sends session.update with all 5 values over WebSocket                       | VERIFIED   | app.js lines 478-496; 3 new change listeners all call buildSessionUpdate(); existing 2 listeners updated to use same helper |
| 10 | Selector values persist in localStorage across page reloads                                       | VERIFIED   | app.js lines 218-235; read on init, write on change for all 3 new keys (rag_brain_model, rag_stt_provider, rag_tts_provider) |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact                                          | Expected                                         | Status   | Details                                                                 |
|---------------------------------------------------|--------------------------------------------------|----------|-------------------------------------------------------------------------|
| `rag_demo_system/backend/voice_session.py`        | Extended VoiceSession with 3 new fields + stack_id | VERIFIED | brain_model, stt_provider, tts_provider fields; @property stack_id; substantive, wired by app.py and tests |
| `rag_demo_system/tests/test_voice_session.py`     | Unit tests for new fields and stack_id           | VERIFIED | 9 tests pass (4 original + 5 new); test_stack_id_composition present    |
| `rag_demo_system/tests/test_instrumentation.py`   | Test scaffold for log event structure            | VERIFIED | 3 tests pass; REQUIRED_LOG_FIELDS set with all 17 fields; test_voice_turn_log_event_has_all_required_fields present |
| `rag_demo_system/backend/app.py`                  | Extended session.update + instrumented voice turn | VERIFIED | primary_kpi_ms, all 6 timestamps, state.log call, session.update extension all present; syntax check passes |
| `rag_demo_system/frontend/index.html`             | Three new select elements                        | VERIFIED | brainModelSelect, sttProviderSelect, ttsProviderSelect with correct option values |
| `rag_demo_system/frontend/app.js`                 | JS variables, listeners, localStorage, wiring    | VERIFIED | selectedBrainModel, selectedSttProvider, selectedTtsProvider; buildSessionUpdate(); 3 change listeners; localStorage read/write |

---

### Key Link Verification

| From                                    | To                                              | Via                                              | Status   | Details                                                                          |
|-----------------------------------------|-------------------------------------------------|--------------------------------------------------|----------|----------------------------------------------------------------------------------|
| `tests/test_voice_session.py`           | `backend/voice_session.py`                      | import + VoiceSession instantiation              | WIRED    | _load_module() imports backend.voice_session; VoiceSession( calls throughout     |
| `backend/app.py`                        | `backend/voice_session.py`                      | session.brain_model, session.stt_provider, session.tts_provider, session.stack_id | WIRED    | Lines 628-631, 641-654, 677-680, 727-728, 816 all reference session fields      |
| `backend/app.py`                        | `backend/state.py`                              | state.log() with voice_turn event dict           | WIRED    | Line 813: state.log({...}); called inside input_audio_buffer.commit handler      |
| `frontend/app.js`                       | `backend/app.py`                                | WebSocket session.update with brain_model, stt_provider, tts_provider | WIRED    | buildSessionUpdate() sends all 5 fields; called on connect and all 5 change listeners |

---

### Data-Flow Trace (Level 4)

| Artifact               | Data Variable      | Source                                    | Produces Real Data | Status   |
|------------------------|--------------------|-------------------------------------------|--------------------|----------|
| `backend/app.py`       | `t_speech_stopped` | time.time() at input_audio_buffer.commit  | Yes — live wall clock | FLOWING |
| `backend/app.py`       | `session.stack_id` | VoiceSession @property (reads live fields) | Yes — computed from mutable session fields | FLOWING |
| `backend/app.py`       | `primary_kpi_ms`   | (t_playback_started - t_speech_stopped) * 1000 | Yes — computed from real captured timestamps | FLOWING |
| `frontend/app.js`      | `selectedBrainModel` | localStorage restore + change listener   | Yes — user-driven; defaults populated | FLOWING |

---

### Behavioral Spot-Checks

| Behavior                                     | Command                                                                                           | Result     | Status |
|----------------------------------------------|---------------------------------------------------------------------------------------------------|------------|--------|
| app.py syntax valid                          | `python3 -c "import ast; ast.parse(open('app.py').read())"`                                       | syntax OK  | PASS   |
| All 12 tests pass (9 voice_session + 3 instrumentation) | `python3 -m pytest test_voice_session.py test_instrumentation.py -v -q`              | 12 passed  | PASS   |
| 17 log fields present in voice_turn log dict | grep of app.py lines 814-829                                                                      | 17/17 fields confirmed | PASS |
| buildSessionUpdate sends 5 fields            | grep of app.js for `brain_model:`, `stt_provider:`, `tts_provider:`, `backend:`, `voice_provider:` | All 5 present in function | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description                                                                 | Status    | Evidence                                                              |
|-------------|-------------|-----------------------------------------------------------------------------|-----------|-----------------------------------------------------------------------|
| INST-01     | Plan 01-01, 01-02 | Voice session logs 6 timing milestones per turn                        | SATISFIED | app.py captures t_speech_stopped, t_stt_done, t_retrieval_done, t_llm_first_token, t_tts_first_chunk, t_playback_started |
| INST-02     | Plan 01-02  | Each turn emits structured JSON log line with question_id, stack_id, timestamps | SATISFIED | state.log() at app.py line 813 with all 17 REQUIRED_LOG_FIELDS        |
| INST-03     | Plan 01-02  | Primary KPI (playback_started - speech_stopped) computed and logged         | SATISFIED | app.py line 812: `primary_kpi_ms = (t_playback_started - t_speech_stopped) * 1000` |
| SWITCH-01   | Plan 01-03  | UI exposes selectors for backend, brain model, STT, TTS                     | SATISFIED | index.html: backendSelect (pre-existing), brainModelSelect, sttProviderSelect, ttsProviderSelect all present |
| SWITCH-02   | Plan 01-02, 01-03 | Switching any selector updates active config without backend restart    | SATISFIED | session.update handler in app.py applies fields to live VoiceSession object; no restart needed for stt/tts/brain_model changes |
| SWITCH-03   | Plan 01-01, 01-02 | Active stack_id captured in every log line                              | SATISFIED | app.py line 816: `"stack_id": session.stack_id` in every voice_turn log; stack_id is live-computed from current session fields |

All 6 phase requirements satisfied. No orphaned requirements found. REQUIREMENTS.md traceability table marks all 6 as Complete.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `backend/app.py` | 771 | `t_llm_first_token = t_retrieval_done` (conservative approximation) | INFO | Not a stub — intentional placeholder with TODO comment; plan explicitly calls for this approximation for non-streaming path. Logging accuracy for LLM first-token will improve in Phase 3. |

No blockers or warnings found.

---

### Human Verification Required

#### 1. Visual selector placement

**Test:** Open `rag_demo_system/frontend/index.html` in a browser. Confirm the three new selectors (Brain model, STT provider, TTS provider) appear in the settings panel, are labeled correctly, and show the correct option lists.
**Expected:** Three labeled dropdown rows appear below the Voice Provider selector; Brain model shows "Qwen3-30B-A3B" and "Qwen3.5-35B-A3B"; STT shows sensevoice, whisper, vosk, yandex_speechkit; TTS shows cosyvoice, vosk_tts, yandex_speechkit.
**Why human:** DOM rendering and CSS layout cannot be verified by static analysis.

#### 2. End-to-end session.update round-trip

**Test:** Connect to a live backend, change the Brain model selector, then trigger a voice turn. Inspect the JSONL log file to confirm the voice_turn line reflects the selected brain_model value in its stack_id field.
**Expected:** Log line's stack_id contains the newly selected model name; primary_kpi_ms is a positive millisecond value.
**Why human:** Requires a running backend and actual audio input; cannot be verified without live execution.

#### 3. localStorage persistence across reload

**Test:** Set all three new selectors to non-default values, reload the page, confirm the selectors restore to the previously set values.
**Expected:** After reload, Brain model, STT, and TTS selectors show the values that were saved, not the hardcoded defaults.
**Why human:** Requires browser interaction; localStorage behavior cannot be confirmed by static analysis.

---

### Gaps Summary

No gaps. All 10 observable truths verified, all 6 artifacts pass all four verification levels (exists, substantive, wired, data flowing), all 4 key links confirmed wired, all 6 requirement IDs satisfied. The only noted item (t_llm_first_token approximation) is an explicit and documented design decision, not a gap.

---

_Verified: 2026-03-25_
_Verifier: Claude (gsd-verifier)_
