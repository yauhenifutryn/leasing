---
phase: 02-voice-provider-adapters
verified: 2026-03-25T22:30:00Z
status: passed
score: 15/15 must-haves verified
gaps: []
human_verification:
  - test: "Run qwen3_tts_server.py on GPU server with QWEN3_TTS_MODEL_ID set and POST to /speak"
    expected: "Returns JSON with audio_b64 (PCM16, 24kHz), sample_rate_hz=24000, provider=qwen3_tts"
    why_human: "Requires GPU server with qwen-tts==0.1.1 installed and model weights downloaded; cannot verify in dev environment"
  - test: "Run qwen3_asr_server.py on GPU server with QWEN3_ASR_MODEL_ID set and POST to /transcribe with Russian audio"
    expected: "Returns JSON with text (non-empty Russian transcription), provider=qwen3_asr"
    why_human: "Requires GPU server with qwen-asr==0.0.6 and model weights; dev environment has no GPU"
  - test: "Run voxtral_server.py on GPU server with VOXTRAL_MODEL_ID set and POST to /transcribe with 24kHz Russian audio"
    expected: "Returns JSON with text (non-empty Russian transcription), provider=voxtral. Audio resampled to 16kHz correctly."
    why_human: "Requires GPU server with transformers>=5.2.0 + VoxtralRealtimeForConditionalGeneration and model weights"
---

# Phase 02: Voice Provider Adapters Verification Report

**Phase Goal:** Qwen3-TTS, Qwen3-ASR, and Voxtral are available as selectable providers in the UI, each backed by a validated sidecar or adapter, and no adapter silently falls through to the default provider
**Verified:** 2026-03-25T22:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Selecting qwen3_asr as STT provider routes transcription through the Qwen3-ASR sidecar HTTP endpoint | VERIFIED | `_HARD_FAIL_STT` frozenset + early-return block in `transcribe_audio()`; test `test_transcribe_audio_supports_qwen3_asr_service` passes; calls `http://qwen3-asr.local/transcribe` with correct JSON |
| 2 | Selecting qwen3_tts as TTS provider routes synthesis through the Qwen3-TTS sidecar HTTP endpoint | VERIFIED | `if preferred == "qwen3_tts":` branch in `synthesize_audio_with_provider()`; test `test_synthesize_audio_supports_qwen3_tts_service` passes; calls `http://qwen3-tts.local/speak` with `{text, session_id, language}` |
| 3 | Selecting voxtral as STT provider routes transcription through the Voxtral sidecar HTTP endpoint | VERIFIED | `_HARD_FAIL_STT` frozenset + early-return block in `transcribe_audio()`; test `test_transcribe_audio_supports_voxtral_service` passes; calls `http://voxtral.local/transcribe` with correct JSON |
| 4 | When a new provider is selected but its BASE_URL env var is unset, the voice turn fails with RuntimeError instead of silently falling back to another provider | VERIFIED | Hard-fail block raises `RuntimeError(f"{preferred} service unavailable: {preferred.upper()}_BASE_URL not set")` before any fallback loop is entered; tests `test_qwen3_asr_hard_fail_when_unconfigured` and `test_voxtral_hard_fail_when_unconfigured` pass |
| 5 | When qwen3_tts is selected but QWEN3_TTS_BASE_URL is unset, synthesis fails with RuntimeError | VERIFIED | `if not base_url: raise RuntimeError("Qwen3-TTS service unavailable: QWEN3_TTS_BASE_URL not set")`; test `test_qwen3_tts_hard_fail_when_unconfigured` passes |
| 6 | The frontend dropdown shows qwen3_asr and voxtral as STT options and qwen3_tts as a TTS option | VERIFIED | `sttProviderSelect` has 6 options: `[sensevoice, whisper, vosk, yandex_speechkit, qwen3_asr, voxtral]`; `ttsProviderSelect` has 4 options: `[cosyvoice, vosk_tts, yandex_speechkit, qwen3_tts]` |
| 7 | All 12 contract tests (6 existing + 6 new) pass | VERIFIED | `pytest tests/test_voice_adapters_official.py -v` output: `12 passed in 0.13s`; zero failures or errors |
| 8 | Qwen3-TTS sidecar serves /health and /speak endpoints following the vosk_tts_server.py pattern | VERIFIED | `qwen3_tts_server.py` has `create_app`, `create_unavailable_app`, `_build_default_app`, `app = _build_default_app()`; `/health` returns `{ok, provider}`; `/speak` returns `{ok, provider, session_id, audio_b64, sample_rate_hz}` |
| 9 | Qwen3-ASR sidecar serves /health and /transcribe endpoints following the whisper_server.py pattern | VERIFIED | `qwen3_asr_server.py` has full pattern; temp WAV write with `wave` module; `finally` block calls `wav_path.unlink(missing_ok=True)`; returns `{ok, provider, session_id, text}` |
| 10 | Voxtral sidecar serves /health and /transcribe endpoints following the whisper_server.py pattern | VERIFIED | `voxtral_server.py` has full pattern; `scipy.signal.resample` for 24kHz-to-16kHz; `_target_sr = self._processor.feature_extractor.sampling_rate`; batch/offline API only |
| 11 | Each sidecar loads the correct HuggingFace model ID at startup | VERIFIED | Qwen3-TTS: `"Qwen/Qwen3-TTS-12Hz-1.7B-Base"` (Base variant); Qwen3-ASR: `"Qwen/Qwen3-ASR-1.7B"`; Voxtral: `"mistralai/Voxtral-Mini-4B-Realtime-2602"` |
| 12 | Each sidecar uses its own requirements file specifying model-specific dependencies isolated from the shared backend venv | VERIFIED | `requirements-qwen3-tts.txt` (qwen-tts==0.1.1, no transformers pin); `requirements-qwen3-asr.txt` (qwen-asr==0.0.6); `requirements-voxtral.txt` (transformers>=5.2.0, scipy, mistral-common) — all three independent of each other and of the shared venv |
| 13 | Voxtral sidecar resamples audio from 24kHz to 16kHz | VERIFIED | `scipy.signal.resample(audio_float, num_samples)` when `sample_rate_hz != self._target_sr`; no streaming API keywords in executable code |
| 14 | build_voice_statuses() reports health for all 3 new providers | VERIFIED | Returns dict with keys `qwen3_asr`, `qwen3_tts`, `voxtral`, each using `_service_status()` against their respective env vars |
| 15 | Frontend dropdown selection sends stt_provider/tts_provider string to backend via session.update | VERIFIED | `app.js` `buildSessionUpdate()` reads `selectedSttProvider`/`selectedTtsProvider` (updated by change listeners on both selects) and includes them as `stt_provider`/`tts_provider` in the WebSocket `session.update` message |

**Score:** 15/15 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `rag_demo_system/tests/test_voice_adapters_official.py` | Contract tests for 3 new providers + hard-fail tests | VERIFIED | 254 lines; 12 test functions (6 existing + 6 new); all pass; follows FakeResponse+monkeypatch+calls[] pattern |
| `rag_demo_system/backend/voice_adapters.py` | Adapter dispatch branches + health status entries | VERIFIED | 246 lines; `_HARD_FAIL_STT` frozenset at module level; hard-fail guard in `transcribe_audio()`; `qwen3_tts` branch in `synthesize_audio_with_provider()`; 3 new entries in `build_voice_statuses()` |
| `rag_demo_system/frontend/index.html` | New option elements in STT and TTS dropdowns | VERIFIED | `sttProviderSelect` has qwen3_asr + voxtral; `ttsProviderSelect` has qwen3_tts; grep count = 3 |
| `rag_demo_system/services/qwen3_tts_server.py` | Qwen3-TTS FastAPI sidecar with /health and /speak | VERIFIED | 89 lines; syntax clean; `Qwen3TTSSynthesizer` class; deferred import; correct model variant `12Hz-1.7B-Base`; `generate_voice_clone(language="Russian")`; PCM16 via soundfile in-memory WAV |
| `rag_demo_system/services/qwen3_asr_server.py` | Qwen3-ASR FastAPI sidecar with /health and /transcribe | VERIFIED | 97 lines; syntax clean; `Qwen3ASRTranscriber` class; temp WAV pattern; `language=language` passthrough; `finally` cleanup |
| `rag_demo_system/services/voxtral_server.py` | Voxtral FastAPI sidecar with /health and /transcribe | VERIFIED | 109 lines; syntax clean; `VoxtralTranscriber` class; `scipy.signal.resample`; batch API; no streaming API keywords in executable code |
| `rag_demo_system/requirements-qwen3-tts.txt` | Pip requirements for Qwen3-TTS isolated venv | VERIFIED | qwen-tts==0.1.1, torch>=2.0.0, fastapi==0.115.6, uvicorn==0.30.6, pydantic>=2.10.0, soundfile |
| `rag_demo_system/requirements-qwen3-asr.txt` | Pip requirements for Qwen3-ASR isolated venv | VERIFIED | qwen-asr==0.0.6, torch>=2.0.0, fastapi==0.115.6, uvicorn==0.30.6, pydantic>=2.10.0 |
| `rag_demo_system/requirements-voxtral.txt` | Pip requirements for Voxtral isolated venv | VERIFIED | transformers>=5.2.0, torch>=2.0.0, fastapi==0.115.6, uvicorn==0.30.6, pydantic>=2.10.0, mistral-common, numpy, scipy |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `voice_adapters.py` | `QWEN3_ASR_BASE_URL` env var | `os.getenv(f"{preferred.upper()}_BASE_URL")` in `transcribe_audio()` hard-fail block | WIRED | Dynamic env var lookup using f-string from `_HARD_FAIL_STT`; hard-fail raises `RuntimeError` if not set |
| `voice_adapters.py` | `QWEN3_TTS_BASE_URL` env var | `os.getenv("QWEN3_TTS_BASE_URL")` in `synthesize_audio_with_provider()` | WIRED | Explicit string lookup; hard-fail raises RuntimeError if not set; POSTs to `base_url/speak` |
| `voice_adapters.py` | `VOXTRAL_BASE_URL` env var | `os.getenv(f"{preferred.upper()}_BASE_URL")` in `transcribe_audio()` hard-fail block | WIRED | Same dynamic pattern as qwen3_asr; hard-fail raises RuntimeError if not set |
| `frontend/index.html` | `voice_adapters.py` | `session.update` sends `stt_provider`/`tts_provider` string to backend via WebSocket | WIRED | `app.js` `buildSessionUpdate()` at line 87 reads `selectedSttProvider`/`selectedTtsProvider`; sent on connect, voice toggle, model change, and select change events |

---

### Data-Flow Trace (Level 4)

Not applicable: all artifacts in this phase are backend dispatch modules and server-side scripts, not UI components that render dynamic data. The contract tests provide equivalent end-to-end coverage for the dispatch path.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 12 contract tests pass | `python3 -m pytest tests/test_voice_adapters_official.py -v` | `12 passed in 0.13s` | PASS |
| qwen3_asr hard-fail when BASE_URL unset | Covered by `test_qwen3_asr_hard_fail_when_unconfigured` | `RuntimeError: ...QWEN3_ASR_BASE_URL not set` | PASS |
| qwen3_tts hard-fail when BASE_URL unset | Covered by `test_qwen3_tts_hard_fail_when_unconfigured` | `RuntimeError: ...QWEN3_TTS_BASE_URL not set` | PASS |
| voxtral hard-fail when BASE_URL unset | Covered by `test_voxtral_hard_fail_when_unconfigured` | `RuntimeError: ...VOXTRAL_BASE_URL not set` | PASS |
| All 3 sidecar scripts parse without SyntaxError | `python3 -c "import ast; ast.parse(...)"` x3 | `syntax ok` for all three | PASS |
| Frontend has exactly 3 new option values | `grep -c 'qwen3_asr\|voxtral\|qwen3_tts'` on index.html | `3` | PASS |
| STT dropdown has 6 options, TTS has 4 | Python `re.findall` on index.html | STT: `[sensevoice, whisper, vosk, yandex_speechkit, qwen3_asr, voxtral]`; TTS: `[cosyvoice, vosk_tts, yandex_speechkit, qwen3_tts]` | PASS |
| All 6 sidecar/requirements files exist | `ls rag_demo_system/services/qwen3_*_server.py rag_demo_system/services/voxtral_server.py rag_demo_system/requirements-*.txt` | All 6 FOUND | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| VPROV-01 | 02-01-PLAN.md, 02-02-PLAN.md | Qwen3-TTS adapter integrated into voice_adapters.py with sidecar FastAPI service | SATISFIED | `qwen3_tts` branch in `synthesize_audio_with_provider()`; `qwen3_tts_server.py` sidecar; `requirements-qwen3-tts.txt`; test `test_synthesize_audio_supports_qwen3_tts_service` passes |
| VPROV-02 | 02-01-PLAN.md, 02-02-PLAN.md | Qwen3-ASR adapter integrated into voice_adapters.py with sidecar FastAPI service | SATISFIED | Hard-fail branch in `transcribe_audio()` via `_HARD_FAIL_STT`; `qwen3_asr_server.py` sidecar; `requirements-qwen3-asr.txt`; test `test_transcribe_audio_supports_qwen3_asr_service` passes |
| VPROV-03 | 02-01-PLAN.md, 02-02-PLAN.md | Voxtral STT adapter integrated into voice_adapters.py with sidecar | SATISFIED | Hard-fail branch in `transcribe_audio()` via `_HARD_FAIL_STT`; `voxtral_server.py` sidecar; `requirements-voxtral.txt`; test `test_transcribe_audio_supports_voxtral_service` passes |
| VPROV-04 | 02-01-PLAN.md | All new adapters pass the existing voice adapter contract tests | SATISFIED | `pytest tests/test_voice_adapters_official.py`: 12/12 pass; 6 existing tests still pass (no regression); 6 new tests cover 3 routing paths and 3 hard-fail paths |
| VPROV-05 | 02-01-PLAN.md | Frontend voice provider selector updated to include all new providers | SATISFIED | `sttProviderSelect` contains `qwen3_asr` and `voxtral`; `ttsProviderSelect` contains `qwen3_tts`; `app.js` sends these as `stt_provider`/`tts_provider` in `session.update` |

No orphaned requirements: all 5 requirements mapped to Phase 2 in REQUIREMENTS.md are claimed in plan frontmatter and verified above.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `voxtral_server.py` | 51 | Comment mentions `input_features_generator` and `padding_cache` | Info | Comment only, not executable code; documents what was deliberately excluded (Pitfall 6). No impact. |

No TODOs, FIXMEs, placeholders, empty returns, or stub patterns found in any modified or created file.

---

### Human Verification Required

The following cannot be verified without a GPU server with model weights:

#### 1. Qwen3-TTS End-to-End Synthesis

**Test:** On GPU server, set `QWEN3_TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base` and `QWEN3_TTS_DEVICE=cuda:0`, install `requirements-qwen3-tts.txt`, start `uvicorn rag_demo_system.services.qwen3_tts_server:app`, and POST `{"text": "Здравствуйте", "session_id": "test", "language": "ru"}` to `/speak`.

**Expected:** Response contains `audio_b64` (non-empty base64 PCM16), `sample_rate_hz=24000`, `provider=qwen3_tts`, `ok=True`.

**Why human:** Requires GPU, qwen-tts==0.1.1 package, and Qwen3-TTS model weights. Cannot test in dev environment.

#### 2. Qwen3-ASR End-to-End Transcription

**Test:** On GPU server, install `requirements-qwen3-asr.txt`, start the ASR sidecar, and POST a base64-encoded Russian speech WAV to `/transcribe`.

**Expected:** Response contains non-empty Russian text, `provider=qwen3_asr`, `ok=True`.

**Why human:** Requires GPU, qwen-asr==0.0.6, and Qwen3-ASR-1.7B model weights.

#### 3. Voxtral End-to-End Transcription with 24kHz Audio

**Test:** On GPU server, install `requirements-voxtral.txt`, start the Voxtral sidecar, and POST 24kHz base64-encoded Russian speech audio to `/transcribe`. Verify resampling to 16kHz works correctly in practice.

**Expected:** Response contains non-empty Russian text, `provider=voxtral`, `ok=True`. Audio quality not degraded by 24kHz-to-16kHz resampling.

**Why human:** Requires GPU, transformers>=5.2.0 with VoxtralRealtimeForConditionalGeneration, and Voxtral-Mini-4B-Realtime-2602 model weights.

---

### Gaps Summary

None. All 15 must-have truths are verified, all 9 artifacts pass all applicable levels (exist, substantive, wired), all 5 requirement IDs are satisfied with implementation evidence, all 6 commits are present in git history, and no anti-patterns block the phase goal. The three human verification items above are deployment-time concerns — they require GPU hardware that is not available in the development environment, but the sidecar code structure and interface contracts are fully verified.

---

_Verified: 2026-03-25T22:30:00Z_
_Verifier: Claude (gsd-verifier)_
