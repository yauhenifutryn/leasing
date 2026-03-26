---
phase: 04-qwen3-omni-hybrid
verified: 2026-03-26T14:00:00Z
status: passed
score: 8/8 must-haves verified
re_verification: false
---

# Phase 4: Qwen3-Omni Hybrid Verification Report

**Phase Goal:** Qwen3-Omni hybrid mode retrieves context via the existing RAG engine, injects it into the Omni prompt, is accessible as a UI provider option, and produces JSONL output directly comparable with split pipeline results.
**Verified:** 2026-03-26
**Status:** passed
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Omni sidecar has POST /chat accepting audio_b64, context_chunks, system_prompt; returns audio_b64, text, sample_rate_hz, t_omni_first_audio | VERIFIED | `qwen3_omni_server.py`: ChatRequest/ChatResponse Pydantic models at lines 37-48; `/chat` endpoint at line 194 |
| 2 | Omni sidecar has GET /health returning status ok | VERIFIED | `/health` at line 190 returns `{"ok": True, "provider": "qwen3_omni"}` |
| 3 | Selecting qwen3_omni routes voice turn: STT for RAG query, RAG retrieval, then Omni sidecar /chat with audio + chunks | VERIFIED | `app.py` lines 923-1001: branch on `session.voice_provider == "qwen3_omni"`, calls `transcribe_audio` (line 897), `engine.retrieve` (line 931), then `_requests.post(.../chat, json={"audio_b64": ..., "context_chunks": ...})` (line 940) |
| 4 | Omni path emits JSONL with all 6 timing fields, question_id, stack_id, transcript -- directly comparable with split pipeline | VERIFIED | `app.py` lines 975-992: `state.log({...})` includes all 6 fields: speech_stopped, stt_done, retrieval_done, llm_first_token, tts_first_chunk, playback_started plus question_id, stack_id, transcript, primary_kpi_ms |
| 5 | Qwen3-Omni appears in frontend voice provider dropdown and is selectable | VERIFIED | `frontend/index.html` line 75: `<option value="qwen3_omni">qwen3_omni</option>` |
| 6 | Out-of-scope questions return refusal because strict grounding system prompt is injected | VERIFIED | `qwen3_omni_server.py` lines 91-96: Russian grounding prompt always built from SYSTEM_PROMPT_TEMPLATE with context_block; custom system_prompt override only if non-empty |
| 7 | normalize_voice_provider("qwen3_omni") returns "qwen3_omni" instead of falling back to "local" | VERIFIED | `yandex_realtime.py` line 23: `"qwen3_omni"` in allowlist set; contract test `test_omni_voice_provider_in_normalizer_allowlist` PASSES (7/7 tests pass, 0 xfail) |
| 8 | Contract tests verify all integration contracts without xfail markers | VERIFIED | `pytest rag_demo_system/tests/test_qwen3_omni_adapter.py`: 7 passed, 0 xfail |

**Score:** 8/8 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `rag_demo_system/services/qwen3_omni_server.py` | FastAPI sidecar with /health and /chat | VERIFIED | 215 lines (min_lines: 100), syntax OK, contains ChatRequest, ChatResponse, Qwen3OmniInference, create_app, /health, /chat, SYSTEM_PROMPT_TEMPLATE, os.unlink(tmp_path) in finally, thinker_return_dict_in_generate=True, return_audio=True, Qwen3OmniMoeForConditionalGeneration, process_mm_info |
| `rag_demo_system/requirements-qwen3-omni.txt` | Isolated pip deps with transformers pin | VERIFIED | 10 lines, contains `transformers==4.57.3`, `qwen-omni-utils>=0.0.9`, isolation warning comment "NEVER in the shared .venv" |
| `rag_demo_system/tests/test_qwen3_omni_adapter.py` | 7+ contract tests | VERIFIED | 245 lines (min_lines: 80), all 7 test functions present, all pass, 0 xfail |
| `rag_demo_system/backend/app.py` | Omni dispatch branch in WebSocket handler | VERIFIED | Contains `qwen3_omni` at lines 818 (session.update override) and 923 (dispatch branch); QWEN3_OMNI_BASE_URL hard-fail at line 925-928 |
| `rag_demo_system/backend/yandex_realtime.py` | qwen3_omni in normalize_voice_provider allowlist | VERIFIED | Line 23: `"qwen3_omni"` present in allowlist set |
| `rag_demo_system/backend/voice_adapters.py` | qwen3_omni health entry in build_voice_statuses() | VERIFIED | Line 50: `"qwen3_omni": _service_status("qwen3_omni", os.getenv("QWEN3_OMNI_BASE_URL"))` |
| `rag_demo_system/frontend/index.html` | qwen3_omni option in voiceProviderSelect dropdown | VERIFIED | Line 75: `<option value="qwen3_omni">qwen3_omni</option>` |
| `rag_demo_system/.env.bench.omni_hybrid` | Benchmark env profile with QWEN3_OMNI_BASE_URL | VERIFIED | Contains QWEN3_OMNI_BASE_URL=http://127.0.0.1:8002, VOICE_PROVIDER=qwen3_omni, SENSEVOICE_BASE_URL, supervisorctl stop/start comment |
| `rag_demo_system/scripts/supervisord.conf` | [program:qwen3_omni] supervisor entry | VERIFIED | Lines 79-85: [program:qwen3_omni] with autostart=false, STACK_QWEN3_OMNI_CMD, correct log paths |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `app.py` | `qwen3_omni_server.py` (sidecar) | HTTP POST to QWEN3_OMNI_BASE_URL/chat | WIRED | Line 940-946: `_requests.post(omni_base_url.rstrip("/") + "/chat", json={...})` |
| `app.py` | `voice_adapters.py` | transcribe_audio() call for RAG query in Omni path | WIRED | Line 897: `transcribe_audio(audio_b64, session_id=session_id)` runs before Omni branch (STT for RAG query extraction per D-01) |
| `frontend/index.html` | `app.py` | session.update with voice_provider=qwen3_omni | WIRED | Frontend `<option value="qwen3_omni">` sends to WebSocket; app.py line 803 calls `normalize_voice_provider` which passes "qwen3_omni" through; line 818 Omni session override executes |
| `app.py` | `yandex_realtime.py` | normalize_voice_provider() call | WIRED | Line 37: `from .yandex_realtime import ... normalize_voice_provider`; line 803: `normalize_voice_provider(event.get("voice_provider"))` |
| `app.py` | RAG engine | engine.retrieve() in Omni dispatch branch | WIRED | Line 931-933: `engine.retrieve(text, fast=True, voice_fast=True, session_id=session_id)` inside Omni branch |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `app.py` Omni dispatch | `chunk_texts` (RAG context) | `engine.retrieve(text, ...)` -> `retrieval.get("final")` | Yes -- live RAG engine call, same as split pipeline path | FLOWING |
| `app.py` Omni dispatch | `omni_data` (sidecar response) | `_requests.post(.../chat)` -> `omni_resp.json()` | Yes -- live HTTP POST to sidecar | FLOWING |
| `app.py` JSONL log | `state.log({...})` | All 6 timing fields populated from `time.time()` calls in the Omni branch | Yes -- real timestamps, not hardcoded | FLOWING |
| `qwen3_omni_server.py` | `system_text` (grounding) | `SYSTEM_PROMPT_TEMPLATE.format(context_block="\n\n".join(req.context_chunks))` | Yes -- context_blocks from RAG forwarded from app.py | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Sidecar server parses without syntax errors | `python3 -c "import ast; ast.parse(...qwen3_omni_server.py...)"` | `syntax OK` | PASS |
| 7 contract tests pass with 0 xfail | `pytest test_qwen3_omni_adapter.py -v` | `7 passed in 0.13s` | PASS |
| Frontend contract test includes qwen3_omni assertion | `pytest test_frontend_config_contract.py -v` | `3 passed in 0.02s` | PASS |
| Full collectable test suite remains green | `pytest rag_demo_system/tests/ -q` (121 collected) | `121 passed, 1 skipped in 0.53s` | PASS |
| All 3 backend Python files parse without errors | `python3 -c "import ast; ast.parse(..."` for app.py, yandex_realtime.py, voice_adapters.py | `app.py OK`, `yandex_realtime.py OK`, `voice_adapters.py OK` | PASS |

Note: 8 test files with pre-existing collection errors (`test_batching.py`, etc.) fail due to a stale path reference (`/Users/jenyafutrin/Desktop/leasing/`) and are unrelated to phase 4.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| OMNI-01 | 04-01 (partial), 04-02 | Qwen3-Omni hybrid adapter retrieves chunks via existing RAG engine and injects them into Omni prompt | SATISFIED | `app.py` lines 931-944: `engine.retrieve()` -> `chunk_texts` -> `json={"context_chunks": chunk_texts}` in POST /chat; `qwen3_omni_server.py` lines 91-96: context_chunks joined and injected into grounding system prompt |
| OMNI-02 | 04-02 | Omni hybrid mode accessible as a voice provider option in the UI alongside split pipeline providers | SATISFIED | `index.html` line 75: `<option value="qwen3_omni">qwen3_omni</option>` in voiceProviderSelect; `test_frontend_config_contract.py` line 30 pins the contract |
| OMNI-03 | 04-01 (partial), 04-02 | Omni results use same log format for direct comparison with split pipeline results | SATISFIED | `app.py` lines 975-992: `state.log({...})` with all 6 timing fields; `t_llm_first_token = t_omni_first_audio` and `t_tts_first_chunk = t_omni_first_audio` (D-06 collapsed); `test_omni_jsonl_has_required_fields` contract test PASSES |

All 3 requirement IDs from REQUIREMENTS.md (OMNI-01, OMNI-02, OMNI-03) claimed in plan frontmatter and verified in codebase. REQUIREMENTS.md tracks all three as "Complete" for Phase 4.

No orphaned requirements: REQUIREMENTS.md Phase 4 row covers exactly OMNI-01, OMNI-02, OMNI-03.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | -- | -- | -- | -- |

No TODOs, FIXMEs, placeholder patterns, empty return stubs, or hardcoded empty data found in phase 4 files. The docstring in `test_qwen3_omni_adapter.py` line 4 references "xfail" in past tense (documentation of prior state) -- not an active marker.

---

### Human Verification Required

#### 1. End-to-End Omni Voice Turn (GPU server required)

**Test:** On a machine with the Omni sidecar running (`venvs/omni/` set up, `QWEN3_OMNI_BASE_URL` pointing to the sidecar), select "qwen3_omni" in the frontend dropdown, speak a question about Mikro Leasing products, and observe the response.
**Expected:** Audio response in Russian, answer grounded in RAG context, JSONL file contains a new entry with `stack_id == "our_rag__Qwen3-Omni-30B-A3B__omni__omni"` and all 6 timing fields.
**Why human:** Sidecar requires Qwen3-Omni-30B-A3B-Instruct model weights and A100-class GPU. Cannot verify inference behavior programmatically without the model.

#### 2. Out-of-Scope Refusal Behavior

**Test:** With Omni active, ask a question clearly outside the Mikro Leasing knowledge base (e.g., "What is the capital of France?").
**Expected:** Response contains the Russian refusal phrase "Извините, у меня нет информации по этому вопросу." (or similar), not a hallucinated answer.
**Why human:** Grounding enforcement is a model behavior that depends on the actual model weights and the quality of the SYSTEM_PROMPT_TEMPLATE injection -- not verifiable without a live model.

#### 3. JSONL Comparability with Split Pipeline

**Test:** Run one voice question through both the split pipeline (e.g., `sensevoice + Qwen3-30B + cosyvoice`) and the Omni hybrid path, then compare the JSONL output files using the benchmark comparison script.
**Expected:** Both JSONL files have identical field schemas; the comparison script produces a valid side-by-side KPI table without errors.
**Why human:** Requires the GPU server, two benchmark runs, and visual inspection of comparison output.

---

### Gaps Summary

No gaps found. All 8 observable truths verified, all 9 required artifacts exist and are substantive and wired, all 5 key links confirmed wired, all 3 requirement IDs (OMNI-01, OMNI-02, OMNI-03) satisfied with direct code evidence.

The phase goal is fully achieved at the code level: Qwen3-Omni hybrid retrieves context via the existing RAG engine (`engine.retrieve` reused from the voice_fast path), injects context chunks into the Omni sidecar prompt (SYSTEM_PROMPT_TEMPLATE with context_block), is selectable in the frontend dropdown, and logs JSONL output with identical field schema to the split pipeline.

---

_Verified: 2026-03-26_
_Verifier: Claude (gsd-verifier)_
