---
phase: 04-qwen3-omni-hybrid
plan: 02
subsystem: backend/frontend/config
tags: [qwen3-omni, websocket, voice-dispatch, rag, supervisord, contract-tests]
dependency_graph:
  requires: [04-01]
  provides: [qwen3_omni_voice_path, omni_health_status, omni_env_profile, omni_supervisord]
  affects: [backend/app.py, backend/voice_adapters.py, backend/yandex_realtime.py, frontend/index.html, scripts/supervisord.conf, .env.bench.omni_hybrid]
tech_stack:
  added: []
  patterns: [omni-dispatch-branch, collapsed-timing-d06, hard-fail-env-guard, xfail-removal]
key_files:
  created: []
  modified:
    - rag_demo_system/backend/app.py
    - rag_demo_system/backend/voice_adapters.py
    - rag_demo_system/backend/yandex_realtime.py
    - rag_demo_system/frontend/index.html
    - rag_demo_system/.env.bench.omni_hybrid
    - rag_demo_system/scripts/supervisord.conf
    - rag_demo_system/tests/test_qwen3_omni_adapter.py
    - rag_demo_system/tests/test_frontend_config_contract.py
decisions:
  - "import requests aliased as _requests in app.py to avoid shadowing the local audio_b64 variable that is reused in the split pipeline path"
  - "Omni dispatch uses continue to skip the split pipeline voice_result/TTS path after logging JSONL"
  - "autostart=false in supervisord qwen3_omni entry: Omni and split pipeline brain model cannot co-host on A100 80GB"
metrics:
  duration: "~20 min"
  completed: "2026-03-26"
  tasks_completed: 2
  files_modified: 8
---

# Phase 04 Plan 02: Omni Hybrid Integration (Wire + Connect) Summary

**One-liner:** Qwen3-Omni wired into the backend WebSocket handler via audio-in/audio-out hybrid path with RAG retrieval injection, normalized provider allowlist, health status entry, frontend dropdown, env profile, supervisord entry, and all 7 contract tests passing without xfail markers.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Wire Omni dispatch into backend, normalizer, and health status | bd4b162 | app.py, voice_adapters.py, yandex_realtime.py |
| 2 | Frontend dropdown, env profile, supervisord entry, and test xfail removal | 70b9f22 | index.html, .env.bench.omni_hybrid, supervisord.conf, test_qwen3_omni_adapter.py, test_frontend_config_contract.py |

## What Was Built

### Task 1: Backend Integration

**`yandex_realtime.py`:** Added `"qwen3_omni"` to the `normalize_voice_provider` allowlist set. Prevents silent fallback to `"local"` when the frontend sends `voice_provider="qwen3_omni"` (per D-13, Pitfall 4).

**`voice_adapters.py`:** Added `"qwen3_omni": _service_status("qwen3_omni", os.getenv("QWEN3_OMNI_BASE_URL"))` to `build_voice_statuses()`. The Omni sidecar now appears in the health status dashboard with the same pattern as all other providers.

**`app.py` session.update override:** When `session.voice_provider == "qwen3_omni"`, the handler sets:
- `session.brain_model = "Qwen/Qwen3-Omni-30B-A3B"`
- `session.stt_provider = "omni"`, `session.tts_provider = "omni"`

This produces the correct `stack_id` `our_rag__Qwen3-Omni-30B-A3B__omni__omni` for JSONL log identification.

**`app.py` Omni dispatch branch:** In `input_audio_buffer.commit`, after STT completes (STT runs for RAG query extraction per D-01), a `if session.voice_provider == "qwen3_omni":` branch:
1. Hard-fails if `QWEN3_OMNI_BASE_URL` is not set (RuntimeError)
2. Calls `engine.retrieve(text, fast=True, voice_fast=True, ...)` for RAG chunk retrieval (reuses existing voice settings per D-04)
3. POSTs `{"audio_b64": audio_b64, "context_chunks": chunk_texts}` to the Omni sidecar `/chat` (original user audio, not transcript, per D-02)
4. Collapses `t_llm_first_token = t_tts_first_chunk = t_omni_first_audio` (D-06: Omni generates audio natively, no separate TTS step)
5. Sends `response.output_text.delta` and `response.output_audio.delta` to the browser
6. Logs JSONL with all 6 timing fields: `speech_stopped`, `stt_done`, `retrieval_done`, `llm_first_token`, `tts_first_chunk`, `playback_started` (OMNI-03)
7. Sends `response.done` with RAG `used_knowledge`
8. Issues `continue` to skip the split pipeline path (voice_result, TTS, existing log)

Added `import requests as _requests` to app.py (was absent; `requests` is used via `voice_adapters` indirectly but not available at module scope in app.py).

### Task 2: Frontend, Config, and Tests

**`index.html`:** Added `<option value="qwen3_omni">qwen3_omni</option>` to the `voiceProviderSelect` dropdown. Users can now select Omni from the voice provider picker.

**`.env.bench.omni_hybrid`:** Replaced placeholder content with real values: `VOICE_PROVIDER=qwen3_omni`, `QWEN3_OMNI_BASE_URL=http://127.0.0.1:8002`, `SENSEVOICE_BASE_URL=http://127.0.0.1:8001` (STT needed for RAG query extraction), and comment explaining the supervisorctl stop/start model swap.

**`supervisord.conf`:** Added `[program:qwen3_omni]` with `autostart=false` and `STACK_QWEN3_OMNI_CMD` environment variable pattern. The sidecar is started explicitly via `supervisorctl start qwen3_omni` after stopping the split pipeline brain model.

**`test_qwen3_omni_adapter.py`:** Removed `@pytest.mark.xfail(...)` from `test_omni_voice_provider_in_normalizer_allowlist` and `test_build_voice_statuses_includes_qwen3_omni`. Both tests now pass.

**`test_frontend_config_contract.py`:** Added `assert "qwen3_omni" in index_html` to pin the contract that the frontend must include the Omni option.

## Verification Results

```
python3 -m pytest rag_demo_system/tests/test_qwen3_omni_adapter.py -v
  7 passed in 0.11s  (0 xfail)

python3 -m pytest rag_demo_system/tests/test_frontend_config_contract.py -v
  3 passed in 0.02s

Full suite (excluding rank_bm25-dependent tests that fail pre-existing on dev machine):
  121 passed, 1 skipped in 0.44s
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing import] Added `import requests as _requests` to app.py**
- **Found during:** Task 1, Part D
- **Issue:** `app.py` had no `import requests` despite the plan's Omni dispatch branch using `requests.post()`. The plan noted to "verify this, and if not, add `import requests` at the top" but the existing code didn't have it.
- **Fix:** Added `import requests as _requests` (aliased to avoid name collision with `audio_b64` reuse in the same scope and to match the `_requests` naming used in the test file)
- **Files modified:** `rag_demo_system/backend/app.py`
- **Commit:** bd4b162

## Known Stubs

None. All integration points are wired with real values:
- `QWEN3_OMNI_BASE_URL` is set to a concrete URL in `.env.bench.omni_hybrid` (not a placeholder)
- The dispatch branch hard-fails if the env var is absent (no silent stub behavior)
- The frontend dropdown option is a real selectable value

## Self-Check: PASSED

Files exist:
- FOUND: rag_demo_system/backend/app.py (modified)
- FOUND: rag_demo_system/backend/yandex_realtime.py (modified)
- FOUND: rag_demo_system/backend/voice_adapters.py (modified)
- FOUND: rag_demo_system/frontend/index.html (modified)
- FOUND: rag_demo_system/.env.bench.omni_hybrid (modified)
- FOUND: rag_demo_system/scripts/supervisord.conf (modified)
- FOUND: rag_demo_system/tests/test_qwen3_omni_adapter.py (modified)
- FOUND: rag_demo_system/tests/test_frontend_config_contract.py (modified)

Commits exist:
- FOUND: bd4b162 (Task 1)
- FOUND: 70b9f22 (Task 2)
