---
phase: 04-qwen3-omni-hybrid
plan: 01
subsystem: api
tags: [qwen3-omni, fastapi, sidecar, transformers, pytest, voice-ai, russian]

# Dependency graph
requires:
  - phase: 02-voice-provider-adapters
    provides: sidecar server pattern (qwen3_tts_server.py, qwen3_asr_server.py) that this follows
  - phase: 03-brain-upgrade-and-benchmark-framework
    provides: JSONL instrumentation contract and benchmark stack_id format

provides:
  - Qwen3-Omni FastAPI sidecar server with /health and /chat endpoints
  - Isolated requirements file pinning transformers==4.57.3 for Omni venv
  - 7 contract tests documenting Omni adapter integration points (2 xfail for Plan 02)

affects:
  - 04-02 (wires sidecar into backend voice_adapters and app.py)
  - 05-benchmark-execution (benchmarks Omni path vs split pipeline)

# Tech tracking
tech-stack:
  added:
    - qwen-omni-utils>=0.0.9 (audio preprocessing for Qwen3-Omni)
    - transformers==4.57.3 (pinned, isolated in venvs/omni/ only)
  patterns:
    - Deferred import pattern for GPU-heavy model classes (keeps import errors contained in _build_default_app)
    - try/finally temp file cleanup for base64-decoded audio (Pitfall 6)
    - xfail contract tests that document Plan 02 requirements before implementation

key-files:
  created:
    - rag_demo_system/services/qwen3_omni_server.py
    - rag_demo_system/requirements-qwen3-omni.txt
    - rag_demo_system/tests/test_qwen3_omni_adapter.py
  modified: []

key-decisions:
  - "Deferred imports of Qwen3OmniMoeForConditionalGeneration/Qwen3OmniMoeProcessor keep test-time import safe without GPU or transformers==4.57.3 installed"
  - "SYSTEM_PROMPT_TEMPLATE uses {context_block} placeholder so custom system_prompt overrides grounding text cleanly without modifying context_chunks joining logic"
  - "xfail markers on normalizer and voice_statuses tests document Plan 02 contracts without blocking CI green state"

patterns-established:
  - "Testable guard helpers (_require_omni_base_url, _call_omni_sidecar) defined inline in test file as executable contracts, then extracted to app.py in Plan 02"
  - "Omni first-audio timestamp set immediately after generate() returns -- llm_first_token == tts_first_chunk per D-06 collapsed field convention"

requirements-completed: [OMNI-01, OMNI-03]

# Metrics
duration: 3min
completed: 2026-03-26
---

# Phase 4 Plan 1: Qwen3-Omni Sidecar Server and Contract Tests Summary

**Standalone FastAPI sidecar for Qwen3-Omni-30B-A3B-Instruct with audio-in/audio-out pipeline and 7 contract tests covering dispatch, hard-fail, stack_id format, JSONL field contract, and RAG chunk injection**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-26T13:03:03Z
- **Completed:** 2026-03-26T13:05:27Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `qwen3_omni_server.py`: FastAPI sidecar following Phase 2 sidecar pattern; ChatRequest/ChatResponse Pydantic models; Qwen3OmniInference class with deferred transformers imports; try/finally temp WAV cleanup; Russian grounding system prompt; /health and /chat endpoints
- `requirements-qwen3-omni.txt`: Pins transformers==4.57.3 with isolation warning comment; includes qwen-omni-utils, accelerate, soundfile, numpy
- `test_qwen3_omni_adapter.py`: 7 contract tests; 5 pass immediately; 2 xfail documenting Plan 02 allowlist and voice_statuses integration points

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Qwen3-Omni sidecar server and requirements file** - `c067085` (feat)
2. **Task 2: Create Omni adapter contract tests** - `cd05650` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `rag_demo_system/services/qwen3_omni_server.py` - Standalone FastAPI sidecar for Qwen3-Omni inference (audio-in, audio+text-out)
- `rag_demo_system/requirements-qwen3-omni.txt` - Isolated pip requirements for the Omni venv; pins transformers==4.57.3
- `rag_demo_system/tests/test_qwen3_omni_adapter.py` - 7 contract tests for Omni adapter integration

## Decisions Made

- Deferred imports of `Qwen3OmniMoeForConditionalGeneration` and `Qwen3OmniMoeProcessor` inside `Qwen3OmniInference.__init__` so that the module is importable in the shared test environment without GPU or transformers==4.57.3 present.
- `SYSTEM_PROMPT_TEMPLATE` uses `{context_block}` placeholder; custom `system_prompt` overrides the entire grounding block (simpler than merging two texts).
- `_require_omni_base_url` and `_call_omni_sidecar` defined as inline helpers in the test file to make hard-fail and dispatch logic testable before Plan 02 extracts them into `app.py`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. The sidecar requires `venvs/omni/` setup with `requirements-qwen3-omni.txt` on the GPU server, but that is Phase 5 (server deployment) work.

## Next Phase Readiness

- Plan 02 (04-02) can now wire `qwen3_omni` into `voice_adapters.build_voice_statuses()`, `yandex_realtime.normalize_voice_provider` allowlist, and the `app.py` WebSocket handler dispatch path.
- The two xfail tests will flip to PASSED once Plan 02 implements those integration points.
- No blockers for Plan 02.

---
*Phase: 04-qwen3-omni-hybrid*
*Completed: 2026-03-26*

## Self-Check: PASSED

- `rag_demo_system/services/qwen3_omni_server.py` exists: FOUND
- `rag_demo_system/requirements-qwen3-omni.txt` exists: FOUND
- `rag_demo_system/tests/test_qwen3_omni_adapter.py` exists: FOUND
- Commit `c067085` exists: FOUND
- Commit `cd05650` exists: FOUND
- pytest: 5 passed, 2 xfailed
