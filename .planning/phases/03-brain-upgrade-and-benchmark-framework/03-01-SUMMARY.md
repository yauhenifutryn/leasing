---
phase: 03-brain-upgrade-and-benchmark-framework
plan: 01
subsystem: api
tags: [vllm, fastapi, pydantic, streaming, llm-routing, voice-pipeline, websocket]

# Dependency graph
requires:
  - phase: 01-instrumentation-and-ui-switching
    provides: VoiceSession.brain_model field, voice turn log structure, t_llm_first_token approximation TODO
provides:
  - ChatRequest.brain_model field enabling per-request LLM model override
  - effective_model resolution in chat() that prefers brain_model over env default
  - _voice_chat_streaming_sync() helper for streaming LLM inference in voice WebSocket handler
  - Real t_llm_first_token from streaming first content chunk via time.time()
  - 5 unit tests in test_brain_routing.py covering ChatRequest field and model resolution logic
affects:
  - 03-02 (benchmark logging — relies on accurate t_llm_first_token in voice_turn logs)
  - 03-03 (env profiles — brain model routing is prerequisite for multi-model benchmarking)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Per-request brain_model override via ChatRequest.brain_model field (not global settings mutation)
    - Synchronous streaming helper wrapped in asyncio.to_thread() to avoid blocking the event loop
    - time.time() for voice path epoch timestamps, NOT perf_counter (cross-process correlation requirement)
    - dify_rag backend falls back to non-streaming path; our_rag uses streaming for real first-token timing

key-files:
  created:
    - rag_demo_system/tests/test_brain_routing.py
  modified:
    - rag_demo_system/backend/app.py

key-decisions:
  - "Per-request brain_model override via ChatRequest field, not settings mutation: avoids race conditions on the module-level singleton"
  - "_voice_chat_streaming_sync is synchronous (requests-based iter_openai_compatible_stream_events); wrapped in asyncio.to_thread() to unblock the event loop during voice turns"
  - "dify_rag backend falls back to non-streaming with t_llm_first_token = t_retrieval_done — Dify manages its own API, streaming extraction not possible"
  - "first_token_time = time.time() inside streaming loop, not perf_counter — epoch timestamps required for cross-process log correlation in benchmarks"

patterns-established:
  - "ChatRequest.brain_model: str | None = None — None means use env default, non-None overrides both fast_model and base model"
  - "effective_model = payload.brain_model or (fast_model if fast and fast_model else base_model) — priority order: explicit > fast > default"

requirements-completed: [BRAIN-01]

# Metrics
duration: 4min
completed: 2026-03-26
---

# Phase 3 Plan 1: Brain Model Routing and Streaming Voice Timing Summary

**ChatRequest.brain_model field wired end-to-end from UI selector to vLLM, plus _voice_chat_streaming_sync replacing the non-streaming voice path to capture real t_llm_first_token via streaming LLM inference**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-26T11:47:52Z
- **Completed:** 2026-03-26T11:51:52Z
- **Tasks:** 2
- **Files modified:** 2 (app.py modified, test_brain_routing.py created)

## Accomplishments

- Closed the BRAIN-01 routing gap: `ChatRequest.brain_model` field added and wired through `effective_model` resolution in `chat()`, so selecting Qwen3.5-35B-A3B in the UI now passes that model string to the vLLM endpoint instead of silently ignoring it
- Voice WebSocket handler now passes `session.brain_model` to ChatRequest, completing the session -> request routing chain
- Replaced non-streaming voice chat with `_voice_chat_streaming_sync()` helper that iterates `iter_openai_compatible_stream_events()` and captures `first_token_time = time.time()` at the first content chunk, resolving the Phase 1 TODO at line 770
- `t_llm_first_token` in `voice_turn` log events is now a real first-token timestamp, not equal to `t_retrieval_done`
- Added 5 unit tests in `test_brain_routing.py` covering ChatRequest backward compatibility, non-default model acceptance, and all three effective_model resolution branches

## Task Commits

Each task was committed atomically:

1. **Task 1: Add brain_model to ChatRequest and fix model resolution in chat()** - `f318cbc` (feat)
2. **Task 2: Replace non-streaming voice chat with streaming path for real t_llm_first_token** - `6e8edd7` (feat)

**Plan metadata:** (docs commit follows)

_Note: Task 1 followed TDD flow: RED (test_chat_request_brain_model_field failed with AttributeError), GREEN (added field + resolution logic, all 17 tests pass)_

## Files Created/Modified

- `rag_demo_system/backend/app.py` - Added brain_model to ChatRequest, effective_model resolution in chat(), _voice_chat_streaming_sync() helper, updated voice WS handler
- `rag_demo_system/tests/test_brain_routing.py` - 5 unit tests: ChatRequest field presence, brain_model acceptance, effective_model resolution for 3 scenarios

## Decisions Made

- Per-request `brain_model` override via ChatRequest field, not settings mutation: the `settings` object is a module-level singleton; mutating `settings.llm.fast_model` per-request is a race condition risk
- `_voice_chat_streaming_sync` is synchronous (it calls `iter_openai_compatible_stream_events` which is a blocking `requests.post` with `stream=True`); wrapped in `asyncio.to_thread()` to avoid blocking the FastAPI event loop during voice turns
- `dify_rag` backend falls back to non-streaming path with `t_llm_first_token = t_retrieval_done` — Dify manages its own streaming API and does not expose a first-token hook
- `time.time()` used for `first_token_time` inside streaming loop, consistent with the existing voice path epoch timestamps (established in Phase 1 decision log)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Adapted test import strategy to handle missing infrastructure dependencies**

- **Found during:** Task 1 TDD RED phase
- **Issue:** `_load_app_module()` via `importlib.import_module("backend.app")` immediately failed with `ModuleNotFoundError: No module named 'rank_bm25'` because the app module imports RAGEngine which depends on rank_bm25, qdrant_client, sentence_transformers, etc. — none installed in test environment
- **Fix:** Added `unittest.mock.patch.dict("sys.modules", ...)` with MagicMock stubs for all heavy deps before importing backend.app. This allows ChatRequest (pure pydantic model) to be instantiated without a running Qdrant/BM25 stack
- **Files modified:** `rag_demo_system/tests/test_brain_routing.py`
- **Verification:** Tests 1 and 2 (ChatRequest field tests) now correctly import the module; the RED->GREEN progression confirmed the brain_model field was genuinely missing before the fix
- **Committed in:** `f318cbc` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking import failure in test infrastructure)
**Impact on plan:** Fix was necessary for the TDD flow to work. The mock approach is consistent with how test_voice_session.py avoids loading the full app. No scope creep.

## Issues Encountered

- The plan cited `iter_openai_compatible_stream_events` as the function name but the actual function in `llm.py` is exactly that name — no mismatch. The plan's pseudocode was accurate.
- The existing streaming chat path in `gen()` uses misaligned indentation for the `iter_openai_compatible_stream_events` call (base_url arg at wrong indent level) — left as-is to avoid unrelated whitespace changes.

## Known Stubs

None. The `_voice_chat_streaming_sync` function is fully wired: retrieval -> prompt building -> streaming LLM -> first_token capture -> return dict with `t_llm_first_token`. The `dify_rag` fallback path uses `t_retrieval_done` as the approximation, which is documented behavior, not a stub.

## User Setup Required

None — no external service configuration required beyond what was already needed (vLLM endpoint configured via `RAG_LLM_BASE_URL`/`RAG_LLM_MODEL`).

## Next Phase Readiness

- Brain model routing is complete: UI selector -> session.brain_model -> ChatRequest.brain_model -> effective_model -> vLLM endpoint
- Voice turn logs now have accurate `t_llm_first_token` for benchmarking (not equal to `t_retrieval_done`)
- Ready for 03-02 (benchmark logging) which depends on correct voice_turn log field values
- Blocker noted: `t_llm_first_token` accuracy requires a real vLLM server; on dev machine without vLLM, the streaming call will fail and fall back to `t_retrieval_done`

---
*Phase: 03-brain-upgrade-and-benchmark-framework*
*Completed: 2026-03-26*

## Self-Check: PASSED

- FOUND: rag_demo_system/backend/app.py
- FOUND: rag_demo_system/tests/test_brain_routing.py
- FOUND: .planning/phases/03-brain-upgrade-and-benchmark-framework/03-01-SUMMARY.md
- Commit f318cbc: FOUND (feat(03-01): add brain_model to ChatRequest)
- Commit 6e8edd7: FOUND (feat(03-01): add streaming voice chat helper)
- Commit 77d8826: FOUND (docs(03-01): complete brain model routing plan)
