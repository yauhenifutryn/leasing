---
phase: 03-brain-upgrade-and-benchmark-framework
plan: 03
subsystem: testing
tags: [benchmark, runner, comparison, jsonl, websocket, tts, fastapi, statistics]

# Dependency graph
requires:
  - phase: 03-brain-upgrade-and-benchmark-framework
    plan: 01
    provides: brain model routing, t_llm_first_token in voice_turn log
  - phase: 03-brain-upgrade-and-benchmark-framework
    plan: 02
    provides: bench_questions_ru.jsonl fixture, .env.bench.* profiles

provides:
  - POST /api/tts endpoint for synthesizing question text to base64 audio
  - benchmark_runner.py async CLI that executes fixture questions via WebSocket and writes JSONL results
  - benchmark_compare.py script that reads two JSONL files and outputs a markdown comparison table
  - 6 unit tests for runner output format, warmup flagging, error handling, keyword hit computation
  - 12 unit tests for comparison script: percentiles, warmup exclusion, error row handling, table structure

affects:
  - phase-05-benchmark-execution (runner and compare scripts are the execution toolchain)

# Tech tracking
tech-stack:
  added:
    - websockets (async WebSocket client library used in benchmark_runner.py)
  patterns:
    - Fresh WebSocket connection per question to avoid session-state contamination (Pitfall 2)
    - POST /api/tts REST endpoint as audio synthesis proxy for benchmark runner (Pitfall 3)
    - json.dumps(ensure_ascii=False) for all JSONL output to preserve Cyrillic text (Pitfall 4)
    - session.update sent first, session.updated awaited before audio (Pitfall 5)
    - statistics.quantiles(n=100, method='inclusive') for p50/p95 benchmark KPIs
    - All helper functions importable without side effects (guard CLI with __main__)

key-files:
  created:
    - rag_demo_system/scripts/benchmark_runner.py
    - rag_demo_system/scripts/benchmark_compare.py
    - rag_demo_system/tests/test_benchmark_runner.py
    - rag_demo_system/tests/test_benchmark_compare.py
  modified:
    - rag_demo_system/backend/app.py

key-decisions:
  - "POST /api/tts as REST proxy for benchmark runner: runner must not need direct sidecar access; the backend routes TTS provider selection via synthesize_audio_with_provider"
  - "Fresh WS connection per question: avoids session state leakage across questions (different brain_model or provider combos would contaminate results)"
  - "statistics.quantiles(n=100, method='inclusive') for p50/p95: matches the percentile semantics documented in research D-07; actual p50 of [1..100] is 50.5 not 50.0 with this method"
  - "Keyword matching is case-insensitive substring match: simple, deterministic, no external NLP dependencies; limitation is that inflected Russian forms may not match base form keywords in fixture"

requirements-completed: [BENCH-02, BENCH-03, BENCH-04]

# Metrics
duration: 5min
completed: 2026-03-26
---

# Phase 3 Plan 3: Benchmark Runner CLI, TTS Endpoint, and Comparison Script Summary

**Async benchmark runner CLI executing JSONL fixture via voice WebSocket with warmup flagging and error handling, POST /api/tts REST endpoint, and comparison script producing side-by-side markdown metrics tables**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-26T11:55:43Z
- **Completed:** 2026-03-26T12:00:41Z
- **Tasks:** 2
- **Files modified/created:** 5

## Accomplishments

- Added `POST /api/tts` endpoint to `app.py` (TTSRequest pydantic model, delegates to `synthesize_audio_with_provider`); the runner calls this to convert question text to audio without needing direct sidecar access
- Created `benchmark_runner.py`: async CLI with argparse, `websockets.connect`, fresh connection per question, session.update first + session.updated await before audio (Pitfall 5), warmup flagging for first N questions, error recovery (logs error, continues), `ensure_ascii=False` JSONL output, summary statistics at end
- Created `benchmark_compare.py`: `load_results` (excludes warmup rows), `percentiles` using `statistics.quantiles(n=100, method='inclusive')`, `compute_metrics` (error rows excluded from timing, counted in error_count), `format_comparison_table` with 9 metric rows, delta (B-A), and winner arrows
- 6 unit tests for runner helpers (all pass): required fields, warmup flagging, error result structure, keyword hit rate, empty-keywords null rate, ensure_ascii serialization
- 12 unit tests for comparison script (all pass): percentile correctness, empty-list handling, warmup exclusion via load_results, error row handling, table structure, delta computation, winner arrows

## Task Commits

1. **Task 1: Add POST /api/tts endpoint, benchmark runner CLI, runner tests** - `40d9c3a` (feat)
2. **Task 2: Build benchmark comparison script and comparison tests** - `4cf146c` (feat)

## Files Created/Modified

- `rag_demo_system/backend/app.py` - Added TTSRequest pydantic model and POST /api/tts endpoint
- `rag_demo_system/scripts/benchmark_runner.py` - Async benchmark runner CLI with WebSocket pipeline
- `rag_demo_system/scripts/benchmark_compare.py` - JSONL comparison script with markdown table output
- `rag_demo_system/tests/test_benchmark_runner.py` - 6 unit tests for runner helper functions
- `rag_demo_system/tests/test_benchmark_compare.py` - 12 unit tests for comparison script functions

## Decisions Made

- `POST /api/tts` as REST proxy for benchmark runner: the runner avoids direct sidecar coupling; provider routing stays in the backend where `synthesize_audio_with_provider` already handles the dispatch logic
- Fresh WebSocket connection per question: prevents session brain_model/provider state from one question affecting the next (Pitfall 2 from research)
- `statistics.quantiles(n=100, method='inclusive')` for p50/p95: this returns 99 cut points; p50 uses index 49, p95 uses index 94. The actual p50 for uniform data [1..100] is 50.5 (not 50.0) with inclusive interpolation
- Keyword matching uses case-insensitive substring (not stemming/lemmatisation): deterministic, zero dependencies, documented limitation for Russian inflected forms

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect test expectation for keyword hit rate with inflected Russian**

- **Found during:** Task 1 TDD GREEN phase
- **Issue:** Test `test_keyword_hit_rate_computed_correctly` used "стоимость" as a keyword against answer text containing "стоимости" (genitive case). Python's `in` substring operator correctly reports `False` because "стоимость" is not literally in "стоимости"
- **Fix:** Changed test to use keywords that appear verbatim in the answer: "составляет" instead of "стоимость"
- **Files modified:** `rag_demo_system/tests/test_benchmark_runner.py`
- **Committed in:** `40d9c3a` (Task 1)

**2. [Rule 1 - Bug] Fixed incorrect test expectation for percentiles with statistics.quantiles**

- **Found during:** Task 2 TDD GREEN phase
- **Issue:** Test asserted p50=50.0 and p95=95.0 for [1..100], but `statistics.quantiles(n=100, method='inclusive')` returns p50=50.5 and p95=95.05 for this input (interpolated values, not element values)
- **Fix:** Updated test expectations to match actual library output (50.5, 95.05), verified empirically
- **Files modified:** `rag_demo_system/tests/test_benchmark_compare.py`
- **Committed in:** `4cf146c` (Task 2)

## Issues Encountered

- The full pytest suite (`tests/`) fails due to `ModuleNotFoundError: No module named 'rank_bm25'` — this is a pre-existing constraint (rag stack not installed in dev environment, documented in Phase 1 SUMMARY). The 18 new benchmark tests and 33 other tests that do not require rank_bm25 all pass (51 total).

## User Setup Required

- The runner requires `websockets` Python package: `pip install websockets`
- The runner also requires `python-dotenv` for env profile loading: `pip install python-dotenv`
- These should be added to `requirements.txt` or the venv before running on the server

## Known Stubs

None. Both scripts are fully functional:
- `benchmark_runner.py` is ready to execute against a live backend at the configured ws-url
- `benchmark_compare.py` is ready to process real JSONL result files
- The `POST /api/tts` endpoint is fully wired to `synthesize_audio_with_provider`

## Next Phase Readiness

- Phase 3 is complete: brain routing (01), fixture + profiles (02), runner + comparison (03) are all done
- Phase 5 (benchmark execution) can proceed: runner accepts `--fixture fixtures/bench_questions_ru.jsonl --profile baseline` and the comparison script accepts the two output JSONL files
- Dependency packages (websockets, python-dotenv) must be installed in the server venv before Phase 5

---
*Phase: 03-brain-upgrade-and-benchmark-framework*
*Completed: 2026-03-26*

## Self-Check: PASSED

- FOUND: rag_demo_system/backend/app.py
- FOUND: rag_demo_system/scripts/benchmark_runner.py
- FOUND: rag_demo_system/scripts/benchmark_compare.py
- FOUND: rag_demo_system/tests/test_benchmark_runner.py
- FOUND: rag_demo_system/tests/test_benchmark_compare.py
- FOUND: .planning/phases/03-brain-upgrade-and-benchmark-framework/03-03-SUMMARY.md
- Commit 40d9c3a: FOUND (feat(03-03): add TTS endpoint, benchmark runner CLI, and runner tests)
- Commit 4cf146c: FOUND (feat(03-03): add benchmark comparison script and comparison tests)
