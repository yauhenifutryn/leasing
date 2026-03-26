---
phase: 03-brain-upgrade-and-benchmark-framework
verified: 2026-03-26T12:30:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 3: Brain Upgrade and Benchmark Framework Verification Report

**Phase Goal:** The brain model is switchable via UI selector and env var, a fixed Russian question set and benchmark runner are ready for use, and per-stack env profiles cover every configuration to be benchmarked
**Verified:** 2026-03-26T12:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Selecting Qwen3.5-35B-A3B in the UI causes the backend to pass that model string to vLLM | VERIFIED | `ChatRequest.brain_model` field at app.py:68; `effective_model = payload.brain_model or (...)` at app.py:558; brain_model passed in voice WS handler at app.py:922 |
| 2 | Voice WS handler uses streaming LLM inference and captures real t_llm_first_token via time.time() | VERIFIED | `_voice_chat_streaming_sync` at app.py:151; `first_token_time = time.time()` at app.py:284; wrapped in `asyncio.to_thread` at app.py:917 |
| 3 | t_llm_first_token in voice_turn log is no longer equal to t_retrieval_done | VERIFIED | `t_llm_first_token = first_token_time or t_retrieval_done` at app.py:289; Phase 1 TODO comment at line 770 fully removed (grep returns 0) |
| 4 | An 80+ question Russian fixture file exists covering all five categories | VERIFIED | bench_questions_ru.jsonl contains 85 questions; all 5 categories present: short_factual, long_factual, kb_grounded, ambiguous, out_of_scope |
| 5 | Each question has correct question_id prefix, category, Russian text, and expected_keywords | VERIFIED | Python structural check: 0 OOS with non-empty keywords, 0 non-OOS with fewer than 2 keywords, all prefix-category mappings correct |
| 6 | Seven env profile files exist for every benchmark configuration | VERIFIED | All 7 files confirmed: baseline, qwen3_tts, qwen3_asr, voxtral, brain_upgrade, omni_hybrid, dify_rag |
| 7 | Each profile contains only the overriding variables that differ from baseline | VERIFIED | Each file contains only delta variables; sidecar BASE_URLs present in voice provider profiles; omni_hybrid correctly marked as Phase 4 placeholder |
| 8 | Benchmark runner CLI executes questions from the fixture via WebSocket and writes JSONL results | VERIFIED | benchmark_runner.py contains argparse, websockets.connect, fixture loading, ensure_ascii=False output, warmup flagging |
| 9 | Comparison script reads two JSONL files and outputs a markdown table with latency and quality metrics | VERIFIED | benchmark_compare.py contains load_results, percentiles, compute_metrics, format_comparison_table, statistics.quantiles, __main__ guard |
| 10 | POST /api/tts endpoint exists for the runner to synthesize question text into audio | VERIFIED | TTSRequest at app.py:1031; @app.post("/api/tts") at app.py:1036; wired to synthesize_audio_with_provider at app.py:1046 |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `rag_demo_system/backend/app.py` | ChatRequest.brain_model field, effective_model resolution, streaming voice path, /api/tts endpoint | VERIFIED | All four features confirmed at lines 68, 558, 151-289, 1031-1050 |
| `rag_demo_system/tests/test_brain_routing.py` | Unit tests for brain_model routing | VERIFIED | 5 tests present: test_chat_request_brain_model_field, test_chat_request_accepts_non_default_brain_model, test_effective_model_prefers_brain_model, test_effective_model_falls_back_to_fast_model, test_effective_model_falls_back_to_base_model |
| `rag_demo_system/fixtures/bench_questions_ru.jsonl` | 80+ Russian benchmark questions across 5 categories | VERIFIED | 85 questions, all 5 categories, correct prefixes, Cyrillic text, sane keyword lists |
| `rag_demo_system/.env.bench.baseline` | Baseline benchmark env profile | VERIFIED | Contains BENCH_PROFILE=baseline, BENCH_BACKEND, BENCH_BRAIN_MODEL, BENCH_STT_PROVIDER, BENCH_TTS_PROVIDER |
| `rag_demo_system/.env.bench.brain_upgrade` | Brain upgrade profile with Qwen3.5-35B-A3B | VERIFIED | Contains BENCH_BRAIN_MODEL=Qwen/Qwen3.5-35B-A3B and RAG_LLM_FAST_MODEL=Qwen/Qwen3.5-35B-A3B |
| `rag_demo_system/.env.bench.dify_rag` | Dify RAG backend profile | VERIFIED | Contains BENCH_BACKEND=dify_rag |
| `rag_demo_system/.env.bench.qwen3_tts` | Qwen3-TTS override profile | VERIFIED | Contains BENCH_TTS_PROVIDER=qwen3_tts, QWEN3_TTS_BASE_URL=http://127.0.0.1:50003 |
| `rag_demo_system/.env.bench.qwen3_asr` | Qwen3-ASR override profile | VERIFIED | Contains BENCH_STT_PROVIDER=qwen3_asr, QWEN3_ASR_BASE_URL=http://127.0.0.1:50004 |
| `rag_demo_system/.env.bench.voxtral` | Voxtral override profile | VERIFIED | Contains BENCH_STT_PROVIDER=voxtral, VOXTRAL_BASE_URL=http://127.0.0.1:50005 |
| `rag_demo_system/.env.bench.omni_hybrid` | Phase 4 placeholder profile | VERIFIED | Contains BENCH_PROFILE=omni_hybrid, BENCH_BACKEND=our_rag, commented OMNI vars with Phase 4 note |
| `rag_demo_system/tests/test_benchmark_fixture.py` | Fixture validation tests | VERIFIED | 12 tests including test_fixture_has_80_plus_questions, test_all_categories_present |
| `rag_demo_system/scripts/benchmark_runner.py` | Async benchmark runner CLI | VERIFIED | argparse, websockets.connect, session.update-before-audio, warmup flagging, ensure_ascii=False, error recovery, /api/tts call |
| `rag_demo_system/scripts/benchmark_compare.py` | Comparison script for two JSONL result files | VERIFIED | load_results, percentiles, compute_metrics, format_comparison_table, statistics.quantiles, if __name__ == "__main__" guard |
| `rag_demo_system/tests/test_benchmark_runner.py` | Runner JSONL output format tests | VERIFIED | 6 tests including test_result_has_required_fields, test_warmup_flagging |
| `rag_demo_system/tests/test_benchmark_compare.py` | Comparison script output tests | VERIFIED | 12 tests including test_warmup_excluded; from benchmark_compare import percentiles present |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| app.py voice WS handler | ChatRequest | brain_model=session.brain_model | WIRED | app.py:922: brain_model=session.brain_model in asyncio.to_thread call |
| app.py chat() | call_openai_compatible / iter_openai_compatible_stream_events | effective_model variable | WIRED | app.py:558 effective_model resolution; used at lines 574, 681 |
| app.py _voice_chat_streaming_sync | voice_turn log | first_token_time captured via time.time() | WIRED | app.py:284: first_token_time = time.time(); returned as t_llm_first_token |
| benchmark_runner.py | ws://localhost:8787/ws/voice | websockets.connect() | WIRED | Line 209: async with websockets.connect(ws_url) as ws |
| benchmark_runner.py | bench_questions_ru.jsonl | --fixture CLI argument | WIRED | Lines 431-432: default=Path("fixtures/bench_questions_ru.jsonl") |
| benchmark_runner.py | POST /api/tts | HTTP call to synthesize audio | WIRED | Line 164: url = backend_url.rstrip("/") + "/api/tts" |
| benchmark_compare.py | JSONL result files | positional file_a/file_b arguments | WIRED | argparse positional args at line 267 |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| app.py _voice_chat_streaming_sync | first_token_time | iter_openai_compatible_stream_events() streaming loop | Yes: real tokens from vLLM; first_token_time = time.time() at first non-empty content chunk | FLOWING |
| app.py chat() | effective_model | payload.brain_model or settings fallback chain | Yes: per-request override or env-configured model string | FLOWING |
| app.py tts_endpoint | audio_b64 | synthesize_audio_with_provider() | Yes: delegated to voice provider sidecar | FLOWING |
| benchmark_runner.py | results | WebSocket voice pipeline events (response.done, transcription.completed) | Yes: live WS events; keyword_hit_rate from real answer text | FLOWING |
| benchmark_compare.py | metrics | load_results() + percentiles() + compute_metrics() | Yes: reads real JSONL files; statistics.quantiles on actual latency values | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| benchmark_compare functions importable | python3 -c "from benchmark_compare import percentiles, compute_metrics, format_comparison_table, load_results; r = percentiles([100.0, 200.0, 300.0])" | {'mean': 200.0, 'p50': 200.0, 'p95': 290.0} | PASS |
| benchmark_runner helper functions work | python3 -c "from benchmark_runner import is_warmup, build_result_dict, compute_keyword_hits; print(is_warmup(0,3), is_warmup(3,3))" | True False | PASS |
| benchmark_runner --help shows usage | python3 benchmark_runner.py --help | Displays --fixture, --profile, --output, --ws-url flags | PASS |
| benchmark_compare --help shows usage | python3 benchmark_compare.py --help | Displays file_a, file_b positional args and --output flag | PASS |
| All phase 3 tests pass | pytest test_brain_routing.py test_benchmark_fixture.py test_benchmark_runner.py test_benchmark_compare.py | 35 passed in 0.52s | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| BRAIN-01 | 03-01 | Brain model switchable between Qwen3-30B-A3B and Qwen3.5-35B-A3B via UI selector or env var | SATISFIED | ChatRequest.brain_model field in app.py; effective_model resolution; voice WS handler passes session.brain_model; UI selector wires via session.update brain_model event |
| BENCH-01 | 03-02 | Fixed Russian test question set with 80+ questions across 5 categories | SATISFIED | bench_questions_ru.jsonl: 85 questions, all 5 categories present, prefix-aligned IDs, Cyrillic text, structured keywords |
| BENCH-02 | 03-03 | Benchmark runner executes the full question set against the active configuration and writes JSONL results | SATISFIED | benchmark_runner.py: async CLI, WebSocket pipeline, JSONL output per question, error recovery, summary statistics |
| BENCH-03 | 03-03 | Each result includes question_id, stack_id, transcript, answer, retrieved chunks, timing breakdown | SATISFIED | REQUIRED_RESULT_FIELDS in test_benchmark_runner.py lists all 17 fields; build_result_dict populates all of them |
| BENCH-04 | 03-03 | Comparison script shows side-by-side latency and quality metrics across stacks | SATISFIED | benchmark_compare.py: format_comparison_table produces 9-row markdown table with Primary KPI, LLM TTFB, keyword_hit_rate, error_count, delta, winner |
| DEPLOY-01 | 03-02 | Env profile files for each benchmark stack | SATISFIED | All 7 .env.bench.{name} files exist with correct overrides; sidecar BASE_URLs embedded in voice provider profiles |

No orphaned requirements: REQUIREMENTS.md maps exactly BRAIN-01, BENCH-01, BENCH-02, BENCH-03, BENCH-04, DEPLOY-01 to Phase 3 — all claimed by plans and all verified.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| .env.bench.omni_hybrid | All | Placeholder profile with commented vars | Info | Intentional and documented. Phase 4 will fill in OMNI_MODE and OMNI_BASE_URL. BENCH_PROFILE and BENCH_BACKEND are set and functional. Not a blocker. |

No blocking or warning anti-patterns found. The omni_hybrid placeholder is documented behavior aligned with the plan spec ("Placeholder for Phase 4 Qwen3-Omni hybrid adapter").

One minor discrepancy noted: Plan 03-03 frontmatter lists `exports: ["test_comparison_table_has_expected_metrics", "test_warmup_excluded"]` for test_benchmark_compare.py, but the actual function is named `test_comparison_table_has_primary_kpi_rows`. The test exists and covers the same behavior; this is a naming mismatch in the plan spec only, not a missing or broken test.

---

### Human Verification Required

None. All critical behaviors are verified programmatically:
- Model routing: grep-verifiable pattern in app.py
- Streaming timing: code inspection confirms time.time() call at first content token
- Fixture structure: validated by running Python assertions on the file directly
- Env profiles: file inspection confirms all required variables
- Scripts: --help output and direct function invocation confirmed

---

### Gaps Summary

No gaps. All 10 observable truths verified. All 15 required artifacts exist, are substantive, and are wired. All 6 required requirements satisfied. 35 tests pass. Behavioral spot-checks pass.

---

_Verified: 2026-03-26T12:30:00Z_
_Verifier: Claude (gsd-verifier)_
