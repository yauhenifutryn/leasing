---
phase: 03-brain-upgrade-and-benchmark-framework
plan: 02
subsystem: testing
tags: [benchmark, jsonl, fixture, russian, env-profiles, leasing]

requires:
  - phase: 03-brain-upgrade-and-benchmark-framework
    provides: context, research, phase decisions D-01 through D-12

provides:
  - 85-question Russian benchmark fixture covering all 5 question categories
  - 12-test validation suite for the fixture
  - 7 per-stack benchmark env profiles for controlled benchmark execution

affects:
  - 03-brain-upgrade-and-benchmark-framework-03 (benchmark runner reads fixture and loads profiles)
  - phase-05-benchmark-execution (benchmark runs use these profiles and questions)

tech-stack:
  added: []
  patterns:
    - JSONL fixture format with question_id/category/text_ru/expected_keywords schema
    - Per-stack env profile overlay pattern (.env.bench.{name}) following existing .env.voice.{name} convention
    - Category prefix encoding (sf/lf/kb/amb/oos) in question_id for fast filtering

key-files:
  created:
    - rag_demo_system/fixtures/bench_questions_ru.jsonl
    - rag_demo_system/tests/test_benchmark_fixture.py
    - rag_demo_system/.env.bench.baseline
    - rag_demo_system/.env.bench.qwen3_tts
    - rag_demo_system/.env.bench.qwen3_asr
    - rag_demo_system/.env.bench.voxtral
    - rag_demo_system/.env.bench.brain_upgrade
    - rag_demo_system/.env.bench.omni_hybrid
    - rag_demo_system/.env.bench.dify_rag
  modified: []

key-decisions:
  - "85 questions across 5 categories: 22 short_factual, 16 long_factual, 21 kb_grounded, 16 ambiguous, 10 out_of_scope — exceeds the 80+ requirement by 5"
  - "Questions phrased as spoken Russian (voice/phone interface style) not formal written queries"
  - "kb_grounded questions reference specific KB facts: Fitch B- rating, Mikro Kapital Luxembourg management, 23500+ contracts, 294M EUR portfolio, specific office addresses and phone numbers"
  - "Sidecar BASE_URLs embedded in voice provider profiles to prevent hard-fail RuntimeError from voice_adapters.py when profile is loaded"
  - "omni_hybrid profile is a clearly-marked placeholder with commented-out vars for Phase 4 to fill in"

patterns-established:
  - "Fixture pattern: JSONL one-object-per-line with question_id (prefix-NN), category, text_ru, expected_keywords list"
  - "Category prefix map: sf=short_factual, lf=long_factual, kb=kb_grounded, amb=ambiguous, oos=out_of_scope"
  - "Env profile overlay: .env.bench.{name} contains only overriding variables; runner loads base .env then overlays profile"
  - "out_of_scope questions always have empty expected_keywords list; all other categories require at least 2 keywords"

requirements-completed: [BENCH-01, DEPLOY-01]

duration: 4min
completed: 2026-03-26
---

# Phase 03 Plan 02: Benchmark Fixture and Env Profiles Summary

**85-question Russian JSONL benchmark fixture grounded in Mikro Leasing KB with 7 per-stack .env.bench.{name} overlay profiles for controlled benchmark execution**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-26T11:47:54Z
- **Completed:** 2026-03-26T11:51:28Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Created 85-question Russian fixture (`bench_questions_ru.jsonl`) covering all 5 required categories, phrased as spoken Russian voice queries, with kb_grounded questions referencing real Mikro Leasing KB facts (Fitch B- rating, company founding date, specific addresses/phones, token/bond details)
- Created 12-test validation suite (`test_benchmark_fixture.py`) that verifies structure, category completeness, prefix-category alignment, keyword requirements, Cyrillic presence, and duplicate detection — all 12 pass
- Created all 7 per-stack benchmark env profiles as incremental overlays: baseline, qwen3_tts, qwen3_asr, voxtral, brain_upgrade, omni_hybrid, dify_rag — each containing only the variables that differ from baseline

## Task Commits

1. **Task 1: Generate 80+ question Russian benchmark fixture** - `5955a6a` (feat)
2. **Task 2: Create all 7 per-stack benchmark env profiles** - `9f4341b` (feat)

## Files Created/Modified

- `rag_demo_system/fixtures/bench_questions_ru.jsonl` - 85-question JSONL benchmark fixture for Mikro Leasing voice assistant
- `rag_demo_system/tests/test_benchmark_fixture.py` - 12-test validation suite for fixture structural integrity
- `rag_demo_system/.env.bench.baseline` - Baseline: sensevoice+cosyvoice+our_rag+Qwen3-30B-A3B
- `rag_demo_system/.env.bench.qwen3_tts` - Override TTS to qwen3_tts (QWEN3_TTS_BASE_URL=50003)
- `rag_demo_system/.env.bench.qwen3_asr` - Override STT to qwen3_asr (QWEN3_ASR_BASE_URL=50004)
- `rag_demo_system/.env.bench.voxtral` - Override STT to voxtral (VOXTRAL_BASE_URL=50005)
- `rag_demo_system/.env.bench.brain_upgrade` - Override brain to Qwen3.5-35B-A3B
- `rag_demo_system/.env.bench.omni_hybrid` - Placeholder for Phase 4 Omni adapter
- `rag_demo_system/.env.bench.dify_rag` - Override backend to dify_rag

## Decisions Made

- Chose 85 questions rather than exactly 80 to provide buffer against future question removal during review
- Included 12 fixture validation tests (plan specified 10 core tests) — added `test_category_values_are_valid` and `test_question_ids_have_numeric_suffix` as additional structural guards
- Sidecar BASE_URLs in voice provider profiles: required by hard-fail guards in `voice_adapters.py` (phase 2 decision), not optional

## Deviations from Plan

None — plan executed exactly as written. The question count of 85 vs "80+" is within spec (80+ means at least 80).

## Issues Encountered

None. Fixture file and test suite were created and validated in a single pass. All 12 tests passed on the first run.

## User Setup Required

None — no external service configuration required. The env profiles are templates; sidecar URLs point to localhost ports that will be populated when running benchmark on the server.

## Next Phase Readiness

- Benchmark fixture is ready for Plan 03 (benchmark runner) to consume via `--fixture` flag
- All 7 env profiles are ready for Plan 03 runner to load via `--profile` flag
- The runner will overlay each `.env.bench.{name}` profile on top of the base `.env` using `load_dotenv(override=True)` per research D-10
- The omni_hybrid profile is a marked placeholder; Phase 4 adapter work will uncomment and fill in the OMNI vars
- No blockers for Plan 03 execution

---
*Phase: 03-brain-upgrade-and-benchmark-framework*
*Completed: 2026-03-26*
