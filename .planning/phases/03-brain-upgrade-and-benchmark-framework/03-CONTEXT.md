# Phase 3: Brain Upgrade and Benchmark Framework - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the brain model switchable on the server (vLLM model swap), extract real LLM first-token timing from streaming, build the 80+ question Russian fixture, build the benchmark runner CLI (WebSocket-based, full pipeline), build the comparison script (markdown with winner highlighting), and create all 7 per-stack env profiles. No Omni work, no server deployment, no new voice adapters.

</domain>

<decisions>
## Implementation Decisions

### Question Fixture Design
- **D-01:** Claude generates the 80+ Russian test questions by reading the knowledge base files. User reviews and edits before the fixture is finalized.
- **D-02:** Fixture format is JSONL. One JSON object per line with fields: question_id (category prefix + number, e.g. sf-01, lf-01, kb-01, amb-01, oos-01), category, text_ru, expected_keywords (list of 2-5 Russian terms the answer should contain).
- **D-03:** Five categories: short_factual, long_factual, kb_grounded, ambiguous, out_of_scope. Question IDs use prefixes sf, lf, kb, amb, oos respectively.

### Benchmark Runner Behavior
- **D-04:** Runner communicates with the backend via WebSocket, same as the browser. Full pipeline testing: STT -> RAG -> LLM -> TTS. Question text is synthesized into audio and sent through the real voice path.
- **D-05:** First 3 turns per benchmark run are flagged as warmup=true in the JSONL output. Comparison script excludes warmup turns from averages.
- **D-06:** On failure (timeout, disconnect), the runner logs the error in the JSONL line with error field and null timings, then continues to the next question. No retry, no full-stop.

### Comparison Script Output
- **D-07:** Output is a markdown table. Columns: metric name, Stack A values, Stack B values, delta. Rows: primary KPI (mean/p50/p95), LLM TTFB (mean/p50/p95), keyword hit rate, error count.
- **D-08:** Winners are highlighted per metric row (arrow or marker showing which stack is better on each dimension).

### Env Profile Structure
- **D-09:** Profiles are flat files named .env.bench.{name} in rag_demo_system/. Follows the existing .env.voice.{name} convention.
- **D-10:** Profiles are incremental overrides. Runner loads base .env first, then overlays .env.bench.{name}. Each profile contains only the variables that differ from baseline.
- **D-11:** All 7 profiles created in this phase: baseline, qwen3_tts, qwen3_asr, voxtral, brain_upgrade, omni_hybrid, dify_rag. The omni_hybrid profile is a placeholder until Phase 4 builds the adapter.

### LLM First-Token Timing
- **D-12:** Phase 3 resolves the TODO at app.py:770 by extracting the real t_llm_first_token from the streaming response using the existing iter_openai_stream_events/iter_openai_stream_text helpers in llm_stream.py.

### Claude's Discretion
- Benchmark runner CLI argument design (flags, defaults, help text)
- JSONL result schema field ordering and naming beyond the required fields
- Comparison script internal implementation (how it computes percentiles)
- How the runner synthesizes question text into audio for WebSocket submission
- Profile variable names and values (following existing env var conventions)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Voice AI Playbook
- `docs/voice_ai_playbook_2026-03-25.md` -- Authoritative planning document. "Benchmark and Comparison Plan" section defines controlled comparison methodology. "Should Do" section #1-4 covers benchmark fixtures, runner, profiles, and export format. "Phase C: Controlled Upgrades" defines benchmark execution order.

### Existing Instrumentation Code
- `rag_demo_system/backend/app.py` -- Lines ~767-828: current timing instrumentation with TODO at line 770 for real first-token extraction. Lines ~638-647: brain_model validation with allowlist. Structured JSON log emission at lines ~817-828.
- `rag_demo_system/backend/voice_session.py` -- VoiceSession dataclass with brain_model field and stack_id property.
- `rag_demo_system/backend/llm_stream.py` -- iter_openai_stream_text() and iter_openai_stream_events() for parsing vLLM streaming responses. Key for extracting real first-token timing.

### Existing Env Patterns
- `rag_demo_system/.env.example` -- Base env file template.
- `rag_demo_system/.env.voice.yandex-speechkit` -- Example of an existing env profile (voice provider override).
- `rag_demo_system/.env.voice.oss-russian.example` -- Another env profile example.

### Stack Management
- `rag_demo_system/scripts/stack_cli.py` -- VOICE_PROFILES set and program selection logic. May need extending for benchmark profiles.
- `rag_demo_system/scripts/stack.sh` -- Stack launcher script.

### Knowledge Base (for question generation)
- `experiments/yandex_realtime_voice/mikro_leasing_site_kb.jsonl` -- Knowledge base content for generating benchmark questions.
- `experiments/yandex_realtime_voice/mikro_leasing_site_unified_dedup.md` -- Deduplicated KB content.

### Test Patterns
- `rag_demo_system/tests/test_instrumentation.py` -- Existing instrumentation tests; benchmark tests should follow the same patterns.
- `rag_demo_system/tests/test_voice_session.py` -- VoiceSession test patterns.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `llm_stream.py:iter_openai_stream_events()` -- Parses vLLM streaming SSE responses into dicts. Can timestamp the first yielded event to get real llm_first_token.
- `llm_stream.py:iter_openai_stream_text()` -- Parses streaming text chunks. Alternative entry point for first-token timing.
- `voice_session.py:VoiceSession` -- brain_model field already exists with default "Qwen/Qwen3-30B-A3B". stack_id property already includes brain model slug.
- `app.py` lines 641-646 -- brain_model validation against allowlist already implemented.
- `app.py` lines 817-828 -- Structured JSON log with all timing fields already emitted per voice turn.

### Established Patterns
- **Env var convention:** `{NAME}_BASE_URL` for service URLs, `.env.voice.{name}` for profile overrides.
- **Timing convention:** `time.time()` for all timestamps (cross-process correlation).
- **Log format:** Structured JSON with session_id, backend, brain_model, stt_provider, tts_provider, stack_id, and all timing fields.
- **Test style:** pytest with mock HTTP responses via FakeResponse class.

### Integration Points
- `app.py:770` -- TODO for real first-token extraction. This is where streaming timing hooks into the existing pipeline.
- `scripts/stack.sh` -- May need a --profile flag to load .env.bench.{name} overlays.
- Frontend brain model selector -- Already exists and sends brain_model via session.update.

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond what's in the playbook and decisions above. Open to standard approaches for CLI design and JSONL schema.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 03-brain-upgrade-and-benchmark-framework*
*Context gathered: 2026-03-25*
