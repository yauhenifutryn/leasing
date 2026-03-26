# Phase 4: Qwen3-Omni Hybrid - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the Qwen3-Omni hybrid adapter with RAG context injection, register it as a UI provider option, and confirm it uses the same JSONL log format as the split pipeline. No pure realtime mode, no tool-mediated retrieval, no server deployment, no benchmark execution.

</domain>

<decisions>
## Implementation Decisions

### Audio Pipeline Path
- **D-01:** Audio-in, audio-out. Omni receives raw user audio + text context chunks. STT runs in parallel only to produce a text query for RAG search. Omni generates audio response directly (no separate TTS step).
- **D-02:** The STT transcript is NOT sent to Omni as input. Omni hears the original audio natively. The transcript is used solely as the RAG retrieval query.

### RAG Injection Strategy
- **D-03:** Strict grounding. Prompt instructs Omni to answer ONLY from provided context chunks. If the context does not cover the topic, Omni must refuse or say it cannot help. This satisfies OMNI-02 (out-of-scope refusal).
- **D-04:** Same retrieval settings as the split pipeline voice_fast profile: vector_top_k=3, bm25_top_k=1, final_top_n=2, reranker disabled. This ensures fair comparison: same chunks, different brain.
- **D-05:** All prompts and grounding instructions written in Russian. Chunks are already Russian. Audio input is Russian. Keeping everything monolingual avoids the code-switching risk flagged in the playbook.

### Timing Instrumentation
- **D-06:** Emit all 6 standard JSONL timing fields to keep output compatible with the Phase 3 comparison script (OMNI-03). Mapping:
  - `speech_stopped`: real timestamp (user finished speaking)
  - `stt_done`: real timestamp (STT finished, used for RAG query)
  - `retrieval_done`: real timestamp (RAG chunks returned)
  - `llm_first_token`: set to Omni first audio timestamp (collapsed with tts_first_chunk)
  - `tts_first_chunk`: set to Omni first audio timestamp (same as llm_first_token)
  - `playback_started`: real timestamp (audio sent to browser)
- **D-07:** Primary KPI (playback_started - speech_stopped) remains directly comparable with split pipeline results.

### Serving Infrastructure
- **D-08:** Standalone FastAPI sidecar with its own Python venv. Loads Qwen3-Omni via transformers (not vLLM). Matches the Phase 2 sidecar pattern (Qwen3-TTS, Qwen3-ASR).
- **D-09:** Single POST /chat endpoint. Accepts: audio (base64 WAV), context_chunks (list of text strings), system_prompt (text). Returns: audio_b64 (generated speech), text (transcript of Omni's answer), sample_rate_hz.
- **D-10:** GET /health endpoint for service health checks, following existing sidecar convention.
- **D-11:** Never co-hosted with split pipeline brain models. A100 80GB cannot fit both. Swap via supervisorctl between benchmark tests.

### Env Profile
- **D-12:** Update the existing placeholder .env.bench.omni_hybrid profile (created in Phase 3) with real values once the sidecar is built.

### Hard-Fail Policy (carried from Phase 2)
- **D-13:** If the Omni sidecar is unavailable when selected, raise RuntimeError. No silent fallback to split pipeline. Matches Phase 2 D-02.

### Claude's Discretion
- Sidecar internal structure (model loading, warmup, audio preprocessing)
- Exact Russian system prompt wording (within the strict grounding constraint)
- Audio format conversion details (sample rate, encoding between browser and Omni)
- How the backend dispatches to the Omni path vs split pipeline path in app.py
- Error message wording for hard-fail scenarios

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Voice AI Playbook
- `docs/voice_ai_playbook_2026-03-25.md` -- Authoritative planning document. "Experimental Track" section (lines 123-138) defines hybrid omni mode. "RAG And Qwen3-Omni" section (lines 256-284) defines three integration levels and recommends prompt-injected context. "Phase D: Native Realtime Experiment" (lines 698-703) defines Phase D scope. Source Notes (lines 711-712) has official Qwen3-Omni repo URL.

### Qwen3-Omni Official Repo
- `https://github.com/QwenLM/Qwen3-Omni` -- Official repo. Researcher MUST check for: model loading code, audio input format, audio output format, inference API, HuggingFace repo ID, VRAM requirements, and Russian language support.

### Existing Voice Pipeline Code
- `rag_demo_system/backend/app.py` -- WebSocket handler. Lines 882-980: input_audio_buffer.commit flow (STT -> RAG -> LLM -> TTS -> log). Lines 99-107: _voice_pipeline() mapping. Lines 800-812: session.update handling. The Omni path will need a parallel branch alongside this existing flow.
- `rag_demo_system/backend/voice_session.py` -- VoiceSession dataclass with voice_provider, stt_provider, tts_provider, brain_model fields and stack_id property. Omni needs to integrate with this session model.
- `rag_demo_system/backend/voice_adapters.py` -- Current adapter dispatch. transcribe_audio() and synthesize_audio_with_provider() functions. Omni bypasses these (single /chat call replaces both).

### Existing Sidecar Examples
- `rag_demo_system/backend/qwen3_tts_sidecar.py` (or equivalent) -- Pattern for standalone FastAPI sidecar with /health endpoint and per-service venv.
- `rag_demo_system/backend/qwen3_asr_sidecar.py` (or equivalent) -- Another sidecar pattern example.

### Benchmark Framework
- `rag_demo_system/scripts/benchmark_runner.py` -- Benchmark runner CLI. Omni must produce JSONL output the runner can consume.
- `rag_demo_system/scripts/benchmark_compare.py` -- Comparison script. Expects all 6 timing fields per line.
- `rag_demo_system/.env.bench.omni_hybrid` -- Placeholder env profile to be completed in this phase.

### Contract Tests
- `rag_demo_system/tests/test_voice_adapters_official.py` -- Existing contract test patterns. Omni adapter needs equivalent tests.

### Frontend
- `rag_demo_system/frontend/app.js` -- buildSessionUpdate() and voice provider selector. Omni needs to appear as a provider option.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `voice_adapters.py:transcribe_audio()` -- STT dispatcher. Omni still uses STT for the RAG text query, so this function is reused as-is.
- `voice_adapters.py:build_voice_statuses()` -- Health check aggregator. Omni sidecar needs a health entry added here.
- `app.py:_voice_chat_streaming_sync()` -- RAG + LLM call. The RAG retrieval part can be factored out and reused by Omni (it needs the same chunk retrieval).
- `voice_session.py:VoiceSession.stack_id` -- Property that builds stack_id from backend/brain/stt/tts. Omni needs a distinct stack_id (e.g., our_rag__Qwen3-Omni-30B-A3B__omni__omni).

### Established Patterns
- **Sidecar convention:** Standalone FastAPI with /health, per-service venv, {NAME}_BASE_URL env var.
- **Hard-fail:** New providers raise RuntimeError when BASE_URL unconfigured (Phase 2 pattern).
- **Env profile overlay:** .env.bench.{name} with load_dotenv(override=True).
- **Timing convention:** time.time() for all timestamps (cross-process correlation).
- **JSONL log format:** All 6 timing fields + question_id, stack_id, transcript, answer, retrieved chunks.

### Integration Points
- `app.py` WebSocket handler: Omni path branches off after session.update identifies voice_provider as "qwen3_omni". Needs a new code path parallel to the split pipeline flow.
- `_voice_pipeline()`: May need updating to recognize "qwen3_omni" as a special case (not a simple stt/tts pair).
- Frontend voice provider dropdown: Needs "Qwen3-Omni" option added.
- `build_voice_statuses()`: Needs Omni sidecar health entry.

</code_context>

<specifics>
## Specific Ideas

- The Omni sidecar /chat endpoint should return both audio AND text (Omni's spoken answer as text) so the JSONL log can include the answer field for quality comparison.
- The benchmark runner already sends audio via WebSocket. For Omni, the same runner should work if the backend routes Omni-selected sessions through the Omni path instead of the split pipeline path.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 04-qwen3-omni-hybrid*
*Context gathered: 2026-03-26*
