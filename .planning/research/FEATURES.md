# Feature Research

**Domain:** On-premises voice AI assistant — voice pipeline instrumentation, benchmarking, and provider upgrade
**Researched:** 2026-03-25
**Confidence:** MEDIUM (web access unavailable; Qwen3-ASR, Qwen3-TTS, Voxtral are < August 2025 training knowledge; Qwen3-Omni is at training cutoff boundary; all model-specific API claims flagged per-item)

---

## Context: What Already Exists

The milestone adds NEW capabilities to an existing, working system. The existing system on branch `codex/split-voice-providers` already provides:

- Split voice pipeline: browser PCM16 -> STT -> RAG -> LLM -> TTS -> browser playback via `WS /ws/voice`
- 4 switchable voice providers: `local` (SenseVoice + CosyVoice), `yandex_speechkit`, `oss_russian` (Vosk), `yandex_realtime`
- 2 RAG backends: `our_rag` (Qdrant + BM25 + rerank) and `dify_rag`
- `voice_fast` retrieval profile (vector_top_k=3, bm25_top_k=1, final_top_n=2, reranker disabled)
- Supervisor-managed one-command launcher (`stack.sh`, `supervisord.conf`)
- Frontend voice provider selector UI
- Provider abstraction in `voice_adapters.py` using a URL-based service contract (`/transcribe`, `/speak`, `/health`)

The abstraction point for all new STT/TTS providers is `voice_adapters.py`. New providers follow the same contract: an HTTP microservice exposing `/transcribe` (POST) and `/speak` (POST) and `/health` (GET), with base URL configured via env var. The `transcribe_audio()` and `synthesize_audio_with_provider()` functions in `voice_adapters.py` handle dispatch by provider name.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that must work correctly for the milestone to be coherent. Missing these makes the benchmark data meaningless or the new providers unusable.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| End-to-end timing instrumentation (6 milestones) | Without timestamps, no benchmark is possible; the whole milestone is premised on measured comparison | MEDIUM | 6 milestones: `speech_stopped_at`, `stt_done_at`, `retrieval_done_at`, `llm_first_token_at`, `tts_first_chunk_at`, `playback_started_at`. Primary KPI: `playback_started_at - speech_stopped_at`. The WebSocket handler in `app.py` already has a `time.perf_counter()` pattern in the HTTP chat path; same pattern must be applied inside the `input_audio_buffer.commit` WS branch. The 6 timestamps span both client-side (browser JS) and server-side (Python), requiring a hybrid approach: server emits timing events over the WS, client captures playback start. |
| Benchmark result log format (per-run JSONL) | Logs are the only artifact that persists after a benchmark run; must be machine-readable for comparison | LOW | Each record: `{run_id, stack_id, question_id, question_text, transcript, answer, retrieved_chunk_ids, citations, timings, evaluator_note, ts}`. JSONL (one record per line) is the correct format: appendable, greppable, easy to load into pandas. |
| Fixed Russian test question set (80+ questions, 5 categories) | Reproducibility requires identical inputs across all benchmark runs | MEDIUM | 5 categories from the playbook: (1) short factual / 20 questions, (2) longer factual / 20 questions, (3) precise KB-grounding required / 20 questions, (4) ambiguous user questions / 10 questions, (5) out-of-scope questions / 10 questions. Questions must be real leasing domain queries in Russian (vehicle leasing, financial requirements, documentation, timelines, refusals). Out-of-scope questions are needed to test grounding discipline: the correct answer is a refusal, not a hallucinated response. |
| Qwen3-TTS adapter in `voice_adapters.py` | The TTS upgrade is the highest-ROI single swap and must be benchmarkable | HIGH | Confidence: LOW (Qwen3-TTS was announced near training cutoff; API details unverified via live sources). Model: `Qwen/Qwen3-TTS-12Hz-1.7B` (as referenced in the playbook). Russian language: the playbook explicitly states Russian is supported. Streaming: the model name suffix `12Hz` implies a streaming frame-rate encoding, but whether the inference server exposes a streaming HTTP endpoint or returns a complete audio buffer is unverified. Implementation strategy: wrap in a sidecar service (same `/speak` + `/health` contract as CosyVoice). Env var: `QWEN3_TTS_BASE_URL`. Fallback: existing `cosyvoice` adapter. |
| Qwen3-ASR adapter in `voice_adapters.py` | Required for STT benchmark phase; enables Qwen-family coherence | HIGH | Confidence: LOW (Qwen3-ASR announced near training cutoff; API details unverified). Model: `Qwen/Qwen3-ASR-1.7B`. Russian language: the playbook states Russian is supported. Implementation strategy: same sidecar pattern as SenseVoice (`/transcribe` + `/health` contract). Env var: `QWEN3_ASR_BASE_URL`. Fallback: existing `sensevoice` adapter. |
| UI brain model selector | Without a UI control, switching from `Qwen3-30B-A3B` to `Qwen3.5-35B-A3B` requires an env var change and backend restart, making the benchmark matrix harder to run | LOW | The frontend already has a voice provider selector. Add a brain model selector with the same pattern. Options: `qwen3_30b` (default) and `qwen3_5_35b`. The backend reads `RAG_LLM_MODEL`; a second env var `RAG_LLM_ALT_MODEL` can hold the alternate. The WS `session.update` event already carries backend state; extend it to carry `brain_model`. |

### Differentiators (Competitive Advantage)

Features that go beyond the table stakes and create meaningful analytical value for the benchmark.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Benchmark runner script (automated multi-question execution) | Manual question-by-question testing is error-prone and slow; a runner guarantees identical conditions across all 80+ questions | MEDIUM | A Python script (not a test file) that: (1) loads the question fixture JSON, (2) posts each question to `/api/chat` with `mode=voice_fast` and the selected backend/model, (3) captures the full response including `timings`, (4) appends to the JSONL log, (5) sleeps between questions to avoid cache artifacts. The script must accept a `--stack-id` argument so runs are labeled. Dependency: the JSONL log format must be finalized first. |
| Voxtral STT adapter (optional, third STT candidate) | Adds a second competitive STT candidate alongside Qwen3-ASR; gives STT benchmark more data points | HIGH | Confidence: LOW (Voxtral Realtime was announced by Mistral AI near training cutoff; API contract details unverified). The playbook lists Voxtral as a benchmark candidate, not a requirement. Implementation strategy: same sidecar contract. Env var: `VOXTRAL_BASE_URL`. This is explicitly optional per the playbook and PROJECT.md. Do not block other work on this. |
| Qwen3-Omni hybrid adapter (retrieved context injected into prompt) | Tests whether a native audio-in/audio-out model can match the split pipeline quality with external RAG grounding | HIGH | Confidence: MEDIUM (Qwen3-Omni architecture is documented; prompt-injection as a RAG integration method is established). Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`. Russian language: explicitly stated as supported in the playbook. Hybrid mode mechanics: (1) browser audio arrives at the backend, (2) backend calls `our_rag` retrieval with the transcribed query, (3) retrieved chunks are injected as text context into the Omni model's input alongside the audio, (4) Omni generates audio output. This is NOT the split pipeline; retrieval is done externally but generation is native Omni. The adapter is a new `voice_adapters.py` path, not a sidecar service. Env var: `QWEN3_OMNI_BASE_URL`. Critical constraint: this path bypasses the standard TTS step; the Omni model produces audio directly. |
| Per-stack env profiles (one `.env` file per benchmark stack) | Enables switching between full benchmark stacks without manually editing env vars | LOW | The stack launcher already uses `STACK_VOICE_PROFILE`. Extend it with a profile loader: `stack.sh load-profile <profile_name>` that copies the appropriate `.env.<profile_name>` into `.env`. Required profiles: `baseline`, `qwen3_tts`, `qwen3_asr`, `voxtral` (optional), `brain_upgrade`, `omni_hybrid`. Each profile pins all relevant env vars to reproduce that exact stack. |
| Benchmark comparison report (per-run diff summary) | Makes it practical to compare two runs without manually reading raw JSONL | MEDIUM | A lightweight Python script that takes two JSONL log files and outputs a markdown table comparing mean/p50/p95 for each timing metric. No external analytics tools needed. |
| Timing events pushed to browser over WS | Enables the frontend to display per-turn latency without requiring a separate log query | LOW | The `response.done` WS event already carries `timings` from the HTTP chat path. The issue is that the voice WebSocket handler (`input_audio_buffer.commit` branch) calls the HTTP chat endpoint as an internal function call, not an HTTP request, so `timings` is available. Ensure the `response.done` event sent over WS includes the full timing dict including the newly-added STT and TTS timing fields. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Qwen3-Omni pure native realtime mode as a primary path | Eliminates the STT/TTS overhead, attractive for latency | Weakens RAG grounding severely; the model generates from its own internal knowledge when chunks are not injected; hallucination rate increases; debugging is opaque; current RAG control is lost entirely | Hybrid mode first: inject retrieved chunks as text context into Omni input. Only test pure realtime after hybrid mode is measured and found viable. |
| LiveKit or Pipecat migration during this milestone | Improves WebRTC transport quality and enables telephony | Adds a new infrastructure layer before the model benchmark is done; cannot isolate whether latency changes come from models or transport; invalidates current baseline | Stay on the existing FastAPI + WebSocket transport for all benchmark runs. Revisit LiveKit after the model benchmark is complete and production shape is decided. |
| Co-hosting all models simultaneously on the benchmark server | Eliminates model swap overhead between test phases | A100 80GB or H100 94GB cannot co-host `Qwen3-30B-A3B`, `Qwen3.5-35B-A3B`, `Qwen3-Omni-30B-A3B-Instruct`, `Qwen3-TTS-1.7B`, and `Qwen3-ASR-1.7B` simultaneously at full weight; VRAM exhaustion causes OOM or model offloading that invalidates latency measurements | Swap models between test phases. One model class loads at a time. The benchmark discipline requires this anyway: change one component per phase. |
| Automated evaluator for grounding correctness | Desirable for scalable quality assessment | Adds significant complexity (an LLM-as-judge setup); creates a dependency on a second LLM or external API during benchmark execution; risks evaluator bias | Manual evaluator notes in the JSONL log per question. Human review for correctness is the correct approach at this scale. Add automation only after the manual process is validated. |
| Streaming TTS output to browser during LLM generation | Reduces perceived latency significantly | Requires chunk-by-chunk sentence splitting, audio chunk buffering, and synchronization of LLM streaming with TTS chunk requests; breaks the current clean request-response pattern in `synthesize_audio_with_provider()`; also breaks the `response.done` WS event timing assumptions | Measure latency with the current batch TTS approach first. Streaming TTS is a separate optimization phase. |

---

## Feature Dependencies

```
[Timing instrumentation] (server side)
    └──required by──> [Benchmark result log format]
                          └──required by──> [Benchmark runner script]
                                                └──required by──> [Comparison report]

[Qwen3-TTS adapter]
    └──depends on──> [Per-stack env profiles]
    └──feeds──> [TTS benchmark phase]

[Qwen3-ASR adapter]
    └──depends on──> [Per-stack env profiles]
    └──feeds──> [STT benchmark phase]

[Brain model selector (UI + backend)]
    └──required for──> [Brain upgrade benchmark phase]

[Qwen3-Omni hybrid adapter]
    └──depends on──> [Timing instrumentation]
    └──depends on──> [existing our_rag retrieval path]
    └──conflicts with──> [standard TTS step] (Omni generates audio natively; no TTS call)
    └──must be built after──> [split pipeline baseline is measured]

[Voxtral STT adapter] (optional)
    └──same pattern as──> [Qwen3-ASR adapter]
    └──no hard dependencies; can be deferred]
```

### Dependency Notes

- **Timing instrumentation is the prerequisite for everything**: without it, no benchmark run produces valid data. It must be the first implementation task.
- **Qwen3-Omni conflicts with the standard TTS step**: the Omni adapter generates audio output natively. The `synthesize_audio_with_provider()` call in the WS handler must be skipped for the `omni_hybrid` provider path. This is a non-trivial divergence in the WS handler flow.
- **Per-stack env profiles enable but do not block adapters**: adapters can be developed and unit-tested against mock services without the profile loader. The profile loader is needed only for actual benchmark execution.
- **Brain model selector depends on the existing `RAG_LLM_MODEL` env var pattern**: adding a second env var `RAG_LLM_ALT_MODEL` and a `session.update` field for `brain_model` is sufficient. No schema changes needed.

---

## MVP Definition

### Launch With (Milestone v1.0 Complete When)

These are the features that constitute a complete and valid benchmark run.

- [ ] Timing instrumentation: all 6 milestones captured and emitted in `response.done` WS event and JSONL log
- [ ] Benchmark JSONL log format defined and implemented
- [ ] Fixed Russian question set: 80+ questions in 5 categories, saved as a JSON fixture
- [ ] Qwen3-TTS adapter: `/speak` + `/health` sidecar contract, `QWEN3_TTS_BASE_URL` env var, integrated in `voice_adapters.py`
- [ ] Qwen3-ASR adapter: `/transcribe` + `/health` sidecar contract, `QWEN3_ASR_BASE_URL` env var, integrated in `voice_adapters.py`
- [ ] Brain model selector: UI control, WS `session.update` field, backend reads selected model
- [ ] Per-stack env profiles: one `.env` file per benchmark phase
- [ ] Benchmark runner script: executes the full question set against a specified stack, appends to JSONL

### Add After Baseline Benchmark Is Complete (v1.x)

Features added after the split pipeline baseline is measured.

- [ ] Qwen3-Omni hybrid adapter: add after split pipeline baseline gives a latency/quality reference point
- [ ] Comparison report script: useful once two or more JSONL logs exist to compare
- [ ] Timing events pushed to browser: nice-to-have for real-time display; not needed for benchmark validity

### Future Consideration (v2+)

- [ ] Voxtral STT adapter: optional per the playbook; add only if the benchmark server setup makes it easy; do not block v1.0 on it
- [ ] Streaming TTS: separate optimization phase after the batch TTS baseline is measured
- [ ] LiveKit migration: separate milestone, explicitly out of scope for v1.0
- [ ] Qwen3-Omni pure native realtime mode: only after hybrid mode proves viable

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Timing instrumentation | HIGH | MEDIUM | P1 |
| Benchmark JSONL log format | HIGH | LOW | P1 |
| Russian question fixture | HIGH | MEDIUM | P1 |
| Qwen3-TTS adapter | HIGH | HIGH | P1 |
| Qwen3-ASR adapter | HIGH | HIGH | P1 |
| Brain model selector | HIGH | LOW | P1 |
| Per-stack env profiles | MEDIUM | LOW | P1 |
| Benchmark runner script | HIGH | MEDIUM | P2 |
| Qwen3-Omni hybrid adapter | MEDIUM | HIGH | P2 |
| Comparison report script | MEDIUM | LOW | P2 |
| Timing events to browser | LOW | LOW | P2 |
| Voxtral STT adapter | MEDIUM | HIGH | P3 |
| Streaming TTS | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Required for a valid benchmark run. Milestone is not done without these.
- P2: Adds analytical value once P1 items are working.
- P3: Deferred to a later phase.

---

## Benchmark Question Set Design (Russian Enterprise KB)

This is a domain-specific sub-feature that requires its own design rationale.

### What a Good Voice AI Benchmark Question Set Looks Like

**Principle:** Questions must be phrased as a real user would speak them, not as search queries. Russian spoken language has case inflections, particles, and ellipsis that differ from written text. The STT output will reflect spoken Russian; the retrieval must handle it.

**Category 1: Short factual (20 questions)**
- One-breath questions with a clear single answer in the KB
- Example: "Какой минимальный аванс при лизинге грузовика?"
- Tests: STT accuracy on short utterances, retrieval recall on unambiguous queries, TTS naturalness on short answers
- Correct behavior: single-sentence answer, correct figure, citation present

**Category 2: Longer factual (20 questions)**
- Multi-part questions or questions requiring synthesis of 2-3 KB fragments
- Example: "Расскажите о требованиях к компании для получения лизинга на спецтехнику"
- Tests: retrieval recall on broader queries, LLM ability to synthesize without hallucinating, TTS naturalness on longer answers

**Category 3: Precise KB grounding required (20 questions)**
- Questions where the answer is in the KB but phrased differently from the question
- Example: "Могу ли я оформить лизинг, если компания работает только год?" (KB may say "minimum 12 months of operation")
- Tests: whether retrieval finds semantically similar but lexically different content, whether the LLM interprets KB correctly without adding hallucinated detail

**Category 4: Ambiguous user questions (10 questions)**
- Questions that could have multiple interpretations
- Example: "Сколько стоит лизинг?" (cost of what — monthly payment, total cost, advance?)
- Tests: whether the system asks a clarifying question or picks the most common interpretation, whether TTS handles uncertainty naturally

**Category 5: Out-of-scope questions (10 questions)**
- Questions on topics not in the KB
- Example: "Где ваш офис в Санкт-Петербурге?" or "Можно ли взять в лизинг яхту?"
- Tests: strict grounding discipline (the correct answer is a Russian refusal phrase), no hallucination, TTS naturalness on refusal phrases
- Critical: grounding correctness is the top priority; a system that answers out-of-scope questions confidently fails this category regardless of STT/TTS quality

### Evaluator Note Format

Each run row in the JSONL log should include an `evaluator_note` field. During manual review, the evaluator fills in:
- `grounding`: `correct` / `partial` / `wrong` / `hallucinated`
- `naturalness`: `1-5` (Russian native listener score)
- `completeness`: `complete` / `incomplete` / `truncated`

---

## Implementation Notes Per Feature

### Timing Instrumentation

The 6 timing milestones span client and server. The server-side split:

1. `speech_stopped_at`: when the `input_audio_buffer.commit` WS event is received (Python `time.perf_counter()` at WS event entry)
2. `stt_done_at`: immediately after `transcribe_audio()` returns
3. `retrieval_done_at`: immediately after the `engine.retrieve()` call inside the WS handler (currently the WS handler calls `chat()` which calls `engine.retrieve()`; timing is already tracked inside `engine.retrieve()` but not propagated to the WS response)
4. `llm_first_token_at`: first token from the LLM (already tracked in the streaming HTTP chat path; must be surfaced in the WS handler path which currently uses non-streaming LLM calls)
5. `tts_first_chunk_at`: when `synthesize_audio_with_provider()` returns (the current TTS call is batch, not streaming; `tts_first_chunk_at` = `tts_done_at` for now)
6. `playback_started_at`: client-side; the browser fires this when it begins audio playback after receiving `response.output_audio.delta`

The current WS handler calls `chat()` with `stream=False`. This means `llm_first_token_at` is not available in the WS path. For the benchmark, use `llm_total_ms` from the chat response `timings` dict (already present in the HTTP response). Instrument the WS handler to record wall-clock timestamps at each stage and include them in the `response.done` WS event.

### voice_adapters.py Extension Pattern

All new STT adapters follow this pattern (same as the existing Vosk integration):

```python
# In transcribe_audio():
if name == "qwen3_asr":
    base_url = os.getenv("QWEN3_ASR_BASE_URL")
    if not base_url:
        continue
    resp = requests.post(
        base_url.rstrip("/") + "/transcribe",
        json={"audio_b64": audio_b64, "session_id": session_id, "language": "ru", "sample_rate_hz": 24000},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("text"):
        data.setdefault("provider", "qwen3_asr")
        return data
    continue
```

All new TTS adapters follow this pattern (same as the existing Vosk TTS integration):

```python
# In synthesize_audio_with_provider():
if preferred == "qwen3_tts":
    base_url = os.getenv("QWEN3_TTS_BASE_URL")
    if not base_url:
        raise RuntimeError("QWEN3_TTS_BASE_URL is not configured")
    resp = requests.post(
        base_url.rstrip("/") + "/speak",
        json={"text": text, "session_id": session_id, "language": "ru"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    data.setdefault("provider", "qwen3_tts")
    data.setdefault("session_id", session_id)
    return data
```

The health check for new providers is already handled generically by `_service_status()` in `voice_adapters.py` as long as the env var name follows the `<NAME>_BASE_URL` convention and is registered in `build_voice_statuses()`.

### Qwen3-Omni Hybrid Adapter

The Omni adapter diverges from the sidecar pattern because it handles both STT and generation in one call, and produces audio output directly (no separate TTS step). The WS handler must be aware of this:

1. When `voice_provider == "omni_hybrid"`: skip the standard `transcribe_audio()` -> `chat()` -> `synthesize_audio_with_provider()` sequence.
2. Instead: send the raw audio buffer directly to the Omni service, along with retrieved context as a text injection.
3. The Omni service returns text transcript + generated audio in a single response (or streaming chunks).
4. The WS handler emits the transcript and audio events as normal.

This requires a new branch in the WS handler, not just a new entry in `voice_adapters.py`. The retrieval step still runs in our backend before calling Omni (hybrid mode), so `our_rag` retrieval is preserved.

### `normalize_voice_provider()` Must Be Updated

The function in `yandex_realtime.py` currently whitelists: `local`, `yandex_realtime`, `yandex_speechkit`, `oss_russian`. Each new provider name must be added to this whitelist, or the function must be moved to `voice_adapters.py` and made extensible.

### Russian Language Verification Status Per Model

| Model | Russian Support Claim | Verification Source | Confidence |
|-------|----------------------|---------------------|------------|
| Qwen3-TTS-12Hz-1.7B | Explicitly stated in playbook | Internal playbook (cites official GitHub) | LOW (not independently verified via live source in this session) |
| Qwen3-ASR-1.7B | Stated in playbook | Internal playbook (cites Qwen3-ASR-Toolkit GitHub) | LOW |
| Qwen3-Omni-30B-A3B-Instruct | Stated in playbook; Russian audio I/O described | Internal playbook | LOW |
| Voxtral Realtime | Stated as "Russian-capable" in playbook | Internal playbook (cites Mistral AI announcement) | LOW |
| Qwen3.5-35B-A3B | Strong multilingual text model; Russian text quality HIGH | Qwen model family documentation (training knowledge) | MEDIUM |
| Qwen3-30B-A3B | Currently deployed, Russian text quality confirmed in production | In-repo evidence (active deployment) | HIGH |

**Critical constraint from PROJECT.md:** "All STT/TTS models must have explicit Russian language support. NVIDIA Canary-Qwen-2.5B is English-only; do not use." Apply this check to any new model before adding it to the test matrix.

---

## Sources

- In-repo: `rag_demo_system/backend/voice_adapters.py` (existing adapter pattern)
- In-repo: `rag_demo_system/backend/app.py` (WS handler, timing pattern in HTTP chat path)
- In-repo: `rag_demo_system/backend/voice_session.py` (session state model)
- In-repo: `.planning/PROJECT.md` (milestone scope, constraints, Russian language requirement)
- In-repo: `docs/voice_ai_playbook_2026-03-25.md` (benchmark plan, timing milestones, question set design, model recommendations)
- In-repo: `rag_demo_system/README.md` (deployment model, provider contract)
- In-repo: `rag_demo_system/tests/test_voice_adapters_official.py` (existing adapter test patterns to follow)
- Training knowledge (cutoff August 2025): Qwen model family general capabilities, voice pipeline benchmarking patterns, JSONL log format conventions

---
*Feature research for: voice pipeline instrumentation, benchmarking, and provider upgrade (milestone v1.0)*
*Researched: 2026-03-25*
