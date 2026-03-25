# Roadmap: Micro Leasing Voice AI — v1.0

## Overview

This milestone extends an already-working split pipeline (SenseVoice STT, Qdrant+BM25 RAG, vLLM brain, CosyVoice TTS) with timing instrumentation, new voice provider adapters, a brain upgrade path, a Qwen3-Omni hybrid experiment, and a reproducible benchmark framework. All code is built locally in Phases 1-4. Phase 5 is the only server phase: it provisions the GPU VM, runs the smoke test, and then executes the full benchmark matrix in a controlled order. The benchmark order within Phase 5 is fixed: RAG comparison first, brain comparison second, STT/TTS comparison third, Omni vs. split pipeline last. This order isolates one variable per benchmark run and ensures each result has a valid baseline to compare against.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Instrumentation and UI Switching** - Wire six per-turn latency milestones into voice_session.py, emit structured JSON logs, and expose UI selectors for all pipeline variables with stack_id tagging
- [x] **Phase 2: Voice Provider Adapters** - Build Qwen3-TTS sidecar + adapter, Qwen3-ASR sidecar + adapter, and Voxtral adapter; all pass contract tests and appear in the frontend selector (completed 2026-03-25)
- [ ] **Phase 3: Brain Upgrade and Benchmark Framework** - Make brain model switchable via UI and env var, create the 80+ question Russian fixture, build the benchmark runner CLI and comparison script, and write per-stack env profiles
- [ ] **Phase 4: Qwen3-Omni Hybrid** - Implement the Omni hybrid adapter with RAG context injection, register it as a UI provider option, and confirm it uses the same log format as the split pipeline
- [ ] **Phase 5: Server Deployment and Benchmarks** - Provision the GPU server from a fresh VM, validate with smoke test, then run the full benchmark matrix in smart order on the server

## Phase Details

### Phase 1: Instrumentation and UI Switching
**Goal**: Every voice turn emits a structured JSON log with six latency milestones and a computed primary KPI, all pipeline variables are selectable from the UI, and every log line is automatically tagged with the active stack_id
**Depends on**: Nothing (first phase)
**Requirements**: INST-01, INST-02, INST-03, SWITCH-01, SWITCH-02, SWITCH-03
**Success Criteria** (what must be TRUE):
  1. After each voice turn, a JSON log line appears in the backend output with question_id, stack_id, and all six timestamp fields: speech_stopped, stt_done, retrieval_done, llm_first_token, tts_first_chunk, playback_started
  2. The primary KPI (playback_started minus speech_stopped) is computed and present in every log line without requiring manual calculation
  3. The UI shows live selectors for RAG backend (our_rag / dify_rag), brain model (Qwen3-30B / Qwen3.5-35B), STT provider, and TTS provider; switching any selector takes effect without requiring a full backend restart
  4. The active stack_id is derived from the current selector combination and embedded in every log line automatically, so no manual tagging is needed during benchmarking
**Plans:** 1/3 plans executed
Plans:
- [x] 01-01-PLAN.md — VoiceSession dataclass extension + test scaffolding
- [x] 01-02-PLAN.md — Backend instrumentation (timestamps, session.update, structured log)
- [x] 01-03-PLAN.md — Frontend UI selectors (brain model, STT, TTS)

### Phase 2: Voice Provider Adapters
**Goal**: Qwen3-TTS, Qwen3-ASR, and Voxtral are available as selectable providers in the UI, each backed by a validated sidecar or adapter, and no adapter silently falls through to the default provider
**Depends on**: Phase 1
**Requirements**: VPROV-01, VPROV-02, VPROV-03, VPROV-04, VPROV-05
**Success Criteria** (what must be TRUE):
  1. Selecting Qwen3-TTS in the UI routes synthesis through the Qwen3-TTS sidecar; the browser plays Russian speech and domain-specific leasing terms (лизингополучатель, аванс, до 84 месяцев) are intelligible
  2. Selecting Qwen3-ASR in the UI routes transcription through the Qwen3-ASR sidecar; spoken Russian produces a correct Russian text transcript
  3. Selecting Voxtral in the UI routes transcription through the Voxtral adapter (sidecar or cloud client); spoken Russian produces a correct Russian text transcript
  4. All three new adapters pass the existing voice adapter contract test suite with no failures (VPROV-04 gate)
  5. The provider field in each log line matches the intended provider when the fallback env vars (SENSEVOICE_BASE_URL, WHISPER_BASE_URL) are unset, confirming no silent fallback during benchmarking
**Plans:** 2/2 plans complete
Plans:
- [x] 02-01-PLAN.md — Contract tests, adapter branches, health status, frontend options
- [x] 02-02-PLAN.md — Qwen3-TTS, Qwen3-ASR, Voxtral sidecar servers + requirements files

### Phase 3: Brain Upgrade and Benchmark Framework
**Goal**: The brain model is switchable via UI selector and env var, a fixed Russian question set and benchmark runner are ready for use, and per-stack env profiles cover every configuration to be benchmarked
**Depends on**: Phase 1
**Requirements**: BRAIN-01, BENCH-01, BENCH-02, BENCH-03, BENCH-04, DEPLOY-01
**Success Criteria** (what must be TRUE):
  1. Selecting Qwen3.5-35B-A3B in the UI or setting the env var causes the backend to route inference to the upgraded model, confirmed by stack_id in log output
  2. An 80+ question Russian fixture file exists covering all five categories (short factual, long factual, KB-grounded, ambiguous, out-of-scope) with questions phrased as spoken Russian
  3. The benchmark runner CLI executes the full question set against the active configuration and writes a JSONL results file where every line contains question_id, stack_id, transcript, answer, retrieved chunks, and all timing fields; the first three turns per stack are flagged as warmup
  4. The comparison script reads two JSONL files and outputs a side-by-side markdown table of latency and quality metrics (mean/p50/p95 for primary KPI and llm_ttfb_ms)
  5. Per-stack env profile files exist for every benchmark configuration: baseline, qwen3_tts, qwen3_asr, voxtral, brain_upgrade, omni_hybrid, dify_rag
**Plans:** 3 plans
Plans:
- [ ] 03-01-PLAN.md — Brain model routing fix + streaming LLM first-token timing
- [ ] 03-02-PLAN.md — 80+ question Russian fixture + 7 env profiles
- [ ] 03-03-PLAN.md — Benchmark runner CLI + comparison script + TTS endpoint

### Phase 4: Qwen3-Omni Hybrid
**Goal**: Qwen3-Omni hybrid mode retrieves context via the existing RAG engine, injects it into the Omni prompt, is accessible as a UI provider option, and produces JSONL output directly comparable with split pipeline results
**Depends on**: Phase 2, Phase 3
**Requirements**: OMNI-01, OMNI-02, OMNI-03
**Success Criteria** (what must be TRUE):
  1. Selecting Qwen3-Omni in the UI routes the voice turn through the Omni hybrid path; the existing RAG engine retrieves chunks first and they are injected into the Omni prompt before inference
  2. Out-of-scope questions (category 5 from the benchmark fixture) return a refusal or out-of-scope response rather than hallucinated answers, confirming the injected context is being respected
  3. Omni JSONL output contains the same fields as split pipeline output (question_id, stack_id, transcript, answer, retrieved chunks, all timing fields) so the comparison script can process both without modification
**Plans**: TBD

### Phase 5: Server Deployment and Benchmarks
**Goal**: The GPU server is provisioned from a fresh VM, all services pass the smoke test, and the benchmark matrix is executed in smart order — RAG comparison first, brain comparison second, Omni hybrid third (the most promising experiment, now with a baseline to compare against), and full STT/TTS matrix only as a fallback if Omni does not perform well enough
**Depends on**: Phase 2, Phase 3, Phase 4
**Requirements**: DEPLOY-02, DEPLOY-03
**Success Criteria** (what must be TRUE):
  1. The provisioning script runs on a fresh Vast.ai or Azure H100 VM and brings all services to a running state without manual steps; the smoke test script confirms every service returns healthy status before any benchmark run begins
  2. The RAG comparison benchmark (our_rag vs. dify_rag, same brain and providers) completes on the server and produces two JSONL result files with valid timing data across all 80+ questions
  3. The brain comparison benchmark (Qwen3-30B-A3B vs. Qwen3.5-35B-A3B, winning RAG, same providers) completes on the server with VRAM confirmed below 80 GB limit via nvidia-smi before each model load
  4. The Omni hybrid benchmark completes on the server using the winning RAG and brain; Omni results and the best split pipeline results exist as JSONL files and the comparison script produces a valid side-by-side table
  5. If Omni does not meet quality/latency bar: STT/TTS provider comparison (Qwen3-ASR, Qwen3-TTS, Voxtral vs. baseline providers) runs as fallback and produces JSONL result files for each provider permutation
**Plans**: TBD

## Progress

**Execution Order:**
Phases 1-4 are local code work. Phase 5 is the only server phase and runs after all local code is complete.
Phase 3 depends only on Phase 1 (not Phase 2) and can be developed in parallel with Phase 2.
Phase 4 depends on Phase 2 and Phase 3.
Phase 5 depends on Phase 2, Phase 3, and Phase 4.

**Benchmark execution order within Phase 5 (smart order):**
1. RAG comparison (our_rag vs. dify_rag) — gives RAG winner
2. Brain comparison (Qwen3-30B vs. Qwen3.5-35B, winning RAG) — gives brain winner
3. Omni hybrid vs. best split pipeline — the most promising experiment, tested third with a baseline to compare against
4. Only if Omni fails: full STT/TTS matrix (Qwen3-ASR, Qwen3-TTS, Voxtral vs. baseline) — fallback plan, not mandatory

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Instrumentation and UI Switching | 1/3 | In Progress|  |
| 2. Voice Provider Adapters | 2/2 | Complete   | 2026-03-25 |
| 3. Brain Upgrade and Benchmark Framework | 0/3 | Not started | - |
| 4. Qwen3-Omni Hybrid | 0/TBD | Not started | - |
| 5. Server Deployment and Benchmarks | 0/TBD | Not started | - |
