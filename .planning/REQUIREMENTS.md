# Requirements: Micro Leasing Voice AI

**Defined:** 2026-03-25
**Core Value:** Accurate, low-latency Russian voice answers grounded in the company knowledge base, with full on-premises control over every pipeline component.

## v1.0 Requirements

Requirements for milestone v1.0: Voice Stack Upgrade and Benchmark. Each maps to roadmap phases.

### Instrumentation

- [x] **INST-01**: Voice session logs 6 timing milestones per turn: speech_stopped, stt_done, retrieval_done, llm_first_token, tts_first_chunk, playback_started
- [x] **INST-02**: Each voice turn emits a structured JSON log line with question_id, stack_id, all timestamps, and derived latencies
- [x] **INST-03**: Primary KPI (playback_started - speech_stopped) is computed and logged for every turn

### Configuration Switching

- [x] **SWITCH-01**: UI exposes selectors for RAG backend (our_rag / dify_rag), brain model (Qwen3.5-35B / Qwen3-30B), STT provider, and TTS provider
- [x] **SWITCH-02**: Switching any selector updates the active configuration without restarting the backend (where possible) or with clear restart guidance
- [x] **SWITCH-03**: The active configuration (stack_id) is captured in every log line so results are automatically tagged by setup

### Benchmark

- [x] **BENCH-01**: Fixed Russian test question set with 80+ questions across 5 categories (short factual, long factual, KB-grounded, ambiguous, out-of-scope)
- [ ] **BENCH-02**: Benchmark runner executes the full question set against the currently active configuration and writes JSONL results
- [ ] **BENCH-03**: Each result includes question_id, stack_id, transcript, answer, retrieved chunks, timing breakdown
- [ ] **BENCH-04**: Comparison script shows side-by-side latency and quality metrics across stacks

### Voice Providers

- [x] **VPROV-01**: Qwen3-TTS adapter integrated into voice_adapters.py with sidecar FastAPI service
- [x] **VPROV-02**: Qwen3-ASR adapter integrated into voice_adapters.py with sidecar FastAPI service
- [x] **VPROV-03**: Voxtral STT adapter integrated into voice_adapters.py (sidecar or API client depending on self-host availability)
- [x] **VPROV-04**: All new adapters pass the existing voice adapter contract tests
- [x] **VPROV-05**: Frontend voice provider selector updated to include all new providers

### Brain

- [ ] **BRAIN-01**: Brain model switchable between Qwen3-30B-A3B (fallback) and Qwen3.5-35B-A3B (target) via UI selector or env var

### Qwen3-Omni

- [ ] **OMNI-01**: Qwen3-Omni hybrid adapter retrieves chunks via existing RAG engine and injects them into Omni prompt
- [ ] **OMNI-02**: Omni hybrid mode accessible as a voice provider option in the UI alongside split pipeline providers
- [ ] **OMNI-03**: Omni results use the same log format so they are directly comparable with split pipeline results

### Deployment

- [x] **DEPLOY-01**: Env profile files for each benchmark stack (baseline, qwen3_tts, qwen3_asr, voxtral, brain_upgrade, omni_hybrid, dify_rag)
- [ ] **DEPLOY-02**: Server deployment script that provisions the stack from a fresh GPU VM
- [ ] **DEPLOY-03**: Smoke test script validates all services are healthy before benchmark execution

## v2 Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### Visualization

- **VIS-01**: Per-provider timing comparison dashboard in the UI
- **VIS-02**: Automated regression detection when benchmark runs get slower

### Advanced Benchmark

- **ABENCH-01**: Automated answer quality scoring with annotated correct answers
- **ABENCH-02**: Multi-run statistical analysis with confidence intervals

### Advanced Omni

- **AOMNI-01**: Qwen3-Omni pure native realtime mode (audio-in, audio-out, no split pipeline)
- **AOMNI-02**: Tool-mediated retrieval inside Omni session (model calls retrieval tool)

### Infrastructure

- **INFRA-01**: LiveKit/Pipecat migration for production WebRTC transport
- **INFRA-02**: Telephony/SIP integration for phone-based access
- **INFRA-03**: Automated VRAM profiling and model hot-swap scripts

### Alternative RAG

- **ARAG-01**: Evaluate RAGFlow or LightRAG as alternative to our_rag if dify_rag wins comparison

## Out of Scope

| Feature | Reason |
|---------|--------|
| New repo or separate project | UI, transport, toggles, deployment scripts already exist in this repo |
| Merging to main | Stays on feature branch until server benchmarks confirm value |
| LiveKit/Pipecat migration | Not before baseline measurements are done (playbook constraint) |
| Telephony/SIP | Browser-first for this milestone |
| Qwen3-Omni pure native realtime | Only after hybrid mode proves viable (playbook Phase D gate) |
| Third-party RAG systems (RAGFlow, LightRAG) | our_rag vs dify_rag comparison first; no proven advantage for Russian low-latency small KB |
| NVIDIA Canary-Qwen-2.5B | English-only; does not meet Russian language requirement |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INST-01 | Phase 1 | Complete |
| INST-02 | Phase 1 | Complete |
| INST-03 | Phase 1 | Complete |
| SWITCH-01 | Phase 1 | Complete |
| SWITCH-02 | Phase 1 | Complete |
| SWITCH-03 | Phase 1 | Complete |
| VPROV-01 | Phase 2 | Complete |
| VPROV-02 | Phase 2 | Complete |
| VPROV-03 | Phase 2 | Complete |
| VPROV-04 | Phase 2 | Complete |
| VPROV-05 | Phase 2 | Complete |
| BRAIN-01 | Phase 3 | Pending |
| BENCH-01 | Phase 3 | Complete |
| BENCH-02 | Phase 3 | Pending |
| BENCH-03 | Phase 3 | Pending |
| BENCH-04 | Phase 3 | Pending |
| DEPLOY-01 | Phase 3 | Complete |
| OMNI-01 | Phase 4 | Pending |
| OMNI-02 | Phase 4 | Pending |
| OMNI-03 | Phase 4 | Pending |
| DEPLOY-02 | Phase 5 | Pending |
| DEPLOY-03 | Phase 5 | Pending |

**Coverage:**
- v1.0 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-03-25*
*Last updated: 2026-03-25 after roadmap rewrite — traceability updated to 5-phase structure*
