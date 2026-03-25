# Project Research Summary

**Project:** Leasing Voice AI — Provider Adapters and Benchmark Infrastructure
**Domain:** On-premises Russian voice assistant: split STT/RAG/LLM/TTS pipeline with switchable providers and reproducible benchmark harness
**Researched:** 2026-03-25
**Confidence:** MEDIUM (architecture analysis HIGH; new model API details LOW; Voxtral self-host status unconfirmed)

## Executive Summary

This milestone extends an already-working split voice pipeline (SenseVoice STT, Qdrant+BM25 RAG, vLLM LLM brain, CosyVoice TTS) by adding three new model adapters (Qwen3-TTS, Qwen3-ASR, Voxtral STT), upgrading the brain to Qwen3.5-35B-A3B, and building a reproducible benchmark harness against a fixed 80+ question Russian leasing KB test set. A second experimental track (Qwen3-Omni hybrid mode) runs the Omni multimodal model with context injection as an alternative to the split pipeline, but only after the split pipeline baseline is measured. All new providers follow the established sidecar FastAPI microservice pattern (`/health`, `/transcribe`, `/speak` HTTP contracts) so the backend never loads model weights in-process. The benchmark harness produces per-turn JSONL logs with six timing milestones that enable quantitative comparison across all stack permutations.

The recommended build order is strict: timing instrumentation first, then TTS and STT adapters, then the benchmark runner and brain upgrade, and finally the Omni experiment. This ordering is non-negotiable because timing instrumentation is the prerequisite for all benchmark data; the Omni experiment must not run until a split pipeline baseline exists for comparison. VRAM management is the dominant infrastructure risk: Qwen3.5-35B-A3B at bfloat16 consumes ~70 GB VRAM and must not co-host with Qwen3-Omni-30B (another ~60 GB). Model swaps between phases must be explicit via supervisord.

The biggest execution risks are correctness risks, not infrastructure risks: (1) new TTS/STT adapters silently returning wrong audio formats or falling through the fallback chain; (2) Qwen3-Omni ignoring injected context and answering from pretraining; (3) Russian domain-specific vocabulary (leasing terminology, numerals) degrading TTS/STT quality below acceptable levels. Each risk has a concrete mitigation: contract unit tests for adapters, out-of-scope question grounding tests for Omni, and domain-specific sentence lists scored by a native speaker for language quality. Do not skip these checks.

## Key Findings

### Recommended Stack

The milestone makes minimal changes to the existing verified stack. The brain model ID changes from `Qwen/Qwen3-30B-A3B` to `Qwen/Qwen3.5-35B-A3B` under the same vLLM serving path — no changes to `llm.py` or base URLs required, only the `RAG_LLM_MODEL` env var. Three new sidecar services are added: `Qwen3-TTS-12Hz-1.7B` and `Qwen3-ASR-1.7B` (both served via HuggingFace `transformers` in isolated venvs), and Voxtral (served via `mistral-inference` if self-hostable, otherwise a thin `requests`-based cloud adapter). Each new service requires a separate venv because the shared `rag_demo_system` venv pins `transformers==4.37.2`, which is too old for Qwen3-series models. Benchmark instrumentation requires no new libraries — `time`, `json`, and `pytest` are already present.

**Core technologies:**
- `vLLM >=0.5.0`: serve Qwen3.5-35B-A3B — same MoE serving path as current brain, zero code change
- `transformers >=4.47.0` (isolated venvs): run Qwen3-TTS and Qwen3-ASR — HuggingFace models, cannot use vLLM for audio I/O
- `fastapi==0.115.6 + uvicorn==0.30.6`: sidecar HTTP wrapper for each new model — exact same pattern as `whisper_server.py` and `vosk_tts_server.py`
- `soundfile >=0.12.1`: PCM16/WAV encode-decode in TTS sidecar — standard audio I/O bridge
- `scipy >=1.13.0`: audio resampling in TTS sidecar if Qwen3-TTS default rate differs from 24 kHz pipeline expectation
- `mistral-inference` or `transformers`: Voxtral STT (self-host path); plain `requests` if cloud-API only
- `time` + `json` (stdlib): benchmark timing instrumentation and JSONL log — no new dependencies

**Critical version constraint:** Never upgrade the shared `rag_demo_system` venv `transformers==4.37.2` pin. It was deliberately pinned against `sentence-transformers==3.4.1`. All new model services must use per-service venvs.

See `.planning/research/STACK.md` for full version compatibility table and per-service venv install commands.

### Expected Features

**Must have (P1 — milestone is invalid without these):**
- End-to-end timing instrumentation: six milestones per voice turn (`t_audio_commit`, `t_stt_done`, `t_retrieval_done`, `t_llm_first_token`, `t_tts_first_chunk`, plus client-side `playback_started_at`) — without this no benchmark is valid
- Benchmark JSONL log format: one record per turn, machine-readable, appendable, with `run_id`, `stack_id`, `question_id`, timings, transcript, answer, and `evaluator_note`
- Fixed Russian question fixture: 80+ questions in five categories (short factual, longer factual, KB grounding, ambiguous, out-of-scope) phrased as spoken Russian
- Qwen3-TTS adapter: sidecar microservice + `voice_adapters.py` branch + `QWEN3_TTS_BASE_URL` env var
- Qwen3-ASR adapter: sidecar microservice + `voice_adapters.py` branch + `QWEN3_ASR_BASE_URL` env var
- Brain model selector: UI control + `session.update` field + backend reads selected model from session
- Per-stack env profiles: one `.env.<profile>` file per benchmark stack (baseline, qwen3_tts, qwen3_asr, brain_upgrade, omni_hybrid)
- Benchmark runner script: HTTP-based CLI that executes the full question set against a specified stack, appends to JSONL

**Should have (P2 — analytical value once P1 is working):**
- Qwen3-Omni hybrid adapter: audio-in + RAG context injection + audio-out in one model call; requires split pipeline baseline as control
- Benchmark comparison report script: markdown table comparing mean/p50/p95 across two JSONL log files
- Timing events surfaced in `response.done` WebSocket event for frontend display

**Defer (P3 / v2+):**
- Voxtral STT adapter: optional; do not block v1.0 on unconfirmed self-host availability
- Streaming TTS output to browser: separate optimization phase after batch TTS baseline is measured
- LiveKit or Pipecat migration: explicitly out of scope; cannot change transport during benchmark window
- Qwen3-Omni pure native realtime mode: only after hybrid mode passes grounding evaluation

See `.planning/research/FEATURES.md` for full feature dependency graph and benchmark question set category design.

### Architecture Approach

The architecture follows a strict sidecar-per-model pattern: each new model (Qwen3-TTS, Qwen3-ASR, Voxtral, Qwen3-Omni) runs as a standalone FastAPI process. The main backend (`app.py`) calls them by URL through `voice_adapters.py`. Provider dispatch uses a flat `if/elif` chain in `transcribe_audio()` and `synthesize_audio_with_provider()` — no class hierarchy, no plugin registry. Timing lives as per-turn fields on `VoiceSession` (reset at turn start, emitted in `response.done`), never as global state. The Qwen3-Omni hybrid path branches before the STT call in the WebSocket loop; it does not go through the standard STT/chat/TTS sequence. The benchmark runner is a standalone HTTP client that drives the backend via HTTP, never imports backend modules.

**Major components:**
1. `voice_adapters.py` (modified): dispatch layer for all STT/TTS providers; add `qwen3_tts`, `qwen3_asr`, `voxtral` branches and `build_voice_statuses()` entries
2. `voice_session.py` (modified): add five timing fields (`t_audio_commit`, `t_stt_done`, `t_retrieval_done`, `t_llm_first_token`, `t_tts_first_chunk`) plus `reset_turn_timings()` and `turn_timing_snapshot()` methods
3. `app.py` (modified): wire `time.time()` calls at each stage, add `qwen3_stack`/`voxtral_stack` pipeline slugs, add Omni hybrid branch before STT dispatch
4. `services/qwen3_tts_server.py` + `services/qwen3_asr_server.py` (new): FastAPI sidecars following `whisper_server.py` pattern
5. `qwen3_tts.py` + `qwen3_asr.py` + `voxtral.py` + `qwen3_omni.py` (new backend modules): thin HTTP clients called by `voice_adapters.py`
6. `yandex_realtime.py` (modified): expand `normalize_voice_provider()` allowlist with new slugs
7. `benchmark/` directory (new): `runner.py` (HTTP CLI), `questions.json` (fixture), `evaluator.py` (grading helpers), `results/` (gitignored)

**Build order from architecture research (dependency-ordered):** timing instrumentation → Qwen3-TTS adapter → Qwen3-ASR adapter → Voxtral adapter → benchmark runner → Qwen3-Omni hybrid → brain switch config.

See `.planning/research/ARCHITECTURE.md` for full data flow diagrams, exact code snippets for each integration point, and anti-patterns to avoid.

### Critical Pitfalls

1. **New adapter returns wrong audio format** — Qwen3-TTS may return opus frames or float32 PCM instead of the PCM16 base64 blob the frontend expects. Write a unit test asserting `audio_b64` (non-empty), `sample_rate_hz` (integer, 24000), and `provider` key before integrating. Never hardcode sample rate; read from model response.

2. **Fallback chain hides which STT provider ran** — `transcribe_audio()` falls through to SenseVoice or Whisper silently if the new adapter returns `{"text": ""}`. During benchmarking, unset `SENSEVOICE_BASE_URL` and `WHISPER_BASE_URL` to force exclusive use of the target provider. Assert `provider` field matches intended provider on every benchmark row.

3. **GPU OOM on model swap** — Qwen3.5-35B-A3B uses ~70 GB VRAM; Qwen3-Omni-30B uses ~60 GB. Never load both simultaneously on an 80 GB A100. Use `supervisorctl stop qwen` and confirm `nvidia-smi` shows headroom before launching the next model. Set `--gpu-memory-utilization 0.85` on the brain to leave room for 1.7B sidecars.

4. **Qwen3-Omni ignores injected context and halluccinates from pretraining** — Multimodal models are not fine-tuned on strict RAG instruction following. Always run the same question set through both split pipeline and Omni hybrid and compare. Require out-of-scope questions to return refusal text; if they do not, Omni grounding has failed.

5. **Timing instrumentation adds measured latency** — Synchronous `state.log` writes inside the async WebSocket handler inflate benchmark measurements. Validate that instrumentation adds under 5 ms by comparing 10 turns with/without. Use in-memory buffering and post-response flush if overhead exceeds threshold.

6. **Russian domain vocabulary quality regressions** — "Supports Russian" from documentation is not sufficient. Test Qwen3-TTS and Qwen3-ASR on leasing-specific sentences (`лизингополучатель`, `аванс`, `до 84 месяцев`) scored by a native speaker before committing either as a benchmark candidate.

7. **No benchmark warm-up** — First 3 turns per stack show 2-3x inflated latency due to KV-cache cold start. Always discard first 3 turns; flag them `"warmup": true` in JSONL or exclude from analysis.

See `.planning/research/PITFALLS.md` for full pitfall descriptions including warning signs and recovery strategies.

## Implications for Roadmap

Based on the combined research, seven phases are suggested. The ordering is driven by hard dependency chains: timing instrumentation must precede all benchmark data capture; new providers must be tested in isolation before the benchmark runner executes them; the brain upgrade and Omni experiment require the split pipeline baseline as a reference point.

### Phase 0: Timing and Benchmark Foundation

**Rationale:** Timing instrumentation and the JSONL log format are prerequisites for every subsequent phase. Without them, no benchmark run produces valid data. This phase has no external model dependencies — it only touches Python files and fixtures already in the repo.

**Delivers:** `VoiceSession` timing fields, `turn_timing_snapshot()`, `time.time()` instrumentation in `voice_ws()`, JSONL log format, 80+ question Russian fixture file, per-stack env profile loader.

**Addresses features:** Timing instrumentation (P1), Benchmark JSONL log format (P1), Russian question fixture (P1), Per-stack env profiles (P1).

**Avoids pitfalls:** Timing instrumentation overhead (validate delta < 5 ms before recording Phase 1 baseline); ensures benchmark warm-up discipline is in the runner from day one.

**Research flag:** Standard patterns — no deeper research needed. Direct code instrumentation with `time.time()` and `json.dumps()` to JSONL. Architecture doc provides exact field names and dataclass change.

### Phase 1: TTS Upgrade (Qwen3-TTS)

**Rationale:** TTS is the highest-ROI single swap per the playbook (Russian quality improvement) and has no dependency on ASR or brain changes. Isolating TTS first keeps the comparison clean: one variable changes while STT and brain stay constant.

**Delivers:** `services/qwen3_tts_server.py` sidecar, `qwen3_tts.py` backend module, `voice_adapters.py` branch for `qwen3_tts`, `QWEN3_TTS_BASE_URL` env var, supervisord `[program:qwen3_tts]` with `autostart=false`, contract tests asserting audio format shape.

**Addresses features:** Qwen3-TTS adapter (P1).

**Avoids pitfalls:** Wrong audio format (unit test before integration), streaming TTS buffering trap (measure batch approach first), Russian domain vocabulary regression (domain-specific sentence list scored by native speaker before calling Phase 1 complete), audio buffer not cleared on provider switch (integration test).

**Research flag:** Needs phase research for Qwen3-TTS exact HuggingFace model card and inference API (model repo name `Qwen3-TTS-12Hz-1.7B` unconfirmed; streaming vs. batch endpoint behavior unknown). Confirm before writing the sidecar server.

### Phase 2: STT Upgrade (Qwen3-ASR)

**Rationale:** Same isolation principle as Phase 1: swap only STT while TTS and brain remain constant. Qwen3-ASR follows identical sidecar pattern as Phase 1, so implementation is low-surprise.

**Delivers:** `services/qwen3_asr_server.py` sidecar, `qwen3_asr.py` backend module, `voice_adapters.py` branch for `qwen3_asr`, `QWEN3_ASR_BASE_URL` env var, contract tests.

**Addresses features:** Qwen3-ASR adapter (P1).

**Avoids pitfalls:** Fallback chain hiding provider identity (disable SENSEVOICE and WHISPER env vars during all benchmark runs for this phase), Russian vocabulary regression (same domain sentence test as Phase 1 but for STT WER).

**Research flag:** Needs phase research for Qwen3-ASR exact HF repo name and whether it ships a CTranslate2-compatible format (faster-whisper path) or requires raw transformers. Confirm before writing the sidecar.

### Phase 3: Benchmark Runner and Baseline Capture

**Rationale:** Once TTS and STT adapters work and timing instrumentation is in place, the benchmark runner can execute a full question set against the existing baseline stack, producing the reference JSONL log all subsequent phases will compare against.

**Delivers:** `benchmark/runner.py` CLI, `benchmark/evaluator.py`, baseline JSONL run for the `local` stack (SenseVoice + Qwen3-30B-A3B + CosyVoice), initial human-reviewed `evaluator_note` entries, optional comparison report script.

**Addresses features:** Benchmark runner script (P1), Comparison report script (P2).

**Avoids pitfalls:** No warm-up (runner enforces 3 discarded turns per stack), WiFi jitter (run over wired/localhost only), Qdrant reindex during run (lock KB before benchmark).

**Research flag:** Standard patterns — HTTP client driving existing `/api/chat` endpoint. No new research needed.

### Phase 4: Brain Upgrade (Qwen3.5-35B-A3B)

**Rationale:** Brain upgrade is an env-var change (`RAG_LLM_MODEL`) with no code change to `llm.py`. It runs after baseline capture so the delta is visible. VRAM must be checked before launch because the new model is ~70 GB bfloat16, which is the VRAM limit on an 80 GB A100 when sidecars are running.

**Delivers:** Qwen3.5-35B-A3B running on the same vLLM endpoint, `brain_upgrade` env profile, JSONL benchmark run for the upgraded brain, answer length distribution and `llm_ttfb_ms` comparison report.

**Addresses features:** Brain model selector (P1), brain upgrade benchmark phase.

**Avoids pitfalls:** GPU OOM (`nvidia-smi` check before launch; `--gpu-memory-utilization 0.85`; stop small sidecars if needed), brain upgrade changes answer length/latency (record word count distribution and TTFB before declaring upgrade successful).

**Research flag:** Standard patterns for vLLM model swap. One validation item: confirm `vllm serve Qwen/Qwen3.5-35B-A3B` succeeds on the target vLLM version (MoE support landed in 0.4.x; use 0.5.x+).

### Phase 5: Voxtral STT Adapter (Optional)

**Rationale:** Voxtral adds a second STT candidate for the benchmark matrix. It is explicitly optional per the playbook and should not block any other phase. Include here only if self-host availability is confirmed before Phase 3 completes.

**Delivers:** `voxtral.py` backend module, `voice_adapters.py` branch for `voxtral`, either a sidecar server or a cloud-API adapter (depending on self-host status), contract tests.

**Addresses features:** Voxtral STT adapter (P3).

**Avoids pitfalls:** Voxtral on-premises licensing must be verified before committing implementation resources; if API-only, use thin `requests` adapter, not a local sidecar.

**Research flag:** Needs phase research specifically for Voxtral self-host availability and Russian language quality confirmation. This is a LOW-confidence area; do not assume from documentation.

### Phase 6: Qwen3-Omni Hybrid Experiment

**Rationale:** Omni experiment is gated on having a split pipeline baseline (Phases 0-3 complete) and a brain upgrade reference (Phase 4) to compare against. It requires a separate execution path in `app.py` and must be VRAM-isolated from the split pipeline brain.

**Delivers:** `qwen3_omni.py` backend module, new `qwen3_omni_hybrid` branch in `voice_ws()` WebSocket loop, `build_voice_statuses()` entry, `normalize_voice_provider()` allowlist update, grounding test run (out-of-scope questions), JSONL comparison against split pipeline baseline.

**Addresses features:** Qwen3-Omni hybrid adapter (P2).

**Avoids pitfalls:** Omni ignores context and hallucinates (out-of-scope question grounding test required before any demo), GPU OOM (stop split pipeline brain before loading Omni), Omni on same code path as split pipeline (dedicated WS branch, not a TTS replacement).

**Research flag:** Needs phase research for Qwen3-Omni vLLM audio input support status (which vLLM version supports the Omni audio modality and what the multimodal input format looks like). LOW confidence area.

### Phase Ordering Rationale

- Timing instrumentation must be first: without it, no benchmark data is valid. Every subsequent phase depends on this foundation.
- TTS before STT: TTS is higher ROI (Russian quality gap is larger) and the change is isolated. Either could come first; TTS is prioritized per the playbook.
- Adapters before benchmark runner: the runner is meaningless if the adapters it exercises are not yet integrated.
- Brain upgrade after baseline: the JSONL baseline run from Phase 3 is the reference point; upgrading before it exists makes the comparison impossible.
- Omni last: it requires the largest VRAM allocation (cannot co-host with brain), needs the most novel implementation (new WS branch), and its value can only be assessed against a measured baseline.
- Voxtral optional and parallel: it follows the exact same pattern as Qwen3-ASR and can be developed alongside Phases 2-3 if the self-host question resolves early. It does not block anything.

### Research Flags

Phases needing deeper research during planning:

- **Phase 1 (Qwen3-TTS):** Confirm `Qwen3-TTS-12Hz-1.7B` HuggingFace repo name, batch vs. streaming inference endpoint behavior, and whether 24000 Hz is the native output sample rate. LOW confidence.
- **Phase 2 (Qwen3-ASR):** Confirm `Qwen3-ASR-1.7B` HuggingFace repo name and whether CTranslate2/faster-whisper path is available. MEDIUM confidence.
- **Phase 5 (Voxtral):** Confirm self-host availability, Russian language quality, and on-premises licensing. LOW confidence — do not implement until confirmed.
- **Phase 6 (Qwen3-Omni):** Confirm which vLLM version supports Omni audio input modality and the multimodal request format. LOW confidence.

Phases with standard, well-documented patterns (skip research-phase):

- **Phase 0 (Timing):** Pure Python instrumentation of existing `app.py` and `VoiceSession`. Architecture doc provides exact field names and code structure.
- **Phase 3 (Benchmark Runner):** HTTP client pattern against existing endpoints. No external API uncertainties.
- **Phase 4 (Brain Upgrade):** Single env var change plus a vLLM serve command. Verify vLLM version supports the model; otherwise no unknowns.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Existing stack (vLLM, FastAPI, Qdrant) is HIGH confidence from codebase. New model details (Qwen3-TTS API, Qwen3-ASR API, Voxtral self-host) are LOW; flagged per item. |
| Features | MEDIUM | Feature list and priorities derived from internal playbook (HIGH) and codebase analysis (HIGH). Russian quality claims for new models are LOW pending empirical testing. |
| Architecture | HIGH | Derived entirely from direct source reading of the repo. Integration points, code patterns, and data flow are confirmed against actual code. No external assumptions. |
| Pitfalls | MEDIUM | VRAM management and async patterns are HIGH from established knowledge. Qwen3-Omni grounding behavior and Voxtral Russian quality are LOW — no published evaluation on Russian RAG tasks found. |

**Overall confidence:** MEDIUM — the architecture and feature plan are solid; the primary uncertainties are in new model API details and quality claims that require empirical validation.

### Gaps to Address

- **Qwen3-TTS HF model card and API:** Confirm `Qwen/Qwen3-TTS-12Hz-1.7B` as the exact repo ID; confirm native sample rate; confirm whether the inference API returns a single WAV or streaming chunks. Address at start of Phase 1.
- **Qwen3-ASR faster-whisper compatibility:** Determine whether the model ships CTranslate2 weights or HuggingFace-only. Determines whether the sidecar uses `faster-whisper` or raw `transformers` pipeline. Address at start of Phase 2.
- **Voxtral self-host status:** Binary question — self-hostable weights exist or Mistral API only. Determines whether Phase 5 is a local sidecar or a thin cloud adapter. Address before Phase 5 begins (non-blocking for Phases 0-4).
- **Qwen3-Omni vLLM audio input support:** Which vLLM version supports Omni's audio input modality. Determines whether a vLLM upgrade is needed for Phase 6. Address before Phase 6 begins.
- **Russian domain vocabulary quality for Qwen3-TTS and Qwen3-ASR:** Must be empirically validated against leasing-specific sentences before either adapter is accepted as a benchmark candidate. This is a go/no-go gate, not a nice-to-have.
- **VRAM footprint under bfloat16:** Dry-run `vllm serve --dry-run` for both Qwen3.5-35B-A3B and Qwen3-Omni-30B before co-hosting decisions are finalized. Estimates in STACK.md are MEDIUM/LOW confidence.

## Sources

### Primary (HIGH confidence)

- `rag_demo_system/backend/voice_adapters.py` — existing provider dispatch pattern, sidecar contract, fallback chain behavior
- `rag_demo_system/backend/app.py` — WebSocket loop, timing pattern in HTTP chat path, `_voice_pipeline()` mapping
- `rag_demo_system/backend/voice_session.py` — session state machine, dataclass structure
- `rag_demo_system/backend/yandex_realtime.py` — `normalize_voice_provider()` allowlist pattern
- `rag_demo_system/services/whisper_server.py` + `vosk_tts_server.py` — sidecar FastAPI contract to replicate
- `rag_demo_system/tests/test_voice_adapters_official.py` — existing contract test pattern to follow
- `.planning/PROJECT.md` — milestone scope, Russian language requirement, constraints
- `docs/voice_ai_playbook_2026-03-25.md` — benchmark plan, timing milestones, question categories, model recommendations

### Secondary (MEDIUM confidence)

- vLLM release notes and documentation (training knowledge through Aug 2025) — Qwen MoE support in vLLM 0.4.x/0.5.x, KV-cache memory allocation behavior
- Qwen model family documentation (training knowledge) — `Qwen3.5-35B-A3B` capabilities, Russian multilingual text quality, Qwen3-ASR/TTS general architecture

### Tertiary (LOW confidence)

- `github.com/QwenLM/Qwen3-TTS` (cited in playbook, not independently verified in this session) — Russian language support claim, `12Hz` streaming encoding
- `github.com/QwenLM/Qwen3-ASR-Toolkit` (cited in playbook) — Qwen3-ASR-1.7B Russian support claim
- `mistral.ai/news/voxtral-transcribe-2` (cited in playbook) — Voxtral Realtime Russian capability; self-host availability unknown
- Qwen3-Omni hybrid mode grounding behavior — no published evaluation on Russian RAG tasks found; treat as empirically unknown

---
*Research completed: 2026-03-25*
*Ready for roadmap: yes*
