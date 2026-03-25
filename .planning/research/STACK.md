# Stack Research

**Domain:** Voice assistant — new provider adapters and benchmark infrastructure
**Researched:** 2026-03-25
**Confidence:** MEDIUM (web tools unavailable; findings based on codebase analysis + training knowledge through Aug 2025; flag items marked LOW for external validation)

---

## Scope Reminder

This document covers only the NEW stack additions for this milestone. The following are already validated and must not be changed:

- FastAPI + WebSocket transport
- Split pipeline: STT -> RAG -> LLM brain -> TTS
- `voice_adapters.py` abstraction (HTTP microservice contract)
- Qdrant + BM25 hybrid search
- vLLM for brain (OpenAI-compatible API)
- Supervisor process management
- Docker for Qdrant

---

## Recommended Stack

### Brain Upgrade: Qwen3.5-35B-A3B

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| vLLM | >=0.5.0 (latest stable) | Serve `Qwen3.5-35B-A3B` via OpenAI-compatible API | Same serving path as current `Qwen3-30B-A3B`; zero change to `llm.py` or settings; MoE architecture is natively supported by vLLM |
| Qwen3.5-35B-A3B | HF: `Qwen/Qwen3.5-35B-A3B` | LLM brain replacement | Direct upgrade from current model; same MoE family; stronger Russian text generation; same vLLM launch flags |

**Serving configuration delta from current:**

```bash
# Only the model ID changes; everything else stays the same
vllm serve Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8001
```

**Env vars: no change required.** `RAG_LLM_MODEL` switches to `Qwen/Qwen3.5-35B-A3B`; `RAG_LLM_BASE_URL` stays the same.

**VRAM requirement:** ~20–24 GB at bfloat16 for 3B active params with 35B total. Fits comfortably on H100 NVL 94 GB alongside other services. **Confidence: MEDIUM** — verify exact bfloat16 footprint with `vllm serve --dry-run` on target hardware before committing to co-hosting layout.

---

### TTS Adapter: Qwen3-TTS

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| transformers | >=4.47.0 | Load and run `Qwen3-TTS` inference | Qwen3-TTS is distributed as a HuggingFace `transformers`-compatible model; vLLM does not handle audio output generation |
| soundfile | >=0.12.1 | Write WAV output from float32 audio arrays | Standard numpy-to-WAV bridge; already used in the broader Python audio ecosystem here |
| scipy | >=1.13.0 | Resample audio if TTS sample rate != 24 kHz expected by the pipeline | Needed only if Qwen3-TTS default sample rate differs from `COSYVOICE_SAMPLE_RATE_HZ` already used |
| fastapi + uvicorn | already pinned (0.115.6 / 0.30.6) | Wrap inference as HTTP microservice matching `/health` + `/speak` contract | Follows existing pattern in `whisper_server.py` and `vosk_tts_server.py` |

**Model:** `Qwen/Qwen3-TTS` (HuggingFace). The documented variant name is `Qwen3-TTS-12Hz-1.7B` per playbook. **Confidence: MEDIUM** — official HF repo name must be confirmed; the playbook references `Qwen3-TTS-12Hz-1.7B` specifically. Pull the correct ID from `https://huggingface.co/Qwen` before writing the service.

**Russian support:** Explicitly confirmed in playbook sourced from `github.com/QwenLM/Qwen3-TTS`. **Confidence: MEDIUM** — flag for quality validation: run a 10-sentence Russian test set during Phase 1 benchmark before committing it as the default.

**Service contract to implement** (must match `synthesize_audio_with_provider` in `voice_adapters.py`):

```
POST /speak
Body: {"text": str, "session_id": str, "language": "ru"}
Response: {"audio_b64": str, "sample_rate_hz": int, "provider": "qwen3_tts", "session_id": str}

GET /health
Response: {"ok": true, "provider": "qwen3_tts"}
```

**Provider name for `_voice_pipeline()`:** `qwen3_tts` (maps to TTS slot; STT slot stays separate).

**Env var:** `QWEN3_TTS_BASE_URL` (follows existing naming convention `{PROVIDER_UPPER}_BASE_URL`).

**Supervisor program name:** `qwen3_tts` (add to `supervisord.conf` following the `cosyvoice` pattern with `STACK_QWEN3_TTS_CMD`).

**VRAM requirement:** ~4–6 GB for the 1.7B model. Can co-host with brain on H100 94 GB. **Confidence: LOW** — no official bfloat16 memory profile found; estimate based on model size class.

---

### STT Adapter: Qwen3-ASR

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| transformers | >=4.47.0 | Load `Qwen3-ASR-1.7B` for transcription | Qwen3-ASR is a HuggingFace encoder-decoder ASR model; same pattern as whisper-family models |
| fastapi + uvicorn | already pinned | Wrap as HTTP microservice matching `/health` + `/transcribe` contract | Follows `whisper_server.py` exactly |

**Model:** `Qwen/Qwen3-ASR` (HuggingFace). Playbook references `Qwen3-ASR-1.7B`. **Confidence: MEDIUM** — confirm exact HF repo name before writing the service. The playbook source is `github.com/QwenLM/Qwen3-ASR-Toolkit`.

**Service contract to implement** (must match `transcribe_audio` in `voice_adapters.py`):

```
POST /transcribe
Body: {"audio_b64": str, "session_id": str, "language": "ru", "sample_rate_hz": 24000}
Response: {"ok": true, "provider": "qwen3_asr", "session_id": str, "text": str}

GET /health
Response: {"ok": true, "provider": "qwen3_asr"}
```

**Provider name for `transcribe_audio()`:** `qwen3_asr` (goes into the `order` list alongside `sensevoice` and `whisper`).

**Env var:** `QWEN3_ASR_BASE_URL`.

**Supervisor program name:** `qwen3_asr` with `STACK_QWEN3_ASR_CMD`.

**VRAM requirement:** ~4–6 GB for the 1.7B model. **Confidence: LOW** — same caveat as TTS.

**NOTE on `faster-whisper` vs raw `transformers`:** If Qwen3-ASR ships a CTranslate2-compatible format (like Whisper), prefer `faster-whisper` or `ctranslate2` for lower latency. If it ships as a standard HuggingFace checkpoint only, use raw `transformers`. Verify against the official release before writing the adapter. The existing `whisper_server.py` uses `faster-whisper`; matching that pattern reduces code duplication.

---

### STT Adapter: Voxtral

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| mistral-inference or transformers | latest stable | Serve Voxtral for Russian ASR | Voxtral is Mistral's Whisper-architecture STT; follows the same HTTP microservice pattern |
| fastapi + uvicorn | already pinned | Wrap as HTTP microservice | Same `/health` + `/transcribe` contract as above |

**Model:** Mistral Voxtral Realtime. **Confidence: LOW** — playbook cites `mistral.ai/news/voxtral-transcribe-2` as source; as of training cutoff, Voxtral was announced but integration details were not publicly stable. Before implementing: verify whether Voxtral is self-hostable via HuggingFace weights or requires Mistral API only. If API-only, the adapter calls `https://api.mistral.ai` rather than a local microservice.

**Russian support:** Playbook includes it as a Russian STT benchmark candidate. **Confidence: LOW** — must be empirically validated in Phase 3. Do not assume quality without a test run.

**Provider name:** `voxtral`. **Env var:** `VOXTRAL_BASE_URL` (or `VOXTRAL_API_KEY` if cloud-API mode).

**Supervisor program name:** `voxtral` with `STACK_VOXTRAL_CMD` (only needed if self-hosted).

**Fallback position:** If Voxtral is not self-hostable, implement it as a cloud-API adapter (similar to `yandex_speechkit.py`) rather than a local microservice. Same external contract; different internal call.

---

### Experimental: Qwen3-Omni Hybrid Mode

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| transformers | >=4.47.0 with audio extras | Load `Qwen3-Omni-30B-A3B-Instruct` for audio-in text-out (hybrid mode only, not pure realtime) | Qwen3-Omni is a multimodal model; audio-in text-out inference uses standard HuggingFace pipeline; vLLM does not yet support multimodal audio input for this model family as of training cutoff |
| librosa or soundfile | >=0.12.1 | Decode incoming audio to float32 array for model input | Standard audio preprocessing for transformer audio encoders |
| fastapi + uvicorn | already pinned | Wrap inference as HTTP endpoint called by the backend | Backend sends audio + retrieved chunks; model generates text answer |

**Model:** `Qwen/Qwen3-Omni-30B-A3B-Instruct` (HuggingFace). **Confidence: MEDIUM** — playbook confirms this is the target; HF repo existence as of playbook date assumed correct.

**Integration mode for first implementation:** Hybrid only. The backend:
1. Receives audio from client WebSocket
2. Calls existing STT (or skips it and passes raw audio bytes to the Omni service)
3. Runs retrieval as usual
4. Sends `{audio_b64, retrieved_chunks}` to the Omni service
5. Omni service injects chunks into prompt context, runs inference, returns text
6. Text goes to TTS as usual

This is NOT a native audio-out path. Pure audio-out (`Qwen3-Omni` generating speech directly) is explicitly out of scope for this milestone.

**Service contract:**

```
POST /omni/chat
Body: {
  "audio_b64": str,            # optional; may be null if text transcript provided
  "transcript": str | null,    # alternative to audio; use whichever is available
  "retrieved_chunks": list[str],
  "system_prompt": str,
  "session_id": str
}
Response: {"text": str, "provider": "qwen3_omni", "session_id": str}

GET /health
Response: {"ok": true, "provider": "qwen3_omni"}
```

**Env var:** `QWEN3_OMNI_BASE_URL`.

**Supervisor program name:** `qwen3_omni` with `STACK_QWEN3_OMNI_CMD`.

**VRAM requirement:** 30B MoE with 3B active params. Estimate ~20–25 GB bfloat16. **Confidence: LOW** — must be measured. Cannot safely co-host with the brain on a single 94 GB H100 without profiling first. Plan for alternating: run Omni experiments separately from the split pipeline baseline.

---

### Benchmark Instrumentation

No new library required. The existing `app.py` already captures `timings: dict[str, Any]` in the `/api/chat` handler. What is needed is a disciplined extension of this pattern to the voice WebSocket path.

| Addition | Purpose | Implementation |
|----------|---------|----------------|
| `time.perf_counter()` timestamps | Per-stage latency capture | Add to `app.py` voice handler at STT call, retrieval call, LLM first token, TTS call, audio sent |
| Structured log line (JSON to stdout) | Machine-readable benchmark output | `json.dumps({"event": "benchmark_turn", "session_id": ..., "stack_id": ..., "timings": {...}})` |
| `STACK_ID` env var | Tag which voice stack combination is active | Read in `app.py` and include in benchmark log |
| `pytest` fixtures with fixed question set | Reproducible benchmark runs | Add `rag_demo_system/tests/benchmark/` directory with question JSONL and a runner script |

**Python dependency additions for benchmark:** None. `time`, `json`, `uuid` are stdlib. `pytest` is already pinned at `8.3.4`.

**Benchmark log schema (append to `.state/benchmark.jsonl`):**

```json
{
  "ts": "ISO8601",
  "stack_id": "qwen35_qwen3tts_qwen3asr",
  "session_id": "uuid",
  "question_id": "q001",
  "stt_ms": 210,
  "retrieval_ms": 45,
  "llm_first_token_ms": 380,
  "tts_ms": 190,
  "total_perceived_ms": 825,
  "transcript": "...",
  "answer": "...",
  "provider_stt": "qwen3_asr",
  "provider_tts": "qwen3_tts",
  "provider_brain": "qwen/qwen3.5-35b-a3b"
}
```

---

## Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| transformers | >=4.47.0 | Load Qwen3-TTS, Qwen3-ASR, Qwen3-Omni | Required for all three new model adapters |
| soundfile | >=0.12.1 | PCM16/WAV encode-decode for TTS output | Required in Qwen3-TTS service |
| scipy | >=1.13.0 | Audio resampling | Only if Qwen3-TTS default sample rate differs from 24 kHz |
| librosa | >=0.10.2 | Audio loading and preprocessing | Qwen3-Omni service if it needs mel spectrogram input |
| accelerate | >=0.30.0 | Multi-GPU / device_map inference for large models | Required when loading Qwen3-Omni on multi-GPU or CPU offload |

**Note on `transformers` version:** The existing `requirements.txt` pins `transformers==4.37.2`. This is too old for Qwen3-series models. The new model services run in their own venv or Docker image — do not upgrade the shared `rag_demo_system` venv `transformers` pin without auditing downstream compatibility with `sentence-transformers==3.4.1` and `reranker`.

---

## Installation

The correct pattern for this repo is a separate venv or requirements file per model service, matching the existing split:

```bash
# Qwen3-TTS service venv
python -m venv .venv-qwen3-tts
source .venv-qwen3-tts/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.47.0 soundfile>=0.12.1 scipy>=1.13.0 \
            fastapi==0.115.6 uvicorn==0.30.6 accelerate>=0.30.0

# Qwen3-ASR service venv
python -m venv .venv-qwen3-asr
source .venv-qwen3-asr/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.47.0 fastapi==0.115.6 uvicorn==0.30.6 accelerate>=0.30.0
# If faster-whisper compatible format is confirmed:
# pip install faster-whisper>=1.1.1

# Qwen3-Omni service venv (heavy; separate to avoid conflicts)
python -m venv .venv-qwen3-omni
source .venv-qwen3-omni/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.47.0 soundfile>=0.12.1 librosa>=0.10.2 \
            fastapi==0.115.6 uvicorn==0.30.6 accelerate>=0.30.0

# Voxtral service venv (if self-hosted)
python -m venv .venv-voxtral
source .venv-voxtral/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install mistral-inference fastapi==0.115.6 uvicorn==0.30.6
# OR if HuggingFace weights:
# pip install transformers>=4.47.0 fastapi==0.115.6 uvicorn==0.30.6
```

**Requirements files to create** (following `requirements-voice-oss.txt` pattern):

```
rag_demo_system/requirements-qwen3-tts.txt
rag_demo_system/requirements-qwen3-asr.txt
rag_demo_system/requirements-qwen3-omni.txt
rag_demo_system/requirements-voxtral.txt  # pending Voxtral self-host confirmation
```

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| `transformers` for Qwen3-TTS | vLLM for TTS | vLLM generates text tokens; audio waveform synthesis is not a text generation task; vLLM cannot serve TTS models |
| `transformers` for Qwen3-ASR | vLLM for ASR | Same reason: ASR decodes audio to text but the input is audio, not a chat prompt; vLLM's input pipeline does not handle raw audio |
| Separate microservices per new provider | Monolithic single service combining all adapters | Existing architecture uses separate services per provider; monolith would break the supervisor-based profile switching in `stack_cli.py` |
| Hybrid Omni mode first | Pure realtime Omni mode first | Weaker RAG control in pure mode; playbook and architecture decision are explicit that hybrid is the correct first experiment |
| `soundfile` for WAV I/O | `librosa` for WAV I/O | `soundfile` is lower-level and sufficient for simple WAV encode/decode; `librosa` is heavier and only needed when mel spectrograms or resampling chains are required |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| vLLM for TTS or ASR | vLLM handles text-token generation only; audio output/input requires `transformers` or dedicated serving libraries | `transformers` + custom FastAPI microservice |
| Upgrading shared `rag_demo_system` `transformers==4.37.2` pin | Breaks `sentence-transformers==3.4.1` compatibility (its dependency graph was pinned deliberately) | Separate venv per new model service |
| Pipecat or LiveKit in this milestone | Explicitly out of scope per playbook; adds transport-layer complexity before baseline benchmarks exist | Keep existing FastAPI + WebSocket |
| Speaches as a drop-in STT/TTS server | Adds an intermediary layer that obscures model-level latency in benchmarks | Direct model microservices with structured timing logs |
| `mistral-inference` before confirming Voxtral self-host status | Voxtral may be API-only; installing heavy inference deps for a cloud-API model wastes GPU RAM and adds unnecessary complexity | Confirm self-host availability first; use thin `requests`-based adapter if cloud API |
| CosyVoice for Russian TTS going forward | Already in codebase but Chinese-primary model; Russian quality is unverified and inconsistent | `Qwen3-TTS` (Russian explicitly supported) |

---

## Integration with Existing `voice_adapters.py`

The existing adapter uses two lookup mechanisms:

1. `_voice_pipeline(voice_provider)` in `app.py` maps a provider name to `(stt_name, tts_name)`.
2. `transcribe_audio(audio_b64, session_id, preferred=...)` iterates a fallback order.
3. `synthesize_audio_with_provider(text, session_id, preferred=...)` switches on provider name.

**Required changes to `voice_adapters.py`:**

- Add `qwen3_tts` branch to `synthesize_audio_with_provider`: reads `QWEN3_TTS_BASE_URL`, POSTs to `/speak`.
- Add `qwen3_asr` to the fallback list in `transcribe_audio` and the `_voice_pipeline` map.
- Add `voxtral` branch (either microservice or cloud-API path).
- Add `qwen3_tts`, `qwen3_asr`, `voxtral` to `build_voice_statuses()` (reads `QWEN3_TTS_BASE_URL`, `QWEN3_ASR_BASE_URL`, `VOXTRAL_BASE_URL`).

**Required changes to `app.py` `_voice_pipeline()`:**

```python
# Add these cases
if voice_provider == "qwen3":
    return ("qwen3_asr", "qwen3_tts")
```

**Required changes to `supervisord.conf`:** Add `[program:qwen3_tts]`, `[program:qwen3_asr]`, `[program:voxtral]`, `[program:qwen3_omni]` following the existing `STACK_*_CMD` pattern.

**Required changes to `stack_cli.py`:**
- Add `qwen3_tts`, `qwen3_asr`, `voxtral`, `qwen3_omni` to `OPTIONAL_PROGRAM_ENV`.
- Add a new voice profile `qwen3` to `VOICE_PROFILES` that activates `qwen3_asr` + `qwen3_tts`.
- Add `qwen3_omni` profile activating the Omni service.

---

## Stack Patterns by Variant

**If running split pipeline benchmark (Phases 1–3):**
- Brain: vLLM serving `Qwen3.5-35B-A3B` on port 8001
- STT: One of `qwen3_asr`, `voxtral`, or existing `whisper`/`sensevoice`
- TTS: `qwen3_tts` or `yandex_speechkit`
- Env: `STACK_VOICE_PROFILE=qwen3` or `STACK_VOICE_PROFILE=local`

**If running Omni experiment (Phase 5):**
- Stop brain vLLM to free VRAM
- Start `qwen3_omni` service on its own port
- Backend routes to `qwen3_omni` microservice instead of split STT/brain/TTS path
- Keep Qdrant and retrieval running; inject chunks into Omni prompt

**If running baseline control (Phase 0):**
- No new services needed; use existing `stack_cli.py up` with `STACK_VOICE_PROFILE=local`

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| transformers>=4.47.0 (new model services) | Must be isolated from transformers==4.37.2 (main rag venv) | Separate venv per service; never pip install into shared venv |
| torch (CUDA 12.1 wheels, nightly for RTX 5090) | All new model services should match the existing PyTorch CUDA channel | Makefile already pins nightly for RTX 5090; apply same flag when building new venvs on same host |
| fastapi==0.115.6 + uvicorn==0.30.6 | Already pinned in all sub-requirements files | Pin to same version in all new `requirements-*.txt` files for consistency |
| vLLM >=0.5.0 | Qwen3.5-35B-A3B | Qwen MoE support landed in vLLM 0.4.x series; 0.5.x is safer; verify against vLLM release notes before deploy |

---

## Sources

- Codebase analysis: `voice_adapters.py`, `app.py`, `whisper_server.py`, `vosk_tts_server.py`, `stack_cli.py`, `supervisord.conf`, `requirements.txt`, `requirements-voice-oss.txt`, `requirements-voice-fallback.txt` — HIGH confidence (primary source)
- `docs/voice_ai_playbook_2026-03-25.md` — HIGH confidence (internal authoritative playbook, verified against codebase)
- vLLM Qwen MoE support: training knowledge through Aug 2025 — MEDIUM confidence; verify `vllm serve Qwen/Qwen3.5-35B-A3B` works on target vLLM version
- Qwen3-TTS transformers-based inference pattern: training knowledge — MEDIUM confidence; confirm HF model card before writing adapter
- Qwen3-ASR inference pattern: training knowledge — MEDIUM confidence; confirm whether faster-whisper or raw transformers is the correct path
- Voxtral self-host availability: LOW confidence; playbook cites announcement page, not confirmed self-host weights

---

*Stack research for: Voice provider adapters and benchmark infrastructure milestone*
*Researched: 2026-03-25*
