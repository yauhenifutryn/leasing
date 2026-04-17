# Spec 3: Classifier / SessionAgent Latency

**Cluster:** C — Performance
**Depends on:** —
**Blocks:** —

## Context

The classifier LLM call (`app.py:703-798`) often takes longer than the main
response LLM call, which is counter-intuitive: it produces only 80 tokens versus
the main response's several hundred. User observed this in production logs.

Memory `project_perf_parallel_classifier_rag.md` notes classifier + RAG are
already parallelized (saves ~300ms), but classifier latency itself remains the
bottleneck.

## Problem

Root cause analysis:

1. **Scheduler contention.** Classifier and main response share the same vLLM
   server hosting Qwen3.5-35B-A3B-FP8. When main LLM is streaming output, the
   classifier call queues behind it. Visible in latency as bursty spikes.

2. **Full 35B cost for a trivial task.** The classifier is doing simple
   extraction that a 4B model handles well. Paying 35B inference cost for
   classification is wasteful.

3. **Unknown prefix cache status.** vLLM has prefix caching on by default,
   but we have never validated that the classifier system prompt is actually
   being cached. If the prompt varies slightly per call (e.g., dynamic fields
   in the system role), cache is missed.

## Goals

- Validate vLLM prefix caching hit rate for SessionAgent system prompt.
- Deploy Qwen3-4B-Instruct in FP8 on a separate vLLM instance (port 8788).
- Route SessionAgent calls exclusively to the small model.
- Keep main LLM on port 8787 (unchanged).
- Measure p50/p95 SessionAgent latency before vs after.
- Target: p50 < 150ms (from current ~300ms), p95 < 400ms.

## Non-goals

- Streaming classifier first-token routing (rejected in brainstorm — fragile)
- Two-stage classifier (rejected — latency tax)
- Speculative main-LLM start (rejected — resource waste + race conditions)
- CPU fallback via llama-cpp-python (plan B, only if GPU fit fails)

## Design

### GPU budget (H100 80GB)

| Component | GPU memory | Status |
|---|---|---|
| vLLM main (Qwen3.5-35B-A3B-FP8, util 0.55) | ~44 GB | reduced from 0.60 |
| Whisper large-v3 (float16) | ~3 GB | unchanged |
| Silero TTS v5 | ~1 GB | unchanged |
| OS / driver / buffers | ~2 GB | unchanged |
| **vLLM small (Qwen3-4B-Instruct-FP8, util 0.08)** | **~6 GB** | **new** |
| Total committed | ~56 GB | well under 80 GB |
| Free margin | ~24 GB | headroom for KV cache growth |

### vLLM deployment

**New command (add to `regenerate_env_and_restart.sh` or supervisor config):**

```
./.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-FP8 \
  --port 8788 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.08 \
  --enable-prefix-caching \
  --download-dir /workspace/models
```

**Main instance update:** reduce `--gpu-memory-utilization` from 0.60 to 0.55
(leaves clean room for small model).

### Prefix cache validation

Pre-deploy test: fire 10 identical SessionAgent calls with varying user message
only. Query vLLM metrics endpoint (`/metrics`). Expect `gpu_cache_usage` to grow
on first call, hold stable on repeated calls. If prefix cache miss-rate > 10%,
audit the prompt — system role text must be byte-identical across calls.

**Prompt stability check:**
- System prompt must be a module-level constant, not f-string with dynamic
  values.
- All dynamic data (conversation history, tool history, new message) goes in
  the `user` role only.

### Code changes

**`rag_demo_system/.env.example`:**
```
SESSIONAGENT_BASE_URL=http://127.0.0.1:8788/v1
SESSIONAGENT_MODEL=Qwen/Qwen3-4B-Instruct-FP8
```

**`rag_demo_system/backend/config.py`:**
```python
session_agent_base_url: str = Field(default="http://127.0.0.1:8788/v1")
session_agent_model: str = Field(default="Qwen/Qwen3-4B-Instruct-FP8")
```

Fall back to main URL if session agent env missing (graceful degradation).

**`rag_demo_system/backend/app.py`:**
- Extract the classifier call block (lines 703-754) into a helper:
  `async def _run_session_agent(message, chat_session, session) -> dict`.
- Use `settings.session_agent_base_url` and `settings.session_agent_model`.
- Temperature 0.0, max_tokens 200 (up from 80 to accommodate richer schema
  from Spec 2).
- Timeout 3s; on timeout, fall back to keyword heuristic (current fallback).

### Measurement

Add Prometheus-style metrics in `app.py`:
- `session_agent_latency_ms` histogram (p50, p95).
- `session_agent_cache_hit_ratio` gauge (polled from vLLM `/metrics`).
- Log line per call: `[SessionAgent] latency_ms=X cache_hit=Y`.

Baseline capture: before deploying small model, run 50 calls on current 35B
setup, record latencies. After deploy, rerun same 50 calls. Report delta.

## Files to change

- `rag_demo_system/.env.example`
- `rag_demo_system/backend/config.py`
- `rag_demo_system/backend/app.py` (extract session agent helper, new URLs)
- `rag_demo_system/scripts/regenerate_env_and_restart.sh` (second vLLM command)
- `rag_demo_system/scripts/provision_server.sh` (start second vLLM on fresh server)
- Optional: `rag_demo_system/scripts/restart_all.sh` (ensure both instances start)

## Testing

**Pre-deploy (local or dev server):**
1. Boot both vLLM instances; verify `/health` on each.
2. Fire classifier request to 8788 with Qwen3-4B model name; assert JSON output
   parses correctly.
3. Fire 10 identical requests; assert p95 latency < 150ms and prefix cache hits
   registered.

**Deploy validation:**
1. After restart, call `/health` on 8788 returns 200.
2. GPU memory: `nvidia-smi` shows ≤ 60 GB used.
3. Run a real call via SIP; grep logs for `[SessionAgent] latency_ms=`; p50 < 150ms.
4. Cross-check main LLM latency unchanged (no regression from util reduction).

**Regression tests:**
- SessionAgent schema parity: same fields extracted from 4B vs 35B on 20 reference
  utterances. Target ≥ 95% agreement.
- For any disagreements, prefer correctness check — if 4B is wrong on a
  particular pattern, add an example to the prompt.

## Risks

| Risk | Mitigation |
|---|---|
| Qwen3-4B worse at Russian extraction than 35B | Pre-deploy parity test on 20 utterances; if < 95% agreement, try Qwen3-7B (~11 GB) |
| GPU OOM on combined load | Main util lowered to 0.55; if still tight, cut to 0.50 (loses some KV cache but model loads fine) |
| Both vLLM instances compete for scheduling | Independent Python processes, no CUDA contention at inference level; verified empirically post-deploy |
| Prefix cache doesn't activate | Explicit `--enable-prefix-caching`; prompt auditing in config |

## Rollback

Env var `SESSIONAGENT_BASE_URL=http://127.0.0.1:8787/v1` (point back to main).
Model name unchanged. Second vLLM instance stays up but receives no traffic.

## Plan B (if GPU fit fails)

Run Qwen3-1.7B-Instruct on CPU via llama-cpp-python:
- Quantize to Q4_K_M (~1.1 GB on disk, ~2 GB RAM).
- Expected latency: 200-350ms CPU-bound.
- Zero GPU impact.
- Trade-off: slower than GPU 4B but still beats current contention.
