# Phase 3: Brain Upgrade and Benchmark Framework - Research

**Researched:** 2026-03-25
**Domain:** Python benchmark tooling, vLLM model routing, WebSocket test automation, LLM streaming instrumentation
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Question Fixture Design**
- D-01: Claude generates the 80+ Russian test questions by reading the knowledge base files. User reviews and edits before the fixture is finalized.
- D-02: Fixture format is JSONL. One JSON object per line with fields: question_id (category prefix + number, e.g. sf-01, lf-01, kb-01, amb-01, oos-01), category, text_ru, expected_keywords (list of 2-5 Russian terms the answer should contain).
- D-03: Five categories: short_factual, long_factual, kb_grounded, ambiguous, out_of_scope. Question IDs use prefixes sf, lf, kb, amb, oos respectively.

**Benchmark Runner Behavior**
- D-04: Runner communicates with the backend via WebSocket, same as the browser. Full pipeline testing: STT -> RAG -> LLM -> TTS. Question text is synthesized into audio and sent through the real voice path.
- D-05: First 3 turns per benchmark run are flagged as warmup=true in the JSONL output. Comparison script excludes warmup turns from averages.
- D-06: On failure (timeout, disconnect), the runner logs the error in the JSONL line with error field and null timings, then continues to the next question. No retry, no full-stop.

**Comparison Script Output**
- D-07: Output is a markdown table. Columns: metric name, Stack A values, Stack B values, delta. Rows: primary KPI (mean/p50/p95), LLM TTFB (mean/p50/p95), keyword hit rate, error count.
- D-08: Winners are highlighted per metric row (arrow or marker showing which stack is better on each dimension).

**Env Profile Structure**
- D-09: Profiles are flat files named .env.bench.{name} in rag_demo_system/. Follows the existing .env.voice.{name} convention.
- D-10: Profiles are incremental overrides. Runner loads base .env first, then overlays .env.bench.{name}. Each profile contains only the variables that differ from baseline.
- D-11: All 7 profiles created in this phase: baseline, qwen3_tts, qwen3_asr, voxtral, brain_upgrade, omni_hybrid, dify_rag. The omni_hybrid profile is a placeholder until Phase 4 builds the adapter.

**LLM First-Token Timing**
- D-12: Phase 3 resolves the TODO at app.py:770 by extracting the real t_llm_first_token from the streaming response using the existing iter_openai_stream_events/iter_openai_stream_text helpers in llm_stream.py.

### Claude's Discretion
- Benchmark runner CLI argument design (flags, defaults, help text)
- JSONL result schema field ordering and naming beyond the required fields
- Comparison script internal implementation (how it computes percentiles)
- How the runner synthesizes question text into audio for WebSocket submission
- Profile variable names and values (following existing env var conventions)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BRAIN-01 | Brain model switchable between Qwen3-30B-A3B (fallback) and Qwen3.5-35B-A3B (target) via UI selector or env var | app.py routing gap identified: ChatRequest lacks brain_model field; voice handler must pass session.brain_model to LLM call |
| BENCH-01 | Fixed Russian test question set with 80+ questions across 5 categories | KB content read; 5-category structure defined; JSONL format with question_id/category/text_ru/expected_keywords confirmed |
| BENCH-02 | Benchmark runner executes full question set against active configuration and writes JSONL results | websockets 15.0.1 available; asyncio confirmed; audio synthesis strategy identified |
| BENCH-03 | Each result includes question_id, stack_id, transcript, answer, retrieved chunks, timing breakdown | Log schema from existing voice_turn log covers all fields; runner captures from WebSocket events |
| BENCH-04 | Comparison script shows side-by-side latency and quality metrics | statistics.quantiles() confirmed for p50/p95; markdown output strategy clear |
| DEPLOY-01 | Env profile files for each benchmark stack (7 profiles) | Existing .env.voice.{name} convention understood; incremental override pattern confirmed |
</phase_requirements>

---

## Summary

Phase 3 has three distinct workstreams that are independent enough to plan as separate waves: (1) brain model routing fix, (2) LLM first-token timing fix, and (3) the benchmark toolchain. All three involve Python code only; no new services, no new dependencies beyond what is already installed.

The most architecturally significant finding is that BRAIN-01 has a gap that is not obvious from the CONTEXT.md description. The UI selector and session.brain_model field already exist (from Phase 1), but the actual vLLM inference call in `chat()` at app.py:397 uses `settings.llm.fast_model`, which is loaded from environment at process startup. The `ChatRequest` dataclass has no `brain_model` field. As a result, switching the brain model via the UI currently only changes the log/stack_id label, not the actual model called. Closing this gap requires: (a) adding `brain_model: str | None` to `ChatRequest`, (b) having the voice WebSocket handler pass `session.brain_model` to `ChatRequest`, and (c) having `chat()` prefer the per-request `brain_model` over the env setting when the field is present.

The first-token timing fix (D-12) is a localized change at app.py:770. The voice handler currently calls `chat()` with `stream=False`, which uses the synchronous `call_openai_compatible` path that never records a first-token timestamp. To get real t_llm_first_token, the voice handler must switch to calling a streaming-capable function (or a dedicated helper) that records `time.time()` at the first yielded token before collecting the full answer. The existing streaming path in `chat()` already computes `first_token_at` using `time.perf_counter()` -- the voice handler needs a similar pattern that returns the first-token wall-clock time back to the caller.

The benchmark toolchain (runner + comparison script + fixture) is pure new-file work with no changes to existing backend code. The `websockets` library (v15.0.1) is available. The `statistics` stdlib module provides `quantiles()` for p50/p95 without numpy. The `stack_cli.py` already has a `benchmark` command stub (line 183-185) ready to be wired up.

**Primary recommendation:** Plan three focused waves: Wave 1 fixes BRAIN-01 routing and D-12 timing in app.py. Wave 2 creates the 7 env profiles and the benchmark fixture. Wave 3 implements the runner CLI and comparison script.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| websockets | 15.0.1 (installed) | WebSocket client for benchmark runner | Same transport the browser uses; already installed |
| asyncio | stdlib | Async event loop for runner | Required by websockets v15 client API |
| statistics | stdlib | p50/p95 percentile computation | `statistics.quantiles(data, n=100, method='inclusive')` — no extra install |
| argparse | stdlib | CLI argument parsing for runner | Standard Python CLI; no extra install |
| json / jsonlines | stdlib json | JSONL fixture and results files | One JSON object per line; stdlib only |
| pathlib | stdlib | File paths for profiles, fixture, results | Project convention |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 2.1.3 (installed) | Percentile alternative | Acceptable fallback if statistics.quantiles is too slow on large datasets; not needed for 80-question runs |
| python-dotenv | installed (confirmed via .env loading) | Loading .env.bench.{name} overlays | Load base .env then overlay profile |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| statistics.quantiles | numpy.percentile | numpy is heavier; statistics is sufficient for N<=500 per stack |
| argparse | click | click adds a dependency; argparse is zero-cost |
| websockets async | httpx WebSocket | httpx WS is less documented; websockets is the standard |

**Installation:** No new packages required. All tools are in stdlib or already installed.

**Version verification:**
- `websockets`: 15.0.1 (confirmed via `python -c "import websockets; print(websockets.__version__)"`)
- `statistics.quantiles`: available in Python 3.8+; running Python 3.13.5

---

## Architecture Patterns

### Recommended Project Structure
```
rag_demo_system/
├── scripts/
│   ├── benchmark_runner.py    # new: CLI runner
│   └── benchmark_compare.py   # new: comparison script
├── fixtures/
│   └── bench_questions_ru.jsonl  # new: 80+ question fixture
├── .env.bench.baseline        # new: 7 env profiles
├── .env.bench.qwen3_tts
├── .env.bench.qwen3_asr
├── .env.bench.voxtral
├── .env.bench.brain_upgrade
├── .env.bench.omni_hybrid
├── .env.bench.dify_rag
└── backend/
    └── app.py                 # modified: brain_model routing + first-token fix
```

### Pattern 1: Brain Model Routing via ChatRequest

**What:** Add `brain_model: str | None = None` to `ChatRequest`. In `chat()`, resolve the effective model as:
```python
effective_model = payload.brain_model or (settings.llm.fast_model if fast else settings.llm.model)
```
Then pass `effective_model` to both `call_openai_compatible` and `iter_openai_compatible_stream_events`.

**When to use:** Any call that should honor a per-session brain model selection.

**Example:**
```python
# In the voice WebSocket handler, where chat() is called:
response = await chat(
    ChatRequest(
        message=text,
        session_id=session_id,
        stream=False,
        fast=True,
        mode="voice_fast",
        backend=session.backend,
        brain_model=session.brain_model,   # NEW field
    ),
    stream=False,
)
```

### Pattern 2: Real First-Token Timing in Voice Path (D-12)

**What:** The voice handler calls `chat()` with `stream=False`. To get real `t_llm_first_token`, we introduce a dedicated async helper `chat_voice_turn()` that wraps the streaming path and returns `(answer_text, retrieved_chunks, used_knowledge, citations, t_llm_first_token, timings)`. This helper captures `time.time()` at the first yielded token.

**Why not just switch the voice handler to stream=True:** The streaming path returns a `StreamingResponse` HTTP object, which cannot be `await`ed inside a WebSocket handler. A dedicated helper avoids coupling the HTTP response format with the WebSocket flow.

**Implementation approach:**
```python
# In app.py or a new voice_chat_helpers.py:
async def _voice_chat_turn(
    message: str,
    session_id: str,
    backend: str,
    brain_model: str,
) -> dict:
    # Runs retrieval (same as chat()), then calls iter_openai_compatible_stream_events
    # directly, records time.time() at first content-bearing event.
    # Returns a plain dict with answer, chunks, t_llm_first_token, timings.
    ...
```

The existing `iter_openai_stream_events()` in `llm_stream.py` is the right iterator -- timestamp the first yielded event that has non-empty `delta.content`.

### Pattern 3: Incremental .env Profile Overlay

**What:** Runner loads base `.env` first with `python-dotenv`, then overlays `.env.bench.{name}`. Only differing variables appear in the profile file.

**Example (from existing .env.voice.yandex-speechkit pattern):**
```bash
# .env.bench.brain_upgrade  -- only the overriding variable
RAG_LLM_FAST_MODEL=Qwen/Qwen3.5-35B-A3B
RAG_LLM_MODEL=Qwen/Qwen3.5-35B-A3B
```

**Loading order in runner:**
```python
from dotenv import load_dotenv
load_dotenv(base_env_path)             # base .env
load_dotenv(profile_path, override=True)  # .env.bench.{name}
```

### Pattern 4: Benchmark Runner CLI Design

**What:** `benchmark_runner.py` as a standalone async script with argparse.

**CLI flags (Claude's discretion):**
```
benchmark_runner.py --fixture fixtures/bench_questions_ru.jsonl
                    --profile baseline
                    --output results/bench_<stack_id>_<ts>.jsonl
                    --ws-url ws://localhost:8787/ws/voice
                    [--timeout 30]
                    [--warmup 3]
```

**JSONL result schema per line:**
```json
{
  "question_id": "sf-01",
  "stack_id": "our_rag__Qwen3-30B-A3B__sensevoice__cosyvoice",
  "warmup": false,
  "transcript": "...",
  "answer": "...",
  "retrieved_chunks": [...],
  "speech_stopped": 1711000000.0,
  "stt_done": 1711000001.2,
  "retrieval_done": 1711000001.8,
  "llm_first_token": 1711000002.1,
  "tts_first_chunk": 1711000003.0,
  "playback_started": 1711000003.1,
  "primary_kpi_ms": 3100.0,
  "llm_ttfb_ms": 300.0,
  "keyword_hits": ["лизинг", "аванс"],
  "keyword_hit_rate": 0.67,
  "error": null
}
```

**On error (D-06):**
```json
{
  "question_id": "sf-05",
  "stack_id": "...",
  "warmup": false,
  "transcript": null,
  "answer": null,
  "retrieved_chunks": [],
  "speech_stopped": null, "stt_done": null, "retrieval_done": null,
  "llm_first_token": null, "tts_first_chunk": null, "playback_started": null,
  "primary_kpi_ms": null, "llm_ttfb_ms": null,
  "keyword_hits": [], "keyword_hit_rate": null,
  "error": "timeout after 30s"
}
```

### Pattern 5: Audio Synthesis for WebSocket Submission

**What (Claude's discretion):** The runner sends question text as audio via WebSocket, same as the browser. The simplest approach that preserves full pipeline fidelity is: call the backend's own TTS endpoint (POST /api/voice/tts or similar) to convert text to base64 PCM, then send it as `input_audio_buffer.append` events followed by `input_audio_buffer.commit`. If no TTS endpoint exists yet, use a local TTS library (pyttsx3 or gTTS) to synthesize audio offline. The benchmark runner must not require browser presence.

**Recommended approach:** Use the existing `synthesize_audio_with_provider` function via a direct HTTP call to the backend, or call the TTS sidecar directly. This avoids introducing a new dependency and reuses the same TTS path the system will actually use.

**Alternative:** Generate simple sinusoidal audio as a placeholder (silence with brief tones) -- this tests the pipeline end-to-end but produces garbage STT transcripts. Not acceptable for quality measurement.

**Correct approach:** Use text-to-audio via the configured TTS provider (call the backend REST API's TTS endpoint). Confirm the endpoint exists; if not, add a thin `/api/tts` endpoint as part of this phase.

### Anti-Patterns to Avoid

- **Changing `settings.llm.fast_model` globally per request:** `settings` is a module-level singleton. Mutating it is a race condition in concurrent sessions. Always pass model overrides per-request.
- **Using `time.perf_counter()` for cross-process timing:** The existing codebase uses `time.time()` for voice turn timestamps. Use `time.time()` for all timing in the voice path, consistent with Phase 1 decision.
- **Hardcoding warmup count:** Warmup count is 3 (D-05). Pass it as a CLI flag with default=3.
- **Blocking the asyncio loop in the runner:** Audio synthesis and file I/O must be non-blocking or use `asyncio.to_thread`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| p50/p95 percentiles | Custom sort + index math | `statistics.quantiles(data, n=100, method='inclusive')` | Stdlib, correct, no off-by-one risk |
| .env overlay loading | Manual string parsing | `python-dotenv load_dotenv(override=True)` | Already used in project; handles quoting/escaping |
| JSONL reading/writing | Custom line splitter | `json.loads(line)` per line + `json.dumps(obj)` | Stdlib; JSONL is just JSON per line |
| WebSocket client | Custom socket code | `websockets` async client (v15 installed) | Standard async context manager API |
| Async test concurrency | Threads | `asyncio.gather` for parallel question batches | Native async; no thread-safety concerns |

**Key insight:** This phase has zero external dependency additions. All needed tools are in stdlib or already installed. The comparison script needs only `statistics`, `json`, `argparse`, and `pathlib`.

---

## Critical Gap: BRAIN-01 Backend Routing

This gap was not fully described in CONTEXT.md but must be understood before planning.

**Current state:** `VoiceSession.brain_model` exists and flows through to logs and `stack_id`. The UI selector sets it via `session.update`. The allowlist validation at app.py:641-646 works correctly.

**The gap:** In `chat()` at app.py:397:
```python
model = settings.llm.fast_model if fast and settings.llm.fast_model else settings.llm.model
```
`settings.llm.fast_model` is loaded from `RAG_LLM_FAST_MODEL` env var at process startup. It is a module-level value. `ChatRequest` has no `brain_model` field. The voice handler passes no brain_model to `chat()`. So today, even if `session.brain_model` is `"Qwen/Qwen3.5-35B-A3B"`, vLLM still receives the env-configured model.

**Fix required:**
1. Add `brain_model: str | None = None` to `ChatRequest` (pydantic model in app.py)
2. In `chat()`: `effective_model = payload.brain_model or (fast_model or default_model)`
3. In voice handler: pass `brain_model=session.brain_model` to `ChatRequest`
4. Pass `effective_model` to both `call_openai_compatible` and `iter_openai_compatible_stream_events`

**Scope note:** This does NOT require restarting vLLM or hot-swapping the model. vLLM's `/v1/chat/completions` endpoint accepts a `model` field per request. If the requested model is not loaded in the running vLLM instance, vLLM returns an error. The benchmark profiles (DEPLOY-01) handle this: `brain_upgrade` profile sets `RAG_LLM_FAST_MODEL=Qwen/Qwen3.5-35B-A3B` so that when the user launches vLLM with that model, the env and session are aligned.

---

## Common Pitfalls

### Pitfall 1: perf_counter vs time.time in voice path
**What goes wrong:** Using `time.perf_counter()` in the new first-token extraction for voice (like the streaming chat path does) will produce a timestamp incompatible with the existing `time.time()` epoch-based timestamps in the log. `primary_kpi_ms` would be wrong.
**Why it happens:** The streaming `chat()` path already uses `perf_counter` for TTFB (relative timing within the request). The voice WebSocket handler uses `time.time()` for all turn-level timestamps.
**How to avoid:** In the new first-token timing for voice: set `t_llm_first_token = time.time()` at the moment the first content chunk is yielded. Preserve `perf_counter`-based `llm_ttfb_ms` as the relative timing for the `timings` dict if desired, but the log field `llm_first_token` must be `time.time()`.
**Warning signs:** `llm_first_token - retrieval_done` is a huge negative number or implausibly large positive.

### Pitfall 2: Runner sends questions too fast without waiting for response.done
**What goes wrong:** The benchmark runner sends the next question before the backend has finished processing the previous one in the same WebSocket session. The backend processes questions sequentially on one WebSocket; interleaving produces wrong timing attribution.
**Why it happens:** Async code sends the next `input_audio_buffer.commit` before receiving `response.done` for the current turn.
**How to avoid:** Use a sequential async loop: await `response.done` event before sending the next question. Use a fresh WebSocket connection per question as an alternative if session state contamination is a concern.
**Warning signs:** `stack_id` in the result does not match the configured profile.

### Pitfall 3: TTS endpoint not exposed for runner
**What goes wrong:** The runner needs to convert question text to audio to submit via WebSocket. There is currently no standalone `/api/tts` endpoint on the backend (only the internal `synthesize_audio_with_provider` function used inside the voice WebSocket handler).
**Why it happens:** The voice path is entirely WebSocket-based; TTS is not exposed as a REST endpoint today.
**How to avoid:** Either (a) add a thin `POST /api/tts` endpoint that accepts `{"text": "...", "tts_provider": "..."}` and returns `{"audio_b64": "...", "sample_rate_hz": ...}`, or (b) have the runner call the TTS sidecar directly via its own HTTP port. Option (a) is cleaner and reuses `synthesize_audio_with_provider`. The planner must include this as a task.
**Warning signs:** Runner cannot send audio; all questions time out immediately.

### Pitfall 4: JSONL results file encoding for Cyrillic
**What goes wrong:** `json.dumps()` defaults to `ensure_ascii=True`, which escapes all Cyrillic characters as `\uXXXX`. Results files become hard to read and diff.
**Why it happens:** Python json stdlib default.
**How to avoid:** Use `json.dumps(obj, ensure_ascii=False)` everywhere in the runner and comparison script. Consistent with existing app.py which already uses `ensure_ascii=False` for all its WebSocket messages.
**Warning signs:** JSONL lines contain `\u043a\u043e\u043c\u043f\u0430\u043d\u0438\u044f` instead of readable Russian.

### Pitfall 5: stack_id mismatch between runner and backend log
**What goes wrong:** The benchmark runner reads `stack_id` from the `session.updated` event, but if the runner launches before sending `session.update`, the backend uses its startup defaults. Runner JSONL records the wrong `stack_id`.
**Why it happens:** The backend initializes `VoiceSession` with `brain_model="Qwen/Qwen3-30B-A3B"` as default. If the runner doesn't send a `session.update` message first, all turns are attributed to the default stack.
**How to avoid:** Runner MUST send `session.update` as the first event after WebSocket connect, specifying the exact provider/brain_model combination for the current profile. Wait for `session.updated` response before sending any audio.
**Warning signs:** All results have the same stack_id regardless of the configured profile.

### Pitfall 6: Env profiles missing new voice provider variables
**What goes wrong:** The `qwen3_tts`, `qwen3_asr`, and `voxtral` profiles need `*_BASE_URL` variables pointing to the Phase 2 sidecar servers. If those variables are absent, the adapters hard-fail (Phase 2 decision: `_HARD_FAIL_STT` pattern).
**Why it happens:** Profile files are created from the .env.voice.{name} convention which was designed for voice provider switching, not sidecar URL configuration.
**How to avoid:** Each profile must include the relevant `*_BASE_URL` for its voice providers. The planner must check what variables each sidecar adapter reads from the env.
**Warning signs:** Runner gets `RuntimeError: QWEN3_ASR_BASE_URL is not set` during execution.

---

## Code Examples

### Streaming First-Token Extraction in Voice Path
```python
# Source: existing app.py streaming chat path + Phase 1 timing convention
# In a new _voice_chat_turn() helper, after retrieval is done:
first_token_time: float | None = None
streamed_parts: list[str] = []
stream_iter = iter_openai_compatible_stream_events(
    base_url=effective_base_url,
    model=effective_model,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    temperature=settings.llm.temperature,
    max_tokens=settings.llm.fast_max_tokens,
    timeout_sec=settings.llm.timeout_sec,
)
for event in stream_iter:
    choice = (event.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    text = delta.get("content") or ""
    if text:
        if first_token_time is None:
            first_token_time = time.time()  # MUST be time.time(), not perf_counter
        streamed_parts.append(text)
t_llm_first_token = first_token_time or time.time()
answer_text = clean_answer("".join(streamed_parts))
```

### JSONL Question Fixture Format
```jsonl
{"question_id": "sf-01", "category": "short_factual", "text_ru": "Какой минимальный аванс для лизинга авто?", "expected_keywords": ["аванс", "процент", "лизинг"]}
{"question_id": "lf-01", "category": "long_factual", "text_ru": "Расскажите подробно об условиях лизинга недвижимости в Микро Лизинг.", "expected_keywords": ["недвижимость", "лизинг", "условия", "срок"]}
{"question_id": "kb-01", "category": "kb_grounded", "text_ru": "Какой рейтинг компании присвоило агентство Fitch?", "expected_keywords": ["Fitch", "рейтинг", "B-"]}
{"question_id": "amb-01", "category": "ambiguous", "text_ru": "Сколько это стоит?", "expected_keywords": ["аванс", "платёж", "условия"]}
{"question_id": "oos-01", "category": "out_of_scope", "text_ru": "Какой курс доллара сегодня?", "expected_keywords": []}
```

### Benchmark Runner Skeleton
```python
# Source: websockets docs + project WebSocket protocol
import asyncio
import json
import time
import argparse
import websockets

async def run_question(ws_url, question, warmup, timeout_sec):
    result = {
        "question_id": question["question_id"],
        "stack_id": None,
        "warmup": warmup,
        "transcript": None,
        "answer": None,
        "retrieved_chunks": [],
        "speech_stopped": None, "stt_done": None, "retrieval_done": None,
        "llm_first_token": None, "tts_first_chunk": None, "playback_started": None,
        "primary_kpi_ms": None, "llm_ttfb_ms": None,
        "keyword_hits": [], "keyword_hit_rate": None,
        "error": None,
    }
    try:
        async with websockets.connect(ws_url) as ws:
            # Step 1: send session.update for current profile config
            await ws.send(json.dumps({"type": "session.update", ...}))
            updated = await asyncio.wait_for(ws.recv(), timeout=5)
            updated_data = json.loads(updated)
            result["stack_id"] = updated_data.get("stack_id")

            # Step 2: synthesize audio, send as input_audio_buffer.append + commit
            audio_b64 = await get_audio_for_text(question["text_ru"])
            t_speech_stopped = time.time()
            result["speech_stopped"] = t_speech_stopped
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            # Step 3: collect events until response.done
            async for raw in ws:
                event = json.loads(raw)
                etype = event.get("type")
                if etype == "conversation.item.input_audio_transcription.completed":
                    result["transcript"] = event.get("transcription")
                elif etype == "response.done":
                    result["answer"] = ...  # from assistant_response event
                    result["retrieved_chunks"] = event.get("used_knowledge", [])
                    break
    except (asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
        result["error"] = str(exc)
    return result
```

### Comparison Script Percentile Computation
```python
# Source: Python stdlib statistics docs
import statistics

def percentiles(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "p50": None, "p95": None}
    qs = statistics.quantiles(clean, n=100, method="inclusive")
    return {
        "mean": statistics.mean(clean),
        "p50": qs[49],   # 50th percentile
        "p95": qs[94],   # 95th percentile
    }
```

### Env Profile Loading with Override
```python
# Source: python-dotenv docs
from dotenv import load_dotenv
from pathlib import Path

def load_bench_profile(repo_root: Path, profile_name: str) -> None:
    base_env = repo_root / "rag_demo_system" / ".env"
    profile_env = repo_root / "rag_demo_system" / f".env.bench.{profile_name}"
    load_dotenv(base_env)
    load_dotenv(profile_env, override=True)
```

---

## Runtime State Inventory

> Not a rename/refactor phase. Omit.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 | All scripts | Yes | 3.13.5 | -- |
| websockets | Benchmark runner | Yes | 15.0.1 | -- |
| asyncio | Benchmark runner | Yes | stdlib | -- |
| statistics | Comparison script | Yes | stdlib | -- |
| argparse | Runner + compare CLI | Yes | stdlib | -- |
| python-dotenv | Profile overlay loading | Yes (used in project) | -- | Manual os.environ |
| pytest | Test infrastructure | Yes (20 tests pass) | -- | -- |
| numpy | Percentile alt | Yes | 2.1.3 | statistics.quantiles |
| vLLM (Qwen3.5-35B-A3B) | BRAIN-01 at benchmark time | NOT checked -- GPU server only | -- | Use Qwen3-30B-A3B (baseline) |

**Missing dependencies with no fallback:**
- vLLM loaded with `Qwen/Qwen3.5-35B-A3B` -- required for the `brain_upgrade` benchmark profile. Not present on local dev machine. This is expected: the brain upgrade profile is for server execution. Phase 3 creates the profile files and validates the routing code; actual benchmarking happens in Phase 5 on the GPU server.

**Missing dependencies with fallback:**
- None that affect local development and testing.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (confirmed, 20 tests pass in 0.04s) |
| Config file | none -- tests run via `python -m pytest rag_demo_system/tests/` from repo root |
| Quick run command | `python -m pytest rag_demo_system/tests/test_instrumentation.py rag_demo_system/tests/test_voice_session.py rag_demo_system/tests/test_llm_stream.py rag_demo_system/tests/test_stack_cli.py -v` |
| Full suite command | `python -m pytest rag_demo_system/tests/ --ignore=rag_demo_system/tests/test_batching.py -x` |

Note: `test_batching.py` has an import error in the current suite; it is unrelated to Phase 3 work. The `--ignore` flag keeps the full suite runnable.

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BRAIN-01 | `ChatRequest.brain_model` field overrides env model | unit | `pytest rag_demo_system/tests/test_brain_routing.py -x` | Wave 0 |
| BRAIN-01 | Voice handler passes session.brain_model to ChatRequest | unit | `pytest rag_demo_system/tests/test_brain_routing.py -x` | Wave 0 |
| D-12 | t_llm_first_token is extracted from streaming, not approximated | unit | `pytest rag_demo_system/tests/test_first_token_timing.py -x` | Wave 0 |
| BENCH-01 | Fixture file has 80+ questions with required fields | unit | `pytest rag_demo_system/tests/test_bench_fixture.py -x` | Wave 0 |
| BENCH-02/03 | Runner JSONL schema contains all required fields | unit | `pytest rag_demo_system/tests/test_bench_runner_schema.py -x` | Wave 0 |
| BENCH-04 | Comparison script outputs correct markdown table | unit | `pytest rag_demo_system/tests/test_bench_compare.py -x` | Wave 0 |
| DEPLOY-01 | All 7 profile files exist with correct structure | unit | `pytest rag_demo_system/tests/test_bench_profiles.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest rag_demo_system/tests/test_instrumentation.py rag_demo_system/tests/test_voice_session.py rag_demo_system/tests/test_llm_stream.py -q`
- **Per wave merge:** full suite minus test_batching.py
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `rag_demo_system/tests/test_brain_routing.py` -- covers BRAIN-01: ChatRequest.brain_model field, routing logic
- [ ] `rag_demo_system/tests/test_first_token_timing.py` -- covers D-12: first-token extraction from stream
- [ ] `rag_demo_system/tests/test_bench_fixture.py` -- covers BENCH-01: fixture file validation (80+ questions, required fields)
- [ ] `rag_demo_system/tests/test_bench_runner_schema.py` -- covers BENCH-02/03: JSONL output schema contract
- [ ] `rag_demo_system/tests/test_bench_compare.py` -- covers BENCH-04: comparison script output format
- [ ] `rag_demo_system/tests/test_bench_profiles.py` -- covers DEPLOY-01: presence and structure of all 7 profiles

---

## Open Questions

1. **TTS endpoint for benchmark runner (D-04)**
   - What we know: The runner must send audio via WebSocket. TTS synthesis is done inside `synthesize_audio_with_provider()` which is not exposed as a REST endpoint.
   - What's unclear: Does a `POST /api/tts` endpoint need to be added as part of this phase, or does the runner call the TTS sidecar (e.g., CosyVoice on port 50001) directly?
   - Recommendation: Add a thin `POST /api/tts` endpoint to the backend as part of Wave 3. This keeps the runner backend-agnostic and reuses existing provider routing logic. If the TTS sidecar is unavailable, the runner can fall back to a simple audio tone as a last resort (but this breaks real STT testing -- document this limitation).

2. **Keyword hit scoring (BENCH-03 / comparison script)**
   - What we know: D-02 defines `expected_keywords` per question (2-5 Russian terms). D-07 specifies `keyword hit rate` as a comparison metric.
   - What's unclear: Should keyword matching be exact substring match or normalized (lowercase, strip punctuation)? Russian morphology means exact match will miss inflected forms (e.g., "лизинг" vs "лизинга").
   - Recommendation: Normalize both answer and keywords to lowercase before substring match. Do not implement stemming/morphological analysis -- it adds complexity with little improvement for a 2-5 keyword list. Document this as a known limitation.

3. **stack_cli.py benchmark command integration (DEPLOY-01)**
   - What we know: `stack_cli.py` line 183 has a `benchmark` command stub that prints "not implemented yet". D-09 says profiles follow `.env.voice.{name}` convention.
   - What's unclear: Should the `benchmark` command in `stack_cli.py` be wired to call `benchmark_runner.py`, or is the runner a standalone script?
   - Recommendation: Wire `stack_cli.py benchmark` to call `benchmark_runner.py` as a subprocess. This preserves the single entrypoint pattern (`stack.sh benchmark --profile baseline --output results/`). Not strictly required for Phase 3 success criteria, but improves usability.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| t_llm_first_token = t_retrieval_done (approximation) | Real streaming first-token timestamp | Phase 3 (this phase) | llm_ttfb_ms will be non-null in voice turn logs |
| Brain model fixed by env at startup | Per-request brain_model routing via ChatRequest | Phase 3 (this phase) | UI brain selector becomes functionally effective |
| No benchmark fixture or runner | JSONL fixture + async WebSocket runner | Phase 3 (this phase) | Controlled comparison becomes possible in Phase 5 |

**Deprecated/outdated:**
- `t_llm_first_token = t_retrieval_done`: This approximation in app.py:771 is replaced by Phase 3. After the fix, the TODO comment at line 770 is resolved.

---

## Project Constraints (from CLAUDE.md)

No project-level CLAUDE.md exists in this repository. The global CLAUDE.md applies with these directives relevant to this phase:

- **Accuracy over confidence:** All architectural gaps documented above were verified against the actual source files, not assumed from descriptions.
- **Test-driven development:** Per CLAUDE.md section 4.1, tests must be written before implementation. The Wave 0 gaps list above reflects this requirement.
- **`time.time()` for voice timestamps:** Confirmed from Phase 1 decision: "time.time() chosen over perf_counter for voice turn timestamps -- absolute epoch values required for cross-process log correlation in benchmarks". Must not use perf_counter for t_llm_first_token in voice path.
- **No new shared venv modifications:** `transformers==4.37.2` pin in shared venv must not change. All new model services use per-service venvs. Phase 3 has no model services, so this constraint does not apply directly.
- **`ensure_ascii=False`:** Use for all JSON output (Cyrillic content); consistent with existing app.py pattern.
- **No emojis in code or output.**

---

## Sources

### Primary (HIGH confidence)
- Source code audit: `rag_demo_system/backend/app.py` (lines 397, 641-646, 756-771, 817-828) -- brain model routing gap, first-token TODO, voice turn log format
- Source code audit: `rag_demo_system/backend/llm_stream.py` -- iter_openai_stream_events, iter_openai_stream_text
- Source code audit: `rag_demo_system/backend/voice_session.py` -- VoiceSession dataclass, brain_model field, stack_id property
- Source code audit: `rag_demo_system/scripts/stack_cli.py` -- benchmark stub, VOICE_PROFILES, program selection
- Source code audit: `rag_demo_system/.env.voice.yandex-speechkit`, `.env.voice.oss-russian.example` -- profile override pattern
- Runtime verification: `python -m pytest` (20 tests, all pass in 0.04s)
- Runtime verification: `import websockets; websockets.__version__` = 15.0.1
- Runtime verification: `statistics.quantiles()` -- p50/p95 confirmed correct

### Secondary (MEDIUM confidence)
- `docs/voice_ai_playbook_2026-03-25.md` -- Benchmark dataset requirements (20+20+20+10+10 question distribution), timing metrics, acceptance targets
- `experiments/yandex_realtime_voice/mikro_leasing_site_unified_dedup.md` -- KB content sample for question generation context

### Tertiary (LOW confidence)
- None.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified as installed; no external dependencies needed
- Architecture: HIGH -- brain routing gap and first-token timing fix confirmed against source code line numbers
- Pitfalls: HIGH -- each pitfall traced to specific code locations or existing project decisions
- Benchmark design: HIGH -- toolchain patterns derived from stdlib and installed libraries; no speculation

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable stack; no fast-moving dependencies)
