# Architecture Research

**Domain:** Voice AI assistant with RAG — provider adapter integration, timing instrumentation, benchmark infrastructure, Omni hybrid experiment
**Researched:** 2026-03-25
**Confidence:** HIGH (based on direct source reading of all relevant files in the repo)

---

## Current System Overview

```
Browser (vanilla JS + Web Audio API)
        │  WebSocket /ws/voice
        ▼
app.py  ── voice_ws() handler
        │
        ├── VoiceSession (voice_session.py)
        │       state: backend, voice_provider, assistant_speaking,
        │               active_task_id, last_user_message
        │
        ├── _voice_pipeline(voice_provider) → (stt_name, tts_name)
        │       maps provider slug to (stt, tts) pair
        │
        ├── transcribe_audio()        ─── voice_adapters.py
        ├── chat()  ─── RAGEngine (engine.py) / dify_client
        ├── synthesize_audio_with_provider() ─── voice_adapters.py
        │
        └── YandexRealtimeRelay  ── yandex_realtime.py
                (bypass path: raw WS relay when provider == yandex_realtime)

Infrastructure:
  Qdrant        :6333   vector store
  vLLM          :8001   brain model (main)
  vLLM          :8002   fast brain (optional)
  Vosk STT      :8010   services/vosk_server.py
  Vosk TTS      :8011   services/vosk_tts_server.py
  Whisper       :8012   services/whisper_server.py
```

---

## Component Responsibilities

| Component | File | Responsibility | Modification Status |
|-----------|------|----------------|---------------------|
| `voice_adapters.py` | `backend/voice_adapters.py` | All STT/TTS provider calls, status reporting, provider name dispatch | **Modify** — add new provider branches |
| `voice_session.py` | `backend/voice_session.py` | Session state machine, event actions | **Modify** — add timing fields |
| `app.py` | `backend/app.py` | WebSocket loop, orchestration, timing capture between stages | **Modify** — wire timing instrumentation, add new pipeline slugs |
| `yandex_speechkit.py` | `backend/yandex_speechkit.py` | Yandex STT/TTS HTTP calls | No change |
| `yandex_realtime.py` | `backend/yandex_realtime.py` | Yandex WS relay, provider normalization | **Modify** — add new provider names to `normalize_voice_provider()` |
| `engine.py` | `backend/engine.py` | RAG retrieval with voice_fast profile | No change |
| `settings.py` | `backend/settings.py` | Config dataclasses, env loading | No change |
| `qwen3_tts.py` | `backend/qwen3_tts.py` | Qwen3-TTS HTTP wrapper | **New file** |
| `qwen3_asr.py` | `backend/qwen3_asr.py` | Qwen3-ASR HTTP wrapper | **New file** |
| `voxtral.py` | `backend/voxtral.py` | Voxtral Realtime HTTP/WS wrapper | **New file** |
| `qwen3_omni.py` | `backend/qwen3_omni.py` | Qwen3-Omni hybrid mode — retrieval injection + audio generation | **New file** |
| `benchmark/` | `rag_demo_system/benchmark/` | Runner, question fixtures, result logger | **New directory** |
| `services/qwen3_tts_server.py` | `rag_demo_system/services/` | Sidecar FastAPI wrapping Qwen3-TTS model | **New file** |
| `services/qwen3_asr_server.py` | `rag_demo_system/services/` | Sidecar FastAPI wrapping Qwen3-ASR model | **New file** |

---

## Integration Point 1: New Provider Adapters in voice_adapters.py

### How the current dispatch works

`transcribe_audio()` iterates a provider name list (`order = [preferred, "sensevoice", "whisper"]`).
For each name it checks either a special-case branch (yandex_speechkit) or reads `{NAME}_BASE_URL` from env and POSTs to `/transcribe`.

`synthesize_audio_with_provider()` is a flat `if/elif` chain keyed on the `preferred` string.
Both functions return `{"text"|"audio_b64", "provider", "sample_rate_hz", "session_id"}`.

### Where new adapters plug in

**STT — transcribe_audio():**

Add `elif name == "qwen3_asr":` and `elif name == "voxtral":` branches inside the `for name in order:` loop, before the generic `{NAME}_BASE_URL + /transcribe` fallback.

These branches call the respective module-level functions:

```python
from .qwen3_asr import transcribe_audio as transcribe_with_qwen3_asr
from .voxtral import transcribe_audio as transcribe_with_voxtral

# inside the for-loop:
if name == "qwen3_asr":
    data = transcribe_with_qwen3_asr(audio_b64, sample_rate_hz=24000)
    if data.get("text"):
        return data
    continue
if name == "voxtral":
    data = transcribe_with_voxtral(audio_b64, sample_rate_hz=24000)
    if data.get("text"):
        return data
    continue
```

Both new modules must return `{"text": str, "provider": str}`. They can use `{NAME}_BASE_URL` from env for sidecar mode, or call a model API directly.

**TTS — synthesize_audio_with_provider():**

Add `if preferred == "qwen3_tts":` before the existing cosyvoice branch:

```python
from .qwen3_tts import synthesize_audio as synthesize_with_qwen3_tts

if preferred == "qwen3_tts":
    data = synthesize_with_qwen3_tts(text)
    data.setdefault("session_id", session_id)
    return data
```

The new module must return `{"audio_b64": str, "sample_rate_hz": int, "provider": "qwen3_tts"}`.

**Status reporting — build_voice_statuses():**

Add entries for each new provider:

```python
"qwen3_tts": _service_status("qwen3_tts", os.getenv("QWEN3_TTS_BASE_URL")),
"qwen3_asr": _service_status("qwen3_asr", os.getenv("QWEN3_ASR_BASE_URL")),
"voxtral":   _service_status("voxtral",   os.getenv("VOXTRAL_BASE_URL")),
```

`_service_status()` already handles the None case as `not_configured`. No change needed to the helper.

**Provider name normalization — yandex_realtime.normalize_voice_provider():**

This function is the canonical allowlist. Add new slugs:

```python
KNOWN_PROVIDERS = {
    "local", "yandex_realtime", "yandex_speechkit",
    "oss_russian", "qwen3_stack", "voxtral_stack",
}
```

`_voice_pipeline()` in `app.py` maps compound slugs to `(stt, tts)` pairs. Add:

```python
if voice_provider == "qwen3_stack":
    return ("qwen3_asr", "qwen3_tts")
if voice_provider == "voxtral_stack":
    return ("voxtral", "qwen3_tts")
```

---

## Integration Point 2: Timing Instrumentation in voice_session.py

### Current state

`VoiceSession` is a plain dataclass with no timing fields. Timing only exists in `app.py`'s `chat()` handler via `time.perf_counter()` guards around LLM calls.

The playbook defines these required stage timestamps:

- `speech_stopped_at` (browser-side; backend receives `input_audio_buffer.commit`)
- `stt_done_at`
- `retrieval_done_at`
- `llm_first_token_at`
- `tts_first_chunk_at`
- `playback_started_at` (browser-side)

### What to add to VoiceSession

Add optional float fields (not dataclass defaults that change signature order — use `field(default=None)`):

```python
from dataclasses import dataclass, field

@dataclass
class VoiceSession:
    session_id: str
    backend: str = "our_rag"
    voice_provider: str = "local"
    assistant_speaking: bool = False
    interrupted: bool = False
    active_task_id: str | None = None
    last_user_message: str = ""
    # timing fields — all in epoch seconds (time.perf_counter is not epoch; use time.time())
    t_audio_commit: float | None = field(default=None)
    t_stt_done: float | None = field(default=None)
    t_retrieval_done: float | None = field(default=None)
    t_llm_first_token: float | None = field(default=None)
    t_tts_first_chunk: float | None = field(default=None)

    def reset_turn_timings(self) -> None:
        self.t_audio_commit = None
        self.t_stt_done = None
        self.t_retrieval_done = None
        self.t_llm_first_token = None
        self.t_tts_first_chunk = None

    def turn_timing_snapshot(self) -> dict[str, float | None]:
        ref = self.t_audio_commit
        def delta_ms(t: float | None) -> float | None:
            if t is None or ref is None:
                return None
            return (t - ref) * 1000.0
        return {
            "stt_ms":       delta_ms(self.t_stt_done),
            "retrieval_ms": delta_ms(self.t_retrieval_done),
            "llm_ttfb_ms":  delta_ms(self.t_llm_first_token),
            "tts_ms":       delta_ms(self.t_tts_first_chunk),
            "total_ms":     delta_ms(self.t_tts_first_chunk),  # proxy for perceived latency
        }
```

### Where app.py sets these fields

Inside `voice_ws()` at `input_audio_buffer.commit`:

```python
import time

# 1. commit received
session.reset_turn_timings()
session.t_audio_commit = time.time()

# 2. after transcribe_audio() returns
session.t_stt_done = time.time()

# 3. chat() returns — retrieval is inside chat(), extract from timings dict
#    chat() already returns timings["total_ms"] via response["timings"]
#    Set retrieval marker from that response:
if isinstance(response, dict) and response.get("timings"):
    session.t_retrieval_done = time.time()  # approximate; refine if needed

# 4. TTS sidecar call completes
session.t_tts_first_chunk = time.time()
```

Then attach `session.turn_timing_snapshot()` to the `response.done` WS event:

```python
await websocket.send_json({
    "type": "response.done",
    "session_id": session_id,
    "backend": response.get("backend"),
    "timings": {
        **response.get("timings", {}),
        **session.turn_timing_snapshot(),
    },
    ...
})
```

**Critical constraint:** do not change the VoiceSession dataclass field ordering in a way that breaks existing tests. The existing test (`test_session_defaults_to_local_voice_provider`) constructs `VoiceSession(session_id="s4", backend="our_rag")` positionally. New fields must have defaults.

---

## Integration Point 3: Benchmark Runner

### Structure

```
rag_demo_system/
└── benchmark/
    ├── __init__.py
    ├── questions.json          # fixed question fixture set
    ├── runner.py               # CLI entry point
    ├── evaluator.py            # per-turn grading helpers
    └── results/                # gitignored output directory
```

### runner.py design

The runner is a standalone script, not integrated into the FastAPI server. It drives the backend via HTTP, not WebSocket, to isolate timing from WS overhead.

```
for each stack_config in stacks:
    for each question in questions:
        POST /api/voice/chat with {message, session_id, stream: false}
        record: wall time start/end, response timings dict, answer, chunks
        write JSONL result row
```

The runner calls the existing `/api/voice/chat` endpoint. This endpoint internally calls `chat()` with `fast=True, mode="voice_fast"`. The `timings` dict in the response already contains `llm_ttfb_ms`, `llm_total_ms`, `qdrant_ms`, `bm25_ms`, `rerank_ms`. The runner adds wall-clock STT and TTS time by timing its own calls.

**Stack config format:**

```json
{
  "stack_id": "qwen3_tts_v1",
  "description": "Qwen3-ASR + our_rag + Qwen3.5-35B + Qwen3-TTS",
  "env_overrides": {
    "VOICE_STT_PROVIDER": "qwen3_asr",
    "VOICE_TTS_PROVIDER": "qwen3_tts",
    "RAG_LLM_MODEL": "Qwen/Qwen3.5-35B-A3B"
  }
}
```

The runner sets env overrides before calling the backend. For server-side env vars this means running the backend in a subprocess with the overridden env, or using a config reload endpoint if added.

**Result row schema (JSONL):**

```json
{
  "run_id": "2026-03-25T14:00:00Z",
  "stack_id": "qwen3_tts_v1",
  "question_id": "q001",
  "question": "...",
  "answer": "...",
  "chunks": ["chunk_id_1"],
  "citations": [],
  "timings": {"llm_ttfb_ms": 320, "qdrant_ms": 12, ...},
  "wall_ms": 850,
  "evaluator_note": ""
}
```

### How the runner interacts with the backend

The runner does not import any backend module. It is a pure HTTP client. This keeps it runnable without loading the full RAG stack into memory.

If STT timing is needed, the runner calls a separate `/api/voice/transcribe` endpoint (to be added) that wraps `transcribe_audio()` and returns the text plus elapsed time.

---

## Integration Point 4: Qwen3-Omni Hybrid Mode

### Architecture decision

Qwen3-Omni hybrid mode is a separate execution path, not a modification of the split pipeline. It requires a new WebSocket branch in `app.py` or a new endpoint, and a new backend module `qwen3_omni.py`.

### How it fits alongside the split pipeline

```
session.update(voice_provider="qwen3_omni_hybrid")
        │
        ▼
app.py voice_ws() detects provider == "qwen3_omni_hybrid"
        │
        ▼
        1. retrieve chunks via engine.retrieve()  ← same as split pipeline
        2. build context block from chunks         ← same as split pipeline
        3. call qwen3_omni.generate(audio_b64, context_chunks) → {text, audio_b64, sample_rate_hz}
        4. send audio response downstream          ← same WS event types as split pipeline
```

The critical design constraint: the `response.done` WS event type must remain identical to the split pipeline path. The frontend must not need changes to handle Omni output.

### qwen3_omni.py interface

```python
def generate(
    audio_b64: str,
    context_chunks: list[dict],
    session_id: str,
    sample_rate_hz: int = 24000,
) -> dict[str, Any]:
    """
    Returns:
        {
            "text": str,        # transcript of the model's answer
            "audio_b64": str,   # base64 PCM16 audio of the answer
            "sample_rate_hz": int,
            "provider": "qwen3_omni",
            "session_id": str,
        }
    """
```

The function calls the Qwen3-Omni vLLM endpoint with:
- audio input (base64 PCM)
- system prompt including the injected context chunks
- instruction to answer only from the provided context

### Status and pipeline slug

In `build_voice_statuses()`:
```python
"qwen3_omni": _service_status("qwen3_omni", os.getenv("QWEN3_OMNI_BASE_URL")),
```

In `normalize_voice_provider()`:
```python
KNOWN_PROVIDERS = {..., "qwen3_omni_hybrid"}
```

In `_voice_pipeline()`: Omni hybrid does NOT return a `(stt, tts)` pair — the model handles both. Instead, `app.py` detects the Omni slug before the STT call and branches to the Omni handler.

---

## Integration Point 5: Brain Model Switching

### Current state

`app.py` selects model and base_url per-request:

```python
model    = settings.llm.fast_model    if fast and settings.llm.fast_model    else settings.llm.model
base_url = settings.llm.fast_base_url if fast and settings.llm.fast_base_url else settings.llm.base_url
```

`RAG_LLM_BASE_URL`, `RAG_LLM_MODEL`, `RAG_LLM_FAST_BASE_URL`, `RAG_LLM_FAST_MODEL` control this via env.

### What is needed

Brain model switching for benchmarking does not require code changes. It requires running the backend with different env var values per benchmark run. The benchmark runner handles this by spawning the backend with a modified env.

For runtime switching (UI-driven), a `session.update(brain_model="qwen3.5_35b")` message type could be added to the WS loop, and `VoiceSession` extended with a `brain_model` field that overrides the env-derived value in the `chat()` call. This is optional for the first benchmark pass.

---

## Recommended File Structure After Milestone

```
rag_demo_system/
├── backend/
│   ├── app.py                  # modified: timing wiring, new pipeline slugs, Omni branch
│   ├── voice_adapters.py       # modified: qwen3_tts, qwen3_asr, voxtral branches + statuses
│   ├── voice_session.py        # modified: timing fields, reset_turn_timings, turn_timing_snapshot
│   ├── qwen3_tts.py            # NEW: synthesize_audio(text) -> dict
│   ├── qwen3_asr.py            # NEW: transcribe_audio(audio_b64, sample_rate_hz) -> dict
│   ├── voxtral.py              # NEW: transcribe_audio(audio_b64, sample_rate_hz) -> dict
│   ├── qwen3_omni.py           # NEW: generate(audio_b64, context_chunks, ...) -> dict
│   ├── yandex_realtime.py      # modified: expanded normalize_voice_provider allowlist
│   └── [unchanged files]
├── services/
│   ├── qwen3_tts_server.py     # NEW: FastAPI sidecar wrapping Qwen3-TTS model
│   ├── qwen3_asr_server.py     # NEW: FastAPI sidecar wrapping Qwen3-ASR model
│   └── [unchanged files]
├── benchmark/
│   ├── __init__.py             # NEW
│   ├── runner.py               # NEW: HTTP-based benchmark runner CLI
│   ├── questions.json          # NEW: fixed Russian question fixture set
│   ├── evaluator.py            # NEW: per-turn answer grading helpers
│   └── results/                # NEW: gitignored output dir
└── tests/
    ├── test_voice_adapters_qwen3.py   # NEW: contract tests for new adapters
    ├── test_voice_adapters_voxtral.py # NEW: contract tests for voxtral
    ├── test_voice_session_timings.py  # NEW: timing field tests
    └── [unchanged files]
```

---

## Data Flow: Instrumented Split Pipeline Turn

```
Browser sends input_audio_buffer.commit
        │
        ▼ t_audio_commit = time.time()
        app.py: concatenate audio_chunks
        │
        ▼ transcribe_audio(audio_b64, preferred=stt_provider)
        │   → qwen3_asr.py / voxtral.py / sensevoice / whisper / yandex_speechkit
        ▼ t_stt_done = time.time()
        │
        ▼ chat(message, fast=True, mode="voice_fast")
        │   engine.retrieve() → qdrant + bm25 → rerank → final chunks
        │   LLM call → answer text
        │   returns timings{qdrant_ms, bm25_ms, llm_ttfb_ms, ...}
        ▼ t_retrieval_done = time.time()  (approximate: after chat() returns)
        │
        ▼ synthesize_audio_with_provider(answer, preferred=tts_provider)
        │   → qwen3_tts.py / cosyvoice / vosk_tts / yandex_speechkit
        ▼ t_tts_first_chunk = time.time()
        │
        ▼ send response.output_audio.delta
        ▼ send response.done {timings: {stt_ms, retrieval_ms, llm_ttfb_ms, tts_ms, total_ms, ...}}
```

---

## Data Flow: Qwen3-Omni Hybrid Turn

```
Browser sends input_audio_buffer.commit
        │
        ▼ app.py detects voice_provider == "qwen3_omni_hybrid"
        │
        ▼ engine.retrieve(audio_transcript_hint_or_empty, voice_fast=True)
        │   (need text query — either run cheap STT first, or use last_user_message as seed)
        │
        ▼ qwen3_omni.generate(audio_b64, context_chunks)
        │   → single HTTP call to Qwen3-Omni vLLM endpoint
        │   → returns {text, audio_b64, sample_rate_hz}
        │
        ▼ send response.output_text.delta (text)
        ▼ send response.output_audio.delta (audio_b64)
        ▼ send response.done
```

Note: the Omni hybrid turn has a retrieval dependency problem. Qwen3-Omni needs a text query to retrieve chunks, but the query normally comes from STT. Two options:

1. Run a cheap STT pass first (e.g. Vosk or Qwen3-ASR) to get the query text, then pass audio to Omni with injected chunks.
2. Have Omni generate a text hypothesis first, use that for retrieval, then do a second Omni call with context. (More expensive, two-pass.)

Option 1 is simpler and keeps retrieval quality. It means the Omni hybrid path still uses an STT provider for retrieval purposes, but Omni generates the final answer and TTS. This is the recommended starting approach.

---

## Architectural Patterns

### Pattern 1: Sidecar FastAPI for Local Models

**What:** Each heavy local model (Qwen3-TTS, Qwen3-ASR, Voxtral) runs as a separate FastAPI process behind an HTTP contract. The backend `voice_adapters.py` calls it by URL.

**When to use:** Any model that loads large weights, requires a GPU, or needs a separate Python env.

**Trade-offs:** Adds deployment complexity (one more process), but cleanly isolates the backend from model runtime dependencies. Same pattern as the existing Vosk/Whisper/CosyVoice services. Tests can mock the HTTP call without loading the model.

**Contract required:**
```
GET  /health        → {"ok": bool, "provider": str}
POST /transcribe    → {"ok": bool, "text": str, "provider": str, "session_id": str}  (STT)
POST /speak         → {"ok": bool, "audio_b64": str, "sample_rate_hz": int, "provider": str}  (TTS)
```

### Pattern 2: Flat Dispatch in voice_adapters.py

**What:** Provider selection is a flat `if/elif` or loop inside `transcribe_audio()` / `synthesize_audio_with_provider()`. No class hierarchy, no plugin registry.

**When to use:** The current scale (6-8 named providers). Adding a provider is adding one `elif` branch and importing one new module.

**Trade-offs:** Does not scale to 20+ providers cleanly, but is maximally readable and testable with monkeypatching. Do not introduce a class registry until there are strong reasons.

### Pattern 3: Timing via Session Fields, Not Global State

**What:** Per-turn timestamps live on the `VoiceSession` object. They are reset at the start of each turn (`reset_turn_timings()`), computed into relative deltas at the end, and sent in `response.done`.

**When to use:** Always. Avoids shared state bugs when sessions are concurrent.

**Trade-offs:** Slight increase in `VoiceSession` size. The dataclass remains a plain value object with no dependencies on `time` (the caller sets the fields).

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Modifying the WebSocket Protocol for New Providers

**What people do:** Add new WS event types specific to Qwen3-Omni or a new provider, requiring frontend changes.

**Why it's wrong:** The frontend is vanilla JS with no type system. Every new event type requires a new handler branch. The existing `response.output_audio.delta` / `response.done` / `response.output_text.delta` types already cover what the browser needs.

**Do this instead:** Map all new provider outputs to the existing event schema. The browser only needs to know that audio arrived; it does not need to know which model produced it.

### Anti-Pattern 2: Loading Model Weights Inside the Main Backend Process

**What people do:** Import `transformers` or `torchaudio` directly in `voice_adapters.py` to call models inline.

**Why it's wrong:** The backend process is a FastAPI app. Loading 1.7B+ model weights into it adds 5-30 seconds to startup, causes OOM on CPU-only machines, and makes tests expensive.

**Do this instead:** Always use the sidecar FastAPI pattern. The adapter module calls an HTTP URL; the model runs in a separate process.

### Anti-Pattern 3: Using time.perf_counter() for Cross-Stage Timestamps

**What people do:** Use `time.perf_counter()` everywhere for latency measurement.

**Why it's wrong:** `perf_counter()` is relative to an arbitrary epoch and is not comparable across processes or log entries. It is appropriate for within-request intervals (the chat handler already uses it for `llm_ttfb_ms`), but session-level timestamps and benchmark results need absolute epoch time.

**Do this instead:** Use `time.time()` for `VoiceSession` turn timestamps and benchmark log rows. Use `time.perf_counter()` only for within-function microsecond-precision deltas (e.g. embedding time, reranker time).

### Anti-Pattern 4: Putting Omni Hybrid on the Same Code Path as Split Pipeline Before Benchmarking

**What people do:** Route `qwen3_omni_hybrid` through the same STT/chat/TTS path with Omni as just a TTS replacement.

**Why it's wrong:** The entire value proposition of Omni is audio-in / audio-out through one model call. Using it only as TTS wastes the model's capability and still pays for a separate STT call.

**Do this instead:** Give Omni hybrid its own branch in the WS loop that calls `qwen3_omni.generate()`. Keep it strictly separate from the split pipeline path so benchmark comparison is clean.

---

## Build Order (Dependency-Ordered)

This order ensures each step is testable before the next is built.

**Step 1: Timing instrumentation in voice_session.py and app.py**
- No external dependencies
- Unblocks benchmark logging
- Add contract test: `test_voice_session_timings.py`

**Step 2: Qwen3-TTS adapter (qwen3_tts.py + services/qwen3_tts_server.py)**
- Depends on: timing instrumentation (to measure TTS latency)
- Depends on: service contract pattern (exists, copy from vosk_tts_server.py)
- Add to `build_voice_statuses()`, `synthesize_audio_with_provider()`, `_voice_pipeline()`
- Add contract test: `test_voice_adapters_qwen3.py::test_synthesize_qwen3_tts`

**Step 3: Qwen3-ASR adapter (qwen3_asr.py + services/qwen3_asr_server.py)**
- Depends on: same sidecar pattern as Step 2
- Add to `transcribe_audio()`, `build_voice_statuses()`, `_voice_pipeline()`
- Add contract test: `test_voice_adapters_qwen3.py::test_transcribe_qwen3_asr`

**Step 4: Voxtral adapter (voxtral.py)**
- Depends on: STT adapter pattern (same as Step 3)
- Add to `transcribe_audio()`, `build_voice_statuses()`
- Add contract test: `test_voice_adapters_voxtral.py`

**Step 5: Benchmark runner (benchmark/runner.py + questions.json)**
- Depends on: Steps 1-4 (needs providers and timing to be meaningful)
- Standalone HTTP client, no backend import
- Run `python -m benchmark.runner --stack baseline --questions benchmark/questions.json`

**Step 6: Qwen3-Omni hybrid mode (qwen3_omni.py)**
- Depends on: Steps 1-5 (need baseline measurements to compare against)
- Requires separate vLLM endpoint running Qwen3-Omni model
- Add new branch to `voice_ws()` in `app.py`
- Add to `build_voice_statuses()`, `normalize_voice_provider()`

**Step 7: Brain model switch config for benchmarking**
- Depends on: Steps 1-5
- Env-var driven; may require benchmark runner to restart backend between runs
- No code change if env overrides at process level are sufficient

---

## Integration Points: New vs. Existing

| Connection | Type | Change Required |
|-----------|------|-----------------|
| `qwen3_tts.py` calls sidecar | HTTP POST `/speak` | New module; voice_adapters.py modified |
| `qwen3_asr.py` calls sidecar | HTTP POST `/transcribe` | New module; voice_adapters.py modified |
| `voxtral.py` calls sidecar or API | HTTP POST `/transcribe` | New module; voice_adapters.py modified |
| `qwen3_omni.py` calls vLLM | HTTP (multimodal) | New module; app.py modified |
| `voice_session.py` timing | In-process fields | Dataclass extended |
| `app.py` timing capture | In-process `time.time()` calls | 4-5 insertion points in `voice_ws()` |
| `app.py` _voice_pipeline | New slug mappings | 2 new `if` branches |
| `normalize_voice_provider()` | Allowlist expansion | 3-4 new strings added |
| `build_voice_statuses()` | New status entries | 3-4 new dict entries |
| Benchmark runner | HTTP client, standalone | New directory, no backend change |

---

## Sources

- Direct source reading: `rag_demo_system/backend/voice_adapters.py` (all provider patterns)
- Direct source reading: `rag_demo_system/backend/voice_session.py` (session state machine)
- Direct source reading: `rag_demo_system/backend/app.py` (WebSocket loop, timing patterns in chat())
- Direct source reading: `rag_demo_system/backend/yandex_realtime.py` (normalize_voice_provider pattern)
- Direct source reading: `rag_demo_system/services/vosk_server.py` and `whisper_server.py` (sidecar contract)
- Direct source reading: `rag_demo_system/tests/test_voice_adapters_official.py` (contract test pattern)
- Direct source reading: `docs/voice_ai_playbook_2026-03-25.md` (benchmark plan, timing KPIs, build order)

---

*Architecture research for: Voice AI assistant provider integration and benchmark infrastructure*
*Researched: 2026-03-25*
