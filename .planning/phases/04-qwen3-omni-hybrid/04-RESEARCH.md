# Phase 4: Qwen3-Omni Hybrid - Research

**Researched:** 2026-03-26
**Domain:** Qwen3-Omni multimodal inference, FastAPI sidecar, RAG injection, JSONL instrumentation
**Confidence:** MEDIUM (model API verified via HuggingFace model card and official repo; audio bytes input path has one gap noted below)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Audio-in, audio-out. Omni receives raw user audio + text context chunks. STT runs in parallel only to produce a text query for RAG search. Omni generates audio response directly (no separate TTS step).
- **D-02:** The STT transcript is NOT sent to Omni as input. Omni hears the original audio natively. The transcript is used solely as the RAG retrieval query.
- **D-03:** Strict grounding. Prompt instructs Omni to answer ONLY from provided context chunks. If the context does not cover the topic, Omni must refuse or say it cannot help. This satisfies OMNI-02 (out-of-scope refusal).
- **D-04:** Same retrieval settings as the split pipeline voice_fast profile: vector_top_k=3, bm25_top_k=1, final_top_n=2, reranker disabled.
- **D-05:** All prompts and grounding instructions written in Russian. Chunks are already Russian. Audio input is Russian.
- **D-06:** Emit all 6 standard JSONL timing fields. llm_first_token and tts_first_chunk both set to the Omni first-audio timestamp.
- **D-07:** Primary KPI (playback_started - speech_stopped) remains directly comparable with split pipeline results.
- **D-08:** Standalone FastAPI sidecar with its own Python venv. Loads Qwen3-Omni via transformers (not vLLM). Matches Phase 2 sidecar pattern.
- **D-09:** Single POST /chat endpoint. Accepts: audio (base64 WAV), context_chunks (list of text strings), system_prompt (text). Returns: audio_b64 (generated speech), text (transcript of Omni's answer), sample_rate_hz.
- **D-10:** GET /health endpoint following existing sidecar convention.
- **D-11:** Never co-hosted with split pipeline brain models. Swap via supervisorctl between benchmark tests.
- **D-12:** Update .env.bench.omni_hybrid with real values once the sidecar is built.
- **D-13:** Hard-fail: if Omni sidecar unavailable when selected, raise RuntimeError. No silent fallback.

### Claude's Discretion

- Sidecar internal structure (model loading, warmup, audio preprocessing)
- Exact Russian system prompt wording (within the strict grounding constraint)
- Audio format conversion details (sample rate, encoding between browser and Omni)
- How the backend dispatches to the Omni path vs split pipeline path in app.py
- Error message wording for hard-fail scenarios

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OMNI-01 | Qwen3-Omni hybrid adapter retrieves chunks via existing RAG engine and injects them into Omni prompt | Retrieval reuse pattern confirmed; context_chunks injected as text into system prompt before generate() |
| OMNI-02 | Omni hybrid mode accessible as a voice provider option in the UI alongside split pipeline providers | index.html voiceProviderSelect pattern confirmed; normalize_voice_provider allowlist needs updating |
| OMNI-03 | Omni results use the same log format so they are directly comparable with split pipeline results | 6-field JSONL contract confirmed; llm_first_token == tts_first_chunk == Omni first audio timestamp |
</phase_requirements>

---

## Summary

Phase 4 adds a Qwen3-Omni-30B-A3B-Instruct hybrid adapter following a well-established sidecar pattern already used in Phases 2 and 3. The model is a Mixture-of-Experts architecture (30B total, ~3B active parameters) with a "thinker-talker" design: a text reasoning component (thinker) feeds a speech synthesis component (talker). The model requires `transformers==4.57.3` (or git HEAD as of the model card, pinned separately from the shared venv which uses `4.37.2`), plus `qwen-omni-utils`, `soundfile`, and `ffmpeg`.

The critical implementation decision (D-08) to use a standalone venv is well-supported: `transformers==4.57.3` is incompatible with `transformers==4.37.2` in the shared backend venv. The sidecar follows the same pattern as the Qwen3-TTS and Qwen3-ASR sidecars from Phase 2: standalone FastAPI process, `{NAME}_BASE_URL` env var, `/health` and `/chat` endpoints, supervisord entry.

VRAM requirements are substantial. The model card shows ~79 GB for a typical short audio turn in BF16 with flash_attention_2. An H100 NVL (94 GB) is the right fit; an A100 80 GB is too tight. The model must never be loaded alongside the split pipeline brain models (D-11).

**Primary recommendation:** Build the sidecar as `rag_demo_system/backend/qwen3_omni_sidecar.py` in its own venv at `rag_demo_system/venvs/omni/`, following the exact file and class structure of the Qwen3-TTS sidecar. Use `tmpfile + soundfile` to materialize the base64 WAV for `process_mm_info`, then inject retrieved chunks as a Russian-language grounding block in the system prompt, and extract audio + text from `model.generate()` output.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `transformers` | `4.57.3` (pinned) | `Qwen3OmniMoeForConditionalGeneration`, `Qwen3OmniMoeProcessor` | Only supported version per model card; separate from shared venv |
| `qwen-omni-utils` | `0.0.9` (latest as of 2026-02-10) | `process_mm_info()` for audio input preprocessing | Official utility; handles WAV/URL/numpy audio loading |
| `accelerate` | latest compatible | device_map="auto" multi-GPU / offloading | Required for from_pretrained with device_map |
| `soundfile` | latest | Reading/writing WAV PCM audio | Standard NumPy-based audio IO; required in sidecar venv |
| `fastapi` | same as shared venv | HTTP sidecar server | Matches existing sidecar convention |
| `uvicorn` | same as shared venv | ASGI server | Matches existing sidecar convention |
| `flash-attn` | compatible with torch | FlashAttention 2 for VRAM reduction | Saves meaningful VRAM; optional but strongly recommended |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `ffmpeg` (system) | any recent | Required by qwen-omni-utils for audio decoding | Must be installed at OS level before sidecar starts |
| `numpy` | latest compatible | Audio array manipulation | Always needed for soundfile + generate() output |
| `torch` | latest CUDA-compatible | Model execution on GPU | Required; H100 needs CUDA 12.x |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| transformers sidecar | vLLM-Omni fork | vLLM-Omni requires building from a fork (`wangxiongts/vllm`), adds complexity, D-08 explicitly chose transformers |
| soundfile for output | scipy.io.wavfile | soundfile is already used in Qwen3-TTS sidecar; consistency preferred |
| tmpfile for audio input | numpy tuple `(array, sr)` | qwen-omni-utils accepts both; tmpfile is simpler and matches existing WAV-in/WAV-out convention |

**Installation (sidecar venv only, not shared venv):**

```bash
python3 -m venv rag_demo_system/venvs/omni
source rag_demo_system/venvs/omni/bin/activate
pip install transformers==4.57.3 accelerate
pip install qwen-omni-utils -U
pip install soundfile numpy fastapi uvicorn
pip install -U flash-attn --no-build-isolation  # requires compatible CUDA + torch
```

**CRITICAL: Never run `pip install transformers` in the shared `rag_demo_system/.venv`. The pin is `4.37.2` and must stay fixed.**

---

## Architecture Patterns

### Recommended Project Structure

```
rag_demo_system/
├── backend/
│   ├── app.py                      # Add qwen3_omni branch in WebSocket handler
│   ├── voice_adapters.py           # Add qwen3_omni to build_voice_statuses()
│   ├── voice_session.py            # Add qwen3_omni to normalize_voice_provider allowlist
│   ├── qwen3_omni_sidecar.py       # NEW: standalone FastAPI sidecar
│   └── yandex_realtime.py          # normalize_voice_provider -- needs qwen3_omni added
├── frontend/
│   └── index.html                  # Add <option value="qwen3_omni"> to voiceProviderSelect
├── scripts/
│   └── supervisord.conf            # Add [program:qwen3_omni] entry
├── venvs/
│   └── omni/                       # NEW: isolated venv for transformers==4.57.3
├── tests/
│   └── test_qwen3_omni_adapter.py  # NEW: contract tests matching existing pattern
└── .env.bench.omni_hybrid           # EXISTS: fill with real values
```

### Pattern 1: Sidecar Server (follows Phase 2 pattern)

**What:** Standalone FastAPI process with `/health` and `/chat` endpoints, loaded model singleton, per-service venv.

**When to use:** All new model services in this project. Keeps shared venv stable.

**Example (derived from Phase 2 sidecar pattern + Omni model card):**

```python
# Source: HuggingFace Qwen/Qwen3-Omni-30B-A3B-Instruct model card
import base64, io, os, tempfile, time
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

MODEL_PATH = os.getenv("QWEN3_OMNI_MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
SPEAKER = os.getenv("QWEN3_OMNI_SPEAKER", "Chelsie")

app = FastAPI(title="Qwen3-Omni Sidecar")
_model = None
_processor = None


def _load():
    global _model, _processor
    _model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    _processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)


@app.on_event("startup")
async def startup():
    _load()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


class ChatRequest(BaseModel):
    audio_b64: str          # base64-encoded WAV (24kHz mono PCM16 from browser)
    context_chunks: list[str]
    system_prompt: str = ""


class ChatResponse(BaseModel):
    audio_b64: str           # base64-encoded WAV output at 24kHz
    text: str                # transcript of Omni's answer
    sample_rate_hz: int = 24000
    t_omni_first_audio: float


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Write base64 WAV to tempfile so process_mm_info can load it
    wav_bytes = base64.b64decode(req.audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name

    # Build grounding system prompt
    context_block = "\n\n".join(req.context_chunks)
    system_text = (
        req.system_prompt or
        "Вы — голосовой ассистент по лизингу. "
        "Отвечайте СТРОГО на основе предоставленного контекста. "
        "Если информация отсутствует в контексте, скажите что не можете помочь.\n\n"
        f"Контекст:\n{context_block}"
    )

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user",   "content": [{"type": "audio", "audio": tmp_path}]},
    ]

    text_tmpl = _processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = _processor(
        text=text_tmpl, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    )
    inputs = inputs.to(_model.device).to(_model.dtype)

    t_start = time.time()
    text_ids, audio_tensor = _model.generate(
        **inputs,
        speaker=SPEAKER,
        thinker_return_dict_in_generate=True,
        return_audio=True,
        use_audio_in_video=False,
    )
    t_omni_first_audio = time.time()

    answer_text = _processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # Convert audio tensor to base64 WAV
    audio_np = audio_tensor.reshape(-1).detach().cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, samplerate=24000, format="WAV", subtype="PCM_16")
    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    os.unlink(tmp_path)
    return ChatResponse(
        audio_b64=audio_b64,
        text=answer_text,
        sample_rate_hz=24000,
        t_omni_first_audio=t_omni_first_audio,
    )
```

### Pattern 2: Backend Dispatch Branch (app.py)

**What:** After `session.update` sets `voice_provider = "qwen3_omni"`, the WebSocket handler enters a new branch instead of the split pipeline path.

**When to use:** The branch point is at `input_audio_buffer.commit` event handling.

**Key insight:** The Omni path runs STT first (for RAG query only), then runs RAG retrieval, then calls the Omni sidecar once with audio + chunks. The STT transcript is NOT forwarded to Omni as text input -- only the original audio file goes to Omni.

```python
# In app.py WebSocket handler, after t_stt_done:
if session.voice_provider == "qwen3_omni":
    # Omni path: retrieve chunks, call /chat with audio + chunks
    omni_base_url = os.getenv("QWEN3_OMNI_BASE_URL")
    if not omni_base_url:
        raise RuntimeError("Qwen3-Omni sidecar unavailable: QWEN3_OMNI_BASE_URL not set")
    # Retrieve chunks using existing RAG engine (reuse _voice_chat_streaming_sync's retrieval)
    # ... factored retrieval call ...
    t_retrieval_done = time.time()
    omni_resp = requests.post(
        omni_base_url.rstrip("/") + "/chat",
        json={"audio_b64": audio_b64, "context_chunks": chunk_texts, "system_prompt": ""},
        timeout=120,
    )
    omni_resp.raise_for_status()
    omni_data = omni_resp.json()
    t_omni_first_audio = omni_data["t_omni_first_audio"]
    # Collapsed: llm_first_token == tts_first_chunk == t_omni_first_audio (D-06)
    t_llm_first_token = t_omni_first_audio
    t_tts_first_chunk = t_omni_first_audio
    answer_text = omni_data["text"]
    audio_b64_out = omni_data["audio_b64"]
    # ... send audio to browser, set t_playback_started, log JSONL ...
```

### Pattern 3: VoiceSession stack_id for Omni

The existing `stack_id` property builds from `backend__brain__stt__tts`. For Omni, stt and tts are both `omni`:

```python
# Example: our_rag__Qwen3-Omni-30B-A3B__omni__omni
# To achieve this, VoiceSession must set:
#   session.brain_model = "Qwen/Qwen3-Omni-30B-A3B"   (used for stack_id)
#   session.stt_provider = "omni"
#   session.tts_provider = "omni"
# OR: override stack_id property to special-case qwen3_omni voice_provider
```

Both approaches work. The simplest is to set all three fields when voice_provider is detected as qwen3_omni in the session.update handler (mirror the yandex_realtime pattern at lines 813-815 of app.py).

### Pattern 4: normalize_voice_provider allowlist

The current allowlist in `yandex_realtime.py:normalize_voice_provider` is:
`{"local", "yandex_realtime", "yandex_speechkit", "oss_russian"}`

Add `"qwen3_omni"` to this set. Otherwise any session.update with `voice_provider: "qwen3_omni"` silently falls back to `"local"`.

### Anti-Patterns to Avoid

- **Sharing the sidecar venv with the backend venv:** transformers 4.37.2 vs 4.57.3 are incompatible. Always use the separate `venvs/omni/` path.
- **Sending the STT transcript as Omni text input (D-02):** Omni must receive raw audio so it hears the user natively. Transcript is only for RAG retrieval.
- **Co-hosting Omni with brain models:** ~79 GB for Omni alone on a short query. Swap via supervisorctl.
- **Calling synthesize_audio_with_provider after Omni:** Omni generates its own speech. There is no separate TTS step.
- **Batch size > 1 with audio output:** The model card states batch inference does not support returning audio simultaneously. The sidecar must process one request at a time (batch_size=1).
- **Omitting system prompt when audio generation is enabled:** The transformers docs warn that missing or wrong system prompt can break audio output. Always provide the grounding system prompt.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio format conversion | Custom WAV parser | `soundfile.write(buf, array, samplerate=24000)` | Handles PCM16 WAV correctly; same format as browser expects |
| Audio input loading | Custom WAV decoder | `qwen-omni-utils process_mm_info` | Handles file paths, URLs, numpy; correctly resamples to model's expected rate |
| Speech output from text | Custom TTS step | `model.generate(..., return_audio=True)` | Talker is built-in; adding TTS after Omni defeats the architecture |
| RAG retrieval | New retrieval function | `_voice_chat_streaming_sync` retrieval path (refactored out) | Same chunks as split pipeline; D-04 requires identical retrieval settings |

---

## Common Pitfalls

### Pitfall 1: transformers version conflict corrupts shared venv

**What goes wrong:** Developer runs `pip install transformers==4.57.3` in the shared `.venv`. Backend then fails because `4.37.2`-dependent code breaks.

**Why it happens:** The Omni model card says "install transformers==4.57.3" without emphasizing isolation.

**How to avoid:** Create a separate `venvs/omni/` venv. Never install Omni dependencies into `.venv`. Add a comment in the sidecar startup script and requirements file.

**Warning signs:** Import errors for `Qwen3OmniMoe*` in the main backend; or conversely, existing voice adapter tests start failing after Omni setup.

### Pitfall 2: Audio batch size > 1 silently drops audio output

**What goes wrong:** `model.generate()` returns `None` for the audio tensor when batch_size > 1.

**Why it happens:** The talker component does not support batched audio generation.

**How to avoid:** Enforce batch_size=1 in the sidecar. The `/chat` endpoint accepts one request at a time and returns one response. Do not attempt to batch multiple voice turns.

### Pitfall 3: Missing or wrong system prompt breaks audio output

**What goes wrong:** Omni generates text but returns `None` for audio, or generates garbage audio.

**Why it happens:** The thinker-talker design depends on the system prompt to activate the talker path. Without the expected phrasing, the talker may not engage.

**How to avoid:** Always include a system prompt with an instruction to produce a spoken response. The HuggingFace model card provides a reference English prompt; adapt to Russian. Test audio output explicitly in the contract tests (mock the sidecar response shape).

### Pitfall 4: normalize_voice_provider silently rejects "qwen3_omni"

**What goes wrong:** Frontend sends `voice_provider: "qwen3_omni"`, backend silently sets it to `"local"`, session uses split pipeline instead of Omni.

**Why it happens:** `normalize_voice_provider` has an explicit allowlist that does not include `"qwen3_omni"`.

**How to avoid:** Add `"qwen3_omni"` to the allowlist in `yandex_realtime.py:normalize_voice_provider` before testing any session.update flow.

### Pitfall 5: VRAM exhaustion during co-host with brain model

**What goes wrong:** nvidia-smi shows OOM error when Omni sidecar starts while brain model is already loaded.

**Why it happens:** Omni needs ~79 GB; Qwen3.5-35B needs ~70 GB; H100 NVL has 94 GB. Loading both simultaneously exceeds capacity.

**How to avoid:** Per D-11, use `supervisorctl stop qwen` before `supervisorctl start qwen3_omni`. Document the swap procedure in the .env.bench.omni_hybrid profile comment.

### Pitfall 6: tmpfile not deleted on exception

**What goes wrong:** Temporary WAV files accumulate on disk from failed or timed-out requests.

**Why it happens:** `os.unlink(tmp_path)` at end of function is skipped when an exception is raised earlier.

**How to avoid:** Use `try/finally` block around the sidecar endpoint body, or use `tempfile.NamedTemporaryFile(delete=True)` with explicit context manager that closes before soundfile reads.

### Pitfall 7: STT transcript forwarded to Omni (violates D-02)

**What goes wrong:** Omni receives the STT text as a text input instead of hearing the raw audio. This defeats the "Omni hears the user natively" design.

**Why it happens:** Easy to confuse with the split pipeline pattern where STT text goes to the LLM.

**How to avoid:** In the app.py Omni branch, never include the transcript string in the `json` payload to `/chat`. Only `audio_b64` and `context_chunks` go to the sidecar.

---

## Code Examples

### Loading Audio from base64 WAV (sidecar internal)

```python
# Source: derived from process_mm_info() API; tmpfile approach confirmed safe
import base64, tempfile, os

def _wav_b64_to_tmpfile(audio_b64: str) -> str:
    wav_bytes = base64.b64decode(audio_b64)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.write(wav_bytes)
    f.close()
    return f.name  # caller must os.unlink() when done
```

### Extracting Audio numpy Array from generate() Output

```python
# Source: HuggingFace Qwen/Qwen3-Omni-30B-A3B-Instruct model card
import soundfile as sf, io

def _audio_tensor_to_wav_b64(audio_tensor) -> str:
    audio_np = audio_tensor.reshape(-1).detach().cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, samplerate=24000, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")
```

### Russian Grounding System Prompt Template

```python
# Strict grounding -- satisfies D-03 and OMNI-02 (out-of-scope refusal)
SYSTEM_PROMPT_TEMPLATE = (
    "Вы — голосовой ассистент компании Mikro Leasing. "
    "Отвечайте СТРОГО на основе предоставленного контекста. "
    "Не используйте знания, не содержащиеся в контексте. "
    "Если информация по вопросу отсутствует в контексте, "
    "скажите: 'Извините, у меня нет информации по этому вопросу.'\n\n"
    "Контекст:\n{context_block}"
)
```

### Omni stack_id (follows existing VoiceSession.stack_id property)

```python
# Target: our_rag__Qwen3-Omni-30B-A3B__omni__omni
# In app.py session.update handler, when voice_provider == "qwen3_omni":
if session.voice_provider == "qwen3_omni":
    session.brain_model = "Qwen/Qwen3-Omni-30B-A3B"
    session.stt_provider = "omni"
    session.tts_provider = "omni"
# stack_id property then yields: our_rag__Qwen3-Omni-30B-A3B__omni__omni
```

### JSONL Timing Field Mapping for Omni (D-06)

```python
# All 6 standard fields, Omni collapsed mapping
t_speech_stopped   = time.time()   # user audio committed
# STT runs in parallel for RAG query:
t_stt_done         = time.time()   # STT transcript available (RAG query ready)
# RAG retrieval:
t_retrieval_done   = time.time()   # chunks returned from RAG engine
# Single Omni call (audio-in, audio+text-out):
t_omni_first_audio = omni_data["t_omni_first_audio"]   # from sidecar response
t_llm_first_token  = t_omni_first_audio   # collapsed (D-06)
t_tts_first_chunk  = t_omni_first_audio   # collapsed (D-06)
# After audio sent to browser:
t_playback_started = time.time()
```

### Contract Test Pattern (follows test_voice_adapters_official.py)

```python
# New file: rag_demo_system/tests/test_qwen3_omni_adapter.py
def test_chat_endpoint_dispatches_to_omni_sidecar(monkeypatch):
    """When voice_provider is qwen3_omni and QWEN3_OMNI_BASE_URL is set,
    app dispatches to sidecar /chat with audio_b64 and context_chunks."""
    ...

def test_hard_fail_when_omni_base_url_not_set(monkeypatch):
    """RuntimeError raised if QWEN3_OMNI_BASE_URL is not configured."""
    ...

def test_omni_stack_id_format():
    """stack_id is our_rag__Qwen3-Omni-30B-A3B__omni__omni when voice_provider==qwen3_omni."""
    ...

def test_omni_voice_provider_in_normalizer_allowlist():
    """normalize_voice_provider('qwen3_omni') returns 'qwen3_omni', not 'local'."""
    ...

def test_build_voice_statuses_includes_qwen3_omni():
    """build_voice_statuses() returns a 'qwen3_omni' key."""
    ...
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Qwen2.5-Omni as reference | Qwen3-Omni-30B-A3B-Instruct | March 2026 | Stronger Russian; 10 speech output languages vs fewer in 2.5 |
| transformers git preview builds | transformers==4.57.3 pinned release | 2026 | Stable install; explicit pin reduces breakage |
| No audio output method in transformers | `model.generate(..., return_audio=True)` + `thinker_return_dict_in_generate=True` | 2026 | Enables single-call audio generation from transformers |

**Deprecated/outdated:**
- `Qwen2.5-Omni`: Predecessor model. Do not use. The project targets `Qwen3-Omni-30B-A3B-Instruct`.
- `vLLM-Omni` fork (`wangxiongts/vllm`): Available but D-08 explicitly chose transformers to match sidecar pattern.

---

## Open Questions

1. **Exact audio input sample rate expected by process_mm_info**
   - What we know: Output is 24000 Hz. Browser sends 24000 Hz PCM16 per existing sidecar convention. qwen-omni-utils docs show WAV file input is resampled internally.
   - What is unclear: Whether process_mm_info resamples arbitrary input sample rates, or whether sending non-24000 Hz audio causes degraded recognition.
   - Recommendation: Confirm browser audio sample rate at sidecar startup via a quick warmup test. If browser sends 16000 Hz (Whisper convention), add explicit resample before tmpfile write.

2. **Text transcript field name from generate() output**
   - What we know: `processor.batch_decode(text_ids.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)` returns a list of strings. The first element is the text response.
   - What is unclear: Whether the text includes the spoken response verbatim or a cleaned version. Some Omni models embed audio tokens in the text output.
   - Recommendation: Test the decode output in the warmup request during sidecar startup. Log it to verify it is clean text before writing to JSONL.

3. **Speaker option for Russian output**
   - What we know: Available speakers are "Chelsie" (female), "Ethan" (male), "Aiden" (male). No Russian-specific speaker mentioned in docs.
   - What is unclear: Which speaker produces the most natural Russian intonation. The model supports Russian speech output but speaker quality differences in Russian are unverified.
   - Recommendation: Use "Chelsie" as default (matches female assistant voice common in Russian customer service). Make speaker configurable via `QWEN3_OMNI_SPEAKER` env var.

4. **Flash attention 2 availability on target GPU**
   - What we know: flash-attn requires CUDA and a compatible GPU. H100 is supported. The `--no-build-isolation` flag requires a working CUDA toolkit at build time.
   - What is unclear: Whether the Azure H100 VM image has the CUDA toolkit installed before flash-attn build.
   - Recommendation: Make flash_attention_2 optional with a graceful fallback to `attn_implementation="eager"`. Add a `QWEN3_OMNI_ATTN_IMPL` env var with default `"flash_attention_2"`.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 | Sidecar venv creation | Yes | 3.13.5 | -- |
| ffmpeg | qwen-omni-utils audio decoding | Yes | 8.0 | None (blocking) |
| CUDA + H100 | Model inference (BF16, ~79 GB VRAM) | Not on dev machine | -- | Dev: skip model load, mock sidecar in tests |
| soundfile | Sidecar audio I/O | Not in shared venv | -- | Install in sidecar venv |
| transformers 4.57.3 | Qwen3OmniMoe* classes | Not in shared venv (4.37.2) | -- | Install in sidecar venv only |

**Missing dependencies with no fallback:**
- `ffmpeg` is required for qwen-omni-utils audio decoding. It is present on the dev machine (v8.0) but must be confirmed on the Azure H100 server. Add to deployment checklist.
- GPU with ~79 GB VRAM (H100 NVL 94 GB) for inference. Dev machine has no GPU. The sidecar can be built and contract-tested on CPU with mocked model; actual inference requires the server.

**Missing dependencies with fallback:**
- `soundfile` and `transformers==4.57.3`: install in `venvs/omni/`, not shared venv.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (system Python 3.13, no .venv pin needed) |
| Config file | none (pytest auto-discovers tests/) |
| Quick run command | `python3 -m pytest rag_demo_system/tests/test_qwen3_omni_adapter.py -q` |
| Full suite command | `python3 -m pytest rag_demo_system/tests/ -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OMNI-01 | Dispatch to /chat with audio_b64 + context_chunks when qwen3_omni selected | unit (monkeypatch requests) | `pytest tests/test_qwen3_omni_adapter.py::test_chat_endpoint_dispatches_to_omni_sidecar -x` | Wave 0 |
| OMNI-01 | Hard-fail when QWEN3_OMNI_BASE_URL not set | unit | `pytest tests/test_qwen3_omni_adapter.py::test_hard_fail_when_omni_base_url_not_set -x` | Wave 0 |
| OMNI-01 | RAG chunks present in sidecar request payload | unit | `pytest tests/test_qwen3_omni_adapter.py::test_context_chunks_in_chat_payload -x` | Wave 0 |
| OMNI-02 | "qwen3_omni" option present in voiceProviderSelect HTML | file-content (no mocking) | `pytest tests/test_frontend_config_contract.py -x` | Partial (test exists, assertion needs adding) |
| OMNI-02 | normalize_voice_provider("qwen3_omni") returns "qwen3_omni" | unit | `pytest tests/test_qwen3_omni_adapter.py::test_omni_voice_provider_in_normalizer_allowlist -x` | Wave 0 |
| OMNI-03 | JSONL output has all 6 timing fields including llm_first_token, tts_first_chunk | unit | `pytest tests/test_qwen3_omni_adapter.py::test_omni_jsonl_has_required_fields -x` | Wave 0 |
| OMNI-03 | stack_id format matches our_rag__Qwen3-Omni-30B-A3B__omni__omni | unit | `pytest tests/test_qwen3_omni_adapter.py::test_omni_stack_id_format -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `python3 -m pytest rag_demo_system/tests/test_qwen3_omni_adapter.py -q`
- **Per wave merge:** `python3 -m pytest rag_demo_system/tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- `rag_demo_system/tests/test_qwen3_omni_adapter.py` -- new file, covers all OMNI-01/02/03 unit tests
- `rag_demo_system/tests/test_frontend_config_contract.py` -- exists, needs one additional assertion for `qwen3_omni` in voiceProviderSelect

---

## Sources

### Primary (HIGH confidence)

- `Qwen/Qwen3-Omni-30B-A3B-Instruct` HuggingFace model card -- model loading code, VRAM table, audio I/O format, speaker options, transformers version pin, disable_talker, batch limitation
- `https://huggingface.co/docs/transformers/main/model_doc/qwen3_omni_moe` -- Qwen3OmniMoe class names, generate() parameters, enable_audio_output, apply_chat_template, system prompt requirement
- `https://github.com/QwenLM/Qwen3-Omni` -- Official repo, web_demo.py for conversation format and speaker list
- `https://pypi.org/project/qwen-omni-utils/` -- Package version 0.0.9, base64/URL/numpy audio support
- Codebase: `rag_demo_system/backend/voice_adapters.py`, `app.py`, `voice_session.py`, `yandex_realtime.py` -- all sidecar patterns, dispatch flow, allowlist location

### Secondary (MEDIUM confidence)

- `https://github.com/QwenLM/Qwen3-TTS/issues/237` -- Confirmed transformers 4.57.3 is incompatible with shared venv 4.37.2; separate venv is the only viable path
- WebSearch result: transformers==4.57.3 installation instructions consistent across multiple official sources

### Tertiary (LOW confidence)

- Speaker quality for Russian speech: No benchmark data found. "Chelsie" recommended as default based on common convention only; validate during warmup.
- Audio resampling behavior of process_mm_info with non-24000 Hz input: Not documented explicitly; assumed to resample based on ffmpeg dependency.

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH -- model card explicitly pins transformers==4.57.3, qwen-omni-utils 0.0.9 is the current release; all other deps follow existing sidecar pattern
- Architecture: HIGH -- dispatch pattern, stack_id, allowlist, and env profile patterns are all verified from existing codebase
- Pitfalls: MEDIUM -- most are derived from official docs and existing codebase patterns; speaker quality in Russian is LOW
- VRAM requirements: HIGH -- exact table from official HuggingFace model card (78.85 GB for 15s video with flash_attention_2)

**Research date:** 2026-03-26
**Valid until:** 2026-04-26 (transformers pin and model card are stable; qwen-omni-utils updates monthly)
