# Phase 2: Voice Provider Adapters - Research

**Researched:** 2026-03-25
**Domain:** FastAPI sidecar services for Qwen3-TTS, Qwen3-ASR, and Voxtral Realtime; adapter dispatch in voice_adapters.py; frontend option extension
**Confidence:** HIGH for Qwen3-TTS and Qwen3-ASR; MEDIUM for Voxtral (architecture confirmed, exact venv deps need pinning at install time)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** Researcher must investigate whether Voxtral can be self-hosted before planning the adapter. If self-hostable: build a local sidecar (matching other adapters). If not: build a thin Mistral cloud API client marked as benchmark-only (not production, violates privacy constraint). Outcome depends on research findings.

**D-02:** Hard fail when a selected provider is unavailable. If the user selects Qwen3-ASR and the sidecar is down, the voice turn fails with a clear error message (e.g., "Qwen3-ASR service unavailable"). No silent substitution to another provider. This protects benchmark integrity (VPROV success criteria #5).

**D-03:** Each new model gets its own standalone FastAPI sidecar script with its own Python venv. Matches the existing CosyVoice/SenseVoice pattern. No shared sidecar framework or multi-model service. Each sidecar exposes `/health`, `/transcribe` (STT) or `/speak` (TTS) endpoints following the existing contract.

**D-04:** Per-service venvs are mandatory. The shared `rag_demo_system` venv must not have its `transformers==4.37.2` pin modified. Each sidecar installs its own model-specific dependencies in isolation.

**D-05:** Researcher must confirm exact HuggingFace repo IDs, native sample rates, and recommended loading libraries (e.g., faster-whisper vs. raw transformers for Qwen3-ASR) before planning begins. Do not hardcode playbook model names; they are LOW confidence.

### Claude's Discretion

- Sidecar script internal structure (how model loading, warmup, and request handling are organized within each script)
- HTTP timeout values and request payload formats (follow existing patterns)
- Specific error message wording for hard-fail scenarios

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VPROV-01 | Qwen3-TTS adapter integrated into voice_adapters.py with sidecar FastAPI service | HuggingFace repo ID confirmed: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`; `qwen-tts==0.1.1`; 24kHz output; Russian supported; mirrors vosk_tts_server.py pattern |
| VPROV-02 | Qwen3-ASR adapter integrated into voice_adapters.py with sidecar FastAPI service | HuggingFace repo ID confirmed: `Qwen/Qwen3-ASR-1.7B`; `qwen-asr==0.0.6`; 16kHz input; Russian supported; mirrors whisper_server.py pattern |
| VPROV-03 | Voxtral STT adapter integrated into voice_adapters.py (sidecar or API client) | Self-hostable confirmed (Apache 2.0 open weights); `mistralai/Voxtral-Mini-4B-Realtime-2602`; requires `transformers>=5.2.0`; 16kHz input; Russian supported |
| VPROV-04 | All new adapters pass the existing voice adapter contract tests | 6 existing tests in test_voice_adapters_official.py; 3 new test cases needed; monkeypatch pattern is well-established |
| VPROV-05 | Frontend voice provider selector updated to include all new providers | `sttProviderSelect` and `ttsProviderSelect` `<option>` elements in `frontend/index.html`; `voiceProviderReady()` in `app.js` may need updating |
</phase_requirements>

---

## Summary

Phase 2 adds three new voice provider sidecars and their corresponding adapter branches. The work pattern is firmly established by two complete prior examples in `rag_demo_system/services/`: `whisper_server.py` (STT pattern) and `vosk_tts_server.py` (TTS pattern). Each new sidecar is a standalone FastAPI application with `/health` and `/transcribe` or `/speak` endpoints, loaded into its own Python venv.

**D-01 resolution (Voxtral self-hostability):** Voxtral Mini 4B Realtime 2602 (`mistralai/Voxtral-Mini-4B-Realtime-2602`) is open-weights under Apache 2.0 and fully self-hostable via Hugging Face Transformers (>=5.2.0) or vLLM. **Build a local sidecar**, not a cloud API client.

**VRAM coexistence with A100 80GB:** The brain model (Qwen3-30B-A3B or Qwen3.5-35B-A3B) uses ~60-70 GB. Voice sidecars run sequentially during benchmarking, not simultaneously. Qwen3-TTS (~4 GB), Qwen3-ASR (~4 GB), and Voxtral (~16 GB minimum) each run alone when their profile is active. Never load voice sidecar simultaneously with the other voice models. Only one sidecar is active at a time per profile.

**Primary recommendation:** Follow the `whisper_server.py` / `vosk_tts_server.py` pattern exactly. Each sidecar gets a `requirements-{name}.txt` file, a `.venv-{name}` directory, and a `services/{name}_server.py` file. Adapter dispatch uses the existing `{NAME}_BASE_URL` env var convention with a `RuntimeError` (not fallback) when the URL is unset.

---

## D-01 Resolution: Voxtral Self-Hosting

**Finding:** Voxtral Mini 4B Realtime 2602 is self-hostable. Apache 2.0 license. Weights at `mistralai/Voxtral-Mini-4B-Realtime-2602` on Hugging Face. Transformers support added in v5.2.0. vLLM also supported.

**Decision outcome:** Build a local sidecar for Voxtral (not a cloud API thin client). The sidecar uses `transformers>=5.2.0` with `VoxtralRealtimeForConditionalGeneration`. This goes into its own venv isolating the modern transformers version from the pinned `transformers==4.37.2` in the shared backend venv.

**Privacy implication:** The cloud Voxtral API (Mistral.ai) exists at $0.006/min but must NOT be used (privacy constraint). The local sidecar is the only acceptable approach.

**Confidence:** HIGH (verified from official HuggingFace model card and Transformers docs)

---

## Standard Stack

### Core: New Sidecar Dependencies

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| qwen-tts | 0.1.1 | Load and run Qwen3-TTS inference | Official Qwen package; only way to use `Qwen3TTSModel` |
| qwen-asr | 0.0.6 | Load and run Qwen3-ASR inference | Official Qwen package; only way to use `Qwen3ASRModel` |
| transformers | >=5.2.0 | Load VoxtralRealtimeForConditionalGeneration | `VoxtralRealtimeForConditionalGeneration` added in 5.2.0 |
| torch | current (with CUDA) | GPU inference backend for all three models | All three model packages depend on PyTorch |
| fastapi | 0.115.6 | Sidecar HTTP server | Matches existing sidecar pattern |
| uvicorn | 0.30.6 | ASGI server for FastAPI sidecars | Matches existing sidecar pattern |
| pydantic | 2.10.2 | Request body validation | Matches existing sidecar pattern |
| soundfile | latest | Audio I/O for Qwen3-TTS output | Required by qwen-tts for wav read/write |
| mistral-common | latest | Mistral tokenizer backend for Voxtral | Required by transformers Voxtral support |

### Per-Sidecar Venv Names (following existing convention)

| Sidecar | Venv Name | Requirements File |
|---------|-----------|-------------------|
| Qwen3-TTS | `.venv-qwen3-tts` | `requirements-qwen3-tts.txt` |
| Qwen3-ASR | `.venv-qwen3-asr` | `requirements-qwen3-asr.txt` |
| Voxtral | `.venv-voxtral` | `requirements-voxtral.txt` |

**Installation example (Qwen3-ASR):**
```bash
python3 -m venv .venv-qwen3-asr
.venv-qwen3-asr/bin/pip install --upgrade pip wheel
.venv-qwen3-asr/bin/pip install -r requirements-qwen3-asr.txt
```

**Version verification run:**
```
qwen-tts:    0.1.1  (confirmed from PyPI, 2026-03-25)
qwen-asr:    0.0.6  (confirmed from PyPI, 2026-03-25)
transformers 5.3.0  (confirmed from HuggingFace docs referencing v5.3.0 for Voxtral)
```

---

## Architecture Patterns

### Project Structure (new files only)

```
rag_demo_system/
├── services/
│   ├── whisper_server.py          # existing
│   ├── vosk_server.py             # existing
│   ├── vosk_tts_server.py         # existing
│   ├── qwen3_asr_server.py        # NEW (mirrors whisper_server.py)
│   ├── qwen3_tts_server.py        # NEW (mirrors vosk_tts_server.py)
│   └── voxtral_server.py          # NEW (mirrors whisper_server.py)
├── backend/
│   └── voice_adapters.py          # MODIFY: add branches for 3 new providers
├── frontend/
│   └── index.html                 # MODIFY: add <option> elements
├── requirements-qwen3-asr.txt     # NEW
├── requirements-qwen3-tts.txt     # NEW
├── requirements-voxtral.txt       # NEW
└── tests/
    └── test_voice_adapters_official.py  # MODIFY: add 3 new contract tests
```

### Pattern 1: STT Sidecar (mirrors whisper_server.py exactly)

**What:** Standalone FastAPI app. `TranscribeRequest` Pydantic model. Class-based transcriber loaded at startup. `create_app()` / `create_unavailable_app()` pattern for graceful startup failure.

**When to use:** Qwen3-ASR and Voxtral sidecars both follow this pattern.

**Qwen3-ASR example:**
```python
# Source: derived from rag_demo_system/services/whisper_server.py + Qwen3-ASR HuggingFace docs
from qwen_asr import Qwen3ASRModel
import torch, base64, tempfile, wave
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class TranscribeRequest(BaseModel):
    audio_b64: str
    session_id: str
    language: str = "ru"
    sample_rate_hz: int = 24000   # input from backend; model expects 16kHz, resample internally

class Qwen3ASRTranscriber:
    def __init__(self, model_id: str, device: str) -> None:
        self._model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device,
        )

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate_hz: int, language: str) -> str:
        # write PCM16 bytes to temp WAV at actual sample_rate_hz,
        # qwen-asr will internally handle 16kHz conversion
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = Path(f.name)
        try:
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate_hz)
                w.writeframes(audio_bytes)
            results = self._model.transcribe(audio=str(wav_path), language="ru")
            return results[0].text.strip() if results else ""
        finally:
            wav_path.unlink(missing_ok=True)
```

**Voxtral example note:** Replace `Qwen3ASRModel` with `VoxtralRealtimeForConditionalGeneration` + `AutoProcessor`. Input must be resampled to 16kHz (processor's `feature_extractor.sampling_rate`).

### Pattern 2: TTS Sidecar (mirrors vosk_tts_server.py exactly)

**What:** Standalone FastAPI app. `SpeakRequest` Pydantic model. Class-based synthesizer. Returns `{"audio_b64": ..., "sample_rate_hz": 24000, "provider": "qwen3_tts", "session_id": ...}`.

**Qwen3-TTS example:**
```python
# Source: derived from rag_demo_system/services/vosk_tts_server.py + Qwen3-TTS HuggingFace docs
from qwen_tts import Qwen3TTSModel
import torch, base64, io
import soundfile as sf

class Qwen3TTSSynthesizer:
    def __init__(self, model_id: str, device: str) -> None:
        self._model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.bfloat16,
        )

    def synthesize(self, text: str, language: str) -> tuple[bytes, int]:
        # generate returns (wavs, sr) where sr is 24000
        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language="Russian",
            ref_audio=None,   # Base model: no ref audio needed for standard TTS
        )
        # convert numpy array to raw PCM16 bytes
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV", subtype="PCM_16")
        buf.seek(44)  # skip WAV header to return raw PCM16
        return buf.read(), sr
```

**Output sample rate:** 24000 Hz (confirmed from multiple sources). This is what `sample_rate_hz` field in the response must contain.

**Note on Base vs CustomVoice vs VoiceDesign variants:**
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`: standard TTS without reference audio (recommended for this use case)
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`: requires a reference audio clip for voice cloning
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`: requires a text voice description
- **Use `Qwen3-TTS-12Hz-1.7B-Base` for the sidecar.** It is the simplest and requires no reference audio at inference time.

### Pattern 3: Adapter Branch in voice_adapters.py

**What:** Each new STT provider gets an `if name == "{provider}":` branch in `transcribe_audio()`. Each new TTS provider gets an `if preferred == "{provider}":` branch in `synthesize_audio_with_provider()`. Missing `{NAME}_BASE_URL` raises `RuntimeError`, never silently falls through (D-02).

**STT adapter branch (qwen3_asr):**
```python
# Source: rag_demo_system/backend/voice_adapters.py (extend existing pattern)
# In transcribe_audio(), order list already contains "qwen3_asr" when preferred="qwen3_asr"
base_url = os.getenv("QWEN3_ASR_BASE_URL")
if not base_url:
    raise RuntimeError("Qwen3-ASR service unavailable: QWEN3_ASR_BASE_URL not set")
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
```

**Key difference from current code:** Current `transcribe_audio()` has a fallback chain. For new providers under D-02, when the preferred provider is one of the new ones and its `BASE_URL` is missing, we must raise immediately rather than continuing the loop. The safest approach: check `if not base_url and name == preferred: raise RuntimeError(...)` before the loop continues.

**TTS adapter branch (qwen3_tts):**
```python
if preferred == "qwen3_tts":
    base_url = os.getenv("QWEN3_TTS_BASE_URL")
    if not base_url:
        raise RuntimeError("Qwen3-TTS service unavailable: QWEN3_TTS_BASE_URL not set")
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

### Pattern 4: Health Status Registration

**What:** Each new sidecar gets an entry in `build_voice_statuses()` using the existing `_service_status()` helper.

```python
# Source: rag_demo_system/backend/voice_adapters.py build_voice_statuses()
def build_voice_statuses() -> dict[str, dict[str, Any]]:
    return {
        # ... existing entries ...
        "qwen3_asr": _service_status("qwen3_asr", os.getenv("QWEN3_ASR_BASE_URL")),
        "qwen3_tts": _service_status("qwen3_tts", os.getenv("QWEN3_TTS_BASE_URL")),
        "voxtral":   _service_status("voxtral",   os.getenv("VOXTRAL_BASE_URL")),
    }
```

### Pattern 5: Contract Tests

**What:** Each new adapter gets a test in `test_voice_adapters_official.py` using the `FakeResponse` helper and `monkeypatch`. Test verifies: correct URL called, correct JSON payload, correct response fields, correct provider string.

**Template (qwen3_asr):**
```python
def test_transcribe_audio_supports_qwen3_asr_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("QWEN3_ASR_BASE_URL", "http://qwen3-asr.local")

    def fake_post(url, headers=None, data=None, timeout=None, files=None, json=None):
        calls.append({"url": url, "json": json})
        return FakeResponse(json_data={"text": "Лизинг одобрен", "provider": "qwen3_asr"})

    monkeypatch.setattr(voice_adapters.requests, "post", fake_post)

    data = voice_adapters.transcribe_audio("AQIDBA==", session_id="s1", preferred="qwen3_asr")

    assert data["text"] == "Лизинг одобрен"
    assert data["provider"] == "qwen3_asr"
    assert calls[0]["url"] == "http://qwen3-asr.local/transcribe"
    assert calls[0]["json"]["sample_rate_hz"] == 24000
```

### Pattern 6: Frontend Options

**What:** Two new `<option>` elements added to `sttProviderSelect`, one to `ttsProviderSelect`, in `frontend/index.html`. No JavaScript changes needed (selectors already persist to `localStorage` and send via `buildSessionUpdate()`).

```html
<!-- In sttProviderSelect -->
<option value="qwen3_asr">qwen3_asr</option>
<option value="voxtral">voxtral</option>

<!-- In ttsProviderSelect -->
<option value="qwen3_tts">qwen3_tts</option>
```

`voiceProviderReady()` in `app.js` currently only checks sidecar health for specific named providers (`yandex_realtime`, `yandex_speechkit`, `oss_russian`). All other providers pass through silently. The new providers fall into the default path so no `voiceProviderReady()` changes are needed unless the planner wants explicit health-indicator support (Claude's discretion).

### Anti-Patterns to Avoid

- **Shared venv for new models:** Never install `qwen-tts`, `qwen-asr`, or `transformers>=5.2` into the main `.venv`. Doing so breaks the `transformers==4.37.2` pin that other backend code depends on.
- **Silent fallback:** Never add new providers to the fallback `order` list in `transcribe_audio()` without also adding hard-fail logic when they are the `preferred` provider and their BASE_URL is unset. D-02 requires this.
- **Wrong model variant for TTS:** Do not use `Qwen3-TTS-12Hz-1.7B-CustomVoice` or `VoiceDesign` in the sidecar; they require reference audio or a voice description at inference time. Use `Base`.
- **Wrong sample rate in TTS response:** Always return `sample_rate_hz: 24000` from the Qwen3-TTS sidecar. The backend's `_pcm16_b64_to_wav_bytes()` uses this value for WAV header construction.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| TTS inference for Qwen3-TTS | Custom PyTorch inference loop | `qwen-tts` library | Handles tokenizer 12Hz timing, streaming, and audio reconstruction internally |
| ASR inference for Qwen3-ASR | faster-whisper or raw transformers pipeline | `qwen-asr` library | Official library; vLLM backend option available; handles language auto-detect |
| Voxtral model loading | Manual weight download + custom inference | `transformers>=5.2.0` with `VoxtralRealtimeForConditionalGeneration` | Official support since v5.2.0; streaming processor included |
| Audio resampling | Manual scipy/numpy resample | Pass PCM16 in temp WAV file; let model library handle resampling | `qwen-asr` resamples to 16kHz internally; avoids numpy dependency in sidecar venv |
| Health check logic | Custom logic | Existing `_service_status()` in `voice_adapters.py` | Reuse; it is already correct |
| Test mock HTTP | Custom mock class | Existing `FakeResponse` in `test_voice_adapters_official.py` | Reuse; consistent test pattern |

**Key insight:** The `qwen-tts` and `qwen-asr` packages are thin wrappers that abstract tokenizer timing, chunking, and GPU dispatch. Using raw transformers for these models requires reimplementing 12Hz token scheduling for TTS and the vLLM dispatch path for ASR.

---

## VRAM Budget (H100 NVL 94GB or A100 80GB)

| Model | Dtype | Estimated VRAM | Notes |
|-------|-------|----------------|-------|
| Qwen3-30B-A3B (brain) | bfloat16 | ~60 GB | Active while sidecar is idle |
| Qwen3.5-35B-A3B (brain target) | bfloat16 | ~70 GB | Active while sidecar is idle |
| Qwen3-TTS-1.7B | bfloat16 | ~4-8 GB | Run only when tts_provider=qwen3_tts |
| Qwen3-ASR-1.7B | bfloat16 | ~4 GB (est.) | Run only when stt_provider=qwen3_asr |
| Voxtral Mini 4B | bfloat16 | ~16 GB minimum, ~35 GB observed | Run only when stt_provider=voxtral |

**Coexistence conclusion:** On a 94 GB H100, Qwen3.5-35B-A3B (70 GB) + Voxtral (16 GB minimum) totals 86 GB: tight but feasible at minimum VRAM mode. On A100 80 GB: do not load Voxtral simultaneously with the larger brain model (86 GB would OOM). Load sidecars only when their profile is active; use supervisorctl to stop/start.

**Confidence:** MEDIUM (model sizes calculated from parameter counts; observed VRAM numbers from community reports, not official benchmarks)

---

## Common Pitfalls

### Pitfall 1: transformers Version Conflict
**What goes wrong:** Installing `transformers>=5.2.0` in the shared `.venv` breaks the `transformers==4.37.2` pin used by the RAG backend.
**Why it happens:** `pip install transformers` upgrades the shared package.
**How to avoid:** Each new sidecar has its own venv. Never run `pip install` commands without first activating `.venv-{name}`.
**Warning signs:** Import errors in the RAG backend; `sentence-transformers` breaking.

### Pitfall 2: Wrong TTS Model Variant
**What goes wrong:** `Qwen3-TTS-12Hz-1.7B-CustomVoice` raises an error at inference time because `ref_audio` is not provided.
**Why it happens:** Three model variants exist; `CustomVoice` and `VoiceDesign` require extra inputs.
**How to avoid:** Use `Qwen/Qwen3-TTS-12Hz-1.7B-Base` as the HuggingFace repo ID.
**Warning signs:** Model raises `ValueError` about missing reference audio on first `/speak` call.

### Pitfall 3: Voxtral Input Sample Rate Mismatch
**What goes wrong:** The backend sends PCM16 at 24kHz; Voxtral expects 16kHz. Transcription output is garbage.
**Why it happens:** The voice pipeline records at 24kHz (existing backend convention). Voxtral's feature extractor uses 16kHz (confirmed from `VoxtralRealtimeFeatureExtractor` config: `sampling_rate=16000`).
**How to avoid:** The Voxtral sidecar must resample the incoming audio from 24kHz to 16kHz before passing to the processor. Use `audio.resample(processor.feature_extractor.sampling_rate)` from `mistral_common` or `librosa`/`scipy`.
**Warning signs:** Non-zero transcription output that is phonetically implausible or contains random tokens.

### Pitfall 4: Qwen3-ASR Returns Wrong Language
**What goes wrong:** `Qwen3ASRModel.transcribe()` with `language=None` auto-detects and may return a non-Russian transcript or produce correct Russian text tagged as wrong language.
**Why it happens:** Auto-detect works but we want to force Russian for this use case.
**How to avoid:** Pass `language="ru"` explicitly. Confirmed from docs: `language=None` means auto-detect; `language="ru"` forces Russian.
**Warning signs:** Transcripts in English when the user speaks Russian.

### Pitfall 5: Silent Fallback Bug in Existing transcribe_audio()
**What goes wrong:** When `preferred="qwen3_asr"` and `QWEN3_ASR_BASE_URL` is unset, the existing loop continues and falls through to `sensevoice` or `whisper`. The log shows `provider: sensevoice` instead of `qwen3_asr`. Benchmark results are silently corrupted.
**Why it happens:** Current loop logic appends fallbacks after `preferred` and continues if `base_url` is falsy.
**How to avoid:** For the three new providers, add an explicit check before the loop: `if preferred in ("qwen3_asr", "qwen3_tts", "voxtral") and not os.getenv(f"{preferred.upper()}_BASE_URL"): raise RuntimeError(...)`. This implements D-02.
**Warning signs:** `provider` field in log says `sensevoice` even though UI selector shows `qwen3_asr`.

### Pitfall 6: Voxtral Streaming API Mismatch
**What goes wrong:** The sidecar tries to use the streaming chunked inference API which requires session state between calls; the stateless `/transcribe` endpoint breaks.
**Why it happens:** The Transformers docs show a complex streaming API with `input_features_generator`. For batch (non-streaming) transcription this is not needed.
**How to avoid:** Use the offline/batch API for the sidecar (pass complete audio, call `model.generate()`). Streaming is only needed if the sidecar ever processes live mic chunks. The current sidecar pattern receives complete audio from the backend.
**Warning signs:** `AttributeError` or `TypeError` on `model.generate()` call; missing `padding_cache` argument.

---

## Code Examples

### Whisper Server (reference: full existing pattern)
```python
# Source: rag_demo_system/services/whisper_server.py
# This is the exact pattern to replicate for Qwen3-ASR and Voxtral.
# Key elements: create_app(), create_unavailable_app(), _build_default_app()
# The app = _build_default_app() at module level allows uvicorn to import it.
```

### Voxtral Batch Inference (offline transcription, non-streaming)
```python
# Source: https://huggingface.co/docs/transformers/model_doc/voxtral_realtime
import torch
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
import numpy as np

processor = AutoProcessor.from_pretrained("mistralai/Voxtral-Mini-4B-Realtime-2602")
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    "mistralai/Voxtral-Mini-4B-Realtime-2602",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# audio_array: numpy float32 at 16kHz
inputs = processor(audio_array, return_tensors="pt")
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

### Qwen3-ASR Batch Inference
```python
# Source: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
from qwen_asr import Qwen3ASRModel
import torch

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
results = model.transcribe(audio="path/to/audio.wav", language="ru")
text = results[0].text.strip()
```

### Qwen3-TTS Base Inference
```python
# Source: https://github.com/QwenLM/Qwen3-TTS
from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
wavs, sr = model.generate_voice_clone(
    text="Добрый день, чем могу помочь?",
    language="Russian",
)
# sr == 24000, wavs[0] is a numpy float array
```

### Hard-Fail Pattern in voice_adapters.py (D-02)
```python
# Source: design from D-02 constraint
NEW_PROVIDERS = {"qwen3_asr", "voxtral"}  # STT providers requiring hard fail

def transcribe_audio(audio_b64: str, session_id: str, preferred: str = "sensevoice") -> dict[str, Any]:
    if preferred in NEW_PROVIDERS:
        base_url = os.getenv(f"{preferred.upper()}_BASE_URL")
        if not base_url:
            raise RuntimeError(f"{preferred} service unavailable: {preferred.upper()}_BASE_URL not set")
        # proceed directly without fallback loop
        resp = requests.post(...)
        ...
    # existing fallback logic for other providers unchanged
    ...
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Whisper-based ASR (whisper_server.py) | Qwen3-ASR via qwen-asr library | 2025 model release | Dedicated Russian-class model replaces English-focused Whisper |
| CosyVoice TTS (existing) | Qwen3-TTS via qwen-tts library | 2025-01-22 first release | Lower latency, official Russian support |
| Voxtral API-only (old assumption) | Voxtral as self-hosted local sidecar | 2026-02-15 HF Transformers support | No privacy constraint violation; local inference possible |

**Deprecated/outdated notes:**
- Playbook reference to `Qwen3-TTS-12Hz-1.7B` without variant suffix was underspecified: three variants exist. Use `Qwen3-TTS-12Hz-1.7B-Base`.
- D-05 warning about playbook model names being LOW confidence was correct for Voxtral (confirmed: weights exist, local sidecar is viable).

---

## Open Questions

1. **Qwen3-TTS Base model: does `generate_voice_clone` require ref_audio=None to work?**
   - What we know: `Base` variant is documented for voice cloning without custom voice. The `generate_voice_clone` method exists on the model.
   - What's unclear: Whether `ref_audio=None` falls back to a default speaker or requires a preset speaker token.
   - Recommendation: Test sidecar startup with a simple Russian text call immediately after model load. Use `generate_voice_clone(text=..., language="Russian")` with no `ref_audio`. If it fails, use `model.generate()` with explicit speaker tokens from the Base checkpoint.

2. **qwen-asr 0.0.6: does it support a WAV file path or does it require the raw array?**
   - What we know: Docs show `model.transcribe(audio="path/to/audio.wav")` where path is a file path string.
   - What's unclear: Whether it can accept a raw numpy array directly (useful for skipping temp file writes).
   - Recommendation: Use temp WAV file path (same as whisper_server.py) for maximum compatibility; optimize later.

3. **Voxtral VRAM on A100 80GB: is 16 GB minimum a realistic floor?**
   - What we know: Model card states >=16 GB required; community deployments report ~35 GB observed.
   - What's unclear: Whether the 35 GB figure includes model weights + KV cache for long audio.
   - Recommendation: Plan for 35 GB observed; document that Voxtral and the brain model cannot run simultaneously on A100 80 GB.

---

## Environment Availability

Step 2.6: Environment availability for the LOCAL development machine is not relevant here since all sidecars are designed for GPU server deployment. The local machine is macOS Darwin without a CUDA GPU.

For the **GPU server** (Azure H100 94GB or A100 80GB, Ubuntu), the following are required:

| Dependency | Required By | Available (server) | Version | Fallback |
|------------|------------|-----------|---------|----------|
| CUDA + cuDNN | All 3 sidecars | Expected (GPU VM provisioned in Phase 5) | CUDA 12.x | No fallback |
| Python 3.10+ | qwen-tts, qwen-asr | Expected | 3.10-3.12 | No fallback |
| python3-venv | Per-service venv creation | Expected (standard Ubuntu) | any | No fallback |
| pip | All sidecar installs | Expected | 23+ | No fallback |
| HuggingFace Hub access | Model weight download | Must be confirmed on server | -- | Pre-download weights |
| CUDA-capable PyTorch | All 3 sidecars | Installed per sidecar venv | 2.x | No fallback for GPU inference |

**Note:** `flash-attn` is optional but recommended for Qwen3-TTS and Qwen3-ASR to reduce VRAM. It requires CUDA and GCC at install time; include it in requirements files as an optional extra.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.3.4 |
| Config file | none (run from `rag_demo_system/` directory) |
| Quick run command | `python3 -m pytest tests/test_voice_adapters_official.py -x -q` |
| Full suite command | `python3 -m pytest tests/ -x -q` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| VPROV-01 | qwen3_tts branch calls `/speak`, returns audio_b64 + sample_rate_hz=24000 + provider="qwen3_tts" | unit | `python3 -m pytest tests/test_voice_adapters_official.py::test_synthesize_audio_supports_qwen3_tts_service -x` | No - Wave 0 |
| VPROV-02 | qwen3_asr branch calls `/transcribe`, returns text + provider="qwen3_asr" | unit | `python3 -m pytest tests/test_voice_adapters_official.py::test_transcribe_audio_supports_qwen3_asr_service -x` | No - Wave 0 |
| VPROV-03 | voxtral branch calls `/transcribe`, returns text + provider="voxtral" | unit | `python3 -m pytest tests/test_voice_adapters_official.py::test_transcribe_audio_supports_voxtral_service -x` | No - Wave 0 |
| VPROV-04 | All 6 existing + 3 new contract tests pass | unit | `python3 -m pytest tests/test_voice_adapters_official.py -x` | Partial (6 of 9 tests exist) |
| VPROV-05 | No automated test; verify visually that selectors show new options | manual | n/a | n/a |
| VPROV-02/03 D-02 | Hard-fail raises RuntimeError when BASE_URL unset and provider is preferred | unit | `python3 -m pytest tests/test_voice_adapters_official.py::test_qwen3_asr_hard_fail_when_unconfigured -x` | No - Wave 0 |

### Sampling Rate

- **Per task commit:** `python3 -m pytest tests/test_voice_adapters_official.py -x -q`
- **Per wave merge:** `python3 -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_voice_adapters_official.py` -- add 3 new provider contract tests (VPROV-01, VPROV-02, VPROV-03) + 2 hard-fail tests (D-02)
- [ ] File already exists; append to it, do not create a new file

*(No new test files needed; no new conftest needed; existing infrastructure covers all phase requirements)*

---

## Sources

### Primary (HIGH confidence)

- `rag_demo_system/services/whisper_server.py` - exact STT sidecar pattern to follow
- `rag_demo_system/services/vosk_tts_server.py` - exact TTS sidecar pattern to follow
- `rag_demo_system/backend/voice_adapters.py` - integration points verified by code read
- `rag_demo_system/tests/test_voice_adapters_official.py` - contract test structure verified
- `rag_demo_system/frontend/index.html` - selector HTML structure verified
- https://huggingface.co/Qwen/Qwen3-ASR-1.7B - repo ID, Russian support, sample rate (16kHz), qwen-asr library
- https://huggingface.co/docs/transformers/model_doc/voxtral_realtime - VoxtralRealtime API, sampling_rate=16000, transformers>=5.2.0
- PyPI `pip3 index versions qwen-tts` and `qwen-asr` - version numbers verified locally on 2026-03-25

### Secondary (MEDIUM confidence)

- https://github.com/QwenLM/Qwen3-TTS - TTS output sample rate 24kHz (cross-verified via multiple community sources), Russian in 10-language list
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base - Base variant confirmed, inference API structure
- https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 - self-hostable, Apache 2.0, >=16 GB VRAM
- https://mistral.ai/news/voxtral-transcribe-2 - Russian support confirmed in 13-language list

### Tertiary (LOW confidence)

- Community reports of ~35 GB observed VRAM for Voxtral (single source; no official benchmark)
- `Qwen3-TTS-12Hz-1.7B-Base` inference with `ref_audio=None` (inferred from API structure, not explicitly tested)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all packages confirmed from PyPI, all repo IDs confirmed from HuggingFace
- Architecture: HIGH - derived directly from existing codebase patterns (whisper_server.py, vosk_tts_server.py)
- Pitfalls: HIGH for transformers conflict and silent fallback; MEDIUM for VRAM figures
- Voxtral self-hosting: HIGH (confirmed Apache 2.0, HF model card, transformers docs)

**Research date:** 2026-03-25
**Valid until:** 2026-06-25 (90 days; qwen-tts and qwen-asr are early-stage packages, check for updates before final implementation)
