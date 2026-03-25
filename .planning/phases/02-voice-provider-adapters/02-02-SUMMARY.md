---
phase: 02-voice-provider-adapters
plan: 02
subsystem: voice-sidecars
tags: [fastapi, sidecar, qwen3-tts, qwen3-asr, voxtral, inference-server]
dependency_graph:
  requires:
    - rag_demo_system/services/whisper_server.py (STT sidecar pattern)
    - rag_demo_system/services/vosk_tts_server.py (TTS sidecar pattern)
  provides:
    - rag_demo_system/services/qwen3_tts_server.py
    - rag_demo_system/services/qwen3_asr_server.py
    - rag_demo_system/services/voxtral_server.py
    - rag_demo_system/requirements-qwen3-tts.txt
    - rag_demo_system/requirements-qwen3-asr.txt
    - rag_demo_system/requirements-voxtral.txt
  affects:
    - Phase 5 deployment scripts (sidecar venv setup)
    - Plan 02-01 adapters (these sidecars are the HTTP backends they call)
tech_stack:
  added:
    - qwen-tts==0.1.1 (Qwen3-TTS inference, isolated venv)
    - qwen-asr==0.0.6 (Qwen3-ASR inference, isolated venv)
    - transformers>=5.2.0 + VoxtralRealtimeForConditionalGeneration (isolated venv)
    - scipy (audio resampling 24kHz to 16kHz for Voxtral)
    - soundfile (WAV I/O for Qwen3-TTS PCM16 conversion)
    - mistral-common (Voxtral tokenizer backend)
  patterns:
    - create_app / create_unavailable_app / _build_default_app / app module-level pattern
    - deferred imports inside __init__ to isolate ImportError to _build_default_app
    - per-service isolated venv with dedicated requirements-{name}.txt
key_files:
  created:
    - rag_demo_system/services/qwen3_tts_server.py
    - rag_demo_system/services/qwen3_asr_server.py
    - rag_demo_system/services/voxtral_server.py
    - rag_demo_system/requirements-qwen3-tts.txt
    - rag_demo_system/requirements-qwen3-asr.txt
    - rag_demo_system/requirements-voxtral.txt
  modified: []
decisions:
  - "Qwen3-TTS synthesize() uses generate_voice_clone(language='Russian') hardcoded: qwen-tts API requires full language names, not ISO codes"
  - "Qwen3-TTS PCM16 extraction uses soundfile in-memory WAV write then 44-byte header skip, matching vosk_tts_server.py pattern"
  - "Voxtral _target_sr read from processor.feature_extractor.sampling_rate (dynamic, not hardcoded 16000) for forward-compatibility"
  - "Voxtral uses batch/offline API (processor -> model.generate -> batch_decode); streaming API (input_features_generator/padding_cache) explicitly excluded per Pitfall 6"
  - "Voxtral resamples via scipy.signal.resample from sample_rate_hz to target_sr only when they differ; handles any input rate, not just 24kHz"
metrics:
  duration_minutes: 2
  completed_date: "2026-03-25"
  tasks_completed: 3
  files_created: 6
  files_modified: 0
---

# Phase 02 Plan 02: Voice Provider Sidecar Servers Summary

**One-liner:** Three FastAPI GPU sidecar servers (Qwen3-TTS 24kHz PCM16, Qwen3-ASR temp-WAV, Voxtral with 24kHz-to-16kHz scipy resampling) each with isolated pip requirements files following the established whisper_server.py / vosk_tts_server.py pattern.

## Objective

Create the HTTP inference servers that voice adapter branches (Plan 01) call. Each sidecar runs in its own Python venv with model-specific dependencies, keeping the shared backend venv (transformers==4.37.2) unchanged.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Qwen3-TTS sidecar server and requirements | f4e4d03 | qwen3_tts_server.py, requirements-qwen3-tts.txt |
| 2 | Qwen3-ASR sidecar server and requirements | 61e98f2 | qwen3_asr_server.py, requirements-qwen3-asr.txt |
| 3 | Voxtral sidecar server and requirements | 0acca2a | voxtral_server.py, requirements-voxtral.txt |

## What Was Built

### Qwen3-TTS Sidecar (qwen3_tts_server.py)

Serves `/health` and `/speak` following `vosk_tts_server.py`. `Qwen3TTSSynthesizer` loads `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (Base variant, not CustomVoice or VoiceDesign) via `qwen-tts==0.1.1`. `synthesize()` calls `generate_voice_clone(text=..., language="Russian")`, writes the numpy waveform to an in-memory soundfile WAV buffer, skips the 44-byte header, and returns raw PCM16 bytes at 24kHz. The `/speak` response includes `audio_b64` and `sample_rate_hz`.

### Qwen3-ASR Sidecar (qwen3_asr_server.py)

Serves `/health` and `/transcribe` following `whisper_server.py`. `Qwen3ASRTranscriber` loads `Qwen/Qwen3-ASR-1.7B` via `qwen-asr==0.0.6`. `transcribe_pcm16()` writes PCM16 bytes to a temporary WAV file at the input sample rate, calls `model.transcribe(audio=path, language=language)` (language passes through from request, not hardcoded), cleans up the temp file in a `finally` block, and returns `results[0].text.strip()`.

### Voxtral Sidecar (voxtral_server.py)

Serves `/health` and `/transcribe` following `whisper_server.py`. `VoxtralTranscriber` loads `mistralai/Voxtral-Mini-4B-Realtime-2602` via `transformers>=5.2.0` using `VoxtralRealtimeForConditionalGeneration`. `transcribe_pcm16()` converts PCM16 bytes to float32 numpy, resamples via `scipy.signal.resample` from input rate to `processor.feature_extractor.sampling_rate` (16000), then uses the batch/offline API: `processor() -> model.generate() -> batch_decode()`. The streaming API (`input_features_generator`, `padding_cache`) is not used.

## Pitfalls Addressed

| Pitfall | Resolution |
|---------|------------|
| Pitfall 2: Wrong Qwen3-TTS model variant | Used `Base` variant (`Qwen3-TTS-12Hz-1.7B-Base`), not `CustomVoice` or `VoiceDesign` |
| Pitfall 3: Voxtral requires 16kHz input | `scipy.signal.resample` in `transcribe_pcm16()` resamples from any input rate to `_target_sr` |
| Pitfall 6: Voxtral streaming vs. batch API | Batch/offline API used exclusively; streaming API keywords absent from executable code |

## Requirements Files

| File | Primary Dep | Isolation |
|------|-------------|-----------|
| requirements-qwen3-tts.txt | qwen-tts==0.1.1 | No transformers pin; soundfile for WAV I/O |
| requirements-qwen3-asr.txt | qwen-asr==0.0.6 | No transformers pin |
| requirements-voxtral.txt | transformers>=5.2.0 | Separate from shared venv (4.37.2); scipy, mistral-common |

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None. All three sidecars are fully implemented following the specified patterns. Model loading will fail at runtime if the model weights are not available on the GPU server (handled gracefully by `create_unavailable_app`), but that is expected deployment-time behavior, not a stub.

## Self-Check: PASSED

Files exist:
- rag_demo_system/services/qwen3_tts_server.py: FOUND
- rag_demo_system/services/qwen3_asr_server.py: FOUND
- rag_demo_system/services/voxtral_server.py: FOUND
- rag_demo_system/requirements-qwen3-tts.txt: FOUND
- rag_demo_system/requirements-qwen3-asr.txt: FOUND
- rag_demo_system/requirements-voxtral.txt: FOUND

Commits exist:
- f4e4d03: FOUND (feat(02-02): add Qwen3-TTS FastAPI sidecar server and requirements)
- 61e98f2: FOUND (feat(02-02): add Qwen3-ASR FastAPI sidecar server and requirements)
- 0acca2a: FOUND (feat(02-02): add Voxtral FastAPI sidecar server and requirements)
