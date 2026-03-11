# Server Voice Options

This repo now supports two split-brain voice options for the same backend RAG/LLM:

1. `yandex_speechkit`
2. `oss_russian`

In both modes, your own backend still owns:

- transcription routing
- retrieval
- RAG
- LLM answer generation
- citations and memory

The voice layer changes only STT and TTS.

## Option 1: Yandex SpeechKit

Use this when you want:

- better Russian voice quality
- lower latency than a fully local stack
- your own RAG and LLM to stay in control

### What runs

- backend on `:8000`
- Qdrant on `:6333`
- vLLM/Qwen on `:8001`
- no extra STT/TTS local service is required

### Environment

Copy:

```bash
cp rag_demo_system/.env.voice.yandex-speechkit rag_demo_system/.env
```

### Server commands

From repo root:

```bash
cd /workspace/leasing
python3 -m venv rag_demo_system/.venv
source rag_demo_system/.venv/bin/activate
pip install -r rag_demo_system/requirements.txt
```

Then start your LLM endpoint and Qdrant the same way you already do.

Then start the backend:

```bash
cd /workspace/leasing/rag_demo_system
source .venv/bin/activate
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### UI test flow

1. Open the demo UI.
2. In `Voice provider`, choose `yandex_speechkit`.
3. In `Backend`, choose `our_rag` or `dify_rag`.
4. Press `Connect`.
5. Hold to talk.

## Option 2: OSS Russian

Use this when you want:

- no Yandex dependency
- Russian-capable fully local speech I/O
- your own RAG and LLM to stay in control

This option uses:

- `Vosk` for STT
- `Vosk TTS` for Russian TTS

### What runs

- backend on `:8000`
- Qdrant on `:6333`
- vLLM/Qwen on `:8001`
- Vosk STT on `:50010`
- Vosk TTS on `:50011`

### Environment

Copy:

```bash
cp rag_demo_system/.env.voice.oss-russian.example rag_demo_system/.env
```

### Model setup

From repo root:

```bash
mkdir -p /workspace/leasing/models
cd /workspace/leasing/models
curl -L -o vosk-model-small-ru-0.22.zip https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip -o vosk-model-small-ru-0.22.zip
```

`Vosk TTS` uses `vosk-model-tts-ru-0.9-multi` by model name in the service config.

### Python env for OSS voice services

```bash
cd /workspace/leasing
python3 -m venv rag_demo_system/.venv-voice-oss
source rag_demo_system/.venv-voice-oss/bin/activate
pip install -r rag_demo_system/requirements-voice-oss.txt
```

### Start OSS voice services

Terminal 1:

```bash
cd /workspace/leasing/rag_demo_system
source .venv-voice-oss/bin/activate
export VOSK_MODEL_PATH=/workspace/leasing/models/vosk-model-small-ru-0.22
python -m uvicorn services.vosk_server:app --host 0.0.0.0 --port 50010
```

Terminal 2:

```bash
cd /workspace/leasing/rag_demo_system
source .venv-voice-oss/bin/activate
export VOSK_TTS_MODEL_NAME=vosk-model-tts-ru-0.9-multi
export VOSK_TTS_SAMPLE_RATE_HZ=22050
python -m uvicorn services.vosk_tts_server:app --host 0.0.0.0 --port 50011
```

### Start backend

Terminal 3:

```bash
cd /workspace/leasing
python3 -m venv rag_demo_system/.venv
source rag_demo_system/.venv/bin/activate
pip install -r rag_demo_system/requirements.txt
cd rag_demo_system
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### UI test flow

1. Open the demo UI.
2. In `Voice provider`, choose `oss_russian`.
3. In `Backend`, choose `our_rag` or `dify_rag`.
4. Press `Connect`.
5. Hold to talk.

## Notes

- `yandex_realtime` still exists in the codebase, but it is not the preferred path if you do not trust the Yandex model/tool layer.
- `yandex_speechkit` and `oss_russian` both keep the brain on your side.
- `oss_russian` is expected to be weaker than Yandex on voice quality and often on latency, but it is the most practical fully open-source Russian stack in this repo.
