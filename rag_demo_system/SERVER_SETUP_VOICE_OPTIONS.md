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
./rag_demo_system/scripts/setup_vast_voice.sh yandex_speechkit
```

Then start your LLM endpoint and Qdrant the same way you already do.

Copy the env:

```bash
cp rag_demo_system/.env.voice.yandex-speechkit rag_demo_system/.env
```

If you want the repo launcher to start vLLM too, fill `STACK_QWEN_CMD` in `rag_demo_system/.env`.

Then run everything from one terminal:

```bash
cd /workspace/leasing
./rag_demo_system/scripts/stack.sh up
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
./rag_demo_system/scripts/setup_vast_voice.sh oss_russian
```

Copy the env:

```bash
cp rag_demo_system/.env.voice.oss-russian.example rag_demo_system/.env
```

If you want the repo launcher to start vLLM too, fill `STACK_QWEN_CMD` in `rag_demo_system/.env`.

Then run everything from one terminal:

```bash
cd /workspace/leasing
./rag_demo_system/scripts/stack.sh up
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
- `stack.sh up` uses `supervisord` and starts only the auxiliary voice services required by `STACK_VOICE_PROFILE`.
