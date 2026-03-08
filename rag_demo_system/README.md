# RAG Demo System (Micro Leasing)

This demo lives entirely under `rag_demo_system/`. The current MVP target is:

- one browser UI
- one FastAPI backend that also serves the UI
- switchable `our_rag` and `dify_rag`
- Russian voice I/O
- one public HTTPS link for client testing

For the first deployable MVP, the recommended shape is:

- Vast GPU host for:
  - backend and UI on `:8000`
  - Qdrant on `:6333`
  - Qwen / vLLM on `:8001`
  - SenseVoice official server on `:50000`
  - CosyVoice official server on `:50001`
  - Whisper fallback service on `:50002`
- Dify Cloud over API, not self-hosted Dify
- `ngrok` for the public HTTPS link

## What Is In Repo

```text
rag_demo_system/
  backend/                    # FastAPI app and RAG logic
  services/                   # Auxiliary voice services, including Whisper fallback
  frontend/                   # Static UI, served by backend at "/"
  config/                     # YAML config and system prompt
  scripts/                    # Launch and smoke helpers
  tests/                      # Lightweight repo tests
  requirements.txt            # Backend requirements
  requirements-voice-fallback.txt
```

## MVP Deployment Model

### `our_rag`
- indexes `knowledge_base/kb_faq_ru.md` locally into Qdrant
- retrieves with vector + BM25 + rerank
- calls your OpenAI-compatible Qwen endpoint

### `dify_rag`
- calls a Dify Chatflow over API
- Dify Cloud stores and indexes the same KB corpus independently
- the backend normalizes Dify retriever output into the same UI contract

### Voice
- browser sends PCM16 audio chunks over `WS /ws/voice`
- backend calls:
  - SenseVoice official API directly for STT
  - CosyVoice official API directly for TTS
  - optional local Whisper fallback service if SenseVoice is unavailable

## Important Runtime Notes

- The UI is now served by the FastAPI backend itself. For public testing, expose only port `8000`.
- The frontend no longer hardcodes `127.0.0.1:8000`. It uses the current origin by default.
- If you want Dify Cloud to use the same Qwen model as `our_rag`, Dify Cloud must be able to reach your vLLM endpoint. The simplest MVP path is an additional `ngrok` tunnel for port `8001`.
- Dify Cloud is being used to test Dify's retrieval and orchestration behavior. If you later move to self-hosted Dify and keep the same Chatflow design, KB corpus, chunking, retrieval settings, rerank settings, and LLM endpoint, the RAG quality should stay materially the same. Hosting location is not what defines retrieval quality.
- Current local verification gap: full `backend.app` import is still slow/sticky on this Mac environment, so server instructions are designed for the actual GPU host rather than pretending this laptop is the target runtime.

## Environment

Copy `.env.example` to `.env` and fill in the values.

Key variables:

- `RAG_LLM_BASE_URL`
- `RAG_LLM_MODEL`
- `DIFY_API_BASE_URL`
- `DIFY_API_KEY`
- `SENSEVOICE_BASE_URL`
- `SENSEVOICE_API_STYLE=official`
- `COSYVOICE_BASE_URL`
- `COSYVOICE_API_STYLE=official`
- `COSYVOICE_SPK_ID`
- `WHISPER_BASE_URL`

## Useful Endpoints

- `GET /`
- `GET /api/health`
- `GET /api/backends`
- `GET /api/voice/status`
- `POST /api/index`
- `POST /api/retrieve`
- `POST /api/chat`
- `POST /api/voice/chat`
- `WS /ws/voice`

## Smoke Test

```bash
./rag_demo_system/scripts/smoke_test.sh
```

It checks:

- health
- backend availability
- voice service status
- local KB indexing
- `our_rag` chat path

If Dify is configured, use the backend switch in the UI or post the same question twice:

```bash
curl -sS http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Какие требования к лизингу грузового транспорта?","backend":"our_rag"}'

curl -sS http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Какие требования к лизингу грузового транспорта?","backend":"dify_rag"}'
```

## Voice Service Notes

### SenseVoice
- The backend supports the official SenseVoice FastAPI service contract via `SENSEVOICE_API_STYLE=official`.
- It sends WAV-wrapped audio to `/api/v1/asr`.

### CosyVoice
- The backend supports the official CosyVoice FastAPI service contract via `COSYVOICE_API_STYLE=official`.
- It calls `/inference_sft`.
- The adapter strips the WAV container and returns raw PCM16 back to the browser so playback works.

### Whisper fallback
- `services/whisper_server.py` is the local fallback service.
- It exposes:
  - `GET /health`
  - `POST /transcribe`

## Stack Control

Repo-level launcher:

```bash
./rag_demo_system/scripts/stack.sh status
./rag_demo_system/scripts/stack.sh up
./rag_demo_system/scripts/stack.sh down
./rag_demo_system/scripts/stack.sh smoke
```

The supervisor mode is the fallback mode for container-style GPU hosts. It is the relevant mode for Vast if Docker Compose is not usable there.

### ngrok

The repo includes an example config at:

```bash
rag_demo_system/scripts/ngrok.yml.example
```

For the fair benchmark case, run two tunnels:

- `app` for `:8000`
- `llm` for `:8001`
