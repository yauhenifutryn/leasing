# Leasing AI Pipeline

End-to-end audio intelligence and voice assistant system for a leasing company. Starts with raw call recordings, extracts structured insights, builds a knowledge base, and serves it through a production voice assistant over WebSocket.

> **Branch cleanup planned.** Once all active work is complete, the core branches will be merged into `main` and experiments archived as tags. See the [restructure plan](docs/superpowers/plans/2026-04-08-repo-restructure.md).

## Branches

This repository contains several long-lived branches. Each represents a distinct stage or direction of the project. They share a common foundation but have diverged significantly.

| Branch | Status | Commits ahead of main | Description |
|--------|--------|----------------------|-------------|
| `main` | Baseline | -- | Original call analysis pipeline: transcription, NLU, knowledge base generation |
| `feature/voice-pipeline` | Production | 250+ | Full voice assistant with tool use: STT, TTS, VAD, RAG, calculator, SMS, LLM intent routing |
| `claude/qwen-voice-next` | Experimental | 204 | Qwen3-Omni benchmarking, multi-model voice testing |
| `codex/split-voice-providers` | Spike | 3 | Quick experiment: split-brain voice provider options |
| `codex/yandex-realtime-voice-integration` | Spike | 5 | Yandex SpeechKit realtime voice demo |

### main: Call Analysis Pipeline

The original product. Processes raw audio recordings from a leasing call center into a structured knowledge base.

**Pipeline stages:**

1. **Transcription**: WhisperX (GPU) with optional diarization, Whisper CLI as fallback
2. **Per-call analysis**: Structured prompts to LLM for intent extraction, resolution tracking, QA pairs
3. **NLU export**: Flat question/answer pairs for downstream ingestion
4. **Batch rollups**: Deduplicated summaries across groups of 10-20 calls
5. **Global aggregation**: Consolidated intent and FAQ clusters
6. **Embedding deduplication**: SentenceTransformers clustering of similar questions
7. **Knowledge base build**: Final FAQ entries in JSON, YAML, and Markdown

**Key components:**

```
scripts/                  # Pipeline stages (00_setup through 50_build_kb)
prompts/                  # LLM prompts for analysis (Russian)
demo_ui/                  # Local web UI for running pipeline steps
requirements.txt          # Python deps (PyTorch cu118, WhisperX, pyannote)
Makefile                  # All pipeline targets (make transcribe, make kb, etc.)
```

**Makefile targets:**

| Target | Description |
|--------|-------------|
| `make check` | Verify ffmpeg and API keys |
| `make transcribe` | WhisperX GPU transcription |
| `make transcribe-cpu` | CPU fallback (slow) |
| `make analyze-calls` | Per-call LLM analysis |
| `make nlu-export` | Flat Q&A export (JSONL) |
| `make rollup` | Batch-level rollups |
| `make aggregate` | Global aggregation |
| `make dedup` | Embedding-based FAQ deduplication |
| `make kb` | Build final KB (JSON + YAML) |
| `make kb-markdown` | Export KB to Markdown |

Also includes a Streamlit review UI (`scripts/review_app.py`) for human validation of KB entries.

### feature/voice-pipeline: Production Voice Assistant

Browser-based Russian-language voice assistant built on top of the knowledge base. This is the production system, tested with the client.

**Stack:** Whisper STT, Silero TTS (v4_ru), Qwen3.5-35B-A3B-FP8 via vLLM, Qdrant vector search, BM25 + reranker hybrid retrieval.

**Capabilities:**

- WebSocket streaming audio (push-to-talk and continuous modes)
- Silero VAD for voice activity detection in continuous conversation
- Sentence-boundary detection for streaming TTS (speak as tokens arrive)
- Barge-in support (interrupt the bot mid-response)
- Conversation memory across turns
- Intent routing (greeting, off-topic, meta-questions)
- Stress dictionary for proper name pronunciation
- Latin-to-Cyrillic transliteration for brand names

```
rag_demo_system/
├── backend/
│   ├── app.py                  # FastAPI, WebSocket voice handler
│   ├── engine.py               # RAG: embedding + BM25 + reranking
│   ├── llm.py                  # vLLM streaming (OpenAI-compatible)
│   ├── voice_adapters.py       # Whisper STT + Silero TTS
│   ├── voice_session.py        # Session state, barge-in tracking
│   ├── sentence_detector.py    # Streaming sentence boundaries
│   ├── vad.py                  # Silero VAD wrapper
│   ├── audio_input.py          # Transport adapter (WebSocket, future SIP)
│   ├── router.py               # Intent classification
│   ├── memory.py               # Turn history
│   ├── text_utils.py           # Answer cleaning, address validation
│   ├── state.py                # Session store, event logging
│   └── settings.py             # Config loading (app.yaml + .env)
├── config/
│   ├── app.yaml                # All config: LLM, RAG, embedding, reranker
│   ├── system_prompt_ru_v2.txt # System prompt (Russian)
│   ├── stress_dictionary.yaml  # TTS stress marks for proper names
│   └── transliteration.yaml    # Brand name transliteration
├── frontend/
│   └── demo.html               # Browser UI
├── services/
│   ├── whisper_server.py       # STT microservice
│   └── silero_tts_server.py    # TTS microservice
├── scripts/
│   ├── provision_server.sh     # One-command GPU server setup
│   ├── restart_all.sh          # Restart all services
│   └── doctor.sh               # Health check
└── tests/                      # Unit and integration tests
```

**Deployment:** Single GPU server (H100 or equivalent). One-command provisioning via `provision_server.sh`. See [deployment playbook](docs/server_deployment_playbook.md).

### feature/tool-use: Tool Use Layer

**Now merged into feature/voice-pipeline.** Tool use is part of the production voice assistant.

The voice assistant uses mid-conversation tool calling via native OpenAI function calling (`tools=[]`). A fast LLM intent classifier routes each message to either the TOOL path (clean prompt for reliable tool calling) or the RAG path (full KB context for information questions).

**How it works:**

1. Fast LLM call (~200ms) classifies message as TOOL or RAG intent
2. TOOL path: clean system prompt + user message, tools always available
3. RAG path: system prompt + memory + KB fragments for information questions
4. When tool is called: filler phrase plays, tool executes, result injected, LLM presents results
5. Recalculations include previous params so model can adjust specific values
6. After results, bot offers SMS with payment schedule link

**Tools:**

| Tool | Status | Purpose |
|------|--------|---------|
| `calculator` | Ready | Leasing payment schedule via 1C calculator API |
| `send_sms` | Ready | Send schedule link via SMS (sms-assistent.by) |
| `escalate_to_human` | Planned | Session summary + lead creation in AMO CRM |

```
rag_demo_system/backend/tools/
├── __init__.py          # Registry: get_tool_schemas(), get_tool(), init_tools()
├── base.py              # ToolDefinition ABC
├── calculator.py        # 1C calculator API integration
├── sms_sender.py        # SMS delivery
└── filler.py            # Filler phrases per tool
```

### claude/qwen-voice-next: Experimental

Experimental branch exploring Qwen3-Omni as an alternative to the split STT/LLM/TTS pipeline. Includes a benchmarking framework for comparing voice model configurations. Not intended for production.

### codex/split-voice-providers and codex/yandex-realtime-voice-integration

Short-lived spikes. Split-brain voice provider experiment and Yandex SpeechKit realtime demo respectively. Will be archived as tags during cleanup.

## Quick Start

**For the call analysis pipeline (main):**

```bash
git clone git@github.com:yauhenifutryn/leasing.git
cd leasing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
mkdir -p audio/        # drop .wav/.mp3 files here
make transcribe && make analyze-calls && make kb
```

**For the voice assistant (feature/voice-pipeline):**

```bash
git clone --branch feature/voice-pipeline git@github.com:yauhenifutryn/leasing.git
cd leasing/rag_demo_system
cp .env.example .env   # fill in all credentials
HF_TOKEN=hf_... bash scripts/provision_server.sh
```

After provisioning, add tool credentials to `.env`:
```
CALCULATOR_API_TOKEN=...
SMS_API_LOGIN=...
SMS_API_PASSWORD=...
```

The server IP must be whitelisted by the client before external API calls will work.

## GPU Server Notes

- Recommended: **A100 40 GB** (~$0.6/hr on vast.ai) or **H100**
- Alternative: **4090** (cheaper, less stable under sustained load)
- Avoid **5090/Blackwell**: requires bleeding-edge drivers, often breaks
- On the server, use only `conda` environment (`lease`); do not mix with `.venv`
- Use tmux for long-running processes to survive SSH disconnects

## License

This project is proprietary. It is strictly forbidden to use this code for commercial purposes.
The code is open for public viewing solely for portfolio demonstration and evaluation.
See the [LICENSE](LICENSE) file for specific terms.
