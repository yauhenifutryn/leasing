# Leasing AI Pipeline

End-to-end audio intelligence and voice assistant system for a leasing company. Starts with raw call recordings, extracts structured insights, builds a knowledge base, and serves it through a production voice assistant over WebSocket.

> **Branch cleanup planned.** Once all active work is complete, the core branches will be merged into `main` and experiments archived as tags. See the [restructure plan](docs/superpowers/plans/2026-04-08-repo-restructure.md).

## Branches

This repository contains several long-lived branches. Each represents a distinct stage or direction of the project. They share a common foundation but have diverged significantly.

| Branch | Status | Commits ahead of main | Description |
|--------|--------|----------------------|-------------|
| `main` | Baseline | -- | Original call analysis pipeline: transcription, NLU, knowledge base generation |
| `feature/voice-pipeline` | Production | 223 | Full voice assistant: STT, TTS, VAD, RAG, barge-in, session management |
| `feature/tool-use` | Active development | 235 | Tool use layer on top of voice-pipeline: calculator API, SMS, streaming tool loop |
| `claude/qwen-voice-next` | Experimental | 204 | Qwen3-Omni benchmarking, multi-model voice testing |
| `codex/split-voice-providers` | Spike | 3 | Quick experiment: split-brain voice provider options |
| `codex/yandex-realtime-voice-integration` | Spike | 5 | Yandex SpeechKit realtime voice demo |
| `feature/kb-viz` | Standalone add-on | small | KB vector-index visualization (static 2D/3D + optional live-query overlay with client feedback capture). Merges into `main` independently, no voice-pipeline dependencies. See [rag_demo_system/README.md](rag_demo_system/README.md#kb-visualization-client-demo--kb-quality-feedback-tool). |

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

Extends the voice assistant with mid-conversation tool calling. The LLM can invoke external APIs during a conversation using native OpenAI function calling (`tools=[]`).

**How it works:**

1. LLM detects user intent requires a tool (e.g., "calculate payments for a BMW X5")
2. Orchestrator sends a filler phrase to TTS ("One moment, calculating...")
3. Tool executes (HTTP call to external API)
4. Result injected back into LLM context
5. LLM continues streaming the spoken response with the results
6. After calculator results, the bot always offers to send the schedule via SMS

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

Experimental branch exploring Qwen3-Omni as an alternative to the split STT/LLM/TTS pipeline. Includes a benchmarking framework for comparing voice model configurations. Not intended for production. During the cleanup pass this branch is expected to become the tag `experiment/qwen-voice-next` and the branch will be deleted.

### feature/kb-viz: KB Visualization (Client Demo + Feedback Tool)

Standalone add-on. Projects the live Qdrant index via UMAP to 2D and 3D, emits self-contained Plotly HTMLs you can email to a client. Optional overlay service on `:8500` embeds client-typed questions in the same space, shows the top-5 matches, and captures Correct/Wrong verdicts with a comment. Feedback is appended as JSONL to `rag_demo_system/.state/kb_viz_feedback.jsonl` (same shape as the self-improvement pipeline so future aggregation scripts can union both sources).

Intended to merge into `main` independently of `feature/voice-pipeline`, so client demos run from a stable trunk. Does not touch `provision_server.sh` or the root `Makefile`. All targets scoped to `rag_demo_system/Makefile`.

```bash
git clone --branch feature/kb-viz git@github.com:yauhenifutryn/leasing.git leasing-kb-viz
cd leasing-kb-viz
make -C rag_demo_system kb-viz             # static HTMLs, ~40s
# Optional: live-query overlay + feedback capture
make -C rag_demo_system kb-viz-overlay-serve &
export KB_VIZ_PUBLIC_URL=https://<your-server>:8500/overlay_query
make -C rag_demo_system kb-viz-overlay-build
```

Full details in [rag_demo_system/README.md](rag_demo_system/README.md#kb-visualization-client-demo--kb-quality-feedback-tool).

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

**For the voice assistant (feature/tool-use, latest):**

```bash
git clone --branch feature/tool-use git@github.com:yauhenifutryn/leasing.git
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
