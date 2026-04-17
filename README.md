# Leasing AI Pipeline

End-to-end audio intelligence and voice assistant system for a leasing company. Starts with raw call recordings, extracts structured insights, builds a knowledge base, and serves it through a production voice assistant over WebSocket or SIP.

> **Branch cleanup planned.** Once all active work is complete, the core branches will be merged into `main` and experiments archived as tags.

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

Russian-language voice assistant with SIP telephony, tool use, and RAG. This is the production system deployed for the client.

**Stack:** Whisper STT (1.2.1), Silero TTS (v5_4_ru), Qwen3.5-35B-A3B-FP8 main brain via vLLM (0.19.0), Qwen3-4B-Instruct-2507-FP8 dedicated SessionAgent (classifier + profile extractor), Qdrant vector search, BM25 + cross-encoder reranker, Jambonz SIP (0.9.6).

**Capabilities:**

- SIP telephony via Jambonz (separated audio tracks, no echo)
- Multi-account SIP (6 concurrent users with per-user monitoring)
- WebSocket streaming audio (browser push-to-talk and continuous modes)
- Silero VAD for voice activity detection (mono mode, 0.35 threshold)
- Barge-in support (interrupt the bot mid-response)
- Semantic stop-command detection (`стоп`, `помолчи` etc. → `listen_mode`, auto-exit)
- Tool calling: leasing calculator (1C API), SMS sender (sms-assistent.by)
- `ClientProfile` state machine: fields collected once per session; bot does read-back before first calc and single-field change-confirmation on recalc
- Calculator MVP relaxation: no hardcoded defaults (missing fields raise `IncompleteProfileError`), prepaid range 0-40%, term 12-84 months, linear/annuity graph selection forwarded to API
- Currency policy (physical person): USD auto-converted to BYN at a configurable rate with explicit disclosure; EUR/RUB politely rejected
- Dedicated small-model SessionAgent on port 8788 (Qwen3-4B-Instruct-2507-FP8) so classifier does not queue behind the main LLM
- RAG with chunk deduplication (overlapping chunk removal at retrieval time)
- LLM intent routing (greeting, company questions, off-topic, tools)
- DTMF consent collection at call start (keypad 1/2, barge-in supported)
- Conversation memory across turns
- Stress dictionary for proper name pronunciation
- TTS abbreviation expansion (`ул.`→улица, `пр-т`→проспект, `г.`→город, `д.`→дом, `тел.`→телефон)
- Whisper hallucination filtering + domain-biased initial prompt (Ксения, линейный, аннуитет, нагрузка, ипэшник)
- Phone number TTS pronunciation fix
- Post-call quality analytics (automatic per-session transcript + LLM analysis)
- Self-improvement reports: KB gap detection, quality trends, flagged sessions, operational metrics (readback/change-confirm/USD-conversion/stop-command rates)

```
rag_demo_system/
├── backend/
│   ├── app.py                  # FastAPI, WebSocket + Jambonz SIP handlers
│   ├── engine.py               # RAG: embedding + BM25 + reranking + dedup
│   ├── retrieval_utils.py      # Vector filtering, chunk deduplication
│   ├── llm.py                  # vLLM streaming (OpenAI-compatible)
│   ├── voice_adapters.py       # Whisper STT + Silero TTS
│   ├── voice_session.py        # Session state, barge-in tracking, ClientProfile, listen_mode
│   ├── session.py              # ClientProfile dataclass + state machine
│   ├── session_analyzer.py     # Per-call LLM quality analyzer
│   ├── sentence_detector.py    # Streaming sentence boundaries
│   ├── vad.py                  # Silero VAD wrapper
│   ├── audio_input.py          # Transport adapter (WebSocket, SIP)
│   ├── router.py               # Intent classification (SessionAgent 8788 w/ main fallback)
│   ├── memory.py               # Turn history
│   ├── text_utils.py           # Answer cleaning, address validation, abbreviation expansion
│   ├── state.py                # Session store, event logging
│   ├── settings.py             # Config loading (app.yaml + .env) + TurnTakingConfig
│   └── tools/                  # Calculator, SMS sender, filler phrases
├── config/
│   ├── app.yaml                # All config: LLM, RAG, embedding, reranker, dedup
│   ├── system_prompt_ru_v2.txt # System prompt (Russian)
│   ├── stress_dictionary.yaml  # TTS stress marks for proper names
│   └── transliteration.yaml    # Brand name transliteration
├── docker/jambonz/             # Jambonz SIP stack (all images pinned)
├── frontend/
│   ├── demo.html               # Browser voice UI
│   └── sip_monitor.html        # SIP call monitor (per-user filtering)
├── services/
│   ├── whisper_server.py       # STT microservice
│   └── silero_tts_server.py    # TTS microservice
├── scripts/
│   ├── provision_server.sh               # One-command GPU server setup
│   ├── regenerate_env_and_restart.sh     # Rewrite .env from template + clean restart
│   ├── smoke_test.sh                     # Service verification + KB indexing
│   ├── terminal_tests.sh                 # 8 post-deploy correctness checks
│   ├── deploy_jambonz.sh                 # SIP telephony deployment
│   ├── restart_all.sh                    # Full stack restart
│   ├── doctor.sh                         # Health check
│   ├── kb_gap_report.py                  # Aggregate KB gaps + operational metrics
│   └── quality_report.py                 # Quality trends and flagged sessions
└── tests/                                # Unit and integration tests
```

**Deployment flow (in order):**

1. `provision_server.sh` -- first-time install, downloads models, writes `.env`, starts stack
2. `regenerate_env_and_restart.sh` -- use after config/code changes (picks up exported creds, rewrites `.env`, clean restart)
3. `smoke_test.sh` -- waits for services, indexes KB, verifies chat + SessionAgent
4. `terminal_tests.sh` -- 8 shell-level correctness checks (SA latency/JSON, calculator no-defaults, currency math, Whisper vocab, abbreviation expansion, KB retrieval, end-to-end chat)
5. `deploy_jambonz.sh` -- deploys SIP telephony, creates 6 accounts

**All dependencies pinned:** Docker images (Jambonz 0.9.6, drachtio, rtpengine), Python packages (vLLM 0.19.0, faster-whisper 1.2.1, silero 0.5.5), HuggingFace model revisions pinnable via `QWEN_MAIN_REVISION` / `QWEN_SESSIONAGENT_REVISION`. No `:latest` tags or `>=` ranges.

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

# Export credentials once per shell session
export HF_TOKEN=hf_...
export CALCULATOR_API_TOKEN='...'
export CALCULATOR_API_BASE_URL='https://personal.mikro-leasing.by/calculator/api'
export SMS_API_LOGIN='...'
export SMS_API_PASSWORD='...'
export SMS_SENDER_NAME='MikroLizing'

# First-time install (downloads models, writes .env with your exported creds, starts stack)
bash scripts/provision_server.sh

# Verify services + index KB
bash scripts/smoke_test.sh

# Run 8-test correctness suite
bash scripts/terminal_tests.sh

# SIP telephony (optional)
bash scripts/deploy_jambonz.sh
```

For routine updates on an already-provisioned server:

```bash
cd /ephemeral/leasing/rag_demo_system
git pull origin feature/voice-pipeline
bash scripts/regenerate_env_and_restart.sh   # preserves exported creds + .env values
bash scripts/smoke_test.sh
bash scripts/terminal_tests.sh
```

The server IP must be whitelisted by the client before external API calls will work.

## GPU Server Notes

- Recommended: **H100 80GB** (Sesterce/ShadeCloud, tested)
- Alternative: **A100 80GB** (Jarvis Labs, tested)
- Avoid **5090/Blackwell**: requires bleeding-edge drivers, often breaks
- Use tmux for long-running processes to survive SSH disconnects
- After instance reboot: `bash scripts/restart_all.sh`
- NVIDIA driver 570+ required for GPU Whisper; 550 falls back to CPU (acceptable)

## License

This project is proprietary. It is strictly forbidden to use this code for commercial purposes.
The code is open for public viewing solely for portfolio demonstration and evaluation.
See the [LICENSE](LICENSE) file for specific terms.
