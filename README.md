# Leasing AI Pipeline

End-to-end audio intelligence and voice assistant system for a leasing company. Starts with raw call recordings, extracts structured insights, builds a knowledge base, and serves it through a production voice assistant over SIP telephony and a text chat widget.

> **Handover state (2026-05-12).** Active development branch is `feature/voice-pipeline`. Read [Resuming Development](#resuming-development) below before making any changes — there are known unverified-on-production fixes on this branch.

## Resuming Development

If you cloned this repo to continue the work:

1. **Active branch is `feature/voice-pipeline`** (not `main`). Everything voice + chat + classifier + RAG lives here. `main` is the original 2026-01 call-analysis pipeline and is not where current work belongs.
2. **Last fully production-validated tag**: `pre-master-plan-with-chat-2026-05-08`. If you hit anything unexplained, you can reset to this tag and start clean.
3. **Current HEAD**: `voice-pipeline-baseline-2026-05-09-evening`. Includes a master-plan sweep that fixed ~28 catalogued bugs and added a chat widget, but several items are **shipped but not yet production-tested** (latency on live calls, automatic audio testing harness, NBRB FX API fallback, B4/B6 audio bugs). See the handover documents the previous developer is sending separately.
4. **Server bootstrap**: see `rag_demo_system/scripts/provision_server.sh`. Tokens and step-by-step commands are in the handover package.
5. **Original developer's planning docs are local-only** — they are NOT in this repo. The handover package contains: `ARCHITECTURE_OVERVIEW_RU.md`, `ANALYSIS.md`, `HANDOVER.md`, `PROJECT_LOG.md`, `ONBOARDING.md`, the master-plan progress file, and a Russian handover letter explaining what was tested and what wasn't.

## Branches

After the 2026-05-12 cleanup, the repo has three live branches:

| Branch | Status | Description |
|--------|--------|-------------|
| `main` | Baseline (2026-04 README touch only) | Original call analysis pipeline: WhisperX transcription, per-call NLU, knowledge base build. Not the current product surface. |
| `feature/voice-pipeline` | **Active development**, 864 commits ahead of main | Production voice assistant + chat widget: SIP telephony, RAG, classifier, calculator + SMS tools, dispatcher state machine. |
| `feature/kb-viz` | Sandbox demo, frozen | 3D knowledge-base visualizer (offline HTML, separate from runtime). Kept for occasional client previews. |

### Archived (deleted) branches

These were merged into `feature/voice-pipeline` (history preserved in its commit graph) or were one-off spikes superseded by the current stack. Deleted on 2026-05-12 to keep the repo navigable.

| Branch | Reason |
|---|---|
| `feature/chat-widget` | Merged into voice-pipeline at tag `pre-master-plan-with-chat-2026-05-08` |
| `feature/kb-refinement` | Merged (topical KB swap, tag `kb-topical-shipped-2026-05-03` removed during prune) |
| `feature/section-3-apply-turn` | Merged (apply-turn refactor) |
| `feature/tool-use` | Merged (calculator + SMS tools) |
| `codex/split-voice-providers` | Spike, superseded |
| `codex/yandex-realtime-voice-integration` | Yandex SpeechKit demo, not productized |
| `temp-kb-upload` | Temporary KB upload branch |
| `claude/qwen-voice-next` | Qwen3-Omni multimodal benchmark experiment. **Tip preserved at tag `archive/qwen-voice-next-2026-04-05`** if anyone wants to revisit. |

To inspect any deleted branch's history: `git fetch origin --tags` then `git log archive/qwen-voice-next-2026-04-05`, or use the GitHub web UI on the tagged commit.

## main: Call Analysis Pipeline

The original product (2026-01). Processes raw audio recordings from a leasing call center into a structured knowledge base. Not actively maintained on this repo; kept as historical baseline.

**Pipeline stages:** WhisperX transcription → per-call LLM analysis → NLU export → batch rollups → global aggregation → embedding deduplication → knowledge base build (JSON / YAML / Markdown).

**Layout:**

```
scripts/                  # Pipeline stages (00_setup through 50_build_kb)
prompts/                  # LLM prompts for analysis (Russian)
demo_ui/                  # Local Streamlit review UI
Makefile                  # make transcribe / make analyze-calls / make kb / etc.
```

## feature/voice-pipeline: Production Voice Assistant + Chat Widget

Russian-language voice assistant deployed for a Belarusian leasing company. SIP telephony via Jambonz, plus a browser chat widget that exercises the same dispatcher state machine.

**Stack:**

- Whisper STT (faster-whisper 1.2.1, GPU when driver ≥ 570)
- Silero TTS v5_4_ru
- Brain LLM: Qwen3.5-30B-A3B-FP8 via vLLM 0.19.0 (port 8787)
- Classifier / SessionAgent: Qwen3-4B-Instruct-2507-FP8 via vLLM (port 8788) — dedicated so classifier never queues behind the brain
- RAG: Qdrant + BM25 + cross-encoder reranker, chunk dedup at retrieval
- Tools: leasing calculator (client's 1C API), SMS (sms-assistent.by), live NBRB exchange-rate fetch with public FX API fallback
- Telephony: Jambonz 0.9.6, 6 SIP accounts, separated audio tracks
- Frontend: SIP operator monitor (`/sip_monitor.html`), chat widget (`/chat_widget.html`)

**Key features shipped on this branch:**

- Pydantic `ClassifierOutput` schema with utterance-grounding post-validators
- Dispatcher state machine with single transaction model, change-confirm flow, FireCalc/EmitReadback/EmitClarify/EmitChangeConfirm/FireLLMFallback actions
- Multi-currency calculator (BYN/USD/EUR/RUB/CNY/PLN/CHF/...) with per-currency NBRB rate lookup and configurable Phys-person drift to BYN
- Barge-in with VAD + RMS floor, listen_mode (`стоп`, `помолчи` → auto-exit)
- Per-turn `[LATENCY:]` instrumentation (parsed by `scripts/analyze_latency.sh`)
- SIP monitor: per-session profile panels, real-time event stream, chat sessions appear alongside voice
- Voice harness (`scripts/voice_harness.py`) for automated chat-mode and audio-mode regression scenarios

**Layout:**

```
rag_demo_system/
├── backend/
│   ├── app.py                  # FastAPI: WebSocket + Jambonz SIP + chat endpoints
│   ├── engine.py               # RAG: embedding + BM25 + reranker + dedup
│   ├── llm.py                  # vLLM streaming (OpenAI-compatible)
│   ├── voice_adapters.py       # Whisper STT + Silero TTS
│   ├── voice_session.py        # Per-call state, barge-in, listen_mode
│   ├── session.py              # ClientProfile + state machine
│   ├── turn_dispatcher.py      # apply_turn → execute_action loop
│   ├── classifier_schema.py    # ClassifierOutput Pydantic schema + grounding
│   ├── profile_prompts.py      # NLU prompts, change-confirm/readback builders
│   ├── router.py               # Intent classification routing
│   └── tools/                  # calculator, sms, filler
├── config/app.yaml             # All config: LLM, RAG, embedding, reranker, classifier
├── docker/jambonz/             # Jambonz SIP stack
├── frontend/                   # demo.html, sip_monitor.html, chat_widget.html
├── services/                   # whisper_server.py, silero_tts_server.py
├── scripts/                    # provision/deploy/smoke/restart/voice_harness/...
└── tests/                      # 1158 unit tests, 16-scenario chat regression
```

**All dependencies pinned:** Docker images (Jambonz 0.9.6, drachtio, rtpengine), Python packages (vLLM 0.19.0, faster-whisper 1.2.1, silero 0.5.5), HuggingFace model revisions via `QWEN_MAIN_REVISION` / `QWEN_SESSIONAGENT_REVISION`.

## Quick Start

### Call analysis pipeline (`main`)

```bash
git clone -b main https://github.com/yauhenifutryn/leasing.git
cd leasing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
mkdir -p audio/        # drop .wav/.mp3 files here
make transcribe && make analyze-calls && make kb
```

### Voice assistant + chat widget (`feature/voice-pipeline`)

Fresh GPU VM (Sesterce / Jarvis Labs / bare-metal H100 / H200 / A100):

```bash
ssh -i <your_key> <user>@<NEW_IP>

# On the server, paste your token block FIRST (HF_TOKEN, CALCULATOR_API_TOKEN,
# SMS_API_LOGIN, SMS_API_PASSWORD, SMS_SENDER_NAME, JAMBONZ_SIP_PASSWORD_*).
# Exact values are in the local handover package, NOT in this repo.

cd /workspace 2>/dev/null || cd /ephemeral 2>/dev/null || cd ~
git clone -b feature/voice-pipeline https://github.com/yauhenifutryn/leasing.git
cd leasing/rag_demo_system

bash scripts/provision_server.sh    # ~15-25 min: deps + venv + vLLM + model dl
bash scripts/smoke_test.sh          # validates the four supervisord services
bash scripts/deploy_jambonz.sh      # Docker SIP stack + Zoiper accounts wired
```

Sanity check from your laptop:

```bash
curl http://<NEW_IP>:8000/api/health   # → {"ok":true,...}
```

Then:

- Chat: `http://<NEW_IP>:8000/chat_widget.html`
- Operator monitor: `http://<NEW_IP>:8000/sip_monitor.html`
- Voice: Zoiper → `voice.<NEW_IP>.nip.io` UDP. `deploy_jambonz.sh` prints the realm + accounts at the end.

### Routine updates on an already-provisioned server

```bash
cd /ephemeral/leasing/rag_demo_system
git pull origin feature/voice-pipeline
bash scripts/regenerate_env_and_restart.sh   # rebuilds .env from current env, clean restart
bash scripts/smoke_test.sh
```

### Backend-only restart (vLLM / Whisper / Silero stay hot)

```bash
.venv/bin/supervisorctl -c scripts/supervisord.conf restart backend
```

### Forgot tokens before provisioning?

```bash
# Re-export tokens in the same shell, then:
bash scripts/set_tokens.sh   # upserts into .env and restarts backend only
```

## Key Scripts

| Script | Purpose |
|---|---|
| `scripts/provision_server.sh` | Idempotent first-boot setup (deps, venv, vLLM build, model download, supervisord, .env). |
| `scripts/regenerate_env_and_restart.sh` | Rewrites `.env` from current shell env and does a clean restart. |
| `scripts/set_tokens.sh` | Upserts Calculator/SMS/CRM tokens + `USD_BYN_RATE` into `.env` and restarts backend only. |
| `scripts/restart_all.sh` | Full stack restart with shutdown / GPU verify / vLLM warm-up wait / per-service health checks. Use instead of `supervisorctl restart all`. |
| `scripts/stack.sh` | Supervisor lifecycle wrappers (`up` / `down`). |
| `scripts/doctor.sh` | Stack health diagnosis. |
| `scripts/smoke_test.sh` | End-to-end smoke: services up, KB indexed, classifier roundtrip, chat path. |
| `scripts/deploy_jambonz.sh` | Docker SIP stack deploy + 6 Zoiper accounts wired. |
| `scripts/voice_harness.py` | Automated scenario runner. `--mode chat` exercises the full backend dispatcher with no audio. `--mode audio` drives `/ws/jambonz` end-to-end through Silero + Whisper. |
| `scripts/analyze_latency.sh` | Parses `[LATENCY:]` markers from backend.log; produces per-stage distributions + action-kind shares + dispatch overhead. |
| `scripts/dump_latency.py` | Per-metric latency distribution from `.state/logs.jsonl`. |
| `scripts/simulate_dialogue.py` | Offline classifier + dispatcher exerciser (no audio, no LLM). |

## GPU Server Notes

- Recommended: **H100 80GB** (Sesterce / ShadeCloud, tested)
- Alternative: **A100 80GB** (Jarvis Labs, tested)
- Avoid **5090 / Blackwell**: bleeding-edge drivers, often breaks
- Use `tmux` for long-running processes to survive SSH disconnects
- After reboot: `bash scripts/restart_all.sh`
- NVIDIA driver 570+ required for GPU Whisper; 550 falls back to CPU (acceptable)

## License

This project is proprietary. It is strictly forbidden to use this code for commercial purposes.
The code is open for public viewing solely for portfolio demonstration and evaluation.
See the [LICENSE](LICENSE) file for specific terms.
