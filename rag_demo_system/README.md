# RAG Demo System (Micro Leasing)

Voice assistant for Micro Leasing. One browser UI, one FastAPI backend, Russian voice I/O, knowledge-grounded answers.

## Current Stack

| Component | Choice | Version | Port |
|-----------|--------|---------|------|
| Brain LLM | Qwen3.5-35B-A3B-FP8 (vLLM) | 0.19.0 | 8787 |
| SessionAgent (classifier + profile extractor) | Qwen3-4B-Instruct-2507-FP8 (vLLM, dedicated instance) | 0.19.0 | 8788 |
| RAG | our_rag (Qdrant + e5-large + mmarco reranker + dedup) | - | - |
| STT | Whisper (faster-whisper, large-v3, GPU) + domain-biased prompt | 1.2.1 | 50002 |
| TTS | Silero v5_4_ru, speaker "xenia" (CPU) + abbreviation expansion | 0.5.5 | 50006 |
| SIP | Jambonz (FreeSWITCH + drachtio) | 0.9.6 | 5060 |
| Vector DB | Qdrant | latest | 6333 |
| Backend/UI | FastAPI | 0.115.6 | 8000 |
| Persona | Ксения (female voice assistant) | - | - |

The SessionAgent runs on its own vLLM instance so classifier calls do not
queue behind the 35B main model. On GPUs below ~75 GB it is automatically
disabled and the main LLM handles classification.

## Quick Start (Any GPU Server)

```bash
# Clone
git clone --branch feature/voice-pipeline https://github.com/yauhenifutryn/leasing.git
cd leasing/rag_demo_system

# Export credentials once per shell session (picked up by provision + regen scripts).
# Values never get committed to git; they are written into .env by the scripts.
export HF_TOKEN=hf_YOUR_TOKEN
export CALCULATOR_API_TOKEN='...'
export CALCULATOR_API_BASE_URL='https://personal.mikro-leasing.by/calculator/api'
export SMS_API_LOGIN='...'
export SMS_API_PASSWORD='...'
export SMS_SENDER_NAME='MikroLizing'

# Provision (downloads both models, writes .env with your creds, starts stack).
# Idempotent -- safe to re-run.
bash scripts/provision_server.sh

# Verify services + index KB
bash scripts/smoke_test.sh

# Run the 8-test correctness suite (SessionAgent, calculator, currency,
# Whisper vocab, abbreviations, KB retrieval, end-to-end chat)
bash scripts/terminal_tests.sh

# Deploy SIP telephony (creates 6 accounts: test, sergey, ilya, john, mike, victor)
bash scripts/deploy_jambonz.sh

# Expose to the internet (browser UI)
ngrok http 8000
```

For routine updates on an already-provisioned server:

```bash
git pull origin feature/voice-pipeline
bash scripts/regenerate_env_and_restart.sh   # preserves existing .env creds
bash scripts/smoke_test.sh
bash scripts/terminal_tests.sh
```

Works on: bare metal servers, Jarvis Labs (KVM), Vast.ai (Docker), RunPod, Lambda Labs, Hetzner.

## Server Requirements

| Spec | Minimum | Recommended |
|------|---------|-------------|
| GPU | H100 80GB / A100 80GB | H100 80GB+ |
| CPU | 16 vCPUs | 24+ vCPUs |
| RAM | 64 GB | 128+ GB |
| Disk | 200 GB | 500+ GB |
| OS | Ubuntu 22.04 | Ubuntu 22.04/24.04 |
| NVIDIA Driver | 550+ | Latest |

## Repository Structure

```
rag_demo_system/
  backend/           # FastAPI app, RAG engine, voice pipeline
  services/          # Voice sidecars (Whisper, Silero TTS)
  frontend/          # Browser UI (demo.html)
  config/            # YAML config, system prompt, reference voice
  scripts/           # Provisioning, stack control, diagnostics
  models/            # Silero VAD model (auto-downloaded)
  tests/             # Unit tests
```

## Scripts

| Script | Purpose |
|--------|---------|
| `provision_server.sh` | Full server setup (apt, venvs, models, stack). Idempotent. |
| `regenerate_env_and_restart.sh` | Rewrite `.env` from template + clean restart. Preserves tool-use creds (calculator, SMS, CRM) across regens by reading exported env vars first, then falling back to existing `.env` values. |
| `smoke_test.sh` | Verify all services (including SessionAgent), index KB if needed |
| `terminal_tests.sh` | 8 post-deploy correctness tests: SA latency/JSON, calculator no-defaults, calculator full profile, USD->BYN math, Whisper vocab, abbreviation expansion, KB retrieval, end-to-end chat |
| `deploy_jambonz.sh` | SIP telephony (Jambonz stack, accounts, monitor) |
| `restart_all.sh` | Full restart after instance reboot or code change |
| `doctor.sh` | Diagnose and auto-fix issues |
| `fix_cuda_and_verify.sh` | CUDA diagnostic for KVM/VM instances |
| `system_snapshot.sh` | Capture system info before/after install |
| `stack.sh` | Stack control (up/down/status/smoke) |
| `kb_gap_report.py` | Weekly aggregate: KB gaps + operational metrics (readback/change-confirm/stop-respect/USD-conversion/linear-success rates) |

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/demo.html` | GET | Browser voice assistant UI |
| `/sip_monitor.html` | GET | SIP call monitor (add `?user=X` to filter) |
| `/api/health` | GET | Backend health check |
| `/api/backends` | GET | Available RAG backends |
| `/api/voice/status` | GET | Voice services status |
| `/api/index` | POST | Index knowledge base |
| `/api/retrieve` | POST | RAG retrieval (debug) |
| `/api/chat` | POST | Text chat |
| `/api/jambonz/credentials` | GET | SIP account credentials |
| `/ws/voice` | WS | Browser voice WebSocket |
| `/ws/jambonz` | WS | Jambonz control WebSocket |
| `/ws/jambonz-audio` | WS | Jambonz audio WebSocket |
| `/ws/sip-monitor` | WS | SIP monitor event stream |

## Environment Variables

Key variables in `.env` (auto-generated by `provision_server.sh` / `regenerate_env_and_restart.sh`):

**Core models:**
- `RAG_LLM_BASE_URL` / `RAG_LLM_MODEL` - main brain vLLM (port 8787, Qwen 35B)
- `SESSIONAGENT_BASE_URL` / `SESSIONAGENT_MODEL` - dedicated SessionAgent (port 8788, Qwen 4B). Empty = fall back to main LLM (small-GPU mode).
- `WHISPER_BASE_URL` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` - STT config
- `SILERO_TTS_BASE_URL` / `SILERO_TTS_SPEAKER` / `SILERO_TTS_MODEL` - TTS config

**Tool use:**
- `CALCULATOR_API_BASE_URL` / `CALCULATOR_API_TOKEN` - 1C leasing calculator
- `USD_BYN_RATE` - MVP physlico USD->BYN conversion rate (default 3.0; remove when calculator API adds NBRB conversion)
- `SMS_API_LOGIN` / `SMS_API_PASSWORD` / `SMS_SENDER_NAME` - sms-assistent.by
- `CRM_WEBHOOK_URL` / `CRM_WEBHOOK_TOKEN` - lead escalation (pending)

**Model pinning (optional, for full reproducibility):**
- `QWEN_MAIN_REVISION` / `QWEN_SESSIONAGENT_REVISION` - HF commit SHAs. Empty = track `main` branch.

**Runtime:**
- `HF_HOME` - Model weights directory (auto-detected: `/ephemeral/models`, `/workspace/models`, or `$HOME/models`)
- `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` - lock starts to cached weights after first download
- `VAD_SILENCE_MS` / `SILERO_VAD_PATH` - turn-taking controls
- `CUDA_HOME` - CUDA toolkit path (for flashinfer JIT)

## Troubleshooting

| Situation | Command |
|-----------|---------|
| First time setup | `export HF_TOKEN=... && bash scripts/provision_server.sh` |
| Routine code/config update | `git pull && bash scripts/regenerate_env_and_restart.sh` |
| After instance restart | `bash scripts/restart_all.sh` |
| Verify services + RAG | `bash scripts/smoke_test.sh` |
| Verify correctness end-to-end | `bash scripts/terminal_tests.sh` |
| Something is broken | `bash scripts/doctor.sh` |
| CUDA issues on VM | `bash scripts/fix_cuda_and_verify.sh` |
| SessionAgent not responding | Check `.state/sessionagent.err.log`. On small GPUs (<75 GB) it is disabled intentionally -- classifier falls back to main LLM. |
| vLLM "No available memory for cache blocks" | Raise `gpu-memory-utilization` for the crashing instance; see defaults in `scripts/regenerate_env_and_restart.sh` (tuned per GPU size). |

## License

Proprietary. Open for portfolio demonstration and evaluation only.
See the [LICENSE](../LICENSE) file for terms.
