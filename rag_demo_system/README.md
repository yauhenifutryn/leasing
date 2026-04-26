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

## Architecture (post `refactor-v1`, 2026-04-26)

The orchestrator (`rag_demo_system/backend/app.py::_stream_voice_response`)
delegates every turn to a small structural pipeline instead of the legacy
5-gate block. There is exactly one orchestration path now — no feature
flag, no parallel legacy.

```
   user utterance
        │
        ▼
┌─────────────────┐
│ SessionAgent    │  Qwen3-4B-FP8 → strict-JSON `ClassifierOutput`
│ (classifier)    │  (intent, slots, semantic flags)
└────────┬────────┘
         │  parsed + utterance-grounded
         ▼
┌─────────────────────────────────────────┐
│ apply_turn (turn_dispatcher.py)         │
│   step 1: CHANGE_PENDING + confirm      │
│   step 2: READBACK_PENDING + confirm    │
│   pre-compute: grounded patches         │
│                + utterance fallbacks    │
│                + year-form disambig     │
│                + implied flips          │
│                + partition_patches      │
│   step 4: delta on captured field       │
│           → EmitChangeConfirm           │
│   step 5: first-time additive captures  │
│   step 5b: COLLECTING + calc-intent     │
│           → EmitClarify(missing_fields) │
│   step 6: COLLECTING + complete         │
│           → EmitReadback                │
│   step 6: CONFIRMED + complete          │
│           → FireCalc                    │
│   step 6b: action=sms → FireSMS         │
│   step 7: action=invalid → FireOOR      │
│   else: FireLLMFallback                 │
└────────┬────────────────────────────────┘
         │  TurnAction
         ▼
┌─────────────────────────────────────────┐
│ execute_action (turn_dispatcher.py)     │
│   yields TTS chunks, calls calc/SMS,    │
│   streams LLM, manages barge-in         │
└─────────────────────────────────────────┘
```

**Key invariants:**
- **One state machine**, in `backend/profile_state.py`. `ProfileState` ∈
  {COLLECTING, READBACK_PENDING, CHANGE_PENDING, CONFIRMED}. State
  transitions happen *only* in apply_turn.
- **One mutation point.** Profile fields are written by apply_turn
  (step 1 confirm-apply via `apply_pending_change`, step 5 first-time
  additive). No other code path mutates the profile.
- **Sole-source classifier.** SessionAgent is the only producer of
  `ClassifierOutput`. Fast-path / skip-classifier / classifier-error
  branches all synthesise a minimal `ClassifierOutput` so apply_turn
  always has a non-None input.
- **Year-form grounding is semantic.** "X лет/года/год" is disambiguated
  by current `ProfileState` + filled fields, not by surface regex.
  Age-phase fills `age_years`; term-phase fills `term_months`.
- **Utterance fallbacks** (`extract_subject`, `extract_client_type`, …)
  fire only when the classifier omitted a slot AND the profile field is
  empty — explicit-change turns can never be overridden silently.
- **Memory + RAG threading.** The orchestrator stamps `session.memory_block`
  before dispatch; FireLLMFallback reads it and prepends to the LLM
  prompt so follow-up RAG / conversation turns retain prior context.
- **Stale-turn discard.** Each turn carries a monotonic `turn_id`;
  apply_turn dispatch drops the turn if a newer one finalised first.

**Production adapters** (`backend/execute_adapters.py`) bridge the pure
turn dispatch to FastAPI / WebSocket / vLLM / FunAudioLLM / Jambonz:
`LLMStreamBackend`, `TtsSink`, `CalcAdapter`, `RagFuture`.

## Quick Start (Any GPU Server)

Canonical fresh-server flow (7 steps, in order):

```bash
# 1. SSH into the VM (from your Mac)
#    ssh -i ~/.ssh/jarvislabs sesterce@<IP>

# 2. Export the HuggingFace token (needed to download models during provision)
export HF_TOKEN=hf_YOUR_TOKEN

# 3. Clone + checkout
cd /ephemeral
git clone https://github.com/yauhenifutryn/leasing.git
cd leasing && git checkout feature/voice-pipeline
cd rag_demo_system

# 4. Provision (installs apt deps, GPU driver check, 2 venvs, downloads
#    models ~60 GB, writes .env with blank tool-use tokens, starts stack)
bash scripts/provision_server.sh

# 5. Now provision has told you exactly what to export. Paste the tokens
#    you got from the client and apply them:
export CALCULATOR_API_TOKEN='...' \
       SMS_API_LOGIN='...' \
       SMS_API_PASSWORD='...'
bash scripts/set_tokens.sh     # patches .env + restarts backend only

# 6. Smoke test (verifies services up, calculator + SMS auth works, KB indexed)
bash scripts/smoke_test.sh

# 7. Deploy SIP (6 accounts: test, sergey, ilya, john, mike, victor)
bash scripts/deploy_jambonz.sh

# Optional: browser UI over the internet
ngrok http 8000
```

For routine code updates on an already-provisioned server:

```bash
cd /ephemeral/leasing
git pull
cd rag_demo_system
bash scripts/restart_all.sh       # restarts all services, reloads .env
bash scripts/smoke_test.sh
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
| `provision_server.sh` | Full server setup (apt, venvs, models, stack). Idempotent. Writes `.env` with blank tool-use tokens; use `set_tokens.sh` after. |
| `set_tokens.sh` | Patch Calculator / SMS / CRM tokens from shell env into `.env`, restart backend only (vLLM stays up). Run this after provision on a fresh server. |
| `regenerate_env_and_restart.sh` | Rewrite the full `.env` from template + clean restart of the whole stack. Heavier than `set_tokens.sh`; use only when vLLM GPU-util or model paths need rewriting. |
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
- `VAD_SILENCE_MS` / `SILERO_VAD_PATH` - turn-taking controls. Default is `900` ms (end-of-speech silence threshold). Raised from 500 ms in Fix 1.4 (2026-04-19) so the client can pause mid-sentence to think without the bot jumping in; trade-off is ~0.4 s added perceived latency at the end of every user turn. The smart adaptive-endpointing successor ships in Section 4 of the post-MVP master plan.
- `CUDA_HOME` - CUDA toolkit path (for flashinfer JIT)

## Troubleshooting

| Situation | Command |
|-----------|---------|
| First time setup | `export HF_TOKEN=... && bash scripts/provision_server.sh` (then `export CALCULATOR_API_TOKEN=... SMS_API_LOGIN=... SMS_API_PASSWORD=... && bash scripts/set_tokens.sh`) |
| Routine code/config update | `git pull && bash scripts/restart_all.sh` |
| Rotate tool tokens | `export CALCULATOR_API_TOKEN=... && bash scripts/set_tokens.sh` |
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
