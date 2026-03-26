# Phase 5: Server Deployment and Benchmarks - Research

**Researched:** 2026-03-26
**Domain:** GPU server provisioning (TensorDock A100 80GB), bash automation, supervisord service management, HuggingFace model download, benchmark orchestration
**Confidence:** HIGH (all findings grounded in existing repo code and CONTEXT.md decisions)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Server Provisioning**
- D-01: Target is a TensorDock VM with A100 80GB. The provisioning script is written for this provider specifically.
- D-02: Full automation. Single script: apt packages, Docker, clone repo, create all venvs (backend + per-sidecar), download models from HuggingFace, write .env, start stack. SSH in, run one command, walk away.
- D-03: Model downloads use `huggingface-cli download` with HF_HOME set to a shared models directory. Handles resume on disconnect and caching. Requires HF_TOKEN env var for gated models.
- D-04: Driver handling: check `nvidia-smi` first. If it fails, trigger the ubuntu-drivers install procedure (`sudo apt install -y ubuntu-drivers-common && sudo ubuntu-drivers install`), reboot if needed, then verify again. If drivers are already present, skip the install entirely.

**Smoke Test Scope**
- D-05: Expanded smoke test validates three additional checks beyond the existing HTTP endpoints: (1) sidecar health endpoints, (2) vLLM readiness via a trivial completion request, (3) VRAM headroom via nvidia-smi parsing.
- D-06: Smoke test is profile-aware. Reads the active .env.bench profile to determine which sidecars should be running. Only fails on sidecars the current profile actually needs. Avoids false failures from intentionally-stopped services.
- D-07: Existing smoke_test.sh checks (UI root, /api/health, /api/backends, /api/voice/status, KB indexing, chat stream) are preserved and extended, not replaced.

**Benchmark Execution Flow**
- D-08: Single orchestrator script runs the full benchmark matrix sequentially. Steps: (1) RAG comparison, (2) brain comparison, (3) Omni hybrid vs best split, (4) STT/TTS matrix (conditional).
- D-09: Between benchmark steps, the orchestrator swaps models via `supervisorctl stop/start`. Stops old model service, updates .env with new profile, starts the new one, waits for /health 200, verifies VRAM with nvidia-smi before proceeding. Does NOT restart the full stack.
- D-10: After step 3 (Omni hybrid benchmark), the orchestrator always pauses. Prints Omni vs split pipeline comparison results on screen and waits for user input ('continue' to run step 4, 'skip' to finish). User decides whether STT/TTS matrix is needed based on Omni results.
- D-11: The orchestrator runs the comparison script automatically after each benchmark step. After step 1: our_rag vs dify_rag. After step 2: brain models. After step 3: Omni vs best split. Results are visible incrementally.

**Result Collection**
- D-12: JSONL result files are written to `rag_demo_system/results/` directory within the repo checkout. Naming: `bench_{profile}_{timestamp}.jsonl`. Directory is gitignored.
- D-13: Results are transferred off the server via scp/rsync manually. The orchestrator prints file paths and a ready-to-paste scp command at the end of each step.

### Claude's Discretion
- Exact provisioning script structure (bash functions, error handling, logging)
- Which apt packages beyond the playbook list are needed
- Exact HuggingFace model repo IDs for download commands (researcher confirms these)
- How the orchestrator detects the "winning" configuration from each step to carry forward
- Smoke test VRAM threshold value
- Result file timestamp format

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DEPLOY-02 | Server deployment script that provisions the stack from a fresh GPU VM | Covered by: provisioning script patterns from setup_vast_voice.sh, apt packages from playbook Steps 2-3, venv patterns for all 5 per-sidecar venvs, model download via huggingface-cli, driver detection logic (D-04), supervisord start |
| DEPLOY-03 | Smoke test script validates all services are healthy before benchmark execution | Covered by: existing smoke_test.sh extension pattern, sidecar /health endpoint convention, vLLM /health + trivial completion pattern, nvidia-smi VRAM check pattern, profile-aware check logic (D-06) |
</phase_requirements>

---

## Summary

Phase 5 is a pure infrastructure and execution phase. No new Python adapters or UI code. The work is two scripts plus an orchestrator:

1. **`provision_server.sh`** -- extends the existing `setup_vast_voice.sh` pattern to cover the full A100 80GB GPU server: driver check/install, apt packages, Docker, repo clone, all per-sidecar venvs, model downloads via `huggingface-cli`, `.env` generation, and stack start.

2. **`smoke_test.sh` (extended)** -- the existing 62-line script gains three new check groups: sidecar `/health` endpoints (profile-aware), vLLM `/health` plus a trivial completion request, and VRAM headroom parsed from `nvidia-smi`. Profile awareness is the critical new requirement: the script must read the active profile to know which sidecars to check.

3. **`benchmark_orchestrator.sh`** (new) -- sequential four-step runner that calls `benchmark_runner.py` with the correct `--profile`, swaps models between steps via `supervisorctl stop/start`, waits for `/health` 200, verifies VRAM, calls `benchmark_compare.py` after each step, and pauses for user input after step 3 before optionally proceeding to the STT/TTS matrix.

All three scripts operate within an already-defined, well-understood system. The codebase has every primitive already built: `stack_cli.py:sync_supervisor_programs()`, `benchmark_runner.py`, `benchmark_compare.py`, 7 `.env.bench.*` profiles, 85 fixture questions, and a working `smoke_test.sh`.

**Primary recommendation:** Build all three scripts as bash files that call the existing Python tooling. Do not refactor the Python layer. The provisioning script is the most complex deliverable; the orchestrator is the most operationally critical.

---

## Standard Stack

### Core
| Library / Tool | Version | Purpose | Why Standard |
|----------------|---------|---------|--------------|
| bash | system | All three scripts | Provisioning scripts must run on a fresh VM before any Python venv exists |
| ubuntu-drivers-common | apt | NVIDIA driver detection and install | Playbook Step 2, D-04; Azure and TensorDock supported path |
| huggingface-cli | bundled in `huggingface-hub==0.24.5` (already in backend venv) | Model download with resume/cache | D-03; handles interrupted downloads, HF_HOME caching |
| supervisord / supervisorctl | installed into `rag_demo_system/.venv` via `requirements.txt` + `supervisor` package | Service management between benchmark steps | Already in `setup_vast_voice.sh` and `stack_cli.py` |
| nvidia-smi | system (GPU driver) | VRAM headroom check | D-05, D-09; smoke test and pre-model-load verification |
| python-dotenv | already in backend venv | Profile overlay in benchmark_runner.py | Already used; no new dependency |
| scp / rsync | system | Result file transfer | D-13 |

### Supporting
| Library / Tool | Version | Purpose | When to Use |
|----------------|---------|---------|-------------|
| Docker + docker-compose-plugin | apt | Qdrant container | Playbook Step 3; Qdrant runs via docker-compose |
| curl | apt | Health check polling in bash scripts | Smoke test, orchestrator health wait loop |
| git | apt | Repo clone | Provisioning step |
| python3-venv | apt | Per-sidecar venv creation | 5 sidecars need isolated venvs |
| jq | apt (optional) | JSON parsing in bash for nvidia-smi output | Useful for VRAM check; fallback is python one-liner |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| bash provisioning script | Python provisioning script | Python requires a working venv before it can run; bash is the only safe choice for a fresh VM entrypoint |
| bash orchestrator | Python orchestrator | Python is cleaner for complex logic; however, the orchestrator's main loop is simple sequential steps. Bash is sufficient and avoids a new script dependency. Either is acceptable; bash keeps it in the same style as smoke_test.sh |
| ubuntu-drivers-common | Manual CUDA toolkit install | Manual path is fragile on Azure/TensorDock; ubuntu-drivers-common is the supported path per playbook |

**Installation (on fresh VM):**
```bash
sudo apt update
sudo apt install -y git curl unzip python3 python3-venv python3-pip docker.io docker-compose-plugin ubuntu-drivers-common jq
```

---

## Architecture Patterns

### Recommended Project Structure (new files only)
```
rag_demo_system/
├── scripts/
│   ├── provision_server.sh          # NEW: full A100 VM provisioning
│   ├── benchmark_orchestrator.sh    # NEW: sequential 4-step benchmark runner
│   ├── smoke_test.sh                # EXTEND: add sidecar, vLLM, VRAM checks
│   ├── setup_vast_voice.sh          # EXISTING: extend with new sidecar venvs
│   ├── stack_cli.py                 # EXISTING: no changes
│   ├── benchmark_runner.py          # EXISTING: no changes
│   ├── benchmark_compare.py         # EXISTING: no changes
│   └── supervisord.conf             # EXISTING: already has qwen3_omni entry
├── results/                         # NEW (gitignored): JSONL output files
├── .env.bench.baseline              # EXISTING
├── .env.bench.dify_rag              # EXISTING
├── .env.bench.brain_upgrade         # EXISTING
├── .env.bench.omni_hybrid           # EXISTING
└── ...
```

### Pattern 1: Driver Detection Before Install (D-04)
**What:** Check if nvidia-smi works before touching drivers. Skip install if already functional.
**When to use:** Provision script entry, before any GPU-dependent step.
**Example:**
```bash
if ! nvidia-smi &>/dev/null; then
  echo "[provision] NVIDIA driver not found — installing ubuntu-drivers-common"
  sudo apt install -y ubuntu-drivers-common
  sudo ubuntu-drivers install
  echo "[provision] Reboot required to load new driver"
  echo "[provision] Re-run this script after reboot."
  exit 1
fi
echo "[provision] nvidia-smi OK: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
```

### Pattern 2: Per-Sidecar Venv Creation
**What:** Each model sidecar (qwen3_tts, qwen3_asr, voxtral, qwen3_omni, sensevoice) has its own venv under `rag_demo_system/.venv-{name}`. Never install sidecar deps into the shared `.venv`.
**When to use:** Provisioning script, after cloning repo.
**Why it matters:** The shared `.venv` pins `transformers==4.37.2` (via `huggingface-hub==0.24.5` transitive). The Omni sidecar requires `transformers==4.57.3`. These are incompatible. Isolation is mandatory.

Sidecar venvs and their requirements files:
| Sidecar | Venv path | Requirements file |
|---------|-----------|------------------|
| backend (shared) | `.venv` | `requirements.txt` + `supervisor` |
| qwen3_tts | `.venv-qwen3-tts` | `requirements-qwen3-tts.txt` |
| qwen3_asr | `.venv-qwen3-asr` | `requirements-qwen3-asr.txt` |
| voxtral | `.venv-voxtral` | `requirements-voxtral.txt` |
| qwen3_omni | `.venv-qwen3-omni` | `requirements-qwen3-omni.txt` |
| oss_voice | `.venv-voice-oss` | `requirements-voice-oss.txt` |

```bash
ensure_venv() {
  local target="$1"
  local req_file="$2"
  if [ ! -d "$target" ]; then
    python3 -m venv "$target"
  fi
  "$target/bin/pip" install --upgrade pip wheel
  "$target/bin/pip" install -r "$req_file"
}
```

### Pattern 3: HuggingFace Model Download with Resume (D-03)
**What:** `huggingface-cli download` handles partial downloads and caches to HF_HOME. Does not re-download if model already present.
**When to use:** Provisioning script, after venvs are created.
**Example:**
```bash
export HF_HOME="${MODELS_DIR:-/workspace/models}"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var required for gated models}"

# huggingface-cli is available after backend venv is installed
HUGGINGFACE_CLI="$APP_DIR/.venv/bin/huggingface-cli"

download_model() {
  local repo_id="$1"
  "$HUGGINGFACE_CLI" download "$repo_id" --local-dir-use-symlinks False
}
```

**Important note:** `huggingface-cli` is available after the backend venv is installed (it is bundled with `huggingface-hub==0.24.5` which is in `requirements.txt`). The provisioning script must install the backend venv before calling any HuggingFace downloads.

### Pattern 4: Model Swap Between Benchmark Steps (D-09)
**What:** Stop the old brain/model supervisor program, update the active .env overlay, start the new one, poll /health until 200, then verify VRAM.
**When to use:** Benchmark orchestrator between steps 1/2/3.
**Example (bash):**
```bash
swap_model() {
  local stop_program="$1"   # e.g. "qwen"
  local start_program="$2"  # e.g. "qwen3_omni"
  local health_url="$3"     # e.g. "http://localhost:8001/health"
  local vram_limit_gb="${4:-75}"

  echo "[orch] Stopping $stop_program"
  "$SUPERVISORCTL" stop "$stop_program" || true

  echo "[orch] Starting $start_program"
  "$SUPERVISORCTL" start "$start_program"

  echo "[orch] Waiting for $health_url"
  wait_healthy "$health_url" 120

  echo "[orch] Verifying VRAM headroom"
  check_vram "$vram_limit_gb"
}

wait_healthy() {
  local url="$1"
  local max_wait="${2:-120}"
  local elapsed=0
  until curl -fsS "$url" >/dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    if [ "$elapsed" -ge "$max_wait" ]; then
      echo "[orch] ERROR: $url did not become healthy within ${max_wait}s"
      exit 1
    fi
  done
  echo "[orch] $url healthy"
}

check_vram() {
  local limit_gb="$1"
  local used_mb
  used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  local used_gb=$((used_mb / 1024))
  echo "[orch] VRAM used: ${used_gb}GB / limit: ${limit_gb}GB"
  if [ "$used_gb" -ge "$limit_gb" ]; then
    echo "[orch] ERROR: VRAM used (${used_gb}GB) >= limit (${limit_gb}GB). Aborting."
    exit 1
  fi
}
```

### Pattern 5: Profile-Aware Sidecar Health Check (D-06)
**What:** Read the active .env.bench profile to determine which sidecars are required, then check only those. A sidecar that is not in the current profile is expected to be stopped; a failing check for it must not fail the smoke test.
**When to use:** Extended smoke_test.sh, orchestrator pre-step verification.
**Example:**
```bash
ACTIVE_PROFILE="${BENCH_PROFILE:-baseline}"

# Map profiles to required sidecar URLs
case "$ACTIVE_PROFILE" in
  baseline|dify_rag|brain_upgrade)
    REQUIRED_SIDECARS=("$SENSEVOICE_BASE_URL" "$COSYVOICE_BASE_URL")
    ;;
  qwen3_tts)
    REQUIRED_SIDECARS=("$SENSEVOICE_BASE_URL" "$QWEN3_TTS_BASE_URL")
    ;;
  qwen3_asr)
    REQUIRED_SIDECARS=("$QWEN3_ASR_BASE_URL" "$COSYVOICE_BASE_URL")
    ;;
  omni_hybrid)
    REQUIRED_SIDECARS=("$SENSEVOICE_BASE_URL" "$QWEN3_OMNI_BASE_URL")
    ;;
  voxtral)
    REQUIRED_SIDECARS=("$VOXTRAL_BASE_URL" "$COSYVOICE_BASE_URL")
    ;;
esac

for sidecar_url in "${REQUIRED_SIDECARS[@]}"; do
  info "Sidecar health: $sidecar_url/health"
  curl -fsS "$sidecar_url/health" >/dev/null
done
```

### Pattern 6: vLLM Readiness Check (D-05)
**What:** vLLM's /health endpoint returns 200 when the model is loaded. Additionally, send a trivial completion request to confirm the model is actually inferring (not just the HTTP server running).
**When to use:** Extended smoke test, orchestrator post-swap verification.
**Example:**
```bash
check_vllm_ready() {
  local vllm_url="${RAG_LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
  # Strip /v1 suffix to get base URL for /health
  local base="${vllm_url%/v1}"

  info "vLLM health check: $base/health"
  curl -fsS "$base/health" >/dev/null

  info "vLLM trivial completion check"
  local resp
  resp=$(curl -fsS -X POST "$vllm_url/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'$RAG_LLM_MODEL'","prompt":"1+1=","max_tokens":3}')
  # Just check it produced a response with "choices"
  echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('choices'), 'no choices'" \
    || { echo "[smoke][error] vLLM trivial completion failed"; exit 1; }
  info "vLLM OK"
}
```

### Pattern 7: Orchestrator "Winning Config" Selection (D-11)
**What:** After each comparison step, the orchestrator determines which profile "won" by parsing the comparison table output. The winning profile name is passed as a parameter to the next step.
**When to use:** Between benchmark orchestrator steps.
**Approach:** The simplest correct approach is to have the orchestrator always carry a `WINNING_RAG` and `WINNING_BRAIN` variable that defaults to `baseline` / `Qwen3-30B-A3B` and is updated after each step if the challenger won. The orchestrator can detect the winner by parsing the `Winner` column of the markdown table (look for the challenger-side label in the majority of metric rows).

### Anti-Patterns to Avoid
- **Restarting the full stack between benchmark steps:** The orchestrator must only stop/start the specific model program via supervisorctl. Restarting the full stack (backend, qdrant) would invalidate the warm vector index and add minutes of startup time.
- **Installing sidecar deps into the shared `.venv`:** Will break the `transformers==4.37.2` pin. The Omni sidecar requires `transformers==4.57.3`.
- **Running model downloads before the backend venv is installed:** `huggingface-cli` comes from the backend venv. Downloads must happen after `install_backend_env` completes.
- **Hardcoding the VRAM threshold:** The threshold should be a variable (e.g., `VRAM_LIMIT_GB=75`) set in the orchestrator config block. A100 80GB has 81920 MiB; leaving 5-6 GB headroom means the safe threshold is 75 GB.
- **Co-hosting `qwen` and `qwen3_omni`:** The `.env.bench.omni_hybrid` profile documents this explicitly: Omni needs ~79 GB, Qwen3.5-35B needs ~70 GB; never start both simultaneously.
- **Calling sidecar health endpoints directly from benchmark_runner.py:** The runner uses `POST /api/tts` via the backend REST proxy (Pitfall 3 in benchmark_runner.py docstring). Direct sidecar calls from the runner are not the established pattern and bypass the stack_id tagging.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Model download with resume | Custom wget/curl loop | `huggingface-cli download` | Built-in resume, caching, progress, HF_HOME, gated model auth |
| Service health wait loop | Complex retry logic | Simple `curl -fsS ... until` loop (30-60 second polling) | vLLM startup is 60-120 seconds; a plain loop is sufficient and transparent |
| Benchmark results parsing for winner | Custom JSON aggregation | `benchmark_compare.py` already outputs markdown; orchestrator parses winner from stdout | benchmark_compare.py is already correct; reading its output is sufficient |
| Supervisor start/stop | Direct process management | `supervisorctl stop/start <program>` | supervisord already manages PID files, log rotation, restart policies |
| Per-question JSONL writing | Custom output format | `benchmark_runner.py --output` | All format decisions (ensure_ascii=False, flush per question, warmup flag) already implemented |
| VRAM measurement | /sys filesystem parsing | `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` | Authoritative source; parseable in one awk/shell line |

**Key insight:** Every complex benchmark primitive is already implemented in Python. The orchestrator's job is to call the right tools in the right order with the right flags, not to reimplement them.

---

## Common Pitfalls

### Pitfall 1: nvidia-smi Not Available at Script Start
**What goes wrong:** The provisioning script fails immediately if NVIDIA drivers are not installed, but some steps (venv creation, git clone) don't need the GPU and should run before the driver check.
**Why it happens:** `set -euo pipefail` causes immediate exit on any command failure.
**How to avoid:** The driver check (D-04) must be one of the first steps but only after basic apt packages are confirmed. The script should: (1) install apt packages first, (2) then check nvidia-smi, (3) install drivers if missing, (4) exit with "reboot and re-run" message if reboot was needed, (5) on re-run, skip driver install if nvidia-smi passes.
**Warning signs:** Script exits with "nvidia-smi: command not found" or "NVIDIA-SMI has failed" on first run on a fresh VM.

### Pitfall 2: Model Download Requires venv to Exist First
**What goes wrong:** `huggingface-cli` is only available after the backend venv is installed and `huggingface-hub` is in it. If the provisioning script calls `huggingface-cli download` before running `install_backend_env`, it fails with "command not found".
**Why it happens:** `huggingface-cli` is not a system package; it is a Python entry point.
**How to avoid:** Provisioning script order: (1) apt packages, (2) driver check, (3) git clone, (4) install all venvs, (5) then download models.

### Pitfall 3: HF_HOME Not Set, Models Download to Wrong Location
**What goes wrong:** Without `HF_HOME`, the `huggingface-cli download` caches to `~/.cache/huggingface/` which may be on a small OS disk, not the data volume.
**Why it happens:** HF default cache is home directory.
**How to avoid:** Always set `export HF_HOME=/path/to/large/volume/models` before any `huggingface-cli` call. TensorDock A100 VMs typically have a separate large volume; confirm its mount point.

### Pitfall 4: Sidecar vLLM Port Conflicts
**What goes wrong:** Different sidecars and the brain model vLLM server each need a unique port. If the orchestrator starts a new model without stopping the old one, the port bind fails.
**Why it happens:** supervisorctl start succeeds (queuing the start), but the actual process fails immediately on port bind. supervisorctl may not surface this as an error if `autorestart=true` is set.
**How to avoid:** Always `supervisorctl stop <old>` and verify the port is free before `supervisorctl start <new>`. The VRAM check after starting the new service also implicitly confirms the new model loaded (VRAM usage will be > 0 for a loaded model).

### Pitfall 5: Smoke Test VRAM Check Threshold
**What goes wrong:** Choosing a VRAM threshold that is too high (e.g., 79 GB) gives almost no warning before OOM. Choosing too low (e.g., 50 GB) fails the check even for a successfully loaded Qwen3-30B-A3B (~60 GB).
**Why it happens:** Each model has a different VRAM footprint:
- Qwen3-30B-A3B: ~60 GB (bfloat16)
- Qwen3.5-35B-A3B: ~70 GB (bfloat16)
- Qwen3-Omni-30B-A3B: ~79 GB (bfloat16, cited in .env.bench.omni_hybrid comment)
**How to avoid:** The smoke test VRAM check should verify that VRAM used is above a "loaded" floor (confirms model is loaded) AND below the 80 GB ceiling. The orchestrator pre-load check only verifies that VRAM is below the ceiling BEFORE the new model loads (i.e., after stopping the old one). Suggested thresholds:
- Smoke test "model is loaded": used_mb > 10,000 (10 GB min; anything less suggests model not loaded)
- Pre-load "VRAM available": used_mb < 5,000 (5 GB max; after stop, VRAM should be nearly empty)
- Pre-load "ceiling": total_vram - used_vram > 60,000 (60 GB free required before loading any large model)

### Pitfall 6: Orchestrator Does Not Wait Long Enough for vLLM to Load
**What goes wrong:** The orchestrator starts `supervisorctl start qwen` and immediately calls the health check loop. vLLM takes 60-120 seconds to load a 30-70 GB model from disk. If the health timeout is too short, the orchestrator incorrectly declares the model unhealthy.
**Why it happens:** vLLM's /health returns 503 until the model is loaded. The HTTP server starts immediately but /health returns non-200 until the model weights are in VRAM.
**How to avoid:** Set the health wait loop timeout to at least 300 seconds (5 minutes) for the brain models. The loop should poll every 10 seconds. Log elapsed time so the user can see progress.

### Pitfall 7: benchmark_runner.py Needs Both --ws-url and --backend-url Pointing at Port 8000
**What goes wrong:** The benchmark runner's default `--ws-url` is `ws://localhost:8787/ws/voice` and default `--backend-url` is `http://localhost:8787`. The actual backend runs on port 8000 (per supervisord.conf `--port 8000`).
**Why it happens:** The defaults in the argparse block were written for a different port configuration.
**How to avoid:** The orchestrator must always pass explicit `--ws-url ws://localhost:8000/ws/voice` and `--backend-url http://localhost:8000` flags when calling `benchmark_runner.py`.

### Pitfall 8: results/ Directory Not in .gitignore
**What goes wrong:** JSONL result files get committed to the repo accidentally. The `.gitignore` at `rag_demo_system/.gitignore` does not currently have a `results/` entry.
**Why it happens:** The results directory does not exist locally and was not added to .gitignore during earlier phases.
**How to avoid:** The provisioning script (or Wave 0 plan) must add `results/` to `rag_demo_system/.gitignore` before any benchmarks run.

### Pitfall 9: Orchestrator SCP Command Needs SERVER_IP Variable
**What goes wrong:** The printed scp command (D-13) is useless if it contains a placeholder like `<server_ip>` instead of the actual IP.
**Why it happens:** The orchestrator doesn't know the server's own IP unless it discovers it.
**How to avoid:** The orchestrator should accept a `--server-ip` argument or read a `SERVER_IP` env var. As a fallback, it can run `curl -s ifconfig.me` or `hostname -I | awk '{print $1}'` to auto-detect the public/private IP. The scp command should use whichever IP the user SSHed in from.

---

## Code Examples

### Verified Patterns from Existing Code

### Supervisorctl Stop/Start (from stack_cli.py)
```python
# Source: rag_demo_system/scripts/stack_cli.py:sync_supervisor_programs()
def sync_supervisor_programs(repo_root: Path, selection: dict[str, list[str]]) -> None:
    ensure_supervisor_running(repo_root)
    for program in selection["stop"]:
        _supervisorctl(repo_root, "stop", program, check=False)
    for program in selection["start"]:
        result = _supervisorctl(repo_root, "start", program, check=False)
        if result.returncode != 0 and "already started" not in (result.stdout + result.stderr):
            raise RuntimeError(...)
```

Bash equivalent for orchestrator:
```bash
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl -c $APP_DIR/scripts/supervisord.conf"
$SUPERVISORCTL stop qwen
$SUPERVISORCTL start qwen3_omni
```

### Existing Smoke Test Extension Points (from smoke_test.sh)
```bash
# Source: rag_demo_system/scripts/smoke_test.sh
# Current checks (lines 9-61): UI root, /api/health, /api/backends, /api/voice/status,
#   KB indexing via POST /api/index, consent step, chat stream with used_knowledge validation.
# These are preserved per D-07.
# New checks are APPENDED after line 61:
#   - load active profile env vars
#   - check required sidecar /health endpoints (profile-aware per D-06)
#   - check vLLM /health + trivial completion (D-05)
#   - check VRAM headroom via nvidia-smi (D-05)
```

### benchmark_runner.py Invocation (from benchmark_runner.py docstring)
```bash
# Source: rag_demo_system/scripts/benchmark_runner.py
python "$APP_DIR/scripts/benchmark_runner.py" \
  --fixture "$APP_DIR/fixtures/bench_questions_ru.jsonl" \
  --profile baseline \
  --output "$APP_DIR/results/bench_baseline_$(date +%Y%m%d_%H%M%S).jsonl" \
  --ws-url ws://localhost:8000/ws/voice \
  --backend-url http://localhost:8000 \
  --timeout 30 \
  --warmup 3
```

### benchmark_compare.py Invocation
```bash
# Source: rag_demo_system/scripts/benchmark_compare.py
python "$APP_DIR/scripts/benchmark_compare.py" \
  "$RESULT_A" \
  "$RESULT_B"
# Prints markdown table to stdout; orchestrator can tee to file and display inline
```

### nvidia-smi VRAM Query
```bash
# Query used VRAM in MiB (integer, no units suffix)
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
FREE_MIB=$((TOTAL_MIB - USED_MIB))
echo "[vram] Used: ${USED_MIB}MiB / Total: ${TOTAL_MIB}MiB / Free: ${FREE_MIB}MiB"
```

---

## Environment Availability

> This section documents the production server environment (TensorDock A100 80GB), not the development machine. Dev machine checks are irrelevant; the scripts run on a fresh Ubuntu VM.

| Dependency | Required By | Available on Target | Version | Notes |
|------------|------------|-----------|---------|--------|
| Ubuntu 22.04 or 24.04 | All | Provision step 0 | LTS | TensorDock supports both; 22.04 recommended for driver compatibility |
| NVIDIA GPU driver | nvidia-smi, VRAM checks | Conditional | A100 | D-04: check first; install via ubuntu-drivers if missing; reboot required |
| nvidia-smi | Smoke test, orchestrator VRAM check | After driver install | Driver-bundled | Available immediately after driver install |
| CUDA toolkit | vLLM, PyTorch | After driver install | 11.8+ or 12.x | Bundled with ubuntu-drivers path on recent Ubuntu; PyTorch wheels are self-contained |
| python3 + python3-venv | All venv creation | apt install | 3.10+ on 22.04 | ubuntu-22.04 ships Python 3.10 |
| git | Repo clone | apt install | any | |
| docker + docker-compose-plugin | Qdrant | apt install | 24+ | docker.io package; ensure docker group membership for non-root |
| curl | Health checks | apt install | any | |
| huggingface-cli | Model download | After backend venv install | bundled with huggingface-hub==0.24.5 | Only available after backend venv creation |
| supervisord/supervisorctl | Service management | After backend venv install | bundled with `supervisor` pip package | Installed into `.venv/bin/` by setup_vast_voice.sh pattern |
| HF_TOKEN env var | Gated models | Must be provided by user | N/A | Required for any gated HuggingFace repo |
| TensorDock SSH access | All | Pre-condition | N/A | User SSHes in; script runs on the VM |

**Missing dependencies with no fallback:**
- HF_TOKEN: must be set in the environment before the provisioning script runs any model downloads. The script should fail early with a clear message if it is unset and gated models are in the download list.

**Missing dependencies with fallback:**
- CUDA toolkit version: if the ubuntu-drivers install gives an older CUDA, PyTorch GPU wheels are self-contained and will work as long as the driver version is compatible. Fallback: install `cuda-toolkit-12-x` via apt after driver install.

---

## Validation Architecture

> `workflow.nyquist_validation` is absent from `.planning/config.json`; treating as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.3.4 (already in requirements.txt) |
| Config file | None present; pytest auto-discovers tests/ |
| Quick run command | `cd rag_demo_system && .venv/bin/pytest tests/ -x -q` |
| Full suite command | `cd rag_demo_system && .venv/bin/pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DEPLOY-02 | Provisioning script is valid bash with no syntax errors | unit (bash -n) | `bash -n rag_demo_system/scripts/provision_server.sh` | Wave 0: create script |
| DEPLOY-02 | Venv creation function creates directory if missing | unit | `pytest tests/test_provision.py::test_ensure_venv_creates -x` | Wave 0: create test |
| DEPLOY-02 | Driver check skips install when nvidia-smi is present | unit (mock) | `pytest tests/test_provision.py::test_driver_check_skip -x` | Wave 0: create test |
| DEPLOY-03 | Smoke test profile-aware check uses correct sidecar list | unit | `pytest tests/test_smoke_test.py::test_profile_sidecar_map -x` | Wave 0: create test |
| DEPLOY-03 | VRAM check correctly parses nvidia-smi CSV output | unit | `pytest tests/test_smoke_test.py::test_vram_parse -x` | Wave 0: create test |
| DEPLOY-03 | Orchestrator wait_healthy exits cleanly on 200 response | unit (mock server) | `pytest tests/test_orchestrator.py::test_wait_healthy -x` | Wave 0: create test |

Note: Smoke test and orchestrator are bash scripts. Unit tests for them use Python's `subprocess.run` with a mock server (simple `http.server` on a random port) to validate the curl-based health check logic. Alternatively, extract the VRAM parsing logic into a small Python helper that can be tested directly.

### Sampling Rate
- Per task commit: `bash -n rag_demo_system/scripts/provision_server.sh && bash -n rag_demo_system/scripts/benchmark_orchestrator.sh`
- Per wave merge: `cd rag_demo_system && .venv/bin/pytest tests/ -q`
- Phase gate: full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `rag_demo_system/scripts/provision_server.sh` -- does not exist; Wave 0 creates skeleton
- [ ] `rag_demo_system/scripts/benchmark_orchestrator.sh` -- does not exist; Wave 0 creates skeleton
- [ ] `rag_demo_system/results/.gitkeep` -- results dir must exist and be gitignored; add `results/` to `rag_demo_system/.gitignore`
- [ ] `rag_demo_system/tests/test_provision.py` -- new test file for provisioning script helpers
- [ ] `rag_demo_system/tests/test_smoke_test.py` -- new test file for extended smoke test logic
- [ ] `rag_demo_system/tests/test_orchestrator.py` -- new test file for orchestrator helper functions

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual server setup steps | Single-script provisioning with driver detection | This phase | Operator runs one command from SSH |
| Smoke test checks HTTP endpoints only | Smoke test also checks sidecar /health, vLLM readiness, VRAM | This phase | Catches model-not-loaded failures before wasting benchmark time |
| Benchmark runs per-stack manually | Orchestrator chains RAG -> brain -> Omni -> (optional) STT/TTS | This phase | Full matrix runs unattended; user only intervenes at step 3 decision point |

**Deprecated / not applicable:**
- `setup_vast_voice.sh` is not replaced; it is extended. The new `provision_server.sh` sources its patterns (ensure_venv, install_backend_env) rather than duplicating them.
- `stack.sh up` is the stack launcher; the orchestrator does NOT call `stack.sh up/down` between steps. It only calls `supervisorctl stop/start` for specific programs.

---

## Open Questions

1. **HuggingFace model repo IDs for new sidecars**
   - What we know: Qwen3-30B-A3B is `Qwen/Qwen3-30B-A3B` (confirmed in .env.bench.baseline). Qwen3.5-35B-A3B is `Qwen/Qwen3.5-35B-A3B` (confirmed in .env.bench.brain_upgrade). Omni model ID is not explicitly confirmed in the codebase.
   - What's unclear: Exact HuggingFace repo IDs for `Qwen3-TTS`, `Qwen3-ASR`, `Voxtral`, and `Qwen3-Omni` sidecars. The CONTEXT.md explicitly marks this as "researcher confirms these" (Claude's Discretion).
   - Recommendation: The provisioning script should accept model IDs as env vars with documented defaults. The planner should create a Wave 0 task: "confirm and document HuggingFace repo IDs for all 5 models". Do not hardcode unconfirmed IDs.

2. **TensorDock large volume mount point**
   - What we know: TensorDock A100 VMs have attached storage volumes. The mount point varies by VM configuration.
   - What's unclear: Default mount point for the data volume on TensorDock (common values: `/workspace`, `/data`, `/mnt/data`). This affects HF_HOME and MODELS_DIR.
   - Recommendation: The provisioning script should accept `MODELS_DIR` as an env var (defaulting to `/workspace/models`). Document that the user should verify their volume mount point and set MODELS_DIR accordingly.

3. **Sidecar STACK_* CMD env vars for new sidecars**
   - What we know: `supervisord.conf` already has a `qwen3_omni` entry using `STACK_QWEN3_OMNI_CMD`. The new sidecars (qwen3_tts, qwen3_asr, voxtral) need corresponding entries.
   - What's unclear: Whether `supervisord.conf` already has entries for qwen3_tts, qwen3_asr, voxtral or whether they need to be added. A review of the file shows it currently has: backend, frontend, qwen, sensevoice, whisper, cosyvoice, vosk, vosk_tts, ngrok, qwen3_omni. The qwen3_tts, qwen3_asr, and voxtral programs are missing.
   - Recommendation: A Wave 0 plan task must add supervisor entries for qwen3_tts, qwen3_asr, and voxtral to `supervisord.conf` before any benchmark using those profiles can work.

4. **Voxtral availability confirmed for self-hosting**
   - What we know: STATE.md flags "Voxtral self-host availability unknown; may become a cloud-API thin adapter instead of local sidecar" as a pre-Phase 2 concern. Phase 2 is marked complete.
   - What's unclear: Was Voxtral confirmed as self-hosted or cloud-API in Phase 2 implementation? The `requirements-voxtral.txt` uses `transformers>=5.2.0`, suggesting a local model load path was implemented.
   - Recommendation: The planner should note this as a verification step: confirm the Voxtral sidecar exists (it was referenced in CONTEXT.md but no sidecar file was found in `rag_demo_system/backend/`). If it is a cloud API adapter, the provisioning script does not need to download a Voxtral model.

---

## Sources

### Primary (HIGH confidence)
- `rag_demo_system/scripts/setup_vast_voice.sh` -- existing provisioning pattern; venv creation, model download, stack start
- `rag_demo_system/scripts/smoke_test.sh` -- existing smoke test to extend
- `rag_demo_system/scripts/stack_cli.py` -- supervisord management primitives
- `rag_demo_system/scripts/supervisord.conf` -- all supervisor program definitions including qwen3_omni
- `rag_demo_system/scripts/benchmark_runner.py` -- CLI flags, env overlay, WebSocket protocol, JSONL format
- `rag_demo_system/scripts/benchmark_compare.py` -- comparison table format
- `rag_demo_system/.env.bench.*` (7 files) -- all profile definitions confirmed present
- `docs/voice_ai_playbook_2026-03-25.md` (lines ~540-648, ~682-696) -- Azure/TensorDock provisioning steps, benchmark order
- `rag_demo_system/requirements*.txt` (7 files) -- per-sidecar dependency sets
- `.planning/phases/05-server-deployment-and-benchmarks/05-CONTEXT.md` -- all locked decisions D-01 through D-13

### Secondary (MEDIUM confidence)
- `rag_demo_system/fixtures/bench_questions_ru.jsonl` -- 85 questions confirmed; fixture is ready
- `.env.bench.omni_hybrid` comment "Omni needs ~79 GB VRAM" -- VRAM sizing for Omni confirmed in profile comment
- `.env.bench.brain_upgrade` with `Qwen/Qwen3.5-35B-A3B` -- brain upgrade model ID confirmed

### Tertiary (LOW confidence)
- HuggingFace model repo IDs for Qwen3-TTS, Qwen3-ASR, Voxtral, Qwen3-Omni sidecars -- not explicitly confirmed in codebase; researcher must verify before provisioning script download commands are finalized
- TensorDock volume mount point -- not documented in repo; `/workspace` assumed based on TensorDock convention

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all tools are either in existing scripts or locked by CONTEXT.md decisions
- Architecture patterns: HIGH -- derived directly from existing codebase, not from external research
- Pitfalls: HIGH -- derived from code inspection, existing decision log entries, and explicit CONTEXT.md warnings (OOM risk in STATE.md)
- HuggingFace model IDs: LOW -- explicitly flagged as researcher-confirmed in CONTEXT.md

**Research date:** 2026-03-26
**Valid until:** 2026-04-25 (stable infra domain; main risk is model availability on HuggingFace)
