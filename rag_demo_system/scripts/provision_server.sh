#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# provision_server.sh
#
# Full provisioning for a fresh TensorDock A100 80GB VM.
# Run once after the VM is created. Idempotent: safe to re-run.
#
# Usage:
#   HF_TOKEN=hf_... bash provision_server.sh
#
# Optional overrides via environment variables:
#   REPO_URL    - Git repo to clone (default: public HTTPS URL)
#   REPO_BRANCH - Branch to check out (default: claude/qwen-voice-next)
#   WORKSPACE   - Root directory for all data (default: /workspace)
#   MODELS_DIR  - Where HuggingFace model weights are stored (default: $WORKSPACE/models)
# =============================================================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/yauhenifutryn/leasing.git}"
REPO_BRANCH="${REPO_BRANCH:-claude/qwen-voice-next}"
WORKSPACE="${WORKSPACE:-/workspace}"
APP_DIR="$WORKSPACE/leasing/rag_demo_system"
MODELS_DIR="${MODELS_DIR:-$WORKSPACE/models}"

# Fail-fast: HF_TOKEN is required before any network work (Pitfall 3)
HF_TOKEN="${HF_TOKEN:?ERROR: HF_TOKEN env var is required for gated model downloads. Export it before running this script.}"

VLLM_PORT=8001
BACKEND_PORT=8000

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
log() {
  echo "[provision][$(date +%H:%M:%S)] $*"
}

# ---------------------------------------------------------------------------
# Step 1: Install system packages
#   ubuntu-drivers-common is installed here so check_nvidia_driver can use it
# ---------------------------------------------------------------------------
install_apt_packages() {
  log "Installing apt packages"
  sudo apt-get update -y
  sudo apt-get install -y \
    git \
    curl \
    unzip \
    python3 \
    python3-venv \
    python3-pip \
    jq

  # Install Docker and ubuntu-drivers only if not in a container (RunPod containers lack both)
  if [ -f /proc/1/cgroup ] && grep -qE 'docker|containerd' /proc/1/cgroup 2>/dev/null || [ -f /.dockerenv ]; then
    log "Running inside a container -- skipping docker.io and ubuntu-drivers-common"
  else
    sudo apt-get install -y docker.io docker-compose-plugin ubuntu-drivers-common || true
    sudo usermod -aG docker "$USER" || true
  fi
}

# ---------------------------------------------------------------------------
# Step 2: Check for NVIDIA driver; install if missing (Pitfall 1: apt must come first)
#   Exits with a reboot message if driver was freshly installed, because
#   udev does not expose the GPU device until after reboot.
# ---------------------------------------------------------------------------
check_nvidia_driver() {
  if nvidia-smi &>/dev/null; then
    log "NVIDIA driver OK: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    return 0
  fi

  # In a container, drivers come from the host; we cannot install them
  if [ -f /.dockerenv ] || grep -qE 'docker|containerd' /proc/1/cgroup 2>/dev/null; then
    log "ERROR: nvidia-smi not found inside container. The host must have NVIDIA drivers installed."
    log "If using RunPod/Vast.ai, select a GPU-enabled template."
    exit 1
  fi

  log "NVIDIA driver not found -- installing via ubuntu-drivers"
  sudo ubuntu-drivers install

  log "Driver installed. REBOOT REQUIRED before continuing."
  log "After reboot, re-run: HF_TOKEN=\$HF_TOKEN bash $0"
  exit 1
}

# ---------------------------------------------------------------------------
# Step 3: Clone or update the repository
# ---------------------------------------------------------------------------
clone_repo() {
  if [ -d "$WORKSPACE/leasing/.git" ]; then
    log "Repo exists; pulling latest from $REPO_BRANCH"
    git -C "$WORKSPACE/leasing" pull origin "$REPO_BRANCH"
  else
    log "Cloning repo branch $REPO_BRANCH into $WORKSPACE/leasing"
    mkdir -p "$WORKSPACE"
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKSPACE/leasing"
  fi
}

# ---------------------------------------------------------------------------
# Step 4: Create a virtualenv and install requirements
#   Re-entrant: venv creation is skipped if the directory already exists.
# ---------------------------------------------------------------------------
ensure_venv() {
  local target="$1"
  local req_file="$2"

  if [ ! -d "$target" ]; then
    log "Creating venv: $target"
    python3 -m venv "$target"
  fi

  log "Installing requirements into $target from $req_file"
  "$target/bin/pip" install --upgrade pip wheel
  "$target/bin/pip" install -r "$req_file"
}

# ---------------------------------------------------------------------------
# Step 5: Install all 6 per-sidecar venvs
#   Order: backend first (Pitfall 2: huggingface-cli lives in backend venv)
# ---------------------------------------------------------------------------
install_all_venvs() {
  log "=== Installing virtualenvs ==="

  log "Installing backend venv (.venv) with supervisor + hf_transfer"
  ensure_venv "$APP_DIR/.venv" "$APP_DIR/requirements.txt"
  # supervisor is installed into the backend venv so stack.sh can use it
  # hf_transfer is needed because some cloud images set HF_HUB_ENABLE_HF_TRANSFER=1 globally
  "$APP_DIR/.venv/bin/pip" install supervisor hf_transfer

  log "Installing OSS voice venv (.venv-voice-oss) -- sensevoice, cosyvoice, whisper"
  ensure_venv "$APP_DIR/.venv-voice-oss" "$APP_DIR/requirements-voice-oss.txt"

  log "Installing Qwen3-TTS venv (.venv-qwen3-tts)"
  ensure_venv "$APP_DIR/.venv-qwen3-tts" "$APP_DIR/requirements-qwen3-tts.txt"

  log "Installing Qwen3-ASR venv (.venv-qwen3-asr)"
  ensure_venv "$APP_DIR/.venv-qwen3-asr" "$APP_DIR/requirements-qwen3-asr.txt"

  log "Installing Voxtral venv (.venv-voxtral)"
  ensure_venv "$APP_DIR/.venv-voxtral" "$APP_DIR/requirements-voxtral.txt"

  log "Installing Qwen3-Omni venv (.venv-qwen3-omni)"
  ensure_venv "$APP_DIR/.venv-qwen3-omni" "$APP_DIR/requirements-qwen3-omni.txt"

  log "All venvs installed."
}

# ---------------------------------------------------------------------------
# Step 6: Download HuggingFace model weights
#   - HF_HOME is set to MODELS_DIR so all weights land on the large volume
#   - huggingface-cli is taken from the backend venv (installed in Step 5)
#   - --local-dir-use-symlinks False prevents symlink caching that defeats
#     HF_HOME volume placement (RESEARCH.md Pattern 3)
# ---------------------------------------------------------------------------
download_models() {
  export HF_HOME="$MODELS_DIR"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

  local hf_cli="$APP_DIR/.venv/bin/huggingface-cli"
  mkdir -p "$MODELS_DIR"

  log "=== Downloading HuggingFace models to HF_HOME=$HF_HOME ==="

  log "  Qwen3-30B-A3B (split pipeline brain baseline)"
  "$hf_cli" download Qwen/Qwen3-30B-A3B \
    --token "$HF_TOKEN" \
    --local-dir-use-symlinks False

  log "  Qwen3.5-35B-A3B (split pipeline brain upgrade)"
  "$hf_cli" download Qwen/Qwen3.5-35B-A3B \
    --token "$HF_TOKEN" \
    --local-dir-use-symlinks False

  log "  Qwen3-TTS-12Hz-1.7B-Base (TTS sidecar)"
  "$hf_cli" download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --token "$HF_TOKEN" \
    --local-dir-use-symlinks False

  log "  Qwen3-ASR-1.7B (ASR sidecar)"
  "$hf_cli" download Qwen/Qwen3-ASR-1.7B \
    --token "$HF_TOKEN" \
    --local-dir-use-symlinks False

  log "  Voxtral-Mini-4B-Realtime-2602 (Voxtral STT sidecar)"
  "$hf_cli" download mistralai/Voxtral-Mini-4B-Realtime-2602 \
    --token "$HF_TOKEN" \
    --local-dir-use-symlinks False

  log "  Qwen3-Omni-30B-A3B-Instruct (Omni hybrid sidecar)"
  "$hf_cli" download Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --token "$HF_TOKEN" \
    --local-dir-use-symlinks False

  log "  SenseVoiceSmall (SenseVoice STT sidecar)"
  "$hf_cli" download FunAudioLLM/SenseVoiceSmall \
    --token "$HF_TOKEN" \
    --local-dir-use-symlinks False

  log "All models downloaded."
}

# ---------------------------------------------------------------------------
# Step 7: Start Qdrant vector database via Docker
# ---------------------------------------------------------------------------
start_qdrant() {
  if command -v docker &>/dev/null; then
    log "Starting Qdrant via Docker"
    sudo docker run -d \
      --name qdrant \
      --restart unless-stopped \
      -p 6333:6333 \
      -p 6334:6334 \
      -v "$WORKSPACE/qdrant_storage:/qdrant/storage" \
      qdrant/qdrant:latest \
      || log "Qdrant container may already exist -- skipping"
  else
    log "Docker not available -- installing Qdrant via pip into backend venv"
    "$APP_DIR/.venv/bin/pip" install qdrant-client
    # Download and run Qdrant binary
    local QDRANT_DIR="$WORKSPACE/qdrant"
    if [ ! -f "$QDRANT_DIR/qdrant" ]; then
      mkdir -p "$QDRANT_DIR"
      log "Downloading Qdrant binary"
      curl -fsSL "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz" | tar xz -C "$QDRANT_DIR"
    fi
    log "Starting Qdrant binary in background"
    mkdir -p "$WORKSPACE/qdrant_storage"
    nohup "$QDRANT_DIR/qdrant" --storage-path "$WORKSPACE/qdrant_storage" > "$WORKSPACE/qdrant.log" 2>&1 &
    sleep 3
    if curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
      log "Qdrant running on port 6333"
    else
      log "WARNING: Qdrant may not have started. Check $WORKSPACE/qdrant.log"
    fi
  fi
}

# ---------------------------------------------------------------------------
# Step 8: Generate .env with production values
#   STACK_*_CMD vars use per-venv python binaries so each sidecar is isolated.
#   vLLM --download-dir points to MODELS_DIR so weights are not re-downloaded.
# ---------------------------------------------------------------------------
write_env_file() {
  log "Writing .env file to $APP_DIR/.env"
  cat > "$APP_DIR/.env" <<ENVEOF
# Generated by provision_server.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Edit manually to adjust model selection or enable optional services.

RAG_LLM_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1
RAG_LLM_MODEL=Qwen/Qwen3-30B-A3B
RAG_LLM_FAST_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1
RAG_LLM_FAST_MODEL=Qwen/Qwen3-30B-A3B

SENSEVOICE_BASE_URL=http://127.0.0.1:50000
SENSEVOICE_API_STYLE=official
COSYVOICE_BASE_URL=http://127.0.0.1:50001
COSYVOICE_API_STYLE=official

WHISPER_BASE_URL=http://127.0.0.1:50002
QWEN3_TTS_BASE_URL=http://127.0.0.1:50003
QWEN3_ASR_BASE_URL=http://127.0.0.1:50004
VOXTRAL_BASE_URL=http://127.0.0.1:50005
QWEN3_OMNI_BASE_URL=http://127.0.0.1:8002

STACK_MODE=docker
RAG_LAUNCH_MODE=supervisor
STACK_VOICE_PROFILE=oss_russian

STACK_QWEN_CMD=./.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-30B-A3B --port ${VLLM_PORT} --dtype bfloat16 --max-model-len 8192 --download-dir ${MODELS_DIR}
STACK_SENSEVOICE_CMD=./.venv-voice-oss/bin/python -m uvicorn services.whisper_server:app --host 0.0.0.0 --port 50000
STACK_COSYVOICE_CMD=./.venv-voice-oss/bin/python -m uvicorn services.vosk_tts_server:app --host 0.0.0.0 --port 50001
STACK_WHISPER_CMD=./.venv-voice-oss/bin/python -m uvicorn services.whisper_server:app --host 0.0.0.0 --port 50002
STACK_QWEN3_TTS_CMD=./.venv-qwen3-tts/bin/python -m uvicorn services.qwen3_tts_server:app --host 0.0.0.0 --port 50003
STACK_QWEN3_ASR_CMD=./.venv-qwen3-asr/bin/python -m uvicorn services.qwen3_asr_server:app --host 0.0.0.0 --port 50004
STACK_VOXTRAL_CMD=./.venv-voxtral/bin/python -m uvicorn services.voxtral_server:app --host 0.0.0.0 --port 50005
STACK_QWEN3_OMNI_CMD=./.venv-qwen3-omni/bin/python -m uvicorn services.qwen3_omni_server:app --host 0.0.0.0 --port 8002

HF_HOME=${MODELS_DIR}
ENVEOF
}

# ---------------------------------------------------------------------------
# Step 9: Start baseline stack via stack.sh
# ---------------------------------------------------------------------------
start_stack() {
  log "Starting stack via stack.sh"
  cd "$APP_DIR"
  bash scripts/stack.sh up
  cd - > /dev/null
  log "Stack started. Run smoke_test.sh to verify services are up."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  log "=== Provisioning TensorDock A100 80GB VM ==="
  log "WORKSPACE=$WORKSPACE"
  log "APP_DIR=$APP_DIR"
  log "MODELS_DIR=$MODELS_DIR"
  log "REPO_BRANCH=$REPO_BRANCH"

  install_apt_packages       # Step 1: system packages (apt must be first -- Pitfall 1)
  check_nvidia_driver        # Step 2: NVIDIA driver check (after apt -- Pitfall 1)
  clone_repo                 # Step 3: clone / pull repo
  install_all_venvs          # Step 4+5: 6 venvs (before download_models -- Pitfall 2)
  download_models            # Step 6: HF model downloads (needs backend venv -- Pitfall 2)
  start_qdrant               # Step 7: Qdrant vector DB (Docker or binary)
  write_env_file             # Step 8: generate .env
  start_stack                # Step 9: launch supervisor stack

  log "=== Provisioning complete ==="
  log "Next: bash $APP_DIR/scripts/smoke_test.sh"
}

main "$@"
