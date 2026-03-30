#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# provision_server.sh
#
# Full provisioning for GPU servers (H100 NVL 94GB recommended).
# Works on: clean Ubuntu 22.04/24.04, Vast.ai, RunPod, Hetzner, Lambda,
#           or any bare-metal/VM with an NVIDIA GPU.
#
# Winning stack: Whisper STT + Silero TTS + Qwen3.5-35B-A3B-FP8.
# Run once after the VM is created. Idempotent: safe to re-run.
#
# Usage:
#   HF_TOKEN=hf_... bash provision_server.sh
#
# Optional overrides via environment variables:
#   REPO_URL    - Git repo to clone (default: public HTTPS URL)
#   REPO_BRANCH - Branch to check out (default: feature/voice-pipeline)
#   WORKSPACE   - Root directory for all data (default: /workspace)
#   MODELS_DIR  - Where HuggingFace model weights are stored (default: $WORKSPACE/models)
# =============================================================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/yauhenifutryn/leasing.git}"
REPO_BRANCH="${REPO_BRANCH:-feature/voice-pipeline}"
WORKSPACE="${WORKSPACE:-/workspace}"
APP_DIR="$WORKSPACE/leasing/rag_demo_system"
MODELS_DIR="${MODELS_DIR:-$WORKSPACE/models}"

# Fail-fast: HF_TOKEN is required before any network work
HF_TOKEN="${HF_TOKEN:?ERROR: HF_TOKEN env var is required for gated model downloads. Export it before running this script.}"

VLLM_PORT=8787  # Avoid 8001 which RunPod reserves internally
BACKEND_PORT=8000

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
log() {
  echo "[provision][$(date +%H:%M:%S)] $*"
}

# ---------------------------------------------------------------------------
# Step 1: Install system packages
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

  # Install ngrok for exposing the backend to the internet
  if ! command -v ngrok &>/dev/null; then
    log "Installing ngrok"
    curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok-v3-stable-linux-amd64.tgz | tar xz -C /usr/local/bin
  fi

  # Install Docker and ubuntu-drivers only if not in a container
  if [ -f /proc/1/cgroup ] && grep -qE 'docker|containerd' /proc/1/cgroup 2>/dev/null || [ -f /.dockerenv ]; then
    log "Running inside a container -- skipping docker.io and ubuntu-drivers-common"
  else
    sudo apt-get install -y docker.io docker-compose-plugin ubuntu-drivers-common || true
    sudo usermod -aG docker "$USER" || true
  fi
}

# ---------------------------------------------------------------------------
# Step 2: Ensure full NVIDIA stack (driver + CUDA toolkit + cuDNN)
#   On Vast.ai/RunPod containers: already present, just verify.
#   On clean Ubuntu bare-metal: install everything from scratch.
# ---------------------------------------------------------------------------
_is_container() {
  [ -f /.dockerenv ] || grep -qE 'docker|containerd' /proc/1/cgroup 2>/dev/null
}

_install_cuda_toolkit() {
  # Install CUDA 12.8 toolkit + cuDNN via NVIDIA's official apt repo.
  # This is the universal method that works on Ubuntu 22.04 and 24.04.
  # Ref: https://developer.nvidia.com/cuda-downloads
  local DISTRO
  DISTRO=$(. /etc/os-release && echo "${ID}${VERSION_ID}" | tr -d '.')
  # e.g. ubuntu2204 or ubuntu2404

  log "Adding NVIDIA CUDA apt repository for $DISTRO"
  local KEYRING="/usr/share/keyrings/cuda-archive-keyring.gpg"
  curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" \
    -o /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  rm -f /tmp/cuda-keyring.deb
  sudo apt-get update -y

  log "Installing CUDA toolkit 12.8 + cuDNN"
  sudo apt-get install -y cuda-toolkit-12-8 libcudnn9-cuda-12 libcudnn9-dev-cuda-12

  # Add CUDA to PATH and LD_LIBRARY_PATH for this session and future logins
  local CUDA_PROFILE="/etc/profile.d/cuda.sh"
  if [ ! -f "$CUDA_PROFILE" ]; then
    log "Writing CUDA environment to $CUDA_PROFILE"
    sudo tee "$CUDA_PROFILE" > /dev/null <<'CUDAEOF'
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
CUDAEOF
  fi
  export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
}

check_nvidia_driver() {
  if nvidia-smi &>/dev/null; then
    log "NVIDIA driver OK: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  elif _is_container; then
    log "ERROR: nvidia-smi not found inside container. The host must have NVIDIA drivers."
    log "If using RunPod/Vast.ai, select a GPU-enabled template."
    exit 1
  else
    log "NVIDIA driver not found -- installing via ubuntu-drivers"
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers install
    log "Driver installed. REBOOT REQUIRED before continuing."
    log "After reboot, re-run: HF_TOKEN=\$HF_TOKEN bash $0"
    exit 1
  fi

  # Check CUDA toolkit (nvcc). Containers usually have it; bare-metal may not.
  if command -v nvcc &>/dev/null; then
    log "CUDA toolkit OK: $(nvcc --version | grep 'release' | head -1)"
  elif _is_container; then
    # Containers on Vast.ai/RunPod have CUDA in non-standard paths; pip packages
    # bundle their own CUDA libs (nvidia-cublas-cu12, nvidia-cudnn-cu12).
    # nvcc may not be in PATH but CUDA works via pip wheels. This is fine.
    log "CUDA toolkit not in PATH (container). pip CUDA wheels will provide libs."
  else
    log "CUDA toolkit not found on bare-metal -- installing"
    _install_cuda_toolkit
    log "CUDA toolkit installed: $(nvcc --version | grep 'release' | head -1)"
  fi

  # Verify nvidia-smi works after all installs
  if ! nvidia-smi &>/dev/null; then
    log "ERROR: nvidia-smi still not working after driver install."
    log "Reboot the machine, then re-run this script."
    exit 1
  fi
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
# Step 4: Create virtualenvs and install requirements
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
# Step 5: Install 2 venvs: backend+vLLM and voice-oss (Whisper+Silero)
# ---------------------------------------------------------------------------
install_all_venvs() {
  log "=== Installing virtualenvs ==="

  log "Installing backend venv (.venv)"
  if [ ! -d "$APP_DIR/.venv" ]; then
    log "Creating venv: $APP_DIR/.venv"
    python3 -m venv "$APP_DIR/.venv"
  fi
  "$APP_DIR/.venv/bin/pip" install --upgrade pip wheel
  # Install vLLM FIRST. It pulls torch, transformers, pydantic, fastapi, etc.
  # Then install only the small backend-specific packages that vLLM does not
  # already provide. This avoids installing torch/transformers twice.
  log "  Installing vLLM + supervisor + hf_transfer (big install, includes torch)"
  "$APP_DIR/.venv/bin/pip" install vllm supervisor hf_transfer
  log "  Installing backend-only packages (small, no duplicates)"
  "$APP_DIR/.venv/bin/pip" install \
    uvicorn \
    pyyaml \
    requests \
    qdrant-client \
    sentence-transformers \
    rank-bm25 \
    num2words \
    pytest

  log "Installing OSS voice venv (.venv-voice-oss) -- Whisper + Silero"
  ensure_venv "$APP_DIR/.venv-voice-oss" "$APP_DIR/requirements-voice-oss.txt"

  log "All venvs installed."
}

# ---------------------------------------------------------------------------
# Step 6: Download HuggingFace model weights + Silero VAD
# ---------------------------------------------------------------------------
download_models() {
  export HF_HOME="$MODELS_DIR"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

  local hf_cli="$APP_DIR/.venv/bin/huggingface-cli"
  mkdir -p "$MODELS_DIR"

  log "=== Downloading HuggingFace models to HF_HOME=$HF_HOME ==="

  log "  Qwen3.5-35B-A3B-FP8 (brain, half VRAM)"
  "$hf_cli" download Qwen/Qwen3.5-35B-A3B-FP8 \
    --token "$HF_TOKEN" \
    --local-dir-use-symlinks False

  log "  Embedding model (intfloat/multilingual-e5-large)"
  "$APP_DIR/.venv/bin/python" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

  log "  Reranker model (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)"
  "$APP_DIR/.venv/bin/python" -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

  log "  Whisper large-v3 (faster-whisper STT model)"
  "$APP_DIR/.venv-voice-oss/bin/python" -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

  log "  Silero TTS model (v5_4_ru Russian voices)"
  "$APP_DIR/.venv-voice-oss/bin/python" -c "from silero import silero_tts; silero_tts(language='ru', speaker='v5_4_ru'); print('Silero TTS downloaded')"

  log "  Silero VAD model (~2MB)"
  mkdir -p "$APP_DIR/models"
  curl -fsSL "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit" \
    -o "$APP_DIR/models/silero_vad.jit"

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
    local QDRANT_DIR="$WORKSPACE/qdrant"
    if [ ! -f "$QDRANT_DIR/qdrant" ]; then
      mkdir -p "$QDRANT_DIR"
      log "Downloading Qdrant binary"
      curl -fsSL "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz" | tar xz -C "$QDRANT_DIR"
    fi
    log "Starting Qdrant binary in background"
    mkdir -p "$WORKSPACE/qdrant_storage"
    QDRANT__STORAGE__STORAGE_PATH="$WORKSPACE/qdrant_storage" \
      nohup "$QDRANT_DIR/qdrant" > "$WORKSPACE/qdrant.log" 2>&1 &
    sleep 5
    if curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
      log "Qdrant running on port 6333"
    else
      log "WARNING: Qdrant may not have started. Check $WORKSPACE/qdrant.log"
    fi
  fi
}

# ---------------------------------------------------------------------------
# Step 8: Generate .env with production values
#   Winning stack: Whisper STT + Silero TTS + Qwen3.5-35B-A3B-FP8
# ---------------------------------------------------------------------------
write_env_file() {
  log "Writing .env file to $APP_DIR/.env"
  cat > "$APP_DIR/.env" <<ENVEOF
# Generated by provision_server.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# H100 NVL 94GB -- Whisper STT + Silero TTS + Qwen3.5-35B-A3B-FP8

RAG_LLM_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1
RAG_LLM_MODEL=Qwen/Qwen3.5-35B-A3B-FP8
RAG_LLM_FAST_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1
RAG_LLM_FAST_MODEL=Qwen/Qwen3.5-35B-A3B-FP8

WHISPER_BASE_URL=http://127.0.0.1:50002
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

SILERO_TTS_BASE_URL=http://127.0.0.1:50006
SILERO_TTS_SPEAKER=baya
SILERO_TTS_SAMPLE_RATE=24000

VAD_SILENCE_MS=500
SILERO_VAD_PATH=./models/silero_vad.jit

STACK_MODE=docker
RAG_LAUNCH_MODE=supervisor
STACK_VOICE_PROFILE=oss_russian

STACK_QWEN_CMD="./.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-35B-A3B-FP8 --port ${VLLM_PORT} --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.60 --download-dir ${MODELS_DIR}"
STACK_WHISPER_CMD="LD_LIBRARY_PATH=./.venv-voice-oss/lib/python3.12/site-packages/nvidia/cublas/lib:./.venv-voice-oss/lib/python3.12/site-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-} ./.venv-voice-oss/bin/python -m uvicorn services.whisper_server:app --host 0.0.0.0 --port 50002"
STACK_SILERO_TTS_CMD="./.venv-voice-oss/bin/python -m uvicorn services.silero_tts_server:app --host 0.0.0.0 --port 50006"

HF_HOME=${MODELS_DIR}
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
ENVEOF
}

# ---------------------------------------------------------------------------
# Step 9: Start stack via stack.sh
# ---------------------------------------------------------------------------
start_stack() {
  # --- Clean shutdown of any previous stack ---
  log "Stopping any existing stack"
  local supervisorctl_bin="$APP_DIR/.venv/bin/supervisorctl"
  local supervisor_conf="$APP_DIR/scripts/supervisord.conf"
  if [ -f "$supervisorctl_bin" ]; then
    "$supervisorctl_bin" -c "$supervisor_conf" shutdown 2>/dev/null || true
    log "  Waiting for supervisor children to exit..."
    sleep 5
  fi

  # Graceful kill remaining processes (SIGTERM, not SIGKILL).
  # SIGKILL on GPU processes leaks CUDA memory in containers.
  pkill -f supervisord 2>/dev/null || true
  pkill -f "uvicorn backend" 2>/dev/null || true
  pkill -f "vllm" 2>/dev/null || true
  pkill -f "uvicorn services" 2>/dev/null || true
  log "  Waiting 15s for GPU processes to release memory..."
  sleep 15

  # Only force-kill non-GPU processes if still lingering
  pkill -9 -f supervisord 2>/dev/null || true
  pkill -9 -f "uvicorn backend" 2>/dev/null || true
  pkill -9 -f "uvicorn services" 2>/dev/null || true
  # Do NOT kill -9 vllm: leaked CUDA memory cannot be recovered in containers
  if pgrep -f "vllm" >/dev/null 2>&1; then
    log "WARNING: vLLM still running after SIGTERM. Waiting 15s more..."
    sleep 15
    if pgrep -f "vllm" >/dev/null 2>&1; then
      log "WARNING: vLLM refuses to die. Sending SIGKILL (may leak GPU memory)."
      pkill -9 -f "vllm" 2>/dev/null || true
      sleep 3
    fi
  fi

  # Verify all service ports are free
  log "Verifying ports are free"
  local all_free=true
  for port in 8000 $VLLM_PORT 50002 50006; do
    local pid
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
      log "  Port $port still held by PID $pid -- sending SIGTERM"
      kill "$pid" 2>/dev/null || true
      all_free=false
    fi
  done
  if [ "$all_free" = false ]; then
    sleep 5
    for port in 8000 $VLLM_PORT; do
      if lsof -ti :"$port" >/dev/null 2>&1; then
        log "ERROR: Port $port still occupied after cleanup. Cannot start stack."
        log "Run: lsof -i :$port   to investigate manually."
        exit 1
      fi
    done
  fi
  log "All ports free"

  # Check for leaked GPU memory before starting
  if command -v nvidia-smi &>/dev/null; then
    local used_mib total_mib gpu_procs
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "0")
    if [ "${used_mib:-0}" -gt 5000 ] && [ "${gpu_procs:-0}" -eq 0 ]; then
      log "ERROR: Leaked GPU memory detected (${used_mib}MiB used, 0 processes)."
      log "Cannot start vLLM. Restart the instance from your provider's dashboard,"
      log "then re-run: bash rag_demo_system/scripts/provision_server.sh"
      exit 1
    fi
    log "GPU memory OK: ${used_mib}MiB / ${total_mib}MiB used, ${gpu_procs} process(es)"
  fi

  # Remove stale pidfile and socket
  rm -f "$APP_DIR/.state/supervisord.pid" "$APP_DIR/.state/supervisor.sock"

  # Start the stack
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
  log "=== Provisioning H100 NVL 94GB ==="
  log "WORKSPACE=$WORKSPACE"
  log "APP_DIR=$APP_DIR"
  log "MODELS_DIR=$MODELS_DIR"
  log "REPO_BRANCH=$REPO_BRANCH"

  install_apt_packages       # Step 1: system packages
  check_nvidia_driver        # Step 2: NVIDIA driver check
  clone_repo                 # Step 3: clone / pull repo
  install_all_venvs          # Step 4+5: 2 venvs (backend+vLLM, voice-oss)
  download_models            # Step 6: HF model + Silero VAD download
  start_qdrant               # Step 7: Qdrant vector DB
  write_env_file             # Step 8: generate .env
  start_stack                # Step 9: launch supervisor stack

  log "=== Provisioning complete ==="
  log "Next: bash $APP_DIR/scripts/smoke_test.sh"
}

main "$@"
