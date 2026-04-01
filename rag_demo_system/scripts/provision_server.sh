#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# provision_server.sh
#
# Full provisioning for GPU servers (H100 80GB+, H200 141GB, A100 80GB).
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
# Auto-detect workspace: /workspace (Vast.ai/RunPod) or $HOME (bare-metal VMs)
if [ -n "${WORKSPACE:-}" ]; then
  : # explicit override, use as-is
elif [ -d "/workspace" ]; then
  WORKSPACE="/workspace"
else
  WORKSPACE="$HOME"
fi
mkdir -p "$WORKSPACE"
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
  export DEBIAN_FRONTEND=noninteractive
  sudo -E apt-get update -y
  sudo -E apt-get install -y \
    git \
    curl \
    unzip \
    python3 \
    python3-venv \
    python3-pip \
    jq \
    software-properties-common

  # Ensure Python 3.12+ (required for type hints and venv compatibility)
  PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
  if [ "$(echo "$PY_VER 3.12" | awk '{print ($1 < $2)}')" = "1" ]; then
    log "Python $PY_VER found, need 3.12+. Installing..."
    sudo -E add-apt-repository -y ppa:deadsnakes/ppa
    sudo -E apt-get update -y
    sudo -E apt-get install -y python3.12 python3.12-venv python3.12-dev
    # Do NOT change system python3 (breaks apt_pkg). Use python3.12 for venvs only.
    # Fix apt_pkg symlink in case update-alternatives was run previously
    APT_PKG_SO=$(find /usr/lib/python3/dist-packages -name "apt_pkg.cpython-3*-linux-gnu.so" 2>/dev/null | head -1)
    if [ -n "$APT_PKG_SO" ] && [ ! -e /usr/lib/python3/dist-packages/apt_pkg.so ]; then
      sudo ln -sf "$APT_PKG_SO" /usr/lib/python3/dist-packages/apt_pkg.so
    fi
    log "Python 3.12 installed: $(python3.12 --version)"
  else
    log "Python $PY_VER OK"
  fi

  # Install ngrok for exposing the backend to the internet (non-fatal)
  if ! command -v ngrok &>/dev/null; then
    log "Installing ngrok via apt"
    curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null 2>&1 \
      && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list >/dev/null 2>&1 \
      && sudo -E apt-get update -y >/dev/null 2>&1 \
      && sudo -E apt-get install -y ngrok >/dev/null 2>&1 \
      || log "WARNING: ngrok install failed. Install manually: snap install ngrok"
  fi
  # Configure ngrok auth token if provided
  if [ -n "${NGROK_AUTHTOKEN:-}" ] && command -v ngrok &>/dev/null; then
    log "Configuring ngrok auth token"
    ngrok config add-authtoken "$NGROK_AUTHTOKEN" 2>/dev/null || true
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

_clean_system_cuda() {
  # Remove system CUDA runtime libs that conflict with pip-bundled CUDA.
  # PyTorch and vLLM bundle their own CUDA runtime via pip wheels.
  # System CUDA (from apt) causes "Error 802: system not yet initialized"
  # due to version mismatch between system libcudart and pip libcudart.
  #
  # IMPORTANT: We keep nvcc (cuda-nvcc-12-x) and /usr/local/cuda because
  # flashinfer (used by vLLM for Qwen3.5 GDN attention) JIT-compiles CUDA
  # kernels at runtime and needs the CUDA compiler. Only the runtime libs
  # conflict; the compiler does not.
  log "Cleaning system CUDA runtime libs (keeping nvcc for JIT compilation)"

  # Remove CUDA runtime libs from LD_LIBRARY_PATH to prevent conflicts,
  # but keep /usr/local/cuda directory intact for nvcc.
  sudo rm -f /etc/profile.d/cuda.sh 2>/dev/null || true
  # Strip CUDA paths from LD_LIBRARY_PATH but do not unset entirely
  export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v cuda | tr '\n' ':' | sed 's/:$//')
  sudo ldconfig 2>/dev/null || true

  # Ensure nvcc is installed (may have been removed by previous provisioning)
  if ! command -v nvcc &>/dev/null; then
    local CUDA_MAJOR_MINOR
    CUDA_MAJOR_MINOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "12.4")
    local CUDA_PKG="cuda-nvcc-${CUDA_MAJOR_MINOR//./-}"
    log "  Installing $CUDA_PKG for flashinfer JIT compilation"
    sudo apt-get install -y "$CUDA_PKG" 2>/dev/null || true
  fi

  # Ensure /usr/local/cuda symlink exists (flashinfer looks for it)
  if [ ! -e /usr/local/cuda ]; then
    local CUDA_DIR
    CUDA_DIR=$(find /usr/local -maxdepth 1 -name "cuda-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CUDA_DIR" ]; then
      log "  Creating symlink: /usr/local/cuda -> $CUDA_DIR"
      sudo ln -sf "$CUDA_DIR" /usr/local/cuda
    elif [ -d "/usr/local/cuda.disabled" ]; then
      log "  Restoring /usr/local/cuda from .disabled"
      sudo mv /usr/local/cuda.disabled /usr/local/cuda
    fi
  fi
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

  # Clean any system CUDA toolkit to prevent conflicts with pip CUDA wheels.
  # PyTorch/vLLM bundle their own CUDA runtime (nvidia-cublas-cu12, etc).
  # System CUDA from apt causes libcudart version mismatch (Error 802).
  _clean_system_cuda

  # Ensure nvidia-uvm module is loaded (required for CUDA runtime).
  # nvidia-smi works without UVM, but torch.cuda.is_available() does not.
  # This is the #1 root cause of CUDA failures on KVM-virtualized GPU VMs.
  if ! lsmod | grep -q nvidia_uvm; then
    log "Loading nvidia-uvm kernel module (required for CUDA runtime)"
    sudo modprobe nvidia-uvm || true
  fi
  if command -v nvidia-modprobe &>/dev/null; then
    sudo nvidia-modprobe -u -c=0 2>/dev/null || true
  fi

  # Verify CUDA device nodes exist
  for dev in /dev/nvidia-uvm /dev/nvidiactl; do
    if [ ! -e "$dev" ]; then
      log "WARNING: $dev not found. CUDA may not work."
      log "Run: bash scripts/fix_cuda_and_verify.sh for full diagnostics."
    fi
  done

  # Enable GPU persistence mode (prevents Error 802 on some VMs)
  sudo nvidia-smi -pm 1 2>/dev/null || true

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
# Use python3.12 if available, else python3
PYTHON_BIN="$(command -v python3.12 2>/dev/null || command -v python3)"

ensure_venv() {
  local target="$1"
  local req_file="$2"

  if [ ! -d "$target" ]; then
    log "Creating venv: $target ($(${PYTHON_BIN} --version))"
    "${PYTHON_BIN}" -m venv "$target"
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
    log "Creating venv: $APP_DIR/.venv ($(${PYTHON_BIN} --version))"
    "${PYTHON_BIN}" -m venv "$APP_DIR/.venv"
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

  log "  Silero TTS model (v5_4_ru, xenia voice)"
  "$APP_DIR/.venv-voice-oss/bin/python" -c "from silero import silero_tts; silero_tts(language='ru', speaker='v5_4_ru'); print('Silero TTS v5_4_ru downloaded')"

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

  # Auto-detect GPU memory and set optimal vLLM utilization
  local GPU_MIB GPU_GB GPU_UTIL GPU_NAME
  GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  GPU_GB=$(( ${GPU_MIB:-0} / 1024 ))
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')

  # Dynamic utilization: leave ~20GB for voice services (Whisper, embedding, reranker)
  # Model uses ~36GB FP8. Rest is KV cache for concurrent requests.
  if [ "$GPU_GB" -ge 120 ]; then
    # H200 141GB or larger: plenty of room
    GPU_UTIL="0.70"
    log "GPU: ${GPU_NAME} ${GPU_GB}GB -> vLLM utilization ${GPU_UTIL} (large GPU, max concurrency)"
  elif [ "$GPU_GB" -ge 90 ]; then
    # H100 NVL 94GB
    GPU_UTIL="0.60"
    log "GPU: ${GPU_NAME} ${GPU_GB}GB -> vLLM utilization ${GPU_UTIL} (standard)"
  elif [ "$GPU_GB" -ge 75 ]; then
    # H100 80GB or A100 80GB
    GPU_UTIL="0.55"
    log "GPU: ${GPU_NAME} ${GPU_GB}GB -> vLLM utilization ${GPU_UTIL} (tight, reduced concurrency)"
  else
    # Smaller GPU, may not fit
    GPU_UTIL="0.50"
    log "WARNING: GPU ${GPU_NAME} has only ${GPU_GB}GB. Stack needs ~75GB. May not fit."
    log "GPU: ${GPU_NAME} ${GPU_GB}GB -> vLLM utilization ${GPU_UTIL} (minimum)"
  fi

  # Auto-detect CUDA lib path for Whisper.
  # Pip packages bundle CUDA libs under nvidia/ in the venv. The directory
  # name changes between versions: cu12 (cublas/lib, cudnn/lib) vs cu13
  # (cu13/lib). Detect whichever exists.
  local WHISPER_CUDA_LIB_PATH=""
  local VOICE_VENV="$APP_DIR/.venv-voice-oss"
  local PY_VER_DIR
  PY_VER_DIR=$(ls -d "$VOICE_VENV"/lib/python3.* 2>/dev/null | head -1)
  if [ -n "$PY_VER_DIR" ]; then
    local NVIDIA_DIR="$PY_VER_DIR/site-packages/nvidia"
    if [ -d "$NVIDIA_DIR/cu13/lib" ]; then
      # New layout: all CUDA 13 libs in one directory
      WHISPER_CUDA_LIB_PATH="$NVIDIA_DIR/cu13/lib"
      log "Whisper CUDA libs: cu13 layout ($WHISPER_CUDA_LIB_PATH)"
    elif [ -d "$NVIDIA_DIR/cublas/lib" ]; then
      # Old layout: separate directories per library
      WHISPER_CUDA_LIB_PATH="$NVIDIA_DIR/cublas/lib:$NVIDIA_DIR/cudnn/lib"
      log "Whisper CUDA libs: cu12 layout ($WHISPER_CUDA_LIB_PATH)"
    else
      log "WARNING: No CUDA libs found in $NVIDIA_DIR. Whisper may fail on GPU."
    fi
  fi
  # Convert absolute paths to relative (portable across workspace locations)
  WHISPER_CUDA_LIB_PATH=$(echo "$WHISPER_CUDA_LIB_PATH" | sed "s|$APP_DIR/|./|g")

  cat > "$APP_DIR/.env" <<ENVEOF
# Generated by provision_server.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# GPU: ${GPU_NAME} ${GPU_GB}GB -- vLLM utilization: ${GPU_UTIL}
# Whisper STT + Silero TTS + Qwen3.5-35B-A3B-FP8

RAG_LLM_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1
RAG_LLM_MODEL=Qwen/Qwen3.5-35B-A3B-FP8
RAG_LLM_FAST_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1
RAG_LLM_FAST_MODEL=Qwen/Qwen3.5-35B-A3B-FP8

WHISPER_BASE_URL=http://127.0.0.1:50002
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

SILERO_TTS_BASE_URL=http://127.0.0.1:50006
SILERO_TTS_SPEAKER=xenia
SILERO_TTS_MODEL=v5_4_ru
SILERO_TTS_SAMPLE_RATE=24000

VAD_SILENCE_MS=500
SILERO_VAD_PATH=./models/silero_vad.jit

STACK_MODE=docker
RAG_LAUNCH_MODE=supervisor
STACK_VOICE_PROFILE=oss_russian

STACK_QWEN_CMD="./.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-35B-A3B-FP8 --port ${VLLM_PORT} --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization ${GPU_UTIL} --download-dir ${MODELS_DIR}"
STACK_WHISPER_CMD="LD_LIBRARY_PATH=${WHISPER_CUDA_LIB_PATH}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} ./.venv-voice-oss/bin/python -m uvicorn services.whisper_server:app --host 0.0.0.0 --port 50002"
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
  local _GPU_NAME_LOG
  _GPU_NAME_LOG=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "GPU not detected")
  log "=== Provisioning: ${_GPU_NAME_LOG} ==="
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
