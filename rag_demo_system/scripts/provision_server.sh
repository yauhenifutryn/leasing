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
# Auto-detect workspace: find the partition with the most free space.
# Cloud providers mount large storage at various paths:
#   Vast.ai/RunPod: /workspace
#   Sesterce:       /ephemeral
#   Lambda/Hetzner: usually large root, or /mnt/data
#   Client servers: varies
#
# Strategy: if not explicitly set, pick the mount with the most free disk.
# Models are ~60GB, venvs ~15GB, repo ~2GB. Minimum 100GB recommended.
if [ -n "${WORKSPACE:-}" ]; then
  : # explicit override, use as-is
else
  # Find the mount point with the most available space (exclude tmpfs, snap, boot)
  WORKSPACE=$(df --output=avail,target -x tmpfs -x devtmpfs 2>/dev/null \
    | tail -n +2 \
    | grep -v -E '(/boot|/snap|/run)' \
    | sort -rn \
    | head -1 \
    | awk '{print $2}')
  # Fallback chain if df parsing fails
  if [ -z "$WORKSPACE" ] || [ "$WORKSPACE" = "/" ]; then
    if [ -d "/ephemeral" ] && [ "$(df --output=avail /ephemeral 2>/dev/null | tail -1)" -gt 100000000 ] 2>/dev/null; then
      WORKSPACE="/ephemeral"
    elif [ -d "/workspace" ]; then
      WORKSPACE="/workspace"
    else
      WORKSPACE="$HOME"
    fi
  fi
fi
mkdir -p "$WORKSPACE"

# Disk space check: warn if less than 100GB free on the selected workspace
AVAIL_KB=$(df --output=avail "$WORKSPACE" 2>/dev/null | tail -1 | tr -d ' ')
AVAIL_GB=$(( ${AVAIL_KB:-0} / 1048576 ))
if [ "$AVAIL_GB" -lt 100 ] 2>/dev/null; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  WARNING: LOW DISK SPACE                                        ║"
  echo "║                                                                  ║"
  echo "║  Workspace: $WORKSPACE"
  echo "║  Available: ${AVAIL_GB}GB (need at least 100GB for models+venvs) ║"
  echo "║                                                                  ║"
  echo "║  The model download (~60GB) will likely fail.                    ║"
  echo "║  Options:                                                        ║"
  echo "║    1. Mount a larger disk and re-run with WORKSPACE=/mount/path  ║"
  echo "║    2. Free up space: du -sh /* | sort -rh | head -20             ║"
  echo "║    3. Check if there is extra storage: lsblk && df -h            ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Detected disk layout:"
  df -h --output=size,avail,pcent,target -x tmpfs -x devtmpfs 2>/dev/null || df -h
  echo ""
  read -p "Continue anyway? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Set WORKSPACE=/path/to/large/disk and re-run."
    exit 1
  fi
fi

APP_DIR="$WORKSPACE/leasing/rag_demo_system"
REPO_ROOT="$WORKSPACE/leasing"
MODELS_DIR="${MODELS_DIR:-$WORKSPACE/models}"

# Fail-fast: HF_TOKEN is required before any network work
HF_TOKEN="${HF_TOKEN:?ERROR: HF_TOKEN env var is required for gated model downloads. Export it before running this script.}"

VLLM_PORT=8787  # Avoid 8001 which RunPod reserves internally
SESSIONAGENT_PORT=8788  # Qwen3-4B dedicated SessionAgent instance
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

  # Add NVIDIA CUDA apt repository if not already configured.
  # Required for installing cuda-toolkit-12-6+ on systems that only ship 12.4.
  if ! apt-cache policy cuda-toolkit-12-6 2>/dev/null | grep -q "Candidate:"; then
    log "Adding NVIDIA CUDA apt repository"
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb 2>/dev/null \
      && sudo dpkg -i /tmp/cuda-keyring.deb 2>/dev/null \
      && rm -f /tmp/cuda-keyring.deb \
      || log "WARNING: Could not add NVIDIA apt repo. CUDA toolkit install may fail."
  fi

  sudo -E apt-get update -y

  # Core utilities
  sudo -E apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    jq \
    lsof \
    software-properties-common

  # Build tools (required by flashinfer JIT, vLLM native extensions, CTranslate2).
  # Vast.ai Docker images include these; bare VMs do not.
  sudo -E apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    pkg-config

  # Python + dev headers (python3-dev needed for C extension builds)
  sudo -E apt-get install -y \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip

  # Libraries required by vLLM, PyTorch, and audio processing
  sudo -E apt-get install -y \
    ffmpeg \
    libssl-dev \
    libnuma-dev \
    libtcmalloc-minimal4 || true

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

  # Ensure nvcc >= 12.6 is installed for flashinfer JIT compilation.
  # flashinfer 0.6+ GDN kernels require CUDA 12.6+ headers. The system may
  # only have 12.4 (Jarvis Labs, older Ubuntu). The NVIDIA driver 550+ supports
  # CUDA 12.6 even if the system toolkit is older; we just need the compiler.
  local NEED_NVCC=false
  if ! command -v nvcc &>/dev/null; then
    NEED_NVCC=true
  else
    local NVCC_VER
    NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "0.0")
    local NVCC_MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
    local NVCC_MINOR=$(echo "$NVCC_VER" | cut -d. -f2)
    if [ "$NVCC_MAJOR" -lt 12 ] || { [ "$NVCC_MAJOR" -eq 12 ] && [ "$NVCC_MINOR" -lt 6 ]; }; then
      log "  nvcc $NVCC_VER is too old for flashinfer (need >= 12.6)"
      NEED_NVCC=true
    fi
  fi

  if [ "$NEED_NVCC" = true ]; then
    # Install ONLY the compiler (cuda-nvcc), NOT the full toolkit (cuda-toolkit).
    # cuda-toolkit includes runtime libs (libcudart, libcublas) that conflict
    # with the pip-bundled CUDA libs and cause Whisper PTX version mismatches.
    # flashinfer JIT only needs nvcc + headers, not runtime libs.
    log "  Installing CUDA nvcc >= 12.6 for flashinfer JIT (compiler only)"
    if sudo -E apt-get install -y cuda-nvcc-12-8 cuda-cudart-dev-12-8 2>/dev/null; then
      log "  Installed cuda-nvcc-12-8 + headers"
    elif sudo -E apt-get install -y cuda-nvcc-12-6 cuda-cudart-dev-12-6 2>/dev/null; then
      log "  Installed cuda-nvcc-12-6 + headers"
    else
      log "  WARNING: Could not install CUDA nvcc >= 12.6. flashinfer JIT may fail."
      log "  Add NVIDIA apt repo: https://developer.nvidia.com/cuda-downloads"
    fi
  fi

  # Point /usr/local/cuda symlink to the newest installed CUDA toolkit.
  # Systems may have /usr/local/cuda -> /etc/alternatives/cuda -> cuda-12.4
  # even after installing cuda-12.8. Force the symlink to the newest version.
  local CUDA_DIR
  CUDA_DIR=$(find /usr/local -maxdepth 1 -name "cuda-1*" -type d 2>/dev/null | sort -V | tail -1)
  if [ -n "$CUDA_DIR" ]; then
    log "  Setting symlink: /usr/local/cuda -> $CUDA_DIR"
    sudo rm -f /usr/local/cuda
    sudo ln -sf "$CUDA_DIR" /usr/local/cuda
  elif [ ! -e /usr/local/cuda ] && [ -d "/usr/local/cuda.disabled" ]; then
    log "  Restoring /usr/local/cuda from .disabled"
    sudo mv /usr/local/cuda.disabled /usr/local/cuda
  fi

  # Set CUDA_HOME and put the correct nvcc FIRST in PATH.
  # Remove any old cuda paths from PATH to prevent stale nvcc being found.
  if [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH="${CUDA_HOME}/bin:$(echo "$PATH" | tr ':' '\n' | grep -v '/cuda' | tr '\n' ':' | sed 's/:$//')"
    log "  CUDA_HOME=$CUDA_HOME (nvcc: $(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo 'not found'))"
  fi
}

check_nvidia_driver() {
  # On KVM VMs (Jarvis Labs), nvidia kernel modules may not auto-load after
  # reboot. Try loading them before concluding the driver is missing.
  if ! nvidia-smi &>/dev/null; then
    log "nvidia-smi failed. Trying to load kernel modules..."
    sudo modprobe nvidia 2>/dev/null || true
    sudo modprobe nvidia-uvm 2>/dev/null || true
    if command -v nvidia-modprobe &>/dev/null; then
      sudo nvidia-modprobe -u -c=0 2>/dev/null || true
    fi
    sleep 2
  fi

  if nvidia-smi &>/dev/null; then
    log "NVIDIA driver OK: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  elif _is_container; then
    log "ERROR: nvidia-smi not found inside container. The host must have NVIDIA drivers."
    log "If using RunPod/Vast.ai, select a GPU-enabled template."
    exit 1
  else
    # Check if the driver package is installed but just not loading.
    # On KVM VMs, the host controls the driver version. NEVER upgrade it;
    # a mismatched driver breaks GPU passthrough permanently.
    local NVIDIA_PKG_INSTALLED
    NVIDIA_PKG_INSTALLED=$(dpkg -l 2>/dev/null | grep -c "nvidia-driver" || echo "0")

    if [ "$NVIDIA_PKG_INSTALLED" -gt 0 ]; then
      log "ERROR: NVIDIA driver package is installed but GPU is not visible."
      log "This typically happens on KVM VMs after reboot when the host"
      log "does not re-attach the GPU. This is a platform issue."
      log ""
      log "Try these steps:"
      log "  1. Restart the instance from the provider dashboard (not sudo reboot)"
      log "  2. If that fails, delete and recreate the instance"
      log "  3. If on bare metal, check: dmesg | grep -i nvidia"
      log ""
      log "DO NOT reinstall the driver. The existing driver must match the host."
      exit 1
    fi

    # No driver installed at all (true bare metal first-time setup)
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
  log "  Installing vLLM + supervisor + hf_transfer + ninja (big install, includes torch)"
  "$APP_DIR/.venv/bin/pip" install vllm==0.19.0 supervisor hf_transfer ninja
  log "  Installing backend-only packages (small, no duplicates)"
  "$APP_DIR/.venv/bin/pip" install \
    uvicorn \
    pyyaml \
    requests \
    qdrant-client \
    sentence-transformers \
    rank-bm25 \
    num2words \
    pymorphy3 \
    aiortc \
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
  export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

  # Prefer new 'hf' CLI; fall back to deprecated 'huggingface-cli'
  local hf_cli="$APP_DIR/.venv/bin/hf"
  if [ ! -x "$hf_cli" ]; then
    hf_cli="$APP_DIR/.venv/bin/huggingface-cli"
  fi
  mkdir -p "$MODELS_DIR"

  log "=== Downloading HuggingFace models to HF_HOME=$HF_HOME ==="

  log "  Qwen3.5-35B-A3B-FP8 (brain, half VRAM)"
  "$hf_cli" download Qwen/Qwen3.5-35B-A3B-FP8 \
    --token "$HF_TOKEN"

  # Always download the SessionAgent small model. Cheap (~4-5 GB) and avoids
  # a variable-scope bug where SESSIONAGENT_GPU_UTIL (set later in
  # write_env_file) is not yet visible here. If SessionAgent ends up disabled
  # on a small GPU, the weights just sit in cache with no runtime impact.
  log "  Qwen3-4B-Instruct-2507-FP8 (SessionAgent, ~4-5GB)"
  "$hf_cli" download Qwen/Qwen3-4B-Instruct-2507-FP8 \
    --token "$HF_TOKEN"

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
  # Always regenerate .env to ensure paths and flags are current.
  # Tool credentials (CALCULATOR_API_TOKEN, SMS_*) are left blank here;
  # export them and run `bash scripts/set_tokens.sh` afterwards. That
  # helper patches .env in place and restarts the backend only (no vLLM
  # reload). Shell env vars ARE read as a safety net if you happened to
  # export them before running provision, but the canonical flow is to
  # export afterwards per the hint printed at the end of this script.
  if [ -f "$APP_DIR/.env" ]; then
    log "Overwriting existing .env (paths/flags may have changed)"
  fi
  log "Writing .env file to $APP_DIR/.env"

  # Auto-detect GPU memory and set optimal vLLM utilization
  local GPU_MIB GPU_GB GPU_UTIL GPU_NAME
  GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  GPU_GB=$(( ${GPU_MIB:-0} / 1024 ))
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')

  # Dynamic utilization: leave ~20GB for voice services (Whisper, embedding, reranker)
  # + ~6GB for SessionAgent (Qwen3-4B-FP8 + KV cache).
  # Model uses ~36GB FP8. Rest is KV cache for concurrent requests.
  #
  # vLLM 0.19 (2026-04 upgrade) reserves CUDA-graph memory more aggressively
  # against the budget — the prior 0.60/0.62/0.68 thresholds drop to a
  # negative KV-cache figure on a fresh VM ("Available KV cache memory:
  # -0.3 GiB" → engine refuses to start). Bumped each tier ~+0.10 to
  # restore the previously-effective KV cache size. Verified on H100 PCIe
  # 80GB at 0.70 (Codex CP-3.6 follow-up, fresh VM 38.128.232.83).
  local SESSIONAGENT_GPU_UTIL
  if [ "$GPU_GB" -ge 120 ]; then
    # H200 141GB or larger: plenty of room
    GPU_UTIL="0.78"
    SESSIONAGENT_GPU_UTIL="0.10"
    log "GPU: ${GPU_NAME} ${GPU_GB}GB -> main ${GPU_UTIL}, sessionagent ${SESSIONAGENT_GPU_UTIL} (large GPU)"
  elif [ "$GPU_GB" -ge 90 ]; then
    # H100 NVL 94GB
    GPU_UTIL="0.72"
    SESSIONAGENT_GPU_UTIL="0.11"
    log "GPU: ${GPU_NAME} ${GPU_GB}GB -> main ${GPU_UTIL}, sessionagent ${SESSIONAGENT_GPU_UTIL} (standard)"
  elif [ "$GPU_GB" -ge 75 ]; then
    # H100 80GB or A100 80GB: 56GB main + 9.6GB SA = 65.6GB, ~14GB headroom
    GPU_UTIL="0.70"
    SESSIONAGENT_GPU_UTIL="0.12"
    log "GPU: ${GPU_NAME} ${GPU_GB}GB -> main ${GPU_UTIL}, sessionagent ${SESSIONAGENT_GPU_UTIL} (tight)"
  else
    # Smaller GPU, may not fit both models
    GPU_UTIL="0.65"
    SESSIONAGENT_GPU_UTIL="0.00"  # disabled, falls back to main
    log "WARNING: GPU ${GPU_NAME} has only ${GPU_GB}GB. SessionAgent disabled; using main LLM for classifier."
  fi

  # Auto-detect CUDA lib path for Whisper (faster-whisper / CTranslate2).
  # CTranslate2 is compiled against CUDA 12. It CANNOT use cu13 libs even
  # if they exist in the venv. Always prefer the cu12 layout (cublas/lib +
  # cudnn/lib). Only fall back to cu13 if cu12 is not available.
  local WHISPER_CUDA_LIB_PATH=""
  local VOICE_VENV="$APP_DIR/.venv-voice-oss"
  local PY_VER_DIR
  PY_VER_DIR=$(ls -d "$VOICE_VENV"/lib/python3.* 2>/dev/null | head -1)
  if [ -n "$PY_VER_DIR" ]; then
    local NVIDIA_DIR="$PY_VER_DIR/site-packages/nvidia"
    if [ -d "$NVIDIA_DIR/cublas/lib" ] && [ -d "$NVIDIA_DIR/cudnn/lib" ]; then
      # CTranslate2 needs CUDA 12 libs (cublas + cudnn separate directories)
      WHISPER_CUDA_LIB_PATH="$NVIDIA_DIR/cublas/lib:$NVIDIA_DIR/cudnn/lib"
      log "Whisper CUDA libs: cu12 layout ($WHISPER_CUDA_LIB_PATH)"
    elif [ -d "$NVIDIA_DIR/cu13/lib" ]; then
      # Fallback: cu13 layout (may not work with CTranslate2)
      WHISPER_CUDA_LIB_PATH="$NVIDIA_DIR/cu13/lib"
      log "WARNING: Whisper using cu13 libs. CTranslate2 may fail with PTX errors."
    else
      log "WARNING: No CUDA libs found in $NVIDIA_DIR. Whisper may fail on GPU."
    fi
  fi
  # Convert absolute paths to relative (portable across workspace locations)
  WHISPER_CUDA_LIB_PATH=$(echo "$WHISPER_CUDA_LIB_PATH" | sed "s|$APP_DIR/|./|g")

  # Auto-detect Whisper device: GPU requires driver >= 570 for CTranslate2 PTX compat.
  # CTranslate2 pip wheels use PTX ISA 8.7 (CUDA 12.8). Driver 550 only supports 8.4.
  # On driver < 570, fall back to CPU (int8) which runs at ~2-3s per utterance.
  local WHISPER_DEVICE="cuda"
  local WHISPER_COMPUTE="float16"
  local DRIVER_MAJOR
  DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 | tr -d ' ')
  if [ -n "$DRIVER_MAJOR" ] && [ "$DRIVER_MAJOR" -lt 570 ] 2>/dev/null; then
    WHISPER_DEVICE="cpu"
    WHISPER_COMPUTE="int8"
    log "NVIDIA driver ${DRIVER_MAJOR}.x < 570: Whisper will run on CPU (CTranslate2 PTX incompatible)"
    log "  Upgrade driver to 570+ to enable GPU Whisper"
  else
    log "NVIDIA driver ${DRIVER_MAJOR}.x >= 570: Whisper will run on GPU"
  fi

  # Model revision pinning: lock exact HF commits for reproducibility.
  # Leave empty (default) to track `main` branch on HuggingFace. Populate with
  # specific SHAs once tested. Overridable via env:
  #   QWEN_MAIN_REVISION=abc123 QWEN_SESSIONAGENT_REVISION=def456 bash provision_server.sh
  local QWEN_MAIN_REV_FLAG=""
  local QWEN_SESSIONAGENT_REV_FLAG=""
  if [ -n "${QWEN_MAIN_REVISION:-}" ]; then
    QWEN_MAIN_REV_FLAG="--revision ${QWEN_MAIN_REVISION}"
    log "Qwen main: pinned to revision ${QWEN_MAIN_REVISION}"
  fi
  if [ -n "${QWEN_SESSIONAGENT_REVISION:-}" ]; then
    QWEN_SESSIONAGENT_REV_FLAG="--revision ${QWEN_SESSIONAGENT_REVISION}"
    log "Qwen sessionagent: pinned to revision ${QWEN_SESSIONAGENT_REVISION}"
  fi

  # Conditionally emit SessionAgent env + launch command. When SESSIONAGENT_GPU_UTIL=0.00
  # (small GPU), SessionAgent is disabled and backend falls back to main LLM.
  local SESSIONAGENT_ENV_LINES=""
  local SESSIONAGENT_CMD_LINE=""
  if [ "$SESSIONAGENT_GPU_UTIL" != "0.00" ]; then
    SESSIONAGENT_ENV_LINES=$'\n'"SESSIONAGENT_BASE_URL=http://127.0.0.1:${SESSIONAGENT_PORT}/v1"$'\n'"SESSIONAGENT_MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8"
    SESSIONAGENT_CMD_LINE="STACK_SESSIONAGENT_CMD=\"./.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B-Instruct-2507-FP8 ${QWEN_SESSIONAGENT_REV_FLAG} --port ${SESSIONAGENT_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization ${SESSIONAGENT_GPU_UTIL} --enable-prefix-caching --download-dir ${MODELS_DIR}\""
  else
    SESSIONAGENT_ENV_LINES=$'\n'"# SessionAgent disabled on small GPU; classifier falls back to main LLM"$'\n'"SESSIONAGENT_BASE_URL="$'\n'"SESSIONAGENT_MODEL="
    SESSIONAGENT_CMD_LINE="STACK_SESSIONAGENT_CMD=\"\""
  fi

  cat > "$APP_DIR/.env" <<ENVEOF
# Generated by provision_server.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# GPU: ${GPU_NAME} ${GPU_GB}GB -- vLLM utilization: ${GPU_UTIL}
# Whisper STT + Silero TTS + Qwen3.5-35B-A3B-FP8

RAG_LLM_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1
RAG_LLM_MODEL=Qwen/Qwen3.5-35B-A3B-FP8
RAG_LLM_FAST_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1
RAG_LLM_FAST_MODEL=Qwen/Qwen3.5-35B-A3B-FP8
${SESSIONAGENT_ENV_LINES}

WHISPER_BASE_URL=http://127.0.0.1:50002
WHISPER_DEVICE=${WHISPER_DEVICE}
WHISPER_COMPUTE_TYPE=${WHISPER_COMPUTE}

SILERO_TTS_BASE_URL=http://127.0.0.1:50006
SILERO_TTS_SPEAKER=xenia
SILERO_TTS_MODEL=v5_4_ru
SILERO_TTS_SAMPLE_RATE=24000

VAD_SILENCE_MS=900
SILERO_VAD_PATH=./models/silero_vad.jit

STACK_MODE=docker
RAG_LAUNCH_MODE=supervisor
STACK_VOICE_PROFILE=oss_russian

STACK_QWEN_CMD="./.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-35B-A3B-FP8 ${QWEN_MAIN_REV_FLAG} --port ${VLLM_PORT} --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization ${GPU_UTIL} --download-dir ${MODELS_DIR} --enable-auto-tool-choice --tool-call-parser qwen3_xml --gdn-prefill-backend triton"
${SESSIONAGENT_CMD_LINE}
STACK_WHISPER_CMD="LD_LIBRARY_PATH=${WHISPER_CUDA_LIB_PATH} ./.venv-voice-oss/bin/python -m uvicorn services.whisper_server:app --host 0.0.0.0 --port 50002"
STACK_SILERO_TTS_CMD="./.venv-voice-oss/bin/python -m uvicorn services.silero_tts_server:app --host 0.0.0.0 --port 50006"

HF_HOME=${MODELS_DIR}
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
CUDA_HOME=/usr/local/cuda
PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Fix 37: cap parallel ninja / nvcc compilation jobs to avoid OOM kill during
# vLLM's GDN prefill kernel JIT. On a 64 GB box each nvcc invocation takes
# ~500 MB, and the default --threads=1 plus unlimited ninja parallelism runs
# 30+ concurrent nvcc processes on first launch -> OOM killer shoots ninja ->
# GDN kernel falls back to a slow path, causing a ~10 s first-token stall on
# long prompts. MAX_JOBS=2 keeps compile RAM under 2 GB.
MAX_JOBS=2

# ── Tool Use: Calculator API ──
# Picked up from the shell env at provision time. Export before running
# provision and these will be filled in automatically.
# Server IP must be whitelisted by client before these will work.
CALCULATOR_API_BASE_URL='${CALCULATOR_API_BASE_URL:-https://personal.mikro-leasing.by/calculator/api}'
CALCULATOR_API_TOKEN='${CALCULATOR_API_TOKEN:-}'

# ── Tool Use: SMS (sms-assistent.by) ──
SMS_API_LOGIN='${SMS_API_LOGIN:-}'
SMS_API_PASSWORD='${SMS_API_PASSWORD:-}'
SMS_SENDER_NAME='${SMS_SENDER_NAME:-MikroLizing}'

# ── Tool Use: CRM Webhook (phase 2, not needed yet) ──
CRM_WEBHOOK_URL=
CRM_WEBHOOK_TOKEN=
ENVEOF

  # Validate: ensure MODELS_DIR has enough disk space
  local models_avail_gb
  models_avail_gb=$(df --output=avail "$MODELS_DIR" 2>/dev/null | tail -1 | tr -d ' ')
  models_avail_gb=$(( ${models_avail_gb:-0} / 1048576 ))
  if [ "$models_avail_gb" -lt 60 ]; then
    log ""
    log "WARNING: MODELS_DIR=$MODELS_DIR has only ${models_avail_gb}GB free."
    log "Model weights need ~60GB. vLLM may fail to load."
    log "Move models to a larger disk: MODELS_DIR=/ephemeral/models"
    log ""
  else
    log "MODELS_DIR=$MODELS_DIR has ${models_avail_gb}GB free (OK)"
  fi

  # Detect whether tool tokens landed in .env or are still blank.
  local calc_set sms_set
  calc_set="$(grep -E "^CALCULATOR_API_TOKEN=" "$APP_DIR/.env" | grep -vE "=''$|=$" || true)"
  sms_set="$(grep -E "^SMS_API_LOGIN=" "$APP_DIR/.env" | grep -vE "=''$|=$" || true)"

  log ""
  log "╔══════════════════════════════════════════════════════════════╗"
  log "║  TOOL USE SETUP                                              ║"
  log "║                                                              ║"
  log "║  1. Whitelist server IP with client (Ilya):                  ║"
  log "║     curl -s ifconfig.me                                      ║"
  if [ -n "$calc_set" ] && [ -n "$sms_set" ]; then
    log "║                                                              ║"
    log "║  2. Tool tokens ALREADY in .env (exported pre-provision).    ║"
    log "║     Skip step 3, go straight to smoke test.                  ║"
  else
    log "║                                                              ║"
    log "║  2. Tool tokens NOT in .env. Export + apply them:            ║"
    log "║     export CALCULATOR_API_TOKEN='...'                        ║"
    log "║     export SMS_API_LOGIN='...'                               ║"
    log "║     export SMS_API_PASSWORD='...'                            ║"
    log "║     bash scripts/set_tokens.sh   # patches .env + restarts   ║"
  fi
  log "║                                                              ║"
  log "║  3. Smoke test:                                              ║"
  log "║     bash scripts/smoke_test.sh                               ║"
  log "║                                                              ║"
  log "║  4. Deploy SIP telephony:                                    ║"
  log "║     bash scripts/deploy_jambonz.sh                           ║"
  log "╚══════════════════════════════════════════════════════════════╝"
  log ""
}

# ---------------------------------------------------------------------------
# Step 9: Install coturn TURN server for WebRTC media relay
# ---------------------------------------------------------------------------
install_turn_server() {
  if command -v turnserver &>/dev/null; then
    log "coturn already installed, skipping"
    return
  fi
  log "Installing coturn TURN server..."
  if [ -f "$APP_DIR/scripts/setup_turn.sh" ]; then
    bash "$APP_DIR/scripts/setup_turn.sh" || log "WARNING: coturn setup failed (non-fatal, WebRTC relay unavailable)"
  else
    log "WARNING: setup_turn.sh not found, skipping TURN installation"
  fi
}

# ---------------------------------------------------------------------------
# Step 9b: SIP telephony (Jambonz)
# ---------------------------------------------------------------------------
setup_sip_notice() {
  # SIP telephony is handled by deploy_jambonz.sh, which should be run
  # separately after provision_server.sh completes. It deploys Jambonz
  # via Docker and does not require any steps from this script.
  log "SIP telephony setup deferred to deploy_jambonz.sh (run after provision + smoke)"
}

# ---------------------------------------------------------------------------
# Step 10: Start stack via stack.sh
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
  for port in 8000 $VLLM_PORT $SESSIONAGENT_PORT 50002 50006; do
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
    for port in 8000 $VLLM_PORT $SESSIONAGENT_PORT; do
      if lsof -ti :"$port" >/dev/null 2>&1; then
        log "ERROR: Port $port still occupied after cleanup. Cannot start stack."
        log "Run: lsof -i :$port   to investigate manually."
        exit 1
      fi
    done
  fi
  log "All ports free"

  # Clean GPU memory: kill any remaining GPU processes and reclaim VRAM
  if command -v nvidia-smi &>/dev/null; then
    local used_mib total_mib gpu_procs gpu_pids
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep '[0-9]' || true)
    gpu_procs=$(echo "$gpu_pids" | grep -c '[0-9]' 2>/dev/null || echo "0")

    if [ "${used_mib:-0}" -gt 1000 ]; then
      log "GPU memory in use: ${used_mib}MiB with ${gpu_procs} process(es). Cleaning up..."

      # Kill any remaining GPU processes
      if [ -n "$gpu_pids" ]; then
        for pid in $gpu_pids; do
          log "  Killing GPU process PID $pid (SIGTERM)"
          kill "$pid" 2>/dev/null || true
        done
        sleep 10

        # Check again, SIGKILL if needed
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep '[0-9]' || true)
        if [ -n "$gpu_pids" ]; then
          for pid in $gpu_pids; do
            log "  Force killing GPU process PID $pid (SIGKILL)"
            kill -9 "$pid" 2>/dev/null || true
          done
          sleep 5
        fi
      fi

      # Try GPU reset if memory is still held
      used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      if [ "${used_mib:-0}" -gt 1000 ]; then
        log "  Attempting nvidia-smi --gpu-reset..."
        nvidia-smi --gpu-reset 2>/dev/null && sleep 3 || log "  GPU reset not supported on this hardware"
      fi

      # Final check
      used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "0")
      if [ "${used_mib:-0}" -gt 5000 ] && [ "${gpu_procs:-0}" -eq 0 ]; then
        log "ERROR: Leaked GPU memory (${used_mib}MiB) could not be recovered."
        log "Restart the instance from your provider's dashboard, then re-run provision."
        exit 1
      fi
    fi
    log "GPU memory OK: ${used_mib:-0}MiB / ${total_mib}MiB used, ${gpu_procs} process(es)"
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
  write_env_file             # Step 8: generate .env (includes MAX_JOBS=2, Fix 37)
  install_turn_server        # Step 9: coturn for WebRTC relay
  setup_sip_notice           # Step 9b: SIP telephony notice (Jambonz deployed separately)
  # Step 10a: tune vLLM kernels (Fix 37 cache-clear + Fix 38 MoE config).
  # Fix 37 part A already applied via write_env_file (MAX_JOBS=2 in .env);
  # part B (stale cache clear) and Fix 38 (MoE tune) run here. Non-fatal:
  # if tuning fails vLLM still runs, just with default MoE routing.
  log "=== Tuning vLLM kernels (Fix 37 + Fix 38) ==="
  bash "$APP_DIR/scripts/tune_vllm_kernels.sh" || \
    log "WARNING: kernel tuning returned non-zero. vLLM will run with default MoE config."
  start_stack                # Step 10b: launch supervisor stack

  log "=== Provisioning complete ==="
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  Canonical fresh-server flow (you are on step 4):"
  echo ""
  echo "  1. ssh -i ~/.ssh/jarvislabs sesterce@<IP>"
  echo "  2. export HF_TOKEN='hf_...'                (for model download)"
  echo "  3. git clone https://github.com/yauhenifutryn/leasing.git"
  echo "     cd leasing && git checkout feature/voice-pipeline"
  echo "     cd rag_demo_system"
  echo "  4. bash scripts/provision_server.sh       ← DONE"
  echo ""
  echo "  Next (do these now, in order):"
  echo ""
  echo "  5. Export tool credentials + apply them:"
  echo "       export CALCULATOR_API_TOKEN='...' \\"
  echo "              SMS_API_LOGIN='...' SMS_API_PASSWORD='...'"
  echo "       bash scripts/set_tokens.sh"
  echo "  6. bash scripts/smoke_test.sh"
  echo "  7. bash scripts/deploy_jambonz.sh"
  echo ""
  echo "  Later — to pull new code on this already-provisioned server:"
  echo "    cd /ephemeral/leasing && git pull"
  echo "    cd rag_demo_system && bash scripts/restart_all.sh"
  echo "════════════════════════════════════════════════════════════"
  echo ""
}

main "$@"
