#!/usr/bin/env bash
# doctor.sh -- Diagnose and fix the deployment stack.
# Run after provision or whenever something is broken.
# Safe to re-run: every check is idempotent.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"
CONF="$APP_DIR/scripts/supervisord.conf"
VLLM_PORT="${VLLM_PORT:-8787}"
PASS=0
FAIL=0
FIX=0

green()  { printf "\033[32m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }

ok()   { green  "[OK]   $*"; PASS=$((PASS + 1)); }
fail() { red    "[FAIL] $*"; FAIL=$((FAIL + 1)); }
fix()  { yellow "[FIX]  $*"; FIX=$((FIX + 1)); }
info() { echo   "[....] $*"; }

# -----------------------------------------------------------------------
# 1. GPU
# -----------------------------------------------------------------------
echo ""
echo "=== 1. GPU ==="

if ! nvidia-smi &>/dev/null; then
  fail "nvidia-smi not available"
else
  CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "0")

  ok "nvidia-smi works (driver $CUDA_VER, ${USED_MIB}/${TOTAL_MIB} MiB, ${GPU_PROCS} procs)"

  # Leaked memory check
  if [ "$USED_MIB" -gt 5000 ] && [ "$GPU_PROCS" -eq 0 ]; then
    fail "Leaked GPU memory: ${USED_MIB}MiB used but 0 processes"
    red "      FIX: Restart instance from provider dashboard, then re-run provision"
  fi

  # CUDA version check (need 13.0+ for qwen-tts)
  CUDA_TOOLKIT=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "unknown")
  if [ "$CUDA_TOOLKIT" = "unknown" ]; then
    fail "Cannot detect CUDA version"
  else
    MAJOR=$(echo "$CUDA_TOOLKIT" | cut -d. -f1)
    if [ "$MAJOR" -ge 13 ]; then
      ok "CUDA $CUDA_TOOLKIT (>= 13.0 required for qwen-tts)"
    else
      fail "CUDA $CUDA_TOOLKIT < 13.0. Qwen3-TTS requires CUDA 13.0+"
    fi
  fi
fi

# -----------------------------------------------------------------------
# 2. Qdrant
# -----------------------------------------------------------------------
echo ""
echo "=== 2. Qdrant ==="

if curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
  ok "Qdrant healthy"
else
  fail "Qdrant not responding on :6333"
  info "Attempting to start Qdrant..."
  QDRANT_DIR="/workspace/qdrant"
  if [ -f "$QDRANT_DIR/qdrant" ]; then
    pkill -f "qdrant" 2>/dev/null || true
    sleep 1
    mkdir -p /workspace/qdrant_storage
    QDRANT__STORAGE__STORAGE_PATH=/workspace/qdrant_storage nohup "$QDRANT_DIR/qdrant" > /workspace/qdrant.log 2>&1 &
    sleep 3
    if curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
      fix "Qdrant started"
    else
      fail "Qdrant still not responding after start attempt"
    fi
  else
    fail "Qdrant binary not found at $QDRANT_DIR/qdrant"
  fi
fi

# -----------------------------------------------------------------------
# 3. Supervisor
# -----------------------------------------------------------------------
echo ""
echo "=== 3. Supervisor ==="

SUPERVISOR_OK=false
if [ ! -f "$SUPERVISORCTL" ]; then
  fail "supervisorctl not found at $SUPERVISORCTL"
else
  # supervisorctl status returns non-zero if ANY program is STOPPED.
  # Use 'pid' which only fails if supervisord itself is unreachable.
  if "$SUPERVISORCTL" -c "$CONF" pid >/dev/null 2>&1; then
    ok "supervisord running (pid $("$SUPERVISORCTL" -c "$CONF" pid 2>/dev/null))"
    SUPERVISOR_OK=true
  else
    fail "supervisord not running"
    info "Attempting to start supervisor..."
    rm -f "$APP_DIR/.state/supervisord.pid" "$APP_DIR/.state/supervisor.sock"
    cd "$APP_DIR" && bash scripts/stack.sh up && cd - >/dev/null
    sleep 3
    if "$SUPERVISORCTL" -c "$CONF" pid >/dev/null 2>&1; then
      fix "supervisord started"
      SUPERVISOR_OK=true
    else
      fail "supervisord still not running"
    fi
  fi
fi

# -----------------------------------------------------------------------
# 4. Core services (backend + vLLM)
# -----------------------------------------------------------------------
echo ""
echo "=== 4. Core services ==="

# Backend
CODE=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" http://localhost:8000/api/health 2>/dev/null || echo "000")
if [ "$CODE" = "200" ]; then
  ok "Backend :8000"
else
  fail "Backend :8000 (HTTP $CODE)"
  if "$SUPERVISORCTL" -c "$CONF" status backend 2>/dev/null | grep -q RUNNING; then
    info "backend is RUNNING but not responding. Check: tail -20 $APP_DIR/.state/backend.err.log"
  else
    info "Attempting restart..."
    "$SUPERVISORCTL" -c "$CONF" restart backend 2>/dev/null || true
    sleep 3
    CODE=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" http://localhost:8000/api/health 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then fix "Backend restarted"; else fail "Backend still down"; fi
  fi
fi

# vLLM
CODE=$(curl -s --max-time 10 -o /dev/null -w "%{http_code}" "http://localhost:$VLLM_PORT/health" 2>/dev/null || echo "000")
if [ "$CODE" = "200" ]; then
  ok "vLLM :$VLLM_PORT"
else
  fail "vLLM :$VLLM_PORT (HTTP $CODE)"
  QWEN_STATUS=$("$SUPERVISORCTL" -c "$CONF" status qwen 2>/dev/null | awk '{print $2}' || echo "UNKNOWN")
  if [ "$QWEN_STATUS" = "RUNNING" ]; then
    info "vLLM process RUNNING but not healthy. May still be loading model."
    info "Check GPU progress: nvidia-smi"
    info "Check logs: tail -20 $APP_DIR/.state/qwen.err.log"
  else
    info "vLLM status: $QWEN_STATUS"
  fi
fi

# -----------------------------------------------------------------------
# 5. Voice sidecars
# -----------------------------------------------------------------------
echo ""
echo "=== 5. Voice sidecars ==="

check_sidecar() {
  local name="$1" port="$2" prog="$3"
  CODE=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    ok "$name :$port"
  else
    # Check if supervisor even has it configured to run
    if [ "$SUPERVISOR_OK" = true ]; then
      STATUS=$("$SUPERVISORCTL" -c "$CONF" status "$prog" 2>/dev/null | awk '{print $2}' || echo "UNKNOWN")
      if [ "$STATUS" = "STOPPED" ]; then
        info "$name :$port not started (STOPPED in supervisor, may need STACK_*_CMD in .env)"
      else
        fail "$name :$port (HTTP $CODE, supervisor status: $STATUS)"
      fi
    else
      fail "$name :$port (HTTP $CODE)"
    fi
  fi
}

check_sidecar "SenseVoice" 50000 sensevoice
check_sidecar "CosyVoice"  50001 cosyvoice
check_sidecar "Whisper"    50002 whisper

# -----------------------------------------------------------------------
# 6. Qwen3-TTS venv integrity
# -----------------------------------------------------------------------
echo ""
echo "=== 6. Qwen3-TTS venv ==="

TTS_VENV="$APP_DIR/.venv-qwen3-tts"
if [ ! -d "$TTS_VENV" ]; then
  fail "Qwen3-TTS venv not found at $TTS_VENV"
else
  # Check torch version
  TORCH_VER=$("$TTS_VENV/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
  TORCH_MAJOR=$(echo "$TORCH_VER" | grep -oP '^\d+\.\d+' || echo "0.0")

  if [ "$TORCH_VER" = "missing" ]; then
    fail "torch not installed in qwen3-tts venv"
  else
    # Compare: need >= 2.7
    MAJOR=$(echo "$TORCH_MAJOR" | cut -d. -f1)
    MINOR=$(echo "$TORCH_MAJOR" | cut -d. -f2)
    if [ "$MAJOR" -gt 2 ] || { [ "$MAJOR" -eq 2 ] && [ "$MINOR" -ge 7 ]; }; then
      ok "torch $TORCH_VER (>= 2.7 required)"
    else
      fail "torch $TORCH_VER < 2.7. qwen-tts needs torch >= 2.7"
      info "Fixing: installing torch + torchaudio from cu126 index..."
      "$TTS_VENV/bin/pip" install --upgrade \
        "torch>=2.7.0" "torchaudio>=2.7.0" \
        --index-url https://download.pytorch.org/whl/cu126 \
        2>&1 | tail -3
      NEW_VER=$("$TTS_VENV/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "failed")
      NEW_MAJOR=$(echo "$NEW_VER" | grep -oP '^\d+\.\d+' || echo "0.0")
      NM=$(echo "$NEW_MAJOR" | cut -d. -f1)
      NMN=$(echo "$NEW_MAJOR" | cut -d. -f2)
      if [ "$NM" -gt 2 ] || { [ "$NM" -eq 2 ] && [ "$NMN" -ge 7 ]; }; then
        fix "torch upgraded to $NEW_VER"
      else
        fail "torch upgrade failed (got $NEW_VER)"
      fi
    fi
  fi

  # Re-read torch version (may have been upgraded above)
  TORCH_VER=$("$TTS_VENV/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")

  # Check torchaudio matches torch
  TA_VER=$("$TTS_VENV/bin/python" -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "missing")
  TORCH_BASE=$(echo "$TORCH_VER" | grep -oP '^\d+\.\d+')
  TA_BASE=$(echo "$TA_VER" | grep -oP '^\d+\.\d+')
  if [ "$TORCH_BASE" = "$TA_BASE" ]; then
    ok "torchaudio $TA_VER matches torch $TORCH_VER"
  elif [ "$TA_VER" = "missing" ]; then
    fail "torchaudio not installed"
  else
    fail "torchaudio $TA_VER does not match torch $TORCH_VER"
    info "Fixing: installing matching torchaudio..."
    "$TTS_VENV/bin/pip" install --upgrade \
      "torchaudio>=${TORCH_BASE}.0" \
      --index-url https://download.pytorch.org/whl/cu126 \
      2>&1 | tail -3
    NEW_TA=$("$TTS_VENV/bin/python" -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "failed")
    fix "torchaudio upgraded to $NEW_TA"
  fi

  # Check qwen-tts import
  info "Testing qwen-tts import..."
  IMPORT_ERR=$("$TTS_VENV/bin/python" -c "from qwen_tts import Qwen3TTSModel; print('ok')" 2>&1 || true)
  if echo "$IMPORT_ERR" | grep -q "^ok$"; then
    ok "qwen-tts imports successfully"
  else
    fail "qwen-tts import failed: $(echo "$IMPORT_ERR" | tail -1)"
  fi
fi

# -----------------------------------------------------------------------
# 7. Qwen3-TTS service
# -----------------------------------------------------------------------
echo ""
echo "=== 7. Qwen3-TTS service ==="

# Check port first; fall back to supervisor status
TTS_HTTP=$(curl -s --max-time 10 -o /dev/null -w "%{http_code}" http://localhost:50003/health 2>/dev/null || echo "000")
TTS_STATUS=$("$SUPERVISORCTL" -c "$CONF" status qwen3_tts 2>/dev/null | awk '{print $2}' || echo "UNKNOWN")
if [ "$TTS_HTTP" = "200" ] || [ "$TTS_STATUS" = "RUNNING" ]; then
  CODE=$(curl -s --max-time 10 -o /dev/null -w "%{http_code}" http://localhost:50003/health 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    HEALTH=$(curl -s --max-time 5 http://localhost:50003/health 2>/dev/null)
    IS_OK=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null || echo "")
    if [ "$IS_OK" = "True" ]; then
      ok "Qwen3-TTS service healthy"
    else
      REASON=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('reason','unknown'))" 2>/dev/null || echo "unknown")
      fail "Qwen3-TTS not ready: $REASON"
    fi
  else
    fail "Qwen3-TTS :50003 (HTTP $CODE)"
  fi
elif [ "$TTS_STATUS" = "STOPPED" ]; then
  info "Qwen3-TTS not started (autostart=false). Starting..."
  "$SUPERVISORCTL" -c "$CONF" start qwen3_tts 2>/dev/null || true
  info "Waiting 45s for model load..."
  sleep 45
  HEALTH=$(curl -s --max-time 10 http://localhost:50003/health 2>/dev/null || echo '{}')
  IS_OK=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null || echo "")
  if [ "$IS_OK" = "True" ]; then
    fix "Qwen3-TTS started and healthy"
  else
    REASON=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('reason','unknown'))" 2>/dev/null || echo "unknown")
    fail "Qwen3-TTS started but not healthy: $REASON"
    info "Check logs: tail -30 $APP_DIR/.state/qwen3_tts.err.log"
  fi
else
  fail "Qwen3-TTS status: $TTS_STATUS"
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "============================================="
if [ "$FAIL" -eq 0 ]; then
  green "  ALL CHECKS PASSED ($PASS ok, $FIX fixed)"
  echo "  Next: bash scripts/smoke_test.sh"
else
  red   "  $FAIL FAILED, $PASS ok, $FIX fixed"
  echo "  Fix the failures above, then re-run: bash scripts/doctor.sh"
fi
echo "============================================="
exit "$FAIL"
