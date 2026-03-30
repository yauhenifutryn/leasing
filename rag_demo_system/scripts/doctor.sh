#!/usr/bin/env bash
# doctor.sh -- Diagnose and fix the deployment stack.
# Winning stack: Whisper STT + Silero TTS + Qwen3.5-35B-A3B-FP8.
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
# 5. Voice services (Whisper STT + Silero TTS)
# -----------------------------------------------------------------------
echo ""
echo "=== 5. Voice services ==="

check_sidecar() {
  local name="$1" port="$2" prog="$3"
  HEALTH=$(curl -s --max-time 5 "http://localhost:$port/health" 2>/dev/null || echo "")
  if [ -n "$HEALTH" ]; then
    IS_OK=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null || echo "")
    if [ "$IS_OK" = "True" ]; then
      ok "$name :$port"
    else
      REASON=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('reason','unknown'))" 2>/dev/null || echo "unknown")
      fail "$name :$port responding but not ready: $REASON"
    fi
  else
    if [ "$SUPERVISOR_OK" = true ]; then
      STATUS=$("$SUPERVISORCTL" -c "$CONF" status "$prog" 2>/dev/null | awk '{print $2}' || echo "UNKNOWN")
      if [ "$STATUS" = "STOPPED" ]; then
        info "$name :$port not started (STOPPED in supervisor)"
      else
        fail "$name :$port not responding (supervisor: $STATUS)"
      fi
    else
      fail "$name :$port not responding"
    fi
  fi
}

check_sidecar "Whisper"    50002 whisper
check_sidecar "Silero TTS" 50006 silero_tts

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
