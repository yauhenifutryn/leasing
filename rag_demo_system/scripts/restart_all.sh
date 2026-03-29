#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"
CONF="$APP_DIR/scripts/supervisord.conf"
VLLM_PORT=8787

echo "[restart] ============================================="
echo "[restart]   Full Stack Restart"
echo "[restart] ============================================="

# --- Step 1: Graceful shutdown ---
echo ""
echo "[restart] Step 1: Shutting down supervisor..."
"$SUPERVISORCTL" -c "$CONF" shutdown 2>/dev/null || true
sleep 3

# --- Step 2: Graceful kill (SIGTERM, not SIGKILL) ---
# SIGKILL on GPU processes leaks CUDA memory in containers.
echo "[restart] Step 2: Sending SIGTERM to all processes..."
pkill -f supervisord 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "vllm" 2>/dev/null || true
pkill -f "python.*services" 2>/dev/null || true
echo "[restart]   Waiting 15s for GPU processes to release memory..."
sleep 15

# Force-kill non-GPU stragglers only
pkill -9 -f supervisord 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
# Do NOT kill -9 vllm unless absolutely necessary
if pgrep -f "vllm" >/dev/null 2>&1; then
  echo "[restart]   WARNING: vLLM still running. Waiting 15s more..."
  sleep 15
  if pgrep -f "vllm" >/dev/null 2>&1; then
    echo "[restart]   WARNING: Force-killing vLLM (may leak GPU memory)"
    pkill -9 -f "vllm" 2>/dev/null || true
    sleep 3
  fi
fi

# --- Step 3: Kill anything on service ports ---
echo "[restart] Step 3: Clearing service ports..."
for port in 8000 $VLLM_PORT 8002 50000 50001 50002 50003 50004 50005; do
  pid=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "  Killing PID $pid on port $port"
    kill -9 $pid 2>/dev/null || true
  fi
done
sleep 2

# --- Step 4: Check GPU memory ---
echo ""
echo "[restart] Step 4: GPU memory check..."
if nvidia-smi &>/dev/null; then
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  FREE_MIB=$((TOTAL_MIB - USED_MIB))
  echo "[restart]   GPU: used=${USED_MIB}MiB total=${TOTAL_MIB}MiB free=${FREE_MIB}MiB"

  # Check for leaked GPU memory (>5GB used with no processes)
  GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "0")
  if [ "$USED_MIB" -gt 5000 ] && [ "$GPU_PROCS" -eq 0 ]; then
    echo ""
    echo "[restart]   *** LEAKED GPU MEMORY DETECTED ***"
    echo "[restart]   ${USED_MIB}MiB used but no GPU processes running."
    echo "[restart]   Cannot continue. vLLM needs ~60GB free to start."
    echo ""
    echo "[restart]   FIX: Restart the instance from your provider's dashboard"
    echo "[restart]         (Vast.ai: Stop then Start. RunPod: Restart pod.)"
    echo "[restart]         Then re-run this script."
    exit 1
  fi

  if [ "$FREE_MIB" -lt 60000 ]; then
    echo ""
    echo "[restart]   *** NOT ENOUGH GPU MEMORY ***"
    echo "[restart]   Only ${FREE_MIB}MiB free. vLLM needs ~60GB."
    echo "[restart]   FIX: Restart the instance from your provider's dashboard."
    exit 1
  fi
fi

# --- Step 5: Clean state files ---
echo ""
echo "[restart] Step 5: Cleaning state files..."
rm -f "$APP_DIR/.state/supervisord.pid" "$APP_DIR/.state/supervisor.sock"
# Clear old log files to avoid confusion with stale error messages
> "$APP_DIR/.state/qwen.err.log"
> "$APP_DIR/.state/qwen.log"
> "$APP_DIR/.state/backend.err.log"

# --- Step 6: Start Qdrant ---
echo ""
echo "[restart] Step 6: Starting Qdrant..."
if ! curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
  QDRANT_DIR="/workspace/qdrant"
  if [ -f "$QDRANT_DIR/qdrant" ]; then
    pkill -f "qdrant" 2>/dev/null || true
    sleep 1
    mkdir -p /workspace/qdrant_storage
    QDRANT__STORAGE__STORAGE_PATH=/workspace/qdrant_storage nohup "$QDRANT_DIR/qdrant" > /workspace/qdrant.log 2>&1 &
    sleep 3
  fi
fi
curl -fsS http://localhost:6333/healthz >/dev/null 2>&1 && echo "[restart]   Qdrant OK" || echo "[restart]   Qdrant FAILED"

# --- Step 7: Start supervisor (backend + vLLM) ---
echo ""
echo "[restart] Step 7: Starting supervisor stack..."
cd "$APP_DIR"
bash scripts/stack.sh up
sleep 3

# --- Step 8: Wait for vLLM to load ---
echo ""
echo "[restart] Step 8: Waiting for vLLM to load model..."
ELAPSED=0
MAX_WAIT=600
while true; do
  CODE=$(curl -s --max-time 5 --connect-timeout 3 -o /dev/null -w "%{http_code}" "http://localhost:$VLLM_PORT/health" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    echo "[restart]   vLLM ready after ${ELAPSED}s"
    break
  fi

  ELAPSED=$((ELAPSED + 15))
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "[restart]   ERROR: vLLM not ready after ${MAX_WAIT}s"
    echo "[restart]   Check: tail -20 .state/qwen.err.log"
    # Check if it crashed
    QWEN_STATUS=$("$SUPERVISORCTL" -c "$CONF" status qwen 2>/dev/null | awk '{print $2}' || echo "UNKNOWN")
    if [ "$QWEN_STATUS" != "RUNNING" ]; then
      echo "[restart]   vLLM status: $QWEN_STATUS (crashed, NOT restarting to prevent memory leak)"
    fi
    break
  fi

  # Show GPU progress
  GPU_INFO=""
  if nvidia-smi &>/dev/null; then
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    GPU_INFO=" | GPU: ${USED}/${TOTAL}MiB"
  fi
  echo "[restart]   ...loading (${ELAPSED}s/${MAX_WAIT}s, HTTP $CODE${GPU_INFO})"
  sleep 15
done

# --- Step 9: Ensure unused services are stopped ---
echo ""
echo "[restart] Step 9: Stopping unused services..."
for svc in sensevoice cosyvoice vosk vosk_tts qwen3_tts qwen3_asr voxtral qwen3_omni ngrok frontend; do
  "$SUPERVISORCTL" -c "$CONF" stop "$svc" 2>/dev/null || true
done

# --- Step 10: Start active voice services ---
echo ""
echo "[restart] Step 10: Starting Whisper STT + Silero TTS..."
"$SUPERVISORCTL" -c "$CONF" start whisper 2>/dev/null || true
"$SUPERVISORCTL" -c "$CONF" start silero_tts 2>/dev/null || true
sleep 10

# --- Step 11: Health checks ---
echo ""
echo "[restart] Step 11: Health checks..."
echo -n "[restart]   Backend:    "; curl -s --max-time 5 http://localhost:8000/api/health >/dev/null && echo "OK" || echo "FAILED"
echo -n "[restart]   vLLM:       "; curl -s --max-time 5 http://localhost:$VLLM_PORT/health >/dev/null && echo "OK" || echo "FAILED"
echo -n "[restart]   Qdrant:     "; curl -s --max-time 5 http://localhost:6333/healthz >/dev/null && echo "OK" || echo "FAILED"
echo -n "[restart]   Whisper:    "; curl -s --max-time 5 http://localhost:50002/health >/dev/null && echo "OK" || echo "FAILED"
echo -n "[restart]   Silero TTS: "; curl -s --max-time 5 http://localhost:50006/health 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('OK' if d.get('ok') else f'FAILED ({d.get(\"reason\",\"unknown\")})')" 2>/dev/null || echo "FAILED"

echo ""
echo "[restart] ============================================="
echo "[restart]   Restart complete"
echo "[restart]   Run: bash scripts/doctor.sh   (diagnose)"
echo "[restart]   Or:  bash scripts/smoke_test.sh"
echo "[restart] ============================================="
