#!/usr/bin/env bash
# start_benchmark_mode.sh
# Starts only backend + vLLM + Qdrant for text-only benchmarks.
# No voice services (STT/TTS) to avoid CUDA memory pressure.
# Run after instance restart, before benchmark_orchestrator.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"
CONF="$APP_DIR/scripts/supervisord.conf"
VLLM_PORT=8787

echo "[bench-mode] ============================================="
echo "[bench-mode]   Starting Benchmark Mode (text only)"
echo "[bench-mode]   No STT/TTS services"
echo "[bench-mode] ============================================="

# --- GPU check ---
echo ""
echo "[bench-mode] GPU check..."
if nvidia-smi &>/dev/null; then
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "0")
  echo "[bench-mode]   GPU: ${USED_MIB}MiB used, ${GPU_PROCS} process(es)"
  if [ "$USED_MIB" -gt 5000 ] && [ "$GPU_PROCS" -eq 0 ]; then
    echo "[bench-mode]   ERROR: Leaked GPU memory. Restart instance first."
    exit 1
  fi
fi

# --- Qdrant ---
echo ""
echo "[bench-mode] Starting Qdrant..."
if curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
  echo "[bench-mode]   Qdrant already running"
else
  QDRANT_DIR="/workspace/qdrant"
  if [ -f "$QDRANT_DIR/qdrant" ]; then
    pkill -f "qdrant" 2>/dev/null || true
    sleep 1
    mkdir -p /workspace/qdrant_storage
    QDRANT__STORAGE__STORAGE_PATH=/workspace/qdrant_storage nohup "$QDRANT_DIR/qdrant" > /workspace/qdrant.log 2>&1 &
    sleep 3
    curl -fsS http://localhost:6333/healthz >/dev/null 2>&1 && echo "[bench-mode]   Qdrant OK" || echo "[bench-mode]   Qdrant FAILED"
  else
    echo "[bench-mode]   ERROR: Qdrant binary not found"
    exit 1
  fi
fi

# --- Supervisor: backend + vLLM only ---
echo ""
echo "[bench-mode] Starting supervisor..."
rm -f "$APP_DIR/.state/supervisord.pid" "$APP_DIR/.state/supervisor.sock"

# Source .env for STACK_QWEN_CMD
cd "$APP_DIR"
bash scripts/stack.sh up
sleep 3

# Make sure no voice services are running
"$SUPERVISORCTL" -c "$CONF" stop qwen3_tts 2>/dev/null || true
"$SUPERVISORCTL" -c "$CONF" stop whisper 2>/dev/null || true
"$SUPERVISORCTL" -c "$CONF" stop sensevoice 2>/dev/null || true
"$SUPERVISORCTL" -c "$CONF" stop cosyvoice 2>/dev/null || true
"$SUPERVISORCTL" -c "$CONF" stop vosk_tts 2>/dev/null || true
"$SUPERVISORCTL" -c "$CONF" stop qwen3_asr 2>/dev/null || true
"$SUPERVISORCTL" -c "$CONF" stop voxtral 2>/dev/null || true
"$SUPERVISORCTL" -c "$CONF" stop qwen3_omni 2>/dev/null || true

# --- Wait for vLLM ---
echo ""
echo "[bench-mode] Waiting for vLLM to load..."
ELAPSED=0
MAX_WAIT=300
while true; do
  CODE=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" "http://localhost:$VLLM_PORT/health" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    echo "[bench-mode]   vLLM ready after ${ELAPSED}s"
    break
  fi
  ELAPSED=$((ELAPSED + 10))
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "[bench-mode]   ERROR: vLLM not ready after ${MAX_WAIT}s"
    exit 1
  fi
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "[bench-mode]   ...loading (${ELAPSED}s | GPU: ${USED}MiB)"
  sleep 10
done

# --- Health check ---
echo ""
echo "[bench-mode] Health checks..."
echo -n "[bench-mode]   Backend: "; curl -s --max-time 5 http://localhost:8000/api/health >/dev/null && echo "OK" || echo "FAILED"
echo -n "[bench-mode]   vLLM:    "; curl -s --max-time 5 "http://localhost:$VLLM_PORT/health" >/dev/null && echo "OK" || echo "FAILED"
echo -n "[bench-mode]   Qdrant:  "; curl -s --max-time 5 http://localhost:6333/healthz >/dev/null && echo "OK" || echo "FAILED"

# GPU usage
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
FREE_MIB=$((TOTAL_MIB - USED_MIB))
echo "[bench-mode]   GPU: ${USED_MIB}/${TOTAL_MIB}MiB (${FREE_MIB}MiB free for embedding/reranker)"

echo ""
echo "[bench-mode] ============================================="
echo "[bench-mode]   Benchmark mode ready"
echo "[bench-mode]   Run: bash scripts/benchmark_orchestrator.sh"
echo "[bench-mode] ============================================="
