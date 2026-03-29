#!/usr/bin/env bash
# Start Qwen3.5-35B directly and run benchmark. No 30B needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"
CONF="$APP_DIR/scripts/supervisord.conf"
VLLM_PORT=8787

echo "[35b] ============================================="
echo "[35b]   Direct Qwen3.5-35B Benchmark"
echo "[35b] ============================================="

# --- Kill everything ---
echo "[35b] Killing all existing processes..."
"$SUPERVISORCTL" -c "$CONF" shutdown 2>/dev/null || true
sleep 3
pkill -f supervisord 2>/dev/null || true
pkill -f vllm 2>/dev/null || true
pkill -f uvicorn 2>/dev/null || true
echo "[35b]   Waiting 20s for GPU cleanup..."
sleep 20

# GPU check
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "0")
echo "[35b]   GPU: ${USED_MIB}MiB, ${GPU_PROCS} procs"
if [ "$USED_MIB" -gt 5000 ] && [ "$GPU_PROCS" -eq 0 ]; then
  echo "[35b]   ERROR: Leaked GPU memory. Restart instance."
  exit 1
fi

# --- Qdrant ---
echo "[35b] Starting Qdrant..."
if ! curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
  QDRANT_DIR="/workspace/qdrant"
  mkdir -p /workspace/qdrant_storage
  QDRANT__STORAGE__STORAGE_PATH=/workspace/qdrant_storage nohup "$QDRANT_DIR/qdrant" > /workspace/qdrant.log 2>&1 &
  sleep 3
fi
echo "[35b]   Qdrant OK"

# --- Patch .env to 3.5-35B-FP8 ---
echo "[35b] Patching .env for Qwen3.5-35B-A3B-FP8..."
cp "$APP_DIR/.env" "$APP_DIR/.env.brain_backup"
sed -i 's|Qwen/Qwen3-30B-A3B|Qwen/Qwen3.5-35B-A3B-FP8|g' "$APP_DIR/.env"

# Ensure embedding/reranker are on GPU (FP8 model is small enough)
sed -i 's/device: "cpu"/device: "cuda"/g' "$APP_DIR/config/app.yaml"

# --- Start supervisor ---
echo "[35b] Starting supervisor (backend + vLLM)..."
rm -f "$APP_DIR/.state/supervisord.pid" "$APP_DIR/.state/supervisor.sock"
cd "$APP_DIR" && bash scripts/stack.sh up
sleep 3

# Stop all voice services
"$SUPERVISORCTL" -c "$CONF" stop qwen3_tts whisper sensevoice cosyvoice vosk_tts qwen3_asr voxtral qwen3_omni 2>/dev/null || true

# --- Wait for vLLM ---
echo "[35b] Waiting for vLLM (Qwen3.5-35B)..."
ELAPSED=0
MAX_WAIT=600
while true; do
  CODE=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" "http://localhost:$VLLM_PORT/health" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    echo "[35b]   vLLM ready after ${ELAPSED}s"
    break
  fi
  ELAPSED=$((ELAPSED + 15))
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "[35b]   ERROR: vLLM not ready after ${MAX_WAIT}s"
    tail -10 "$APP_DIR/.state/qwen.err.log"
    mv "$APP_DIR/.env.brain_backup" "$APP_DIR/.env" 2>/dev/null || true
    exit 1
  fi
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "[35b]   ...loading (${ELAPSED}s | GPU: ${USED}MiB)"
  sleep 15
done

# Health checks
echo "[35b] Health checks..."
echo -n "[35b]   Backend: "; curl -s --max-time 5 http://localhost:8000/api/health >/dev/null && echo "OK" || echo "FAILED"
echo -n "[35b]   vLLM:    "; curl -s --max-time 5 "http://localhost:$VLLM_PORT/health" >/dev/null && echo "OK" || echo "FAILED"

# --- Ensure KB is indexed ---
echo ""
echo "[35b] Checking KB index..."
KB_COUNT=$(curl -s http://localhost:6333/collections/micro_leasing_kb 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('result',{}).get('points_count',0))" 2>/dev/null || echo "0")
if [ "$KB_COUNT" -lt 3000 ]; then
  echo "[35b]   KB has $KB_COUNT points, indexing..."
  curl -s --max-time 600 -X POST http://localhost:8000/api/index -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1
  KB_COUNT=$(curl -s http://localhost:6333/collections/micro_leasing_kb 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('result',{}).get('points_count',0))" 2>/dev/null || echo "0")
  echo "[35b]   KB indexed: $KB_COUNT points"
else
  echo "[35b]   KB already indexed: $KB_COUNT points"
fi

# --- Run benchmark ---
echo ""
echo "[35b] Running benchmark..."
"$APP_DIR/.venv/bin/python" "$APP_DIR/scripts/benchmark_text.py" \
  --profile brain_upgrade \
  --output "$APP_DIR/results/text_brain_upgrade_$(date +%Y%m%d_%H%M%S).jsonl" \
  --backend-url "http://localhost:8000"

# --- Restore .env ---
echo "[35b] Restoring .env to Qwen3-30B..."
mv "$APP_DIR/.env.brain_backup" "$APP_DIR/.env"

echo ""
echo "[35b] ============================================="
echo "[35b]   Done. Results in results/ directory."
echo "[35b]   .env restored to Qwen3-30B."
echo "[35b]   Run start_benchmark_mode.sh to reload 30B."
echo "[35b] ============================================="
