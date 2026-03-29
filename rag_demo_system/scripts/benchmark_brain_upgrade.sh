#!/usr/bin/env bash
# benchmark_brain_upgrade.sh
# Stops Qwen3-30B, starts Qwen3.5-35B, runs text benchmark, then restores.
# Run after start_benchmark_mode.sh when baseline benchmark is done.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"
CONF="$APP_DIR/scripts/supervisord.conf"
VLLM_PORT=8787

echo "[brain] ============================================="
echo "[brain]   Brain Upgrade Benchmark"
echo "[brain]   Qwen3-30B-A3B -> Qwen3.5-35B-A3B"
echo "[brain] ============================================="

# --- Step 1: Stop current vLLM ---
echo ""
echo "[brain] Step 1: Stopping vLLM (Qwen3-30B)..."
"$SUPERVISORCTL" -c "$CONF" stop qwen 2>/dev/null || true

# Wait for GPU memory to free (SIGTERM, not SIGKILL)
echo "[brain]   Waiting 20s for GPU memory release..."
sleep 20

# Verify GPU is free
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "0")
echo "[brain]   GPU: ${USED_MIB}MiB used, ${GPU_PROCS} process(es)"

if [ "$USED_MIB" -gt 10000 ]; then
  echo "[brain]   WARNING: GPU still has ${USED_MIB}MiB used."
  echo "[brain]   Waiting 30s more..."
  sleep 30
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "[brain]   GPU: ${USED_MIB}MiB used"
  if [ "$USED_MIB" -gt 10000 ]; then
    echo "[brain]   ERROR: GPU memory not freed. Cannot load Qwen3.5-35B."
    echo "[brain]   Restart instance and try again."
    exit 1
  fi
fi

# --- Step 2: Swap model in .env ---
echo ""
echo "[brain] Step 2: Patching .env for Qwen3.5-35B-A3B..."
cp "$APP_DIR/.env" "$APP_DIR/.env.brain_backup"
sed -i 's|Qwen/Qwen3-30B-A3B|Qwen/Qwen3.5-35B-A3B|g' "$APP_DIR/.env"
grep RAG_LLM_MODEL "$APP_DIR/.env"

# --- Step 3: Start vLLM with new model ---
echo ""
echo "[brain] Step 3: Starting vLLM with Qwen3.5-35B-A3B..."
"$SUPERVISORCTL" -c "$CONF" start qwen

echo "[brain]   Waiting for vLLM to load (may take 2-3 min)..."
ELAPSED=0
MAX_WAIT=600
while true; do
  CODE=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" "http://localhost:$VLLM_PORT/health" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    echo "[brain]   vLLM ready after ${ELAPSED}s"
    break
  fi
  ELAPSED=$((ELAPSED + 15))
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "[brain]   ERROR: vLLM not ready after ${MAX_WAIT}s"
    echo "[brain]   Check: tail -20 $APP_DIR/.state/qwen.err.log"
    echo "[brain]   Restoring .env..."
    mv "$APP_DIR/.env.brain_backup" "$APP_DIR/.env"
    exit 1
  fi
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "[brain]   ...loading (${ELAPSED}s | GPU: ${USED}MiB)"
  sleep 15
done

# --- Step 4: Run benchmark ---
echo ""
echo "[brain] Step 4: Running text benchmark..."
"$APP_DIR/.venv/bin/python" "$APP_DIR/scripts/benchmark_text.py" \
  --profile brain_upgrade \
  --output "$APP_DIR/results/text_brain_upgrade_$(date +%Y%m%d_%H%M%S).jsonl" \
  --backend-url "http://localhost:8000"

# --- Step 5: Restore original model ---
echo ""
echo "[brain] Step 5: Restoring Qwen3-30B-A3B..."
"$SUPERVISORCTL" -c "$CONF" stop qwen 2>/dev/null || true
sleep 20
mv "$APP_DIR/.env.brain_backup" "$APP_DIR/.env"
"$SUPERVISORCTL" -c "$CONF" start qwen

echo "[brain]   Waiting for vLLM to reload baseline..."
ELAPSED=0
while true; do
  CODE=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" "http://localhost:$VLLM_PORT/health" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    echo "[brain]   Baseline restored after ${ELAPSED}s"
    break
  fi
  ELAPSED=$((ELAPSED + 15))
  if [ "$ELAPSED" -ge 300 ]; then
    echo "[brain]   WARNING: Baseline not ready after 300s"
    break
  fi
  sleep 15
done

echo ""
echo "[brain] ============================================="
echo "[brain]   Brain upgrade benchmark complete"
echo "[brain]   Compare results in results/ directory"
echo "[brain] ============================================="
