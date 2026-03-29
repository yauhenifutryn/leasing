#!/usr/bin/env bash
# start_omni_benchmark.sh
# Stops vLLM, starts Qwen3-Omni + Whisper, runs voice benchmark.
# Omni handles LLM+TTS internally. Whisper needed for STT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"
CONF="$APP_DIR/scripts/supervisord.conf"
VLLM_PORT=8787
BENCHMARK_RUNNER="$APP_DIR/.venv/bin/python $APP_DIR/scripts/benchmark_runner.py"

echo "[omni] ============================================="
echo "[omni]   Qwen3-Omni Voice Benchmark"
echo "[omni] ============================================="

# --- Kill everything ---
echo "[omni] Stopping all processes..."
"$SUPERVISORCTL" -c "$CONF" shutdown 2>/dev/null || true
sleep 3
pkill -f supervisord 2>/dev/null || true
pkill -f vllm 2>/dev/null || true
pkill -f uvicorn 2>/dev/null || true
echo "[omni]   Waiting 20s for GPU cleanup..."
sleep 20

# GPU check
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "0")
echo "[omni]   GPU: ${USED_MIB}MiB, ${GPU_PROCS} procs"
if [ "$USED_MIB" -gt 5000 ] && [ "$GPU_PROCS" -eq 0 ]; then
  echo "[omni]   ERROR: Leaked GPU memory. Restart instance."
  exit 1
fi

# --- Qdrant ---
echo "[omni] Starting Qdrant..."
if ! curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
  QDRANT_DIR="/workspace/qdrant"
  mkdir -p /workspace/qdrant_storage
  QDRANT__STORAGE__STORAGE_PATH=/workspace/qdrant_storage nohup "$QDRANT_DIR/qdrant" > /workspace/qdrant.log 2>&1 &
  sleep 3
fi
echo "[omni]   Qdrant OK"

# --- Backup and patch .env ---
echo "[omni] Patching .env for Omni mode..."
cp "$APP_DIR/.env" "$APP_DIR/.env.omni_backup"

# Ensure embedding/reranker on GPU
sed -i 's/device: "cpu"/device: "cuda"/g' "$APP_DIR/config/app.yaml"

# --- Start supervisor ---
echo "[omni] Starting supervisor..."
rm -f "$APP_DIR/.state/supervisord.pid" "$APP_DIR/.state/supervisor.sock"
cd "$APP_DIR" && bash scripts/stack.sh up
sleep 3

# Stop vLLM and all voice services except Whisper
"$SUPERVISORCTL" -c "$CONF" stop qwen 2>/dev/null || true
"$SUPERVISORCTL" -c "$CONF" stop qwen3_tts sensevoice cosyvoice vosk_tts qwen3_asr voxtral 2>/dev/null || true

# Start Whisper (needed for STT in voice benchmark)
echo "[omni] Starting Whisper STT..."
"$SUPERVISORCTL" -c "$CONF" start whisper 2>/dev/null || true
sleep 5

# Start Omni sidecar
echo "[omni] Starting Qwen3-Omni sidecar..."
"$SUPERVISORCTL" -c "$CONF" start qwen3_omni

echo "[omni]   Waiting for Omni to load (3-5 min)..."
ELAPSED=0
MAX_WAIT=600
while true; do
  CODE=$(curl -s --max-time 10 -o /dev/null -w "%{http_code}" "http://localhost:8002/health" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    echo "[omni]   Omni ready after ${ELAPSED}s"
    break
  fi
  ELAPSED=$((ELAPSED + 15))
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "[omni]   ERROR: Omni not ready after ${MAX_WAIT}s"
    tail -20 "$APP_DIR/.state/qwen3_omni.err.log"
    mv "$APP_DIR/.env.omni_backup" "$APP_DIR/.env" 2>/dev/null || true
    exit 1
  fi
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "[omni]   ...loading (${ELAPSED}s | GPU: ${USED}MiB)"
  sleep 15
done

# Health checks
echo "[omni] Health checks..."
echo -n "[omni]   Backend: "; curl -s --max-time 5 http://localhost:8000/api/health >/dev/null && echo "OK" || echo "FAILED"
echo -n "[omni]   Omni:    "; curl -s --max-time 10 http://localhost:8002/health >/dev/null && echo "OK" || echo "FAILED"
echo -n "[omni]   Whisper: "; curl -s --max-time 5 http://localhost:50002/health >/dev/null && echo "OK" || echo "FAILED"
echo -n "[omni]   Qdrant:  "; curl -s --max-time 5 http://localhost:6333/healthz >/dev/null && echo "OK" || echo "FAILED"

USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "[omni]   GPU: ${USED_MIB}MiB used"

# --- Ensure KB indexed ---
echo ""
echo "[omni] Checking KB index..."
KB_COUNT=$(curl -s http://localhost:6333/collections/micro_leasing_kb 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('result',{}).get('points_count',0))" 2>/dev/null || echo "0")
if [ "$KB_COUNT" -lt 3000 ]; then
  echo "[omni]   KB has $KB_COUNT points, indexing..."
  curl -s --max-time 600 -X POST http://localhost:8000/api/index -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1
  KB_COUNT=$(curl -s http://localhost:6333/collections/micro_leasing_kb 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('result',{}).get('points_count',0))" 2>/dev/null || echo "0")
  echo "[omni]   KB indexed: $KB_COUNT points"
else
  echo "[omni]   KB already indexed: $KB_COUNT points"
fi

# --- Run voice benchmark with Omni ---
echo ""
echo "[omni] Running voice benchmark (Omni mode)..."
echo "[omni]   This uses the voice WebSocket: STT(Whisper) -> RAG -> Omni(LLM+TTS)"
echo "[omni]   Each question: ~10-30s. Total: ~20-40 min for 85 questions."

OUTPUT="$APP_DIR/results/voice_omni_$(date +%Y%m%d_%H%M%S).jsonl"

$BENCHMARK_RUNNER \
  --fixture "$APP_DIR/fixtures/bench_questions_ru.jsonl" \
  --profile "omni_hybrid" \
  --output "$OUTPUT" \
  --ws-url "ws://localhost:8000/ws/voice" \
  --backend-url "http://localhost:8000" \
  --timeout 60 \
  --warmup 3

echo "[omni]   Results: $OUTPUT"

# --- Restore ---
echo ""
echo "[omni] Restoring..."
"$SUPERVISORCTL" -c "$CONF" stop qwen3_omni 2>/dev/null || true
"$SUPERVISORCTL" -c "$CONF" stop whisper 2>/dev/null || true
sleep 10
mv "$APP_DIR/.env.omni_backup" "$APP_DIR/.env" 2>/dev/null || true

echo ""
echo "[omni] ============================================="
echo "[omni]   Omni benchmark complete."
echo "[omni]   Results: $OUTPUT"
echo "[omni]   Run restart_all.sh to reload split pipeline."
echo "[omni] ============================================="
