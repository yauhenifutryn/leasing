#!/usr/bin/env bash
# start_omni_live.sh
# Kill everything, start Qwen3-Omni for live voice testing.
# No ngrok. Access via SSH tunnel: ssh -p 50576 -L 8000:localhost:8000 root@<IP>
# Then open http://localhost:8000/dev and set voice_provider to qwen3_omni.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================="
echo "  Omni Live Voice Test"
echo "============================================="

# --- Kill everything ---
echo ""
echo "[1/7] Killing all processes..."
pkill -f ngrok 2>/dev/null || true
pkill -f supervisord 2>/dev/null || true
pkill -f vllm 2>/dev/null || true
pkill -f uvicorn 2>/dev/null || true
pkill -f "python.*services" 2>/dev/null || true
echo "  Waiting 20s for GPU cleanup..."
sleep 20

# --- GPU check ---
echo ""
echo "[2/7] Checking GPU..."
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "0")
echo "  GPU: ${USED}MiB used, ${PROCS} processes"
if [ "${USED:-0}" -gt 5000 ] && [ "${PROCS:-0}" -eq 0 ]; then
  echo "  ERROR: Leaked GPU memory. Restart the instance from provider dashboard."
  exit 1
fi

# --- Qdrant ---
echo ""
echo "[3/7] Starting Qdrant..."
if curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
  echo "  Qdrant already running"
else
  QDRANT__STORAGE__STORAGE_PATH=/workspace/qdrant_storage \
    nohup /workspace/qdrant/qdrant > /workspace/qdrant.log 2>&1 &
  sleep 3
  if curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
    echo "  Qdrant started"
  else
    echo "  WARNING: Qdrant may not have started"
  fi
fi

# --- Whisper (needed for STT -> RAG search query) ---
echo ""
echo "[4/7] Starting Whisper STT..."
cd "$APP_DIR"
LD_LIBRARY_PATH=./.venv-voice-oss/lib/python3.12/site-packages/nvidia/cublas/lib:./.venv-voice-oss/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-} \
  nohup ./.venv-voice-oss/bin/python -m uvicorn services.whisper_server:app --host 0.0.0.0 --port 50002 > .state/whisper.log 2>&1 &
sleep 5
if curl -fsS http://localhost:50002/health >/dev/null 2>&1; then
  echo "  Whisper ready"
else
  echo "  Whisper loading (check .state/whisper.log)..."
  sleep 10
fi

# --- Qwen3-Omni (the big one, 3-5 min to load) ---
echo ""
echo "[5/7] Starting Qwen3-Omni (3-5 min to load, ~60GB VRAM)..."
nohup ./.venv-qwen3-omni/bin/python -m uvicorn services.qwen3_omni_server:app --host 0.0.0.0 --port 8002 > .state/qwen3_omni.log 2>&1 &

ELAPSED=0
MAX_WAIT=600
while true; do
  CODE=$(curl -s --max-time 10 -o /dev/null -w "%{http_code}" "http://localhost:8002/health" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    echo "  Omni ready after ${ELAPSED}s"
    break
  fi
  ELAPSED=$((ELAPSED + 15))
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "  ERROR: Omni not ready after ${MAX_WAIT}s"
    echo "  Last log lines:"
    tail -20 "$APP_DIR/.state/qwen3_omni.err.log" 2>/dev/null || true
    exit 1
  fi
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "  ...loading (${ELAPSED}s | GPU: ${USED}MiB)"
  sleep 15
done

# --- Backend ---
echo ""
echo "[6/7] Starting backend..."
nohup ./.venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 > .state/backend.log 2>&1 &
sleep 3
if curl -fsS http://localhost:8000/api/health >/dev/null 2>&1; then
  echo "  Backend ready"
else
  echo "  Backend loading..."
  sleep 5
fi

# --- Health summary ---
echo ""
echo "[7/7] Health checks..."
echo -n "  Qdrant:  "; curl -s --max-time 3 http://localhost:6333/healthz >/dev/null && echo "OK" || echo "FAILED"
echo -n "  Whisper: "; curl -s --max-time 3 http://localhost:50002/health >/dev/null && echo "OK" || echo "FAILED"
echo -n "  Omni:    "; curl -s --max-time 10 http://localhost:8002/health >/dev/null && echo "OK" || echo "FAILED"
echo -n "  Backend: "; curl -s --max-time 3 http://localhost:8000/api/health >/dev/null && echo "OK" || echo "FAILED"

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo ""
echo "  GPU: ${USED}MiB used"

echo ""
echo "============================================="
echo "  Ready for live Omni voice test!"
echo ""
echo "  From your local machine, run:"
echo "    ssh -p 50576 -L 8000:localhost:8000 root@185.151.171.35"
echo ""
echo "  Then open: http://localhost:8000/dev"
echo "  Set voice_provider to: qwen3_omni"
echo "  Press Talk and speak Russian."
echo ""
echo "  No vLLM, no Silero. Omni handles LLM + TTS."
echo "  Whisper handles STT for RAG search only."
echo "============================================="
