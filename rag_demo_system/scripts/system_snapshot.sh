#!/usr/bin/env bash
# system_snapshot.sh - Capture system state before and after provisioning.
# Usage:
#   bash system_snapshot.sh before   # Run BEFORE provision
#   bash system_snapshot.sh after    # Run AFTER provision + smoke test
set -euo pipefail

MODE="${1:-before}"
OUT="$HOME/system_${MODE}_install.txt"

if [ "$MODE" = "before" ]; then
  {
    echo "=== SYSTEM INFO (BEFORE INSTALL) ==="
    date
    echo "--- OS ---"
    cat /etc/os-release | head -5
    echo "--- GPU ---"
    nvidia-smi 2>/dev/null || echo "nvidia-smi not found"
    echo "--- CUDA ---"
    nvcc --version 2>/dev/null || echo "nvcc not found"
    echo "--- Docker ---"
    docker --version 2>/dev/null || echo "docker not found"
    echo "--- Python ---"
    python3 --version 2>/dev/null || echo "python3 not found"
    echo "--- Disk ---"
    df -h /
    echo "--- RAM ---"
    free -h
    echo "--- CPU ---"
    lscpu | grep "Model name" || echo "unknown"
    echo "--- Installed packages ---"
    echo "$(dpkg -l 2>/dev/null | wc -l) packages"
    echo "=== END ==="
  } | tee "$OUT"
  echo ""
  echo "Saved to: $OUT"

elif [ "$MODE" = "after" ]; then
  {
    echo "=== SYSTEM INFO (AFTER INSTALL) ==="
    date
    echo "--- GPU VRAM ---"
    nvidia-smi 2>/dev/null || echo "nvidia-smi not found"
    echo "--- Services ---"
    echo -n "Backend:    "; curl -s --max-time 3 http://localhost:8000/api/health || echo "DOWN"
    echo ""
    echo -n "Whisper:    "; curl -s --max-time 3 http://localhost:50002/health || echo "DOWN"
    echo ""
    echo -n "Silero TTS: "; curl -s --max-time 3 http://localhost:50006/health || echo "DOWN"
    echo ""
    echo -n "Qdrant:     "; curl -s --max-time 3 http://localhost:6333/healthz || echo "DOWN"
    echo ""
    echo -n "vLLM:       "; curl -s --max-time 3 http://localhost:8787/health || echo "DOWN"
    echo ""
    echo "--- KB indexed ---"
    curl -s http://localhost:6333/collections/micro_leasing_kb 2>/dev/null \
      | python3 -c "import json,sys; print(f'Chunks: {json.load(sys.stdin).get(\"result\",{}).get(\"points_count\",0)}')" 2>/dev/null \
      || echo "KB not indexed"
    echo "--- .env config ---"
    cat /workspace/leasing/rag_demo_system/.env 2>/dev/null || echo ".env not found"
    echo "--- Disk ---"
    df -h /
    echo "=== END ==="
  } | tee "$OUT"
  echo ""
  echo "Saved to: $OUT"

else
  echo "Usage: bash system_snapshot.sh [before|after]"
  exit 1
fi
