#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"
CONF="$APP_DIR/scripts/supervisord.conf"

echo "[restart] Shutting down supervisor..."
"$SUPERVISORCTL" -c "$CONF" shutdown 2>/dev/null || true
sleep 3

echo "[restart] Killing stale processes..."
pkill -9 -f supervisord 2>/dev/null || true
pkill -9 -f uvicorn 2>/dev/null || true
pkill -9 -f vllm 2>/dev/null || true
sleep 2

echo "[restart] Clearing held ports..."
for port in 8000 8787 8002 50000 50001 50002 50003 50004 50005; do
  pid=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "  Killing PID $pid on port $port"
    kill -9 $pid 2>/dev/null || true
  fi
done
sleep 2

echo "[restart] Removing stale pidfile and socket..."
rm -f "$APP_DIR/.state/supervisord.pid" "$APP_DIR/.state/supervisor.sock"

echo "[restart] Starting stack..."
cd "$APP_DIR"
bash scripts/stack.sh up
sleep 5

echo "[restart] Starting voice sidecars..."
"$SUPERVISORCTL" -c "$CONF" start sensevoice whisper cosyvoice
sleep 15

echo "[restart] Status:"
"$SUPERVISORCTL" -c "$CONF" status

echo "[restart] Health checks:"
curl -s http://localhost:8000/api/health && echo " backend OK" || echo " backend FAILED"
curl -s http://localhost:8787/health && echo " vLLM OK" || echo " vLLM NOT READY"
curl -s http://localhost:6333/healthz && echo " Qdrant OK" || echo " Qdrant FAILED"
curl -s http://localhost:50000/health && echo " sensevoice" || echo " sensevoice FAILED"
curl -s http://localhost:50001/health && echo " cosyvoice" || echo " cosyvoice FAILED"
curl -s http://localhost:50002/health && echo " whisper" || echo " whisper FAILED"

echo ""
echo "[restart] Done. Run smoke_test.sh to verify."
