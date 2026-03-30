#!/usr/bin/env bash
set -euo pipefail

# Simple stack launcher: starts supervisord, then starts vLLM + voice services.
# Replaces the old stack_cli.py which was removed in the clean branch.

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONF="$APP_DIR/scripts/supervisord.conf"
SUPERVISORD="$APP_DIR/.venv/bin/supervisord"
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"

# Load .env
if [ -f "$APP_DIR/.env" ]; then
  set -a
  . "$APP_DIR/.env"
  set +a
fi

mkdir -p "$APP_DIR/.state"

case "${1:-up}" in
  up)
    # Start supervisord (manages backend, vLLM, whisper, silero_tts)
    rm -f "$APP_DIR/.state/supervisord.pid" "$APP_DIR/.state/supervisor.sock"
    "$SUPERVISORD" -c "$CONF"
    sleep 2

    # Start vLLM (autostart=false, needs explicit start)
    "$SUPERVISORCTL" -c "$CONF" start qwen
    ;;

  down)
    "$SUPERVISORCTL" -c "$CONF" shutdown 2>/dev/null || true
    ;;

  status)
    "$SUPERVISORCTL" -c "$CONF" status
    ;;

  *)
    echo "Usage: stack.sh [up|down|status]"
    exit 1
    ;;
esac
