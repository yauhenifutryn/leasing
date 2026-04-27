#!/usr/bin/env bash
# Convenience wrapper for simulate_dialogue.py — uses the project venv so
# you don't need to remember the .venv/bin/python path.
#
# Usage:
#   bash scripts/simulate.sh CONVERSATIONS_FILE [--base-url URL] [--model MODEL]
#
# Run from repo root or rag_demo_system/. The script auto-detects the venv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$APP_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "venv python not found at $VENV_PY" >&2
  echo "Run scripts/provision_server.sh first to create the venv." >&2
  exit 1
fi

cd "$APP_DIR"
exec "$VENV_PY" scripts/simulate_dialogue.py "$@"
