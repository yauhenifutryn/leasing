#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${VOSK_PYTHON_BIN:-python3}"
HOST="${VOSK_HOST:-0.0.0.0}"
PORT="${VOSK_PORT:-50010}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m uvicorn services.vosk_server:app --host "$HOST" --port "$PORT"
