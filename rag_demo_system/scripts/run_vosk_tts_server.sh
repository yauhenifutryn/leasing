#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${VOSK_TTS_PYTHON_BIN:-python3}"
HOST="${VOSK_TTS_HOST:-0.0.0.0}"
PORT="${VOSK_TTS_PORT:-50011}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m uvicorn services.vosk_tts_server:app --host "$HOST" --port "$PORT"
