#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-${STACK_VOICE_PROFILE:-oss_russian}}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
APP_DIR="$ROOT_DIR/rag_demo_system"
BACKEND_VENV="$APP_DIR/.venv"
OSS_VENV="$APP_DIR/.venv-voice-oss"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS_DIR="${STACK_MODELS_DIR:-$ROOT_DIR/models}"
DEFAULT_VOSK_MODEL_PATH="$MODELS_DIR/vosk-model-small-ru-0.22"
VOSK_MODEL_PATH_VALUE="${VOSK_MODEL_PATH:-$DEFAULT_VOSK_MODEL_PATH}"

case "$PROFILE" in
  local|yandex_speechkit|oss_russian|yandex_realtime) ;;
  *)
    echo "Usage: $0 [local|yandex_speechkit|oss_russian|yandex_realtime]" >&2
    exit 1
    ;;
esac

log() {
  echo "[setup] $*"
}

ensure_venv() {
  local target="$1"
  if [ ! -d "$target" ]; then
    log "Creating virtualenv: $target"
    "$PYTHON_BIN" -m venv "$target"
  fi
}

install_backend_env() {
  ensure_venv "$BACKEND_VENV"
  "$BACKEND_VENV/bin/pip" install --upgrade pip wheel
  "$BACKEND_VENV/bin/pip" install -r "$APP_DIR/requirements.txt" supervisor
}

install_oss_voice_env() {
  ensure_venv "$OSS_VENV"
  "$OSS_VENV/bin/pip" install --upgrade pip wheel
  "$OSS_VENV/bin/pip" install -r "$APP_DIR/requirements-voice-oss.txt"
}

ensure_vosk_model() {
  if [ -d "$VOSK_MODEL_PATH_VALUE" ]; then
    log "Vosk model already present: $VOSK_MODEL_PATH_VALUE"
    return
  fi
  mkdir -p "$MODELS_DIR"
  local archive="$MODELS_DIR/vosk-model-small-ru-0.22.zip"
  log "Downloading Vosk model to $archive"
  curl -L "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip" -o "$archive"
  unzip -o "$archive" -d "$MODELS_DIR" >/dev/null
}

mkdir -p "$APP_DIR/.state"

log "Installing backend environment"
install_backend_env

if [ "$PROFILE" = "oss_russian" ]; then
  log "Installing OSS voice environment"
  install_oss_voice_env
  ensure_vosk_model
fi

cat <<EOF

[setup] Done.
[setup] Next steps:
  1. Copy the right env template into rag_demo_system/.env.
  2. Set STACK_VOICE_PROFILE=$PROFILE in that env file.
  3. If you want this repo to start vLLM itself, set STACK_QWEN_CMD in rag_demo_system/.env.
  4. Start the stack with: ./rag_demo_system/scripts/stack.sh up
  5. Stop the stack with:  ./rag_demo_system/scripts/stack.sh down

EOF
