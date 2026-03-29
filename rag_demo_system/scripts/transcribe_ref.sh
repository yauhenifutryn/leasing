#!/usr/bin/env bash
# Transcribe the TTS reference audio file using the Whisper server.
# Usage: bash scripts/transcribe_ref.sh [path_to_wav]
set -euo pipefail

WAV="${1:-config/ref_voice_ru.wav}"
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FULL_PATH="$APP_DIR/$WAV"

if [ ! -f "$FULL_PATH" ]; then
  echo "File not found: $FULL_PATH"
  exit 1
fi

# Build JSON payload via python to avoid shell argument limits
python3 -c "
import base64, json, requests, sys

with open('$FULL_PATH', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode()

resp = requests.post('http://localhost:50002/transcribe', json={
    'audio_b64': audio_b64,
    'session_id': 'ref',
    'language': 'ru',
    'sample_rate_hz': 16000,
}, timeout=60)
resp.raise_for_status()
text = resp.json().get('text', '')
print(f'Transcript: {text}')
print()
print(f'Add to .env:')
print(f'QWEN3_TTS_REF_TEXT=\"{text}\"')
"
