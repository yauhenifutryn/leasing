#!/usr/bin/env bash
# Transcribe the TTS reference audio file using the Whisper server.
# Usage: bash scripts/transcribe_ref.sh [path_to_wav]
set -euo pipefail

WAV="${1:-config/ref_voice_ru.wav}"
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$APP_DIR/$WAV" ]; then
  echo "File not found: $APP_DIR/$WAV"
  exit 1
fi

# Convert to base64
AUDIO_B64=$(base64 -w0 "$APP_DIR/$WAV" 2>/dev/null || base64 "$APP_DIR/$WAV")

# Send to Whisper
RESULT=$(curl -s -X POST http://localhost:50002/transcribe \
  -H 'Content-Type: application/json' \
  -d "{\"audio_b64\":\"$AUDIO_B64\",\"session_id\":\"ref\",\"language\":\"ru\",\"sample_rate_hz\":16000}")

TEXT=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('text',''))")
echo "Transcript: $TEXT"
echo ""
echo "Add to .env:"
echo "QWEN3_TTS_REF_TEXT=\"$TEXT\""
