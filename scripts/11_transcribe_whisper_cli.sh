#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${1:-audio}"
OUT_DIR="${2:-transcripts_raw}"
mkdir -p "$OUT_DIR"

# Requires: pip install openai-whisper  (or brew install whisper via faster-whisper taps)
# Model choices: small, medium, large
for f in "$IN_DIR"/*; do
  ext="${f##*.}"
  name="$(basename "$f" .${ext})"
  whisper "$f" --language ru --model large --task transcribe --temperature 0 \
    --output_dir "$OUT_DIR" --output_format json
  mv "$OUT_DIR/$name.json" "$OUT_DIR/$name.whisper.json" || true
done
