#!/bin/bash
set -e

# Default arguments
IN_DIR="audio"
OUT_DIR="transcripts_clean"
MODEL="large-v2"
LANGUAGE="ru"
BATCH_SIZE=16

# Run the transcription script
python3 scripts/10_transcribe_whisperx.py \
    --in "$IN_DIR" \
    --out "$OUT_DIR" \
    --model "$MODEL" \
    --language "$LANGUAGE" \
    --batch-size "$BATCH_SIZE" \
    --device cuda \
    --compute-type float16 \
    --disable-diarization  # Remove this if you have a valid HF token and want diarization

echo "Transcription complete. Results in $OUT_DIR"
