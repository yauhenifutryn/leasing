#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${RAG_DEMO_BASE_URL:-http://127.0.0.1:8000}

info() { echo "[smoke] $*"; }
warn() { echo "[smoke][warn] $*"; }

info "Health check..."
curl -fsS "$BASE_URL/api/health" >/dev/null

info "Index KB (if Qdrant running)..."
if ! curl -fsS -X POST "$BASE_URL/api/index" >/dev/null; then
  warn "Index failed (Qdrant may be down). Skipping chat checks."
  exit 0
fi

info "Consent step..."
curl -fsS -X POST "$BASE_URL/api/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"да, согласен"}' >/dev/null

info "Chat stream check..."
RESP=$(curl -fsS -N -X POST "$BASE_URL/api/chat?stream=1" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Какие требования к лизингу грузового транспорта?"}')

JSON_LINE=$(echo "$RESP" | sed -n 's/^data: //p' | tail -n 1)
python - <<PY
import json, sys
raw = sys.argv[1]
try:
    data = json.loads(raw)
except Exception as e:
    print('[smoke][error] invalid JSON in stream')
    sys.exit(1)
used = data.get('used_knowledge') or []
if not used:
    print('[smoke][error] used_knowledge empty')
    sys.exit(1)
if not used[0].get('chunk_id'):
    print('[smoke][error] missing chunk_id')
    sys.exit(1)
print('[smoke] OK')
PY "$JSON_LINE"
