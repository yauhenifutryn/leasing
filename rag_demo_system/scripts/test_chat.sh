#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${RAG_DEMO_BASE_URL:-http://127.0.0.1:8000}

echo "[test] Consent..."
RESP=$(curl -s --max-time 15 -X POST "$BASE_URL/api/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"да, согласен"}')
SID=$(echo "$RESP" | python3 -c "import json,sys;print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)
echo "[test] Session: $SID"

echo ""
echo "[test] Asking about truck leasing..."
RESP=$(curl -s --max-time 60 -X POST "$BASE_URL/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"message\":\"Расскажи о лизинге грузовых автомобилей\",\"backend\":\"our_rag\",\"session_id\":\"$SID\"}")

echo "$RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Answer:', d.get('answer','')[:300])
print('Chunks:', len(d.get('used_knowledge',[])))
print('Consent:', d.get('consent',''))
timings = d.get('timings', {})
if timings:
    print('Timings:')
    for k,v in timings.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.0f}ms')
"

echo ""
echo "[test] Checking TTS..."
TTS_RESP=$(curl -s --max-time 30 -X POST "$BASE_URL/api/tts" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Привет, это тестовое сообщение","tts_provider":"cosyvoice"}' 2>/dev/null || echo "FAILED")
if [ "$TTS_RESP" = "FAILED" ]; then
  echo "[test] TTS endpoint failed or not found"
else
  echo "$TTS_RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
audio = d.get('audio_b64','')
sr = d.get('sample_rate_hz','')
print(f'TTS: {len(audio)} chars base64, sample_rate={sr}')
" 2>/dev/null || echo "[test] TTS response not JSON: $(echo "$TTS_RESP" | head -c 100)"
fi
