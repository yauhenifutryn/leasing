#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${RAG_DEMO_BASE_URL:-http://127.0.0.1:8000}

info() { echo "[smoke] $*"; }
warn() { echo "[smoke][warn] $*"; }

info "UI root check..."
curl -fsS "$BASE_URL/" >/dev/null

info "Health check..."
curl -fsS "$BASE_URL/api/health" >/dev/null

info "Backends check..."
curl -fsS "$BASE_URL/api/backends" >/dev/null

info "Voice status check..."
curl -fsS "$BASE_URL/api/voice/status" >/dev/null

info "Index KB (if Qdrant running)..."
if ! curl -fsS -X POST "$BASE_URL/api/index" -H 'Content-Type: application/json' -d '{"rebuild":false}' >/dev/null; then
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
  -d '{"message":"Какие требования к лизингу грузового транспорта?","backend":"our_rag"}')

JSON_LINE=$(echo "$RESP" | sed -n 's/^data: //p' | tail -n 1)
echo "$JSON_LINE" | python3 -c "
import json, sys
raw = sys.stdin.read().strip()
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
"

if curl -fsS "$BASE_URL/api/backends" | grep -q '"dify_rag".*"available":true'; then
  info "Dify backend check..."
  curl -fsS -X POST "$BASE_URL/api/chat" \
    -H 'Content-Type: application/json' \
    -d '{"message":"Какие требования к лизингу грузового транспорта?","backend":"dify_rag"}' >/dev/null || warn "Dify chat failed"
fi

# --- Extended checks (Phase 5: DEPLOY-03) ---
# Profile-aware sidecar health, vLLM readiness, VRAM headroom

BENCH_PROFILE="${BENCH_PROFILE:-baseline}"
info "Active profile: $BENCH_PROFILE"

# Load profile env vars so BASE_URL values are available
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE_FILE="$APP_DIR/.env.bench.$BENCH_PROFILE"
if [ -f "$PROFILE_FILE" ]; then
  set -a
  source "$PROFILE_FILE"
  set +a
  info "Loaded profile: $PROFILE_FILE"
else
  warn "Profile file not found: $PROFILE_FILE (using defaults)"
fi

# Determine required sidecar URLs based on active profile
REQUIRED_SIDECARS=()
case "$BENCH_PROFILE" in
  baseline|dify_rag|brain_upgrade)
    REQUIRED_SIDECARS=("${SENSEVOICE_BASE_URL:-http://127.0.0.1:50000}" "${COSYVOICE_BASE_URL:-http://127.0.0.1:50001}")
    ;;
  qwen3_tts)
    REQUIRED_SIDECARS=("${SENSEVOICE_BASE_URL:-http://127.0.0.1:50000}" "${QWEN3_TTS_BASE_URL:-http://127.0.0.1:50003}")
    ;;
  qwen3_asr)
    REQUIRED_SIDECARS=("${QWEN3_ASR_BASE_URL:-http://127.0.0.1:50004}" "${COSYVOICE_BASE_URL:-http://127.0.0.1:50001}")
    ;;
  voxtral)
    REQUIRED_SIDECARS=("${VOXTRAL_BASE_URL:-http://127.0.0.1:50005}" "${COSYVOICE_BASE_URL:-http://127.0.0.1:50001}")
    ;;
  omni_hybrid)
    REQUIRED_SIDECARS=("${SENSEVOICE_BASE_URL:-http://127.0.0.1:50000}" "${QWEN3_OMNI_BASE_URL:-http://127.0.0.1:8002}")
    ;;
  *)
    warn "Unknown profile '$BENCH_PROFILE'; skipping sidecar checks"
    REQUIRED_SIDECARS=()
    ;;
esac

for sidecar_url in "${REQUIRED_SIDECARS[@]}"; do
  info "Sidecar health: $sidecar_url/health"
  if ! curl -fsS "$sidecar_url/health" >/dev/null 2>&1; then
    echo "[smoke][error] Sidecar $sidecar_url/health FAILED"
    exit 1
  fi
done
info "All required sidecars healthy"

# vLLM readiness (skip for omni_hybrid which uses its own model server)
if [ "$BENCH_PROFILE" != "omni_hybrid" ]; then
  VLLM_BASE="${RAG_LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
  VLLM_HEALTH="${VLLM_BASE%/v1}/health"
  VLLM_MODEL="${RAG_LLM_MODEL:-Qwen/Qwen3-30B-A3B}"

  info "vLLM health: $VLLM_HEALTH"
  curl -fsS "$VLLM_HEALTH" >/dev/null

  info "vLLM trivial completion check"
  VLLM_RESP=$(curl -fsS -X POST "$VLLM_BASE/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$VLLM_MODEL\",\"prompt\":\"1+1=\",\"max_tokens\":3}")
  echo "$VLLM_RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d.get('choices'), 'vLLM returned no choices'
print('[smoke] vLLM OK: model responded')
" || { echo "[smoke][error] vLLM trivial completion failed"; exit 1; }
fi

# VRAM headroom check
if nvidia-smi &>/dev/null; then
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  FREE_MIB=$((TOTAL_MIB - USED_MIB))
  info "VRAM: used=${USED_MIB}MiB total=${TOTAL_MIB}MiB free=${FREE_MIB}MiB"

  # Verify model is loaded (used > 10GB = 10240 MiB)
  if [ "$USED_MIB" -lt 10240 ]; then
    warn "VRAM used (${USED_MIB}MiB) < 10240MiB -- model may not be loaded"
  fi
else
  warn "nvidia-smi not available; skipping VRAM check"
fi

info "=== Smoke test PASSED ==="
