#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${RAG_DEMO_BASE_URL:-http://127.0.0.1:8000}
TIMEOUT=${SMOKE_TIMEOUT:-15}
LONG_TIMEOUT=${SMOKE_LONG_TIMEOUT:-120}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[smoke] $*"; }
warn()  { echo "[smoke][WARN] $*"; }
fail()  { echo "[smoke][FAIL] $*"; exit 1; }
pass()  { echo "[smoke][OK]   $1"; }

# check_url <label> <url> [timeout] [method] [body]
# Curls a URL with a timeout, prints status. Fails the script on error.
check_url() {
  local label="$1" url="$2" max="${3:-$TIMEOUT}" method="${4:-GET}" body="${5:-}"
  local args=(-s --max-time "$max" --connect-timeout 5 -X "$method")
  if [ -n "$body" ]; then
    args+=(-H 'Content-Type: application/json' -d "$body")
  fi
  info "$label (timeout: ${max}s)..."
  local http_code
  http_code=$(curl "${args[@]}" -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
  if [ "$http_code" = "000" ]; then
    fail "$label -- connection refused or timed out after ${max}s"
  elif [ "$http_code" -ge 400 ]; then
    fail "$label -- HTTP $http_code"
  fi
  pass "$label"
}

# wait_for_url <label> <url> [max_wait] [interval]
# Polls until URL returns 200 or times out. Shows progress every interval.
wait_for_url() {
  local label="$1" url="$2" max_wait="${3:-300}" interval="${4:-10}"
  local elapsed=0
  info "$label -- waiting up to ${max_wait}s..."
  while true; do
    local code
    code=$(curl -s --max-time 5 --connect-timeout 3 -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then
      pass "$label (ready after ${elapsed}s)"
      return 0
    fi
    elapsed=$((elapsed + interval))
    if [ "$elapsed" -ge "$max_wait" ]; then
      fail "$label -- not ready after ${max_wait}s (last HTTP code: $code)"
    fi
    # Show GPU memory progress if available (useful for vLLM model loading)
    local gpu_info=""
    if nvidia-smi &>/dev/null; then
      local used_mib
      used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      local total_mib
      total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      if [ -n "$used_mib" ] && [ -n "$total_mib" ]; then
        local pct=$((used_mib * 100 / total_mib))
        gpu_info=" | GPU: ${used_mib}/${total_mib}MiB (${pct}%)"
      fi
    fi
    echo "[smoke]       ...waiting (${elapsed}s/${max_wait}s, HTTP $code${gpu_info})"
    sleep "$interval"
  done
}

# ---------------------------------------------------------------------------
# Pre-flight: check dependencies before anything else
# ---------------------------------------------------------------------------
info "============================================="
info "  Smoke Test"
info "  Backend: $BASE_URL"
info "============================================="

# Step 0: Check Qdrant is running (required for index)
info ""
info "--- Infrastructure ---"
wait_for_url "Qdrant" "http://localhost:6333/healthz" 30 5

# Step 1: Check vLLM is loaded (can take 3-5 min on first start)
BENCH_PROFILE="${BENCH_PROFILE:-baseline}"
if [ "$BENCH_PROFILE" != "omni_hybrid" ]; then
  VLLM_BASE="${RAG_LLM_BASE_URL:-http://127.0.0.1:8787/v1}"
  VLLM_HEALTH="${VLLM_BASE%/v1}/health"
  wait_for_url "vLLM model loading" "$VLLM_HEALTH" 600 15
fi

# VRAM check
if nvidia-smi &>/dev/null; then
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  FREE_MIB=$((TOTAL_MIB - USED_MIB))
  info "VRAM: used=${USED_MIB}MiB total=${TOTAL_MIB}MiB free=${FREE_MIB}MiB"
  if [ "$USED_MIB" -lt 10240 ]; then
    warn "VRAM used (${USED_MIB}MiB) < 10240MiB -- model may not be loaded"
  fi
else
  warn "nvidia-smi not available; skipping VRAM check"
fi

# ---------------------------------------------------------------------------
# Core backend checks
# ---------------------------------------------------------------------------
info ""
info "--- Backend ---"
check_url "UI root" "$BASE_URL/"
check_url "Health" "$BASE_URL/api/health"
check_url "Backends" "$BASE_URL/api/backends"
check_url "Voice status" "$BASE_URL/api/voice/status"

# Index KB (may take 10-30s on first run depending on KB size)
check_url "Index KB" "$BASE_URL/api/index" "$LONG_TIMEOUT" "POST" '{"rebuild":false}'

# Chat checks: consent + stream must share the same session_id
info "Consent (timeout: ${TIMEOUT}s)..."
CONSENT_RESP=$(curl -s --max-time "$TIMEOUT" -X POST "$BASE_URL/api/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"да, согласен"}' 2>/dev/null || true)
SESSION_ID=$(echo "$CONSENT_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null || true)
if [ -z "$SESSION_ID" ]; then
  fail "Consent -- no session_id returned"
fi
pass "Consent (session: ${SESSION_ID:0:8}...)"

info "Chat stream check (timeout: ${LONG_TIMEOUT}s)..."
RESP=$(curl -s --max-time "$LONG_TIMEOUT" -N -X POST "$BASE_URL/api/chat?stream=1" \
  -H 'Content-Type: application/json' \
  -d "{\"message\":\"Какие требования к лизингу грузового транспорта?\",\"backend\":\"our_rag\",\"session_id\":\"$SESSION_ID\"}" 2>/dev/null || true)
if [ -z "$RESP" ]; then
  fail "Chat stream -- empty response or timed out"
fi
JSON_LINE=$(echo "$RESP" | sed -n 's/^data: //p' | tail -n 1)
echo "$JSON_LINE" | python3 -c "
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    print('[smoke][FAIL] Chat stream -- no JSON data in response')
    sys.exit(1)
try:
    data = json.loads(raw)
except Exception:
    print('[smoke][FAIL] Chat stream -- invalid JSON')
    sys.exit(1)
used = data.get('used_knowledge') or []
if not used:
    print('[smoke][FAIL] Chat stream -- used_knowledge empty')
    sys.exit(1)
if not used[0].get('chunk_id'):
    print('[smoke][FAIL] Chat stream -- missing chunk_id')
    sys.exit(1)
print('[smoke][OK]   Chat stream -- knowledge retrieved')
"

# Optional: Dify backend
if curl -s --max-time 5 "$BASE_URL/api/backends" | grep -q '"dify_rag".*"available":true'; then
  check_url "Dify chat" "$BASE_URL/api/chat" "$TIMEOUT" "POST" '{"message":"Какие требования к лизингу грузового транспорта?","backend":"dify_rag"}'
fi

# ---------------------------------------------------------------------------
# Profile-aware sidecar checks
# ---------------------------------------------------------------------------
info ""
info "--- Sidecars (profile: $BENCH_PROFILE) ---"

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

if [ ${#REQUIRED_SIDECARS[@]} -gt 0 ]; then
  for sidecar_url in "${REQUIRED_SIDECARS[@]}"; do
    check_url "Sidecar $sidecar_url" "$sidecar_url/health" 10
  done
  pass "All required sidecars healthy"
else
  info "No sidecars required for profile: $BENCH_PROFILE"
fi

# vLLM trivial completion (skip for omni_hybrid)
if [ "$BENCH_PROFILE" != "omni_hybrid" ]; then
  VLLM_BASE="${RAG_LLM_BASE_URL:-http://127.0.0.1:8787/v1}"
  VLLM_MODEL="${RAG_LLM_MODEL:-Qwen/Qwen3-30B-A3B}"
  info "vLLM trivial completion (timeout: 30s)..."
  VLLM_RESP=$(curl -s --max-time 30 -X POST "$VLLM_BASE/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$VLLM_MODEL\",\"prompt\":\"1+1=\",\"max_tokens\":3}" 2>/dev/null || true)
  if [ -z "$VLLM_RESP" ]; then
    fail "vLLM completion -- no response"
  fi
  echo "$VLLM_RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d.get('choices'), 'vLLM returned no choices'
print('[smoke][OK]   vLLM completion -- model responded')
" || fail "vLLM completion -- invalid response"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
info ""
info "============================================="
info "  Smoke test PASSED"
info "============================================="
