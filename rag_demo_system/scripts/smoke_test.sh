#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${RAG_DEMO_BASE_URL:-http://127.0.0.1:8000}
TIMEOUT=${SMOKE_TIMEOUT:-15}
LONG_TIMEOUT=${SMOKE_LONG_TIMEOUT:-120}
MAX_RETRIES=${SMOKE_RETRIES:-3}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[smoke] $*"; }
warn()  { echo "[smoke][WARN] $*"; }
fail()  { echo "[smoke][FAIL] $*"; exit 1; }
pass()  { echo "[smoke][OK]   $1"; }

# check_url <label> <url> [timeout] [method] [body]
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
    local gpu_info=""
    if nvidia-smi &>/dev/null; then
      local used_mib total_mib
      used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
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

# retry_curl <label> <max_retries> <curl_args...>
# Retries a curl command up to N times with 3s sleep between attempts.
# Returns the response body on success, or fails on exhaustion.
retry_curl() {
  local label="$1" retries="$2"
  shift 2
  local attempt=1
  while [ "$attempt" -le "$retries" ]; do
    local resp
    resp=$(curl "$@" 2>/dev/null || true)
    if [ -n "$resp" ]; then
      echo "$resp"
      return 0
    fi
    if [ "$attempt" -lt "$retries" ]; then
      echo "[smoke]       ...retry ${attempt}/${retries} for $label (empty response, waiting 3s)" >&2
      sleep 3
    fi
    attempt=$((attempt + 1))
  done
  return 1
}

# ---------------------------------------------------------------------------
# PHASE 1: Infrastructure (Qdrant, vLLM, GPU)
# ---------------------------------------------------------------------------
info "============================================="
info "  Smoke Test"
info "  Backend: $BASE_URL"
info "============================================="

info ""
info "--- Phase 1: Infrastructure ---"

# Qdrant
wait_for_url "Qdrant" "http://localhost:6333/healthz" 30 5

# vLLM
BENCH_PROFILE="${BENCH_PROFILE:-baseline}"
if [ "$BENCH_PROFILE" != "omni_hybrid" ]; then
  VLLM_BASE="${RAG_LLM_BASE_URL:-http://127.0.0.1:8787/v1}"
  VLLM_HEALTH="${VLLM_BASE%/v1}/health"
  VLLM_MODEL="${RAG_LLM_MODEL:-Qwen/Qwen3-30B-A3B}"
  wait_for_url "vLLM model" "$VLLM_HEALTH" 600 15

  # Quick LLM test directly (bypasses backend)
  info "vLLM direct test (timeout: 30s)..."
  VLLM_RESP=$(retry_curl "vLLM direct" 2 \
    -s --max-time 30 -X POST "$VLLM_BASE/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$VLLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=\"}],\"max_tokens\":5,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
  if [ -z "$VLLM_RESP" ]; then
    fail "vLLM direct test -- no response after retries"
  fi
  pass "vLLM direct test"
fi

# VRAM
if nvidia-smi &>/dev/null; then
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  FREE_MIB=$((TOTAL_MIB - USED_MIB))
  info "VRAM: used=${USED_MIB}MiB total=${TOTAL_MIB}MiB free=${FREE_MIB}MiB"
  if [ "$USED_MIB" -lt 10240 ]; then
    warn "VRAM used < 10GB -- model may not be loaded"
  fi
else
  warn "nvidia-smi not available; skipping VRAM check"
fi

# ---------------------------------------------------------------------------
# PHASE 2: Backend health
# ---------------------------------------------------------------------------
info ""
info "--- Phase 2: Backend ---"

# Wait for backend to be responsive (may need a moment after restart)
wait_for_url "Backend health" "$BASE_URL/api/health" 30 3

check_url "Backends" "$BASE_URL/api/backends"
check_url "Voice status" "$BASE_URL/api/voice/status"

# ---------------------------------------------------------------------------
# PHASE 3: Knowledge Base (index only if needed)
# ---------------------------------------------------------------------------
info ""
info "--- Phase 3: Knowledge Base ---"

# Check if Qdrant collection already has documents
KB_COUNT=$(curl -s --max-time 5 "http://localhost:6333/collections/micro_leasing_kb" 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('result',{}).get('points_count', 0))
except:
    print(0)
" 2>/dev/null || echo "0")

if [ "$KB_COUNT" -gt 0 ]; then
  pass "KB already indexed ($KB_COUNT chunks in Qdrant -- skipping re-index)"
else
  info "KB not indexed yet -- indexing now (timeout: ${LONG_TIMEOUT}s)..."
  INDEX_RESP=$(retry_curl "Index KB" "$MAX_RETRIES" \
    -s --max-time "$LONG_TIMEOUT" -X POST "$BASE_URL/api/index" \
    -H 'Content-Type: application/json' -d '{"rebuild":false}')
  if [ -z "$INDEX_RESP" ]; then
    fail "Index KB -- no response after $MAX_RETRIES retries"
  fi
  # Re-check count
  sleep 2
  KB_COUNT=$(curl -s --max-time 5 "http://localhost:6333/collections/micro_leasing_kb" 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('result',{}).get('points_count', 0))
except:
    print(0)
" 2>/dev/null || echo "0")
  if [ "$KB_COUNT" -gt 0 ]; then
    pass "Index KB ($KB_COUNT chunks indexed)"
  else
    warn "Index returned but Qdrant shows 0 chunks -- KB may be empty"
  fi
fi

# ---------------------------------------------------------------------------
# PHASE 4: Chat (consent + RAG query)
# ---------------------------------------------------------------------------
info ""
info "--- Phase 4: Chat ---"

# Consent: retry up to MAX_RETRIES times (backend may be slow after index)
info "Consent (retries: $MAX_RETRIES, timeout: ${TIMEOUT}s each)..."
CONSENT_RESP=$(retry_curl "Consent" "$MAX_RETRIES" \
  -s --max-time "$TIMEOUT" -X POST "$BASE_URL/api/chat" \
  -H 'Content-Type: application/json' -d '{"message":"да, согласен"}')
if [ -z "$CONSENT_RESP" ]; then
  fail "Consent -- no response after $MAX_RETRIES retries (backend may be stuck; try: supervisorctl restart backend)"
fi
SESSION_ID=$(echo "$CONSENT_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null || true)
if [ -z "$SESSION_ID" ]; then
  fail "Consent -- response received but no session_id. Response: $(echo "$CONSENT_RESP" | head -c 200)"
fi
pass "Consent (session: ${SESSION_ID:0:8}...)"

# Chat with RAG: use the same session, retry
info "Chat + RAG (retries: $MAX_RETRIES, timeout: ${LONG_TIMEOUT}s)..."
CHAT_RESP=$(retry_curl "Chat" "$MAX_RETRIES" \
  -s --max-time "$LONG_TIMEOUT" -X POST "$BASE_URL/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"message\":\"Какой минимальный аванс по лизингу?\",\"backend\":\"our_rag\",\"session_id\":\"$SESSION_ID\"}")
if [ -z "$CHAT_RESP" ]; then
  fail "Chat -- no response after $MAX_RETRIES retries"
fi

# Validate response has knowledge
echo "$CHAT_RESP" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
except:
    print('[smoke][FAIL] Chat -- invalid JSON response')
    sys.exit(1)
answer = data.get('answer', '')
used = data.get('used_knowledge') or []
consent = data.get('consent', '')
if consent == 'needed':
    print('[smoke][FAIL] Chat -- consent not carried over (session_id mismatch)')
    sys.exit(1)
if not answer:
    print('[smoke][FAIL] Chat -- empty answer')
    sys.exit(1)
if not used:
    print('[smoke][WARN] Chat -- answer present but no used_knowledge (may be router bypass)')
else:
    print(f'[smoke][OK]   Chat -- answer received, {len(used)} chunks used')
print(f'[smoke]       Answer preview: {answer[:120]}...')
" || fail "Chat -- validation failed"

# ---------------------------------------------------------------------------
# PHASE 5: Sidecars (profile-aware)
# ---------------------------------------------------------------------------
info ""
info "--- Phase 5: Sidecars (profile: $BENCH_PROFILE) ---"

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

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
info ""
info "============================================="
info "  Smoke test PASSED"
info "  Infrastructure: Qdrant + vLLM + GPU"
info "  Backend: health + index + chat"
info "  KB: $KB_COUNT chunks"
info "  Profile: $BENCH_PROFILE"
info "============================================="
