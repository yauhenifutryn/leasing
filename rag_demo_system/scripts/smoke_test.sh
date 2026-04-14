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
info "  Stack: Whisper + Silero TTS + Qwen3.5-35B"
info "============================================="

info ""
info "--- Phase 1: Infrastructure ---"

# Qdrant
wait_for_url "Qdrant" "http://localhost:6333/healthz" 30 5

# vLLM
VLLM_BASE="${RAG_LLM_BASE_URL:-http://127.0.0.1:8787/v1}"
VLLM_HEALTH="${VLLM_BASE%/v1}/health"
VLLM_MODEL="${RAG_LLM_MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
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

wait_for_url "Backend health" "$BASE_URL/api/health" 30 3

check_url "Backends" "$BASE_URL/api/backends"
check_url "Voice status" "$BASE_URL/api/voice/status"

# ---------------------------------------------------------------------------
# PHASE 3: Knowledge Base (index only if needed)
# ---------------------------------------------------------------------------
info ""
info "--- Phase 3: Knowledge Base ---"

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
# PHASE 4: Chat (RAG query, consent is implicit via UI banner)
# ---------------------------------------------------------------------------
info ""
info "--- Phase 4: Chat ---"

info "Chat + RAG (retries: $MAX_RETRIES, timeout: ${LONG_TIMEOUT}s)..."
CHAT_RESP=$(retry_curl "Chat" "$MAX_RETRIES" \
  -s --max-time "$LONG_TIMEOUT" -X POST "$BASE_URL/api/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Какой минимальный аванс по лизингу?","backend":"our_rag"}')
if [ -z "$CHAT_RESP" ]; then
  fail "Chat -- no response after $MAX_RETRIES retries"
fi

echo "$CHAT_RESP" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
except:
    print('[smoke][FAIL] Chat -- invalid JSON response')
    sys.exit(1)
answer = data.get('answer', '')
used = data.get('used_knowledge') or []
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
# PHASE 5: Voice sidecars (Whisper + Silero TTS)
# ---------------------------------------------------------------------------
info ""
info "--- Phase 5: Voice sidecars ---"

check_url "Whisper STT" "http://localhost:50002/health" 10
check_url "Silero TTS"  "http://localhost:50006/health" 10
pass "All voice sidecars healthy"

# ---------------------------------------------------------------------------
# PHASE 6: Tool Use (Calculator API, SMS API, vLLM tool calling)
# ---------------------------------------------------------------------------
info ""
info "--- Phase 6: Tool Use ---"

# Source .env for tool credentials
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
  set +u
  . "$ENV_FILE" 2>/dev/null || true
  set -u
fi

TOOL_OK=true

# 6a. Calculator API connectivity
if [ -n "${CALCULATOR_API_TOKEN:-}" ] && [ -n "${CALCULATOR_API_BASE_URL:-}" ]; then
  info "Calculator API (timeout: 10s)..."
  CALC_RESP=$(curl -s --max-time 10 -H "Authorization: Bearer $CALCULATOR_API_TOKEN" "$CALCULATOR_API_BASE_URL/1.0/subjects/" 2>/dev/null || echo "")
  if echo "$CALC_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d)>0" 2>/dev/null; then
    pass "Calculator API (subjects endpoint)"
  else
    warn "Calculator API -- failed or empty response (IP whitelisted?)"
    TOOL_OK=false
  fi
else
  warn "Calculator API -- CALCULATOR_API_TOKEN or CALCULATOR_API_BASE_URL not set in .env"
  TOOL_OK=false
fi

# 6b. SMS API connectivity (check auth without sending)
if [ -n "${SMS_API_LOGIN:-}" ] && [ -n "${SMS_API_PASSWORD:-}" ]; then
  info "SMS API auth check (timeout: 10s)..."
  # Send to obviously invalid number; expect auth error (-2) = bad creds, positive or -1/-10/-13 = connected
  SMS_RESP=$(curl -s --max-time 10 "https://userarea.sms-assistent.by/api/v1/send_sms/plain?user=${SMS_API_LOGIN}&password=${SMS_API_PASSWORD}&recipient=000&message=smoke_test&sender=${SMS_SENDER_NAME:-MikroLizing}" 2>/dev/null || echo "")
  if [ -n "$SMS_RESP" ] && [ "$SMS_RESP" != "-2" ]; then
    pass "SMS API (auth OK, response: $SMS_RESP)"
  elif [ "$SMS_RESP" = "-2" ]; then
    warn "SMS API -- auth failed (code -2). Check SMS_API_LOGIN/SMS_API_PASSWORD."
    TOOL_OK=false
  else
    warn "SMS API -- no response (network issue?)"
    TOOL_OK=false
  fi
else
  warn "SMS API -- SMS_API_LOGIN or SMS_API_PASSWORD not set in .env"
  TOOL_OK=false
fi

# 6c. vLLM tool calling (structured tool_calls, not text)
info "vLLM tool calling (timeout: 30s)..."
TOOL_RESP=$(curl -s --max-time 30 -X POST "$VLLM_BASE/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$VLLM_MODEL\",\"messages\":[{\"role\":\"system\",\"content\":\"Используй инструменты.\"},{\"role\":\"user\",\"content\":\"Рассчитай лизинг на машину за 30000\"}],\"max_tokens\":200,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"calculator\",\"description\":\"Рассчитать платежи\",\"parameters\":{\"type\":\"object\",\"properties\":{\"subject\":{\"type\":\"string\"},\"cost\":{\"type\":\"number\"}},\"required\":[\"subject\",\"cost\"]}}}],\"chat_template_kwargs\":{\"enable_thinking\":false}}" 2>/dev/null || echo "")

if [ -n "$TOOL_RESP" ]; then
  TOOL_RESULT=$(echo "$TOOL_RESP" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message']
    tc = c.get('tool_calls') or []
    content = c.get('content') or ''
    if tc and len(tc) > 0 and tc[0].get('function',{}).get('name'):
        print(f'STRUCTURED:{tc[0][\"function\"][\"name\"]}')
    elif '<tool_call>' in content or '<function=' in content:
        print('TEXT_XML')
    else:
        print('NONE')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null || echo "ERROR")

  case "$TOOL_RESULT" in
    STRUCTURED:*)
      pass "vLLM tool calling (structured tool_calls: ${TOOL_RESULT#STRUCTURED:})"
      ;;
    TEXT_XML)
      warn "vLLM tool calling -- model outputs XML in content instead of structured tool_calls."
      warn "  Check --tool-call-parser flag. For Qwen3.5 use: --tool-call-parser qwen3_xml"
      warn "  Current .env STACK_QWEN_CMD:"
      grep 'tool-call-parser' "$ENV_FILE" 2>/dev/null || echo "    (not found)"
      TOOL_OK=false
      ;;
    NONE)
      warn "vLLM tool calling -- model did not call any tool. Tool schemas may not be reaching the model."
      TOOL_OK=false
      ;;
    *)
      warn "vLLM tool calling -- unexpected result: $TOOL_RESULT"
      TOOL_OK=false
      ;;
  esac
else
  warn "vLLM tool calling -- no response (timeout or crash)"
  TOOL_OK=false
fi

if [ "$TOOL_OK" = true ]; then
  pass "All tool use checks passed"
else
  warn "Some tool use checks failed. Tool features may not work in the UI."
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
info "  Voice: Whisper STT + Silero TTS"
if [ "$TOOL_OK" = true ]; then
  info "  Tools: Calculator API + SMS API + vLLM tool calling"
else
  info "  Tools: PARTIAL (see warnings above)"
fi
info "============================================="

# ---------------------------------------------------------------------------
# Suggest ngrok for public access
# ---------------------------------------------------------------------------
info ""
if command -v ngrok &>/dev/null; then
  if ngrok config check &>/dev/null 2>&1; then
    info "To expose the UI publicly:"
    info "  ngrok http 8000"
  else
    info "ngrok is installed but not authenticated."
    info "To expose the UI publicly:"
    info "  1. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken"
    info "  2. Run: ngrok config add-authtoken YOUR_TOKEN"
    info "  3. Run: ngrok http 8000"
  fi
else
  info "To expose the UI publicly, install ngrok:"
  info "  curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null"
  info "  echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list"
  info "  sudo apt-get update && sudo apt-get install -y ngrok"
  info "  ngrok config add-authtoken YOUR_TOKEN"
  info "  ngrok http 8000"
fi

info ""
info "To enable SIP telephony:"
info "  bash scripts/deploy_jambonz.sh"
