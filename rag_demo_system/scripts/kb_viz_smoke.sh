#!/usr/bin/env bash
# One-shot smoke test for the KB-viz overlay service.
#
# Does not stream anything. Hits each endpoint in sequence, reports a single
# PASS/FAIL line per check, and exits non-zero if anything is broken. Safe
# to run before emailing/sending a link to a client.
#
# Env vars (optional):
#   KB_VIZ_BASE_URL       default: http://localhost:8500
#   KB_VIZ_OVERLAY_TOKEN  if set, included as Bearer on auth-gated endpoints
#
# Usage:
#   bash rag_demo_system/scripts/kb_viz_smoke.sh
#   KB_VIZ_BASE_URL=https://my-host:8500 KB_VIZ_OVERLAY_TOKEN=abc \
#       bash rag_demo_system/scripts/kb_viz_smoke.sh

set -u
set -o pipefail

BASE="${KB_VIZ_BASE_URL:-http://localhost:8500}"
TOKEN="${KB_VIZ_OVERLAY_TOKEN:-}"

AUTH_HEADER=()
if [ -n "$TOKEN" ]; then
    AUTH_HEADER=(-H "Authorization: Bearer $TOKEN")
fi

fail_count=0
pass_count=0

ok() { echo "[PASS] $1"; pass_count=$((pass_count + 1)); }
bad() { echo "[FAIL] $1" >&2; fail_count=$((fail_count + 1)); }

# ---- 1. Health ----
resp=$(curl -sS -m 5 -w '\n%{http_code}' "$BASE/health" 2>&1)
code=$(echo "$resp" | tail -n1)
body=$(echo "$resp" | sed '$d')
if [ "$code" = "200" ]; then
    ok "GET /health -> 200 ($(echo "$body" | head -c 120)...)"
else
    bad "GET /health -> $code: $body"
fi

# ---- 2. Static HTMLs ----
for path in / /3d; do
    code=$(curl -sS -o /dev/null -m 5 -w '%{http_code}' "$BASE$path")
    if [ "$code" = "200" ]; then
        ok "GET $path -> 200 (HTML served)"
    else
        bad "GET $path -> $code (HTML missing? run kb-viz-overlay-build)"
    fi
done

# ---- 3. Overlay query ----
QUERY_PAYLOAD='{"text":"smoke test query","kind":"3d","top_k":3,"client_id":"smoke-bot"}'
# First-call timeout is generous because it can trigger a 2.2 GB model
# download of intfloat/multilingual-e5-large on a fresh server. Subsequent
# calls return in < 1s (CPU) or < 100ms (CUDA).
resp=$(curl -sS -m 240 -w '\n%{http_code}' \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]}" \
    -d "$QUERY_PAYLOAD" \
    "$BASE/overlay_query" 2>&1)
code=$(echo "$resp" | tail -n1)
body=$(echo "$resp" | sed '$d')
query_id=""
if [ "$code" = "200" ]; then
    query_id=$(echo "$body" | python3 -c "import json,sys; print(json.load(sys.stdin)['query_id'])" 2>/dev/null || echo "")
    if [ -n "$query_id" ]; then
        ok "POST /overlay_query -> 200 (query_id=$query_id)"
    else
        bad "POST /overlay_query -> 200 but missing query_id: $body"
    fi
else
    bad "POST /overlay_query -> $code: $body"
fi

# ---- 4. Feedback (only if overlay_query succeeded) ----
if [ -n "$query_id" ]; then
    FB_PAYLOAD=$(printf '{"query_id":"%s","query_text":"smoke test query","kind":"3d","verdict":"correct","client_id":"smoke-bot","top_k":[]}' "$query_id")
    resp=$(curl -sS -m 5 -w '\n%{http_code}' \
        -H "Content-Type: application/json" \
        "${AUTH_HEADER[@]}" \
        -d "$FB_PAYLOAD" \
        "$BASE/feedback" 2>&1)
    code=$(echo "$resp" | tail -n1)
    body=$(echo "$resp" | sed '$d')
    if [ "$code" = "200" ]; then
        log_path=$(echo "$body" | python3 -c "import json,sys; print(json.load(sys.stdin)['log_path'])" 2>/dev/null || echo "")
        ok "POST /feedback -> 200 (log: $log_path)"
    else
        bad "POST /feedback -> $code: $body"
    fi
fi

# ---- 5. Coverage ----
resp=$(curl -sS -m 5 -w '\n%{http_code}' "${AUTH_HEADER[@]}" "$BASE/coverage" 2>&1)
code=$(echo "$resp" | tail -n1)
body=$(echo "$resp" | sed '$d')
if [ "$code" = "200" ]; then
    cnt=$(echo "$body" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_feedback'])" 2>/dev/null || echo "?")
    ok "GET /coverage -> 200 (total_feedback=$cnt)"
else
    bad "GET /coverage -> $code: $body"
fi

# ---- 6. Profiles ----
resp=$(curl -sS -m 5 -w '\n%{http_code}' "${AUTH_HEADER[@]}" "$BASE/profiles" 2>&1)
code=$(echo "$resp" | tail -n1)
body=$(echo "$resp" | sed '$d')
if [ "$code" = "200" ]; then
    n=$(echo "$body" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['profiles']))" 2>/dev/null || echo "?")
    ok "GET /profiles -> 200 (profiles=$n)"
else
    bad "GET /profiles -> $code: $body"
fi

echo
echo "----"
echo "$pass_count passed, $fail_count failed"
if [ "$fail_count" -gt 0 ]; then
    exit 1
fi
exit 0
