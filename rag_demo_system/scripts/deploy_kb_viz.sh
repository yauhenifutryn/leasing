#!/usr/bin/env bash
# One-shot deploy for the KB-viz overlay.
#
# Clone the branch on a fresh server, run this script, share the printed
# URL with the client. That's it.
#
#   cd /ephemeral
#   git clone https://github.com/yauhenifutryn/leasing.git leasing-kb-viz
#   cd leasing-kb-viz && git checkout feature/kb-viz
#   bash rag_demo_system/scripts/deploy_kb_viz.sh
#
# What it does:
#   1. Detects the server's public IP (or honors $KB_VIZ_PUBLIC_IP).
#   2. Generates a soft-gate token if $KB_VIZ_OVERLAY_TOKEN is unset.
#   3. Installs .venv-viz + deps (one-time, ~60s).
#   4. Dumps the live Qdrant KB to JSON.
#   5. Renders 2D + 3D Plotly HTMLs with the overlay URL + token baked in.
#   6. Starts the FastAPI service in the background, logging to .state/.
#   7. Waits for /health to pass (up to 90s — sentence-transformers import
#      is slow on first run).
#   8. Runs the smoke test from inside the box.
#   9. Prints the share URL plus a cheat-sheet of maintenance commands.
#
# Env vars (all optional):
#   KB_VIZ_PUBLIC_IP        override IP (default: autodetect)
#   KB_VIZ_PORT             listen port (default: 8500)
#   KB_VIZ_OVERLAY_TOKEN    soft-gate bearer token (default: 16 random hex)
#   KB_VIZ_EMBED_DEVICE     cuda|cpu (default: cpu; set cuda if GPU free)
#   QDRANT_URL              default: http://localhost:6333
#   QDRANT_COLLECTION       default: micro_leasing_kb
#
# Idempotent: safe to re-run. Re-runs regenerate HTMLs and restart service.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- 1. Public IP ------------------------------------------------------
PUBLIC_IP="${KB_VIZ_PUBLIC_IP:-}"
if [ -z "$PUBLIC_IP" ]; then
    PUBLIC_IP=$(curl -fsS --max-time 5 https://ifconfig.me 2>/dev/null || true)
fi
if [ -z "$PUBLIC_IP" ]; then
    PUBLIC_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
if [ -z "$PUBLIC_IP" ]; then
    echo "Could not determine public IP. Set KB_VIZ_PUBLIC_IP=<ip> and re-run." >&2
    exit 1
fi
PORT="${KB_VIZ_PORT:-8500}"
TOKEN="${KB_VIZ_OVERLAY_TOKEN:-$(openssl rand -hex 16)}"
PUBLIC_URL="http://${PUBLIC_IP}:${PORT}"
OVERLAY_URL="${PUBLIC_URL}/overlay_query"

echo "============================================================"
echo " KB-Viz deploy"
echo "   Public IP:   ${PUBLIC_IP}"
echo "   Port:        ${PORT}"
echo "   Public URL:  ${PUBLIC_URL}/3d"
echo "   Token:       ${TOKEN:0:8}… (baked into HTML; rotate via re-deploy)"
echo "============================================================"

mkdir -p rag_demo_system/.state rag_demo_system/results

# --- 2. Stop any prior instance ----------------------------------------
if pgrep -f 'uvicorn.*kb_viz_service' >/dev/null 2>&1; then
    echo "[deploy] stopping previous instance"
    pkill -f 'uvicorn.*kb_viz_service' || true
    sleep 2
fi

# --- 3. Bootstrap venv + render HTMLs with overlay baked in ------------
echo "[deploy] step 1/4: bootstrapping venv, dumping Qdrant, rendering HTMLs"
OVERLAY_URL="$OVERLAY_URL" \
OVERLAY_TOKEN="$TOKEN" \
    bash rag_demo_system/scripts/setup_kb_viz.sh

# --- 4. Ensure sentence-transformers present ---------------------------
VENV="$REPO_ROOT/.venv-viz"
if ! "$VENV/bin/python" -c "import sentence_transformers" 2>/dev/null; then
    echo "[deploy] step 2/4: installing sentence-transformers (one-time, ~60-120s)"
    "$VENV/bin/pip" install --quiet sentence-transformers
else
    echo "[deploy] step 2/4: sentence-transformers already installed"
fi

# --- 5. Start service --------------------------------------------------
echo "[deploy] step 3/4: starting service on 0.0.0.0:${PORT}"
LOG="rag_demo_system/.state/kb_viz_service.log"
: > "$LOG"
(
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    export KB_VIZ_OVERLAY_TOKEN="$TOKEN"
    export KB_VIZ_PORT="$PORT"
    export KB_VIZ_EMBED_DEVICE="${KB_VIZ_EMBED_DEVICE:-cpu}"
    nohup "$VENV/bin/python" -m uvicorn \
        rag_demo_system.services.kb_viz_service:app \
        --host 0.0.0.0 --port "$PORT" --log-level info \
        > "$LOG" 2>&1 &
    echo $! > rag_demo_system/.state/kb_viz_service.pid
) &
wait

sleep 2
PID=$(cat rag_demo_system/.state/kb_viz_service.pid 2>/dev/null || echo "?")
echo "[deploy] pid=${PID}, log=${LOG}"

# --- 6. Wait for /health ----------------------------------------------
echo "[deploy] step 4/4: waiting for /health (up to 90s)"
ready=0
for i in $(seq 1 45); do
    if curl -sf -m 2 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        echo "[deploy] service up after ${i} × 2s polls"
        break
    fi
    sleep 2
done
if [ "$ready" -ne 1 ]; then
    echo "[deploy] service failed to come up. Last 40 lines of log:" >&2
    tail -n 40 "$LOG" >&2 || true
    exit 1
fi

# --- 7. Smoke test ----------------------------------------------------
echo
echo "[deploy] running smoke test"
if KB_VIZ_BASE_URL="http://localhost:${PORT}" KB_VIZ_OVERLAY_TOKEN="$TOKEN" \
        bash rag_demo_system/scripts/kb_viz_smoke.sh; then
    smoke_ok=1
else
    smoke_ok=0
fi

# Also test from the public IP to confirm the firewall is open
echo
echo "[deploy] verifying public reachability (${PUBLIC_URL})"
if KB_VIZ_BASE_URL="$PUBLIC_URL" KB_VIZ_OVERLAY_TOKEN="$TOKEN" \
        bash rag_demo_system/scripts/kb_viz_smoke.sh >/dev/null 2>&1; then
    public_ok=1
    echo "[deploy] public reachability OK"
else
    public_ok=0
    echo "[deploy] WARNING: could not reach ${PUBLIC_URL} from inside the box."
    echo "[deploy]          Client link may still work if only the loopback"
    echo "[deploy]          path is blocked. Try from your laptop:"
    echo "[deploy]            curl -s ${PUBLIC_URL}/health"
fi

# --- 8. Final report --------------------------------------------------
cat <<EOT

============================================================
  KB-Viz READY
============================================================
  Client link (3D):    ${PUBLIC_URL}/3d
  Client link (2D):    ${PUBLIC_URL}/
  Pre-branded link:    ${PUBLIC_URL}/3d?user=sasha
  Bearer token:        ${TOKEN}
                       (already baked into the HTML — clients do not
                       need to enter it anywhere)

  Service PID:         ${PID}
  Service log:         ${LOG}
  Feedback log:        rag_demo_system/.state/kb_viz_feedback.jsonl
  Profiles:            rag_demo_system/.state/kb_viz_profiles.json

  Smoke test:          $([ $smoke_ok = 1 ] && echo 'PASS (localhost)' || echo 'FAIL (localhost) — inspect log')
  Public reachability: $([ $public_ok = 1 ] && echo 'PASS' || echo 'FAIL — check firewall / port mapping')

------------------------------------------------------------
  Maintenance (run from repo root)
------------------------------------------------------------
  View logs + feedback + profiles (one-shot, no streaming):
    make -C rag_demo_system kb-viz-logs

  Smoke-test again any time:
    KB_VIZ_OVERLAY_TOKEN=${TOKEN} make -C rag_demo_system kb-viz-smoke

  Status + PID + listening ports:
    make -C rag_demo_system kb-viz-status

  Wipe feedback + profiles (keeps service running):
    make -C rag_demo_system kb-viz-reset-state

  Stop the service:
    make -C rag_demo_system kb-viz-stop

  Re-deploy (new token, re-rendered HTMLs, fresh service):
    bash rag_demo_system/scripts/deploy_kb_viz.sh

  Build emailable static HTMLs (no server, no overlay):
    make -C rag_demo_system kb-viz-static
    # then: scp rag_demo_system/results/kb_viz_3d.html <laptop>:

  Aggregate feedback report:
    make -C rag_demo_system kb-viz-feedback-report
============================================================
EOT

if [ $smoke_ok -eq 0 ]; then
    exit 1
fi
