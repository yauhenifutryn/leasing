#!/usr/bin/env bash
# Start the KB-viz overlay FastAPI service on :8500.
#
# Assumes setup_kb_viz.sh has already generated the HTMLs and UMAP reducers
# in rag_demo_system/results/. Runs in the foreground; wrap with systemd /
# supervisord / nohup for background operation.
#
# Env vars (all optional, defaults in kb_viz_service.py):
#   KB_VIZ_PORT, KB_VIZ_HOST
#   KB_VIZ_OVERLAY_TOKEN   set to require bearer auth
#   KB_VIZ_EMBED_DEVICE    "cuda" or "cpu", default cpu
#
# Usage:
#   bash rag_demo_system/scripts/run_kb_viz_overlay.sh
#   KB_VIZ_EMBED_DEVICE=cuda bash rag_demo_system/scripts/run_kb_viz_overlay.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv-viz"
if [ ! -d "$VENV" ]; then
    echo "venv-viz not found. Run setup_kb_viz.sh first." >&2
    exit 1
fi

# The overlay service needs sentence-transformers + torch, which are already
# installed for production in the main repo venv. Install them into .venv-viz
# on demand so the static-only flow stays lightweight.
"$VENV/bin/python" -c "import sentence_transformers" 2>/dev/null || {
    echo "[kb-viz-overlay] installing sentence-transformers (one-time)"
    "$VENV/bin/pip" install --quiet sentence-transformers
}

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
PORT="${KB_VIZ_PORT:-8500}"
HOST="${KB_VIZ_HOST:-0.0.0.0}"

echo "[kb-viz-overlay] starting uvicorn on ${HOST}:${PORT}"
exec "$VENV/bin/python" -m uvicorn \
    rag_demo_system.services.kb_viz_service:app \
    --host "$HOST" --port "$PORT" --log-level info
