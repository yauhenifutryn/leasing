#!/usr/bin/env bash
# Fresh-server provisioning for the KB visualization flow.
#
# Idempotent: safe to run multiple times. Assumes production is already
# running (Qdrant on :6333, KB indexed). Creates .venv-viz, installs deps,
# dumps embeddings from Qdrant, renders 2D and 3D HTMLs.
#
# Usage:
#   bash rag_demo_system/scripts/setup_kb_viz.sh                  # static only
#   OVERLAY_URL=https://host:8500/overlay_query \
#     bash rag_demo_system/scripts/setup_kb_viz.sh                # overlay injected
#   OVERLAY_URL=... OVERLAY_TOKEN=abc \
#     bash rag_demo_system/scripts/setup_kb_viz.sh                # overlay + bearer
#
# After running, the HTMLs land in rag_demo_system/results/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv-viz"
REQS="rag_demo_system/requirements-kb-viz.txt"
EMB_OUT="rag_demo_system/results/embeddings.json"

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-micro_leasing_kb}"
OVERLAY_URL="${OVERLAY_URL:-}"
OVERLAY_TOKEN="${OVERLAY_TOKEN:-}"

mkdir -p rag_demo_system/results rag_demo_system/.state

echo "[kb-viz] venv: $VENV"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$REQS"

echo "[kb-viz] dumping embeddings from Qdrant"
echo "        url:        $QDRANT_URL"
echo "        collection: $QDRANT_COLLECTION"
"$VENV/bin/python" rag_demo_system/scripts/dump_kb_embeddings.py \
    --qdrant-url "$QDRANT_URL" \
    --collection "$QDRANT_COLLECTION" \
    --out "$EMB_OUT"

render_args=(--in "$EMB_OUT" --out-dir rag_demo_system/results)
if [ -n "$OVERLAY_URL" ]; then
    render_args+=(--overlay-url "$OVERLAY_URL")
    if [ -n "$OVERLAY_TOKEN" ]; then
        render_args+=(--overlay-token "$OVERLAY_TOKEN")
    fi
    echo "[kb-viz] rendering with overlay -> $OVERLAY_URL"
else
    echo "[kb-viz] rendering static only (no overlay)"
fi

"$VENV/bin/python" rag_demo_system/scripts/render_viz.py "${render_args[@]}"

echo "[kb-viz] done. Outputs:"
ls -lah rag_demo_system/results/kb_viz_*.html 2>/dev/null || true
