#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Start Qdrant
cd "$ROOT_DIR/.."
docker compose -f rag_demo_system/docker-compose.yml up -d

# UI path
echo "UI: file://$ROOT_DIR/frontend/index.html"

# Start backend
cd "$ROOT_DIR"
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
