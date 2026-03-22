#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [ -f "$ROOT_DIR/rag_demo_system/.env" ]; then
  set -a
  . "$ROOT_DIR/rag_demo_system/.env"
  set +a
fi

python "$ROOT_DIR/rag_demo_system/scripts/stack_cli.py" "$@"
