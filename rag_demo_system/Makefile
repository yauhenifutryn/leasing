# Standalone Makefile for the RAG demo system.
#
# Deliberately separate from the root Makefile (which handles the audio ->
# transcription -> KB build pipeline). Everything here is scoped to the
# rag_demo_system/ subtree and can be run independently, including on a
# freshly cloned server with no voice-pipeline dependencies.
#
# Run from the repo root:
#     make -C rag_demo_system kb-viz
#     make -C rag_demo_system kb-viz-test

VENV_VIZ ?= ../.venv-viz
PY_VIZ    = $(VENV_VIZ)/bin/python

# ---------------------------------------------------------------------------
# KB Visualization
# ---------------------------------------------------------------------------
# Static HTMLs:   emailable, no server at runtime.
# Overlay (opt):  FastAPI on :8500, live queries + client feedback capture.

# One-shot deploy: bootstrap venv + render HTMLs + start service +
# smoke-test + print share URL. Use this on a fresh server clone.
kb-viz-deploy:
	bash scripts/deploy_kb_viz.sh

kb-viz:
	bash scripts/setup_kb_viz.sh

kb-viz-overlay-build:
	@test -n "$$KB_VIZ_PUBLIC_URL" || \
	    (echo "Set KB_VIZ_PUBLIC_URL=https://<your-server>:8500/overlay_query" && exit 1)
	OVERLAY_URL="$$KB_VIZ_PUBLIC_URL" \
	OVERLAY_TOKEN="$$KB_VIZ_OVERLAY_TOKEN" \
	    bash scripts/setup_kb_viz.sh

kb-viz-overlay-serve:
	bash scripts/run_kb_viz_overlay.sh

kb-viz-test:
	$(PY_VIZ) -m pytest \
	    tests/test_render_viz_config.py \
	    tests/test_render_viz_smoke.py \
	    tests/test_kb_viz_service.py -v

kb-viz-feedback-report:
	$(PY_VIZ) scripts/kb_viz_feedback_report.py

kb-viz-smoke:
	bash scripts/kb_viz_smoke.sh

kb-viz-logs:
	@echo "=== service log (last 30 lines) ==="
	@tail -n 30 .state/kb_viz_service.log 2>/dev/null || echo "(no service log yet: run kb-viz-overlay-serve | tee .state/kb_viz_service.log)"
	@echo
	@echo "=== feedback log (last 10 lines) ==="
	@tail -n 10 .state/kb_viz_feedback.jsonl 2>/dev/null || echo "(no feedback yet)"
	@echo
	@echo "=== profiles ==="
	@cat .state/kb_viz_profiles.json 2>/dev/null || echo "(no profiles yet)"

# Static-only fallback: rebuild the HTMLs without overlay wiring. Use this
# to email emailable files when the overlay service is unavailable.
kb-viz-static:
	bash scripts/setup_kb_viz.sh

# Wipe feedback + profiles but leave the service running. Use before a
# fresh client testing round so the coverage panel starts at zero.
kb-viz-reset-state:
	rm -f .state/kb_viz_feedback.jsonl .state/kb_viz_profiles.json
	@echo "[kb-viz] wiped feedback + profiles"

# Stop any running uvicorn serving the kb-viz app. Safe no-op if none.
kb-viz-stop:
	@pkill -f 'uvicorn.*kb_viz_service' && echo "[kb-viz] stopped" || echo "[kb-viz] not running"

# Show whether the service is up, and its PID + listening ports.
kb-viz-status:
	@pgrep -af 'uvicorn.*kb_viz_service' || echo "[kb-viz] not running"
	@ss -tlnp 2>/dev/null | grep ':8500' || true

# Freeze current overlay state into a single client-shippable demo HTML.
# Run this BEFORE killing the server. Produces ONE pair of files
# regardless of how many queries you bake in — the HTML carries a
# pill picker when multiple queries are provided, so the client
# can switch examples without opening different files:
#   results/kb_viz_demo_snapshot.json     (raw data: coverage + all queries)
#   results/kb_viz_3d_demo.html           (self-contained 3D preview)
#   results/kb_viz_2d_demo.html           (self-contained 2D preview)
#
# Default behaviour bakes every query from DEMO_QUERIES_FILE. Override
# with DEMO_QUERY="..." to bake just one.
#
# Batch from the curated example file (one query per line, # comments ok):
#   make -C rag_demo_system kb-viz-demo-snapshot
#
# Single query, override:
#   make -C rag_demo_system kb-viz-demo-snapshot \
#       DEMO_QUERY="как оплатить через ерип" \
#       DEMO_QUERIES_FILE=
#
# Optional:
#   DEMO_URL            (default http://localhost:8500)
#   DEMO_USER           (default Demo)
DEMO_URL           ?= http://localhost:8500
DEMO_USER          ?= Demo
DEMO_QUERY         ?=
DEMO_QUERIES_FILE  ?= scripts/demo_queries.example.txt

# Build the --query / --queries-file args conditionally so the python
# script never sees an empty string when one variable is unset.
DEMO_QUERY_ARG        = $(if $(strip $(DEMO_QUERY)),--query $(DEMO_QUERY),)
DEMO_QUERIES_FILE_ARG = $(if $(strip $(DEMO_QUERIES_FILE)),--queries-file $(DEMO_QUERIES_FILE),)

kb-viz-demo-snapshot:
	$(PY_VIZ) scripts/make_demo_snapshot.py \
	    --url "$(DEMO_URL)" \
	    --token "$$KB_VIZ_OVERLAY_TOKEN" \
	    $(DEMO_QUERY_ARG) \
	    $(DEMO_QUERIES_FILE_ARG) \
	    --user "$(DEMO_USER)" \
	    --embeddings results/embeddings.json \
	    --out-dir results

# Offline merge: unify existing per-query demo snapshots (JSON) + a
# rendered template HTML into a single unified demo HTML with pill
# picker. No server, no embeddings — everything is done with downloaded
# files. Defaults auto-detect kb_viz_demo_snapshot_*.json and
# kb_viz_3d_demo_1.html in DEMO_MERGE_DIR.
#
# Usage after you've already downloaded per-query files:
#   make -C rag_demo_system kb-viz-demo-merge DEMO_MERGE_DIR=~/Downloads/kb-viz-demos
DEMO_MERGE_DIR ?= results
kb-viz-demo-merge:
	$(PY_VIZ) scripts/merge_demo_snapshots.py \
	    --out-dir "$(DEMO_MERGE_DIR)"

.PHONY: kb-viz-deploy kb-viz kb-viz-overlay-build kb-viz-overlay-serve \
        kb-viz-test kb-viz-feedback-report kb-viz-smoke kb-viz-logs \
        kb-viz-static kb-viz-reset-state kb-viz-stop kb-viz-status \
        kb-viz-demo-snapshot kb-viz-demo-merge
