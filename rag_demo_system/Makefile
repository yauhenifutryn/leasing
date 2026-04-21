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

.PHONY: kb-viz-deploy kb-viz kb-viz-overlay-build kb-viz-overlay-serve \
        kb-viz-test kb-viz-feedback-report kb-viz-smoke kb-viz-logs \
        kb-viz-static kb-viz-reset-state kb-viz-stop kb-viz-status
