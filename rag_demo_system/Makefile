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

.PHONY: kb-viz kb-viz-overlay-build kb-viz-overlay-serve kb-viz-test kb-viz-feedback-report
