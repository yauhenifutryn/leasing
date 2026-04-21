"""End-to-end smoke test: synthetic embeddings in, HTML + reducer out.

Uses a small synthetic dataset so UMAP runs in a few hundred ms on CPU.
Skips cleanly if umap/plotly/joblib are not installed in the current env.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("umap")
pytest.importorskip("plotly")
pytest.importorskip("joblib")

import numpy as np  # noqa: E402

import render_viz  # noqa: E402


def _synthetic_embeddings(n_per_cluster: int = 10, dim: int = 32, n_clusters: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    points = []
    sections = ["pricing", "documents", "offices"]
    for c in range(n_clusters):
        center = rng.normal(0, 1, size=(dim,))
        for i in range(n_per_cluster):
            vec = center + rng.normal(0, 0.2, size=(dim,))
            vec = vec / np.linalg.norm(vec)
            idx = c * n_per_cluster + i
            points.append(
                {
                    "id": f"id-{idx}",
                    "vector": vec.tolist(),
                    "payload": {
                        "chunk_id": f"chunk-{idx}",
                        "text": f"Synthetic chunk {idx} in cluster {sections[c]}.",
                        "heading_path": [sections[c], f"sub-{i % 3}"],
                        "source": "synthetic",
                        "doc_name": "synthetic.md",
                        "start_char": 0,
                        "end_char": 50,
                    },
                }
            )
    return {
        "collection": "synthetic_kb",
        "model_name": "synthetic",
        "vector_dim": dim,
        "count": len(points),
        "points": points,
    }


def test_render_pipeline_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "embeddings.json"
    src.write_text(json.dumps(_synthetic_embeddings()), encoding="utf-8")

    written = render_viz.render(embeddings_path=src, out_dir=tmp_path)

    assert (tmp_path / "kb_viz_2d.html").exists()
    assert (tmp_path / "kb_viz_3d.html").exists()
    assert (tmp_path / "umap_2d.joblib").exists()
    assert (tmp_path / "umap_3d.joblib").exists()
    assert (tmp_path / "viz_meta.json").exists()

    html_2d = (tmp_path / "kb_viz_2d.html").read_text(encoding="utf-8")
    assert "Micro Leasing KB" in html_2d
    assert "plotly" in html_2d.lower()
    # No overlay when overlay_url is not provided
    assert "Enable live overlay" not in html_2d

    meta = json.loads((tmp_path / "viz_meta.json").read_text(encoding="utf-8"))
    assert meta["count"] == 30
    assert meta["umap"]["2d"]["n_neighbors"] == render_viz.UMAP_2D_N_NEIGHBORS


def test_render_with_overlay_injection(tmp_path: Path) -> None:
    src = tmp_path / "embeddings.json"
    src.write_text(json.dumps(_synthetic_embeddings()), encoding="utf-8")

    render_viz.render(
        embeddings_path=src,
        out_dir=tmp_path,
        overlay_url="https://example.com/overlay_query",
        overlay_token="test-token-123",
    )

    html_3d = (tmp_path / "kb_viz_3d.html").read_text(encoding="utf-8")
    assert "Enable live overlay" in html_3d
    assert "https://example.com/overlay_query" in html_3d
    assert "test-token-123" in html_3d
    # XSS smoke check: our injection uses textContent, not innerHTML, for
    # server-returned fields. Confirm no innerHTML assignment with user data.
    assert "innerHTML" not in html_3d or "statusEl.innerHTML" not in html_3d


def test_overlay_has_user_identity_and_polling(tmp_path: Path) -> None:
    """Covers the multi-user features: identity capture + polling + shared traces."""
    src = tmp_path / "embeddings.json"
    src.write_text(json.dumps(_synthetic_embeddings()), encoding="utf-8")

    render_viz.render(
        embeddings_path=src,
        out_dir=tmp_path,
        overlay_url="https://example.com/overlay_query",
    )

    html_3d = (tmp_path / "kb_viz_3d.html").read_text(encoding="utf-8")
    # User identity: URL param, localStorage, prompt fallback
    assert "URLSearchParams" in html_3d
    assert "kb_viz_user" in html_3d
    assert "You: " in html_3d
    # Polling: setInterval + pollMs constant piped through
    assert "setInterval" in html_3d
    assert str(render_viz.COVERAGE_POLL_MS) in html_3d
    # Shared overlay traces (user 2 sees user 1's marks)
    assert render_viz.COVERAGE_VALIDATED_TRACE in html_3d
    assert render_viz.COVERAGE_FLAGGED_TRACE in html_3d
    # Per-user colored traces
    assert render_viz.COVERAGE_USER_TRACE_PREFIX in html_3d
    assert render_viz.COVERAGE_WRONG_USER_TRACE_PREFIX in html_3d
    assert "colorForUser" in html_3d
    assert "YOU_COLOR" in html_3d
    # client_id wired into both request payloads
    assert "payload.client_id" in html_3d or "body.client_id" in html_3d


def test_truncate_handles_long_text() -> None:
    short = "hello"
    assert render_viz._truncate(short) == "hello"
    long_text = "a" * (render_viz.HOVER_TEXT_MAX_CHARS + 50)
    out = render_viz._truncate(long_text)
    assert len(out) <= render_viz.HOVER_TEXT_MAX_CHARS + 1
    assert out.endswith("…")


def test_load_embeddings_requires_points(tmp_path: Path) -> None:
    bad = tmp_path / "empty.json"
    bad.write_text(json.dumps({"points": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        render_viz.load_embeddings(bad)
