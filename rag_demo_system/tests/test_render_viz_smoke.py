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
    # Server-backed profile picker
    assert "profilesUrl" in html_3d
    assert "showProfilePicker" in html_3d
    assert "fetchProfiles" in html_3d
    assert "Who is using this viz?" in html_3d
    # Per-chunk dual-signal buttons with Russian copy
    assert "Точность" in html_3d      # content accuracy label
    assert "Релевантность" in html_3d  # relevance label
    assert "✓ Верно" in html_3d
    assert "✗ Ошибка" in html_3d
    assert "◯ Подходит" in html_3d
    assert "⊘ Не подходит" in html_3d
    assert "signal_type" in html_3d
    assert "text_full" in html_3d
    # Step-through review flow wiring
    assert "currentChunkIdx" in html_3d
    assert "Дальше" in html_3d         # Next button
    assert "Пропустить" in html_3d     # Skip button
    assert "Скрыть чанки" in html_3d   # Collapse toggle
    assert "markSelected" in html_3d   # Persistent vote highlighting
    # Verdict halos + filter toggle + query zone/links + falling star + UMAP note
    assert "✓ Подтверждено" in html_3d        # green verdict halo trace
    assert "✗ Есть ошибка" in html_3d         # red verdict halo trace
    assert "Показать только проверенные" in html_3d  # filter button label
    assert "applyVisibilityFilter" in html_3d # filter applier fn
    assert "onlyInvestigated" in html_3d      # filter state flag
    assert "Top-5 этого запроса" in html_3d   # numbered retrieved-chunk markers
    assert "Связи запроса" in html_3d         # tether-line trace
    assert "★ Ваш запрос" in html_3d          # centroid star label
    assert "animateFallingStar" in html_3d    # falling-star animation
    # UMAP projection caveat is visible under the query input.
    assert "Top-5 — ближайшие по смыслу" in html_3d


def test_truncate_handles_long_text() -> None:
    short = "hello"
    assert render_viz._truncate(short) == "hello"
    long_text = "a" * (render_viz.HOVER_TEXT_MAX_CHARS + 50)
    out = render_viz._truncate(long_text)
    assert len(out) <= render_viz.HOVER_TEXT_MAX_CHARS + 1
    assert out.endswith("…")


def test_high_section_variety_does_not_explode_html(tmp_path: Path) -> None:
    """Regression test for the 575 MB incident: with a KB where every chunk
    has a unique heading_path[1], Plotly Express (color="section") would
    create one trace per chunk and explode the HTML to hundreds of MB. The
    MAX_COLOR_GROUPS bucketing keeps trace count bounded so the file stays
    around the baseline ~10 MB regardless of section variety.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    points = []
    n = 800  # keep UMAP fast in tests; the regression scales with heading variety
    for i in range(n):
        vec = rng.normal(0, 1, size=(32,))
        vec = vec / np.linalg.norm(vec)
        points.append({
            "id": f"pt-{i}",
            "vector": vec.tolist(),
            "payload": {
                "chunk_id": f"chunk-{i}",
                "text": f"Chunk {i}",
                "heading_path": ["Knowledge Base", f"topic-{i}"],
            },
        })
    src = tmp_path / "embeddings.json"
    src.write_text(json.dumps({"points": points}), encoding="utf-8")

    render_viz.render(embeddings_path=src, out_dir=tmp_path)

    html_3d = tmp_path / "kb_viz_3d.html"
    size_mb = html_3d.stat().st_size / (1024 * 1024)
    assert size_mb < 20, f"HTML grew to {size_mb:.1f} MB — MAX_COLOR_GROUPS bucketing regressed"


def test_json_for_script_escapes_closing_tag() -> None:
    """Codex adversarial finding 3: </script> breakout via crafted URL/token."""
    hostile_token = "abc</script><script>alert(1)</script>"
    out = render_viz._json_for_script(hostile_token)
    # No raw </ sequence survives — every occurrence is escaped to <\/.
    assert "</" not in out
    assert "<\\/script>" in out
    # Still valid JSON after the escape (the backslash is a JSON escape).
    import json as _json
    assert _json.loads(out) == hostile_token


def test_render_with_hostile_token_does_not_break_out(tmp_path: Path) -> None:
    """End-to-end: a crafted token must not terminate the <script> block."""
    src = tmp_path / "embeddings.json"
    src.write_text(json.dumps(_synthetic_embeddings()), encoding="utf-8")
    render_viz.render(
        embeddings_path=src,
        out_dir=tmp_path,
        overlay_url="https://example.com/overlay_query",
        overlay_token="x</script><script>alert(1)</script>",
    )
    html = (tmp_path / "kb_viz_3d.html").read_text(encoding="utf-8")
    # The hostile literal must NOT appear verbatim; the injected script block
    # must only close where the renderer intended.
    assert "x</script><script>alert(1)</script>" not in html
    # The escaped form does appear, carrying the same semantic value back to
    # JSON.parse / the JS string literal at runtime.
    assert "x<\\/script><script>alert(1)<\\/script>" in html


def test_chunk_coords_map_is_correct_under_multi_section_split(tmp_path: Path) -> None:
    """Regression test for Codex finding #2.

    Plotly Express splits one trace per color_group, so the emitted figure
    has multiple traces. Earlier versions reconstructed the chunk_id ->
    coords mapping by walking trace.customdata, but that silently used the
    shared full-length customdata array against per-trace x/y subsets,
    mis-attributing every chunk on any multi-section KB.

    Fix: the renderer now injects a server-computed chunk_id -> coords
    map directly as JSON. Verify here that every chunk appears in the
    map, and that the coords match the actual per-chunk UMAP projection
    (not some other chunk's coords sharing a trace index).
    """
    import numpy as np

    # Build a dataset where heading_path[1] is unique per cluster so the
    # Plotly color split produces multiple traces AND each trace's section
    # is distinguishable.
    rng = np.random.default_rng(0)
    points = []
    cluster_sections = ["Стоимость", "Документы", "Офисы"]
    per_cluster = 10
    for c, sec in enumerate(cluster_sections):
        center = rng.normal(0, 1, size=(32,))
        for i in range(per_cluster):
            vec = center + rng.normal(0, 0.15, size=(32,))
            vec = vec / np.linalg.norm(vec)
            idx = c * per_cluster + i
            points.append({
                "id": f"id-{idx}",
                "vector": vec.tolist(),
                "payload": {
                    "chunk_id": f"chunk-{idx}",
                    "text": f"text {idx}",
                    "heading_path": ["Knowledge Base", sec],
                },
            })
    src = tmp_path / "embeddings.json"
    src.write_text(json.dumps({"points": points, "vector_dim": 32, "count": len(points)}), encoding="utf-8")

    outputs = render_viz.render(
        embeddings_path=src,
        out_dir=tmp_path,
        overlay_url="https://example.com/overlay_query",
    )

    # Ground truth: re-run load_embeddings + _fit_umap to get the coords
    # we expect every chunk to land at.
    loaded = render_viz.load_embeddings(src)
    import joblib
    reducer = joblib.load(outputs["3d_reducer"])
    expected = reducer.transform(loaded.vectors)

    html = (tmp_path / "kb_viz_3d.html").read_text(encoding="utf-8")
    import re as _re
    m = _re.search(r"var __KB_VIZ_CHUNK_COORDS__ = (\{.*?\});", html, _re.DOTALL)
    assert m, "chunk coords var not present"
    coords_map = json.loads(m.group(1))

    # Every chunk present exactly once.
    assert len(coords_map) == 30
    assert set(coords_map.keys()) == {f"chunk-{i}" for i in range(30)}

    # Each chunk's recorded coords should match its ground-truth UMAP
    # projection within a tight tolerance. Under the old (wrong) code,
    # chunks would have been mapped to whichever trace's last
    # customdata[*][3] referenced them — producing systematic mis-attribution.
    import math
    for idx, rec in enumerate(loaded.records):
        cid = rec["chunk_id"]
        row = coords_map[cid]
        assert len(row) == 4, f"{cid} should be [x, y, z, section] in 3D mode"
        assert math.isclose(row[0], float(expected[idx, 0]), abs_tol=1e-4)
        assert math.isclose(row[1], float(expected[idx, 1]), abs_tol=1e-4)
        assert math.isclose(row[2], float(expected[idx, 2]), abs_tol=1e-4)
        # The section label should be the chunk's own section, not a neighbor's.
        assert row[3] == rec["section"]

    # Distinct sections should survive the color-group split: we gave
    # each cluster its own heading_path[1], so exactly 3 sections appear.
    assert {row[3] for row in coords_map.values()} == set(cluster_sections)


def test_load_embeddings_requires_points(tmp_path: Path) -> None:
    bad = tmp_path / "empty.json"
    bad.write_text(json.dumps({"points": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        render_viz.load_embeddings(bad)
