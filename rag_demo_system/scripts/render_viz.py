"""Render KB embeddings as 2D and 3D Plotly HTMLs.

Reads embeddings.json produced by dump_kb_embeddings.py, runs UMAP to 2D and
3D, and writes self-contained Plotly HTML files with optional live-query
overlay hooks.

Runs on CPU, no GPU required, no torch, no sentence-transformers. The
embedding model is only needed server-side for the overlay path.

Outputs (default to --out-dir):
    kb_viz_2d.html, kb_viz_3d.html   self-contained interactive plots
    umap_2d.joblib, umap_3d.joblib   fitted UMAP reducers, used by the
                                     kb_viz_service to project new queries

Usage:
    python render_viz.py --in results/embeddings.json
    python render_viz.py --overlay-url https://example.com/overlay_query
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECTION_3D: bool = True
UMAP_2D_N_NEIGHBORS: int = 15
UMAP_2D_MIN_DIST: float = 0.1
UMAP_3D_N_NEIGHBORS: int = 20
UMAP_3D_MIN_DIST: float = 0.2
UMAP_RANDOM_STATE: int = 42

HOVER_TEXT_MAX_CHARS: int = 300
TITLE_2D: str = "Micro Leasing KB · 2D Projection (UMAP)"
TITLE_3D: str = "Micro Leasing KB · 3D Projection (UMAP)"

_DEFAULT_SECTION = "Без раздела"


@dataclass
class LoadedEmbeddings:
    vectors: "np.ndarray"  # type: ignore[name-defined]
    records: list[dict[str, Any]]
    meta: dict[str, Any]


def load_embeddings(path: Path) -> LoadedEmbeddings:
    import numpy as np

    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("points") or []
    if not points:
        raise ValueError(f"No points found in {path}")

    vectors = np.array([p["vector"] for p in points], dtype=np.float32)
    records: list[dict[str, Any]] = []
    for p in points:
        payload_field = p.get("payload") or {}
        heading_path = payload_field.get("heading_path") or []
        section = heading_path[0] if heading_path else _DEFAULT_SECTION
        records.append(
            {
                "point_id": str(p.get("id", "")),
                "chunk_id": str(payload_field.get("chunk_id", p.get("id", ""))),
                "text": str(payload_field.get("text", "")),
                "heading_path": list(heading_path),
                "section": section,
                "source": str(payload_field.get("source", "")),
                "doc_name": str(payload_field.get("doc_name", "")),
            }
        )

    meta = {k: v for k, v in payload.items() if k != "points"}
    return LoadedEmbeddings(vectors=vectors, records=records, meta=meta)


def _truncate(text: str, n: int = HOVER_TEXT_MAX_CHARS) -> str:
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"


def _fit_umap(vectors: "np.ndarray", n_components: int, n_neighbors: int, min_dist: float) -> tuple["np.ndarray", Any]:  # type: ignore[name-defined]
    import numpy as np
    import umap

    effective_n_neighbors = min(n_neighbors, max(2, vectors.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(vectors)
    return np.asarray(coords, dtype=np.float32), reducer


def _build_hover_customdata(records: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            _truncate(r["text"]),
            r["section"] or _DEFAULT_SECTION,
            r["doc_name"] or "",
            r["chunk_id"],
        ]
        for r in records
    ]


def _overlay_post_script(kind: str, embed_url: str, token: str | None) -> str:
    """Inject the overlay UI: query input, feedback buttons, coverage panel.

    Uses DOM API only (createElement + textContent + appendChild) so response
    data from the server is never interpolated as HTML. No XSS surface.
    """
    token_js = json.dumps(token) if token else "null"
    url_js = json.dumps(embed_url)
    kind_js = json.dumps(kind)
    return f"""
(function() {{
  var embedUrl = {url_js};
  var token = {token_js};
  var kind = {kind_js};
  var feedbackUrl = embedUrl.replace(/\\/overlay_query(\\?.*)?$/, '/feedback$1');
  var coverageUrl = embedUrl.replace(/\\/overlay_query(\\?.*)?$/, '/coverage$1');
  var gd = document.getElementsByClassName('plotly-graph-div')[0];
  if (!gd) return;

  function el(tag, props) {{
    var node = document.createElement(tag);
    if (props) {{
      for (var k in props) {{
        if (k === 'style') node.style.cssText = props[k];
        else if (k === 'text') node.textContent = props[k];
        else if (k === 'onclick') node.onclick = props[k];
        else if (k === 'disabled') node.disabled = props[k];
        else node.setAttribute(k, props[k]);
      }}
    }}
    return node;
  }}

  function coverageIcon(cov) {{
    if (!cov) return '?';
    if (cov.wrong > 0 && cov.correct === 0) return '✗';
    if (cov.correct > 0 && cov.wrong === 0) return '✓';
    if (cov.correct > 0 && cov.wrong > 0) return '±';
    return '?';
  }}

  // State for the current query session
  var lastQuery = null;  // {{query_id, text, kind, top_k}}
  var lastCoverage = {{per_chunk: {{}}, per_section: {{}}, total_feedback: 0}};

  var bar = el('div', {{style: 'font-family:system-ui,sans-serif;font-size:13px;padding:10px;background:#fafafa;border-bottom:1px solid #ddd;'}});
  var toggle = el('button', {{text: 'Enable live overlay', style: 'padding:6px 10px;cursor:pointer;'}});
  var note = el('span', {{text: 'Experimental. Calls the GPU server.', style: 'margin-left:10px;color:#888;'}});
  var panel = el('div', {{style: 'display:none;margin-top:10px;'}});
  var input = el('input', {{type: 'text', placeholder: 'Ask a question (Russian)', style: 'width:60%;padding:6px;'}});
  var ask = el('button', {{text: 'Project', style: 'padding:6px 10px;margin-left:6px;cursor:pointer;'}});
  var status = el('div', {{style: 'margin-top:6px;color:#666;'}});
  var matches = el('div', {{style: 'margin-top:6px;color:#333;'}});
  var feedback = el('div', {{style: 'margin-top:8px;display:none;'}});
  var fbBtnOk = el('button', {{text: '✓ Correct', style: 'padding:6px 10px;cursor:pointer;background:#e7f7e7;border:1px solid #7bc97b;margin-right:6px;'}});
  var fbBtnNo = el('button', {{text: '✗ Wrong', style: 'padding:6px 10px;cursor:pointer;background:#fbeaea;border:1px solid #d98080;'}});
  var fbComment = el('div', {{style: 'margin-top:6px;display:none;'}});
  var fbTextarea = el('textarea', {{placeholder: 'What was wrong? (required)', style: 'width:60%;min-height:60px;padding:6px;'}});
  var fbSend = el('button', {{text: 'Send comment', style: 'padding:6px 10px;cursor:pointer;margin-left:6px;vertical-align:top;'}});
  var fbStatus = el('div', {{style: 'margin-top:4px;color:#666;'}});
  var coverage = el('div', {{style: 'margin-top:10px;padding:8px;background:#f4f8ff;border:1px solid #cfe0f5;font-size:12px;color:#234;'}});
  fbComment.appendChild(fbTextarea);
  fbComment.appendChild(fbSend);
  feedback.appendChild(fbBtnOk);
  feedback.appendChild(fbBtnNo);
  feedback.appendChild(fbComment);
  feedback.appendChild(fbStatus);
  panel.appendChild(input);
  panel.appendChild(ask);
  panel.appendChild(status);
  panel.appendChild(matches);
  panel.appendChild(feedback);
  panel.appendChild(coverage);
  bar.appendChild(toggle);
  bar.appendChild(note);
  bar.appendChild(panel);
  gd.parentNode.insertBefore(bar, gd);

  function authHeaders() {{
    var h = {{'Content-Type': 'application/json'}};
    if (token) h['Authorization'] = 'Bearer ' + token;
    return h;
  }}

  async function refreshCoverage() {{
    try {{
      var res = await fetch(coverageUrl, {{headers: token ? {{Authorization: 'Bearer ' + token}} : {{}}}});
      if (!res.ok) throw new Error('HTTP ' + res.status);
      lastCoverage = await res.json();
    }} catch (e) {{
      coverage.textContent = 'Coverage unavailable: ' + e.message;
      return;
    }}
    coverage.textContent = '';
    var header = el('div', {{text: 'Validation coverage: ' + lastCoverage.total_feedback + ' feedback events, ' + lastCoverage.unique_chunks_validated + ' unique chunks touched', style: 'font-weight:600;margin-bottom:4px;'}});
    coverage.appendChild(header);
    var sections = lastCoverage.per_section || {{}};
    var keys = Object.keys(sections);
    if (keys.length === 0) {{
      coverage.appendChild(el('div', {{text: 'No sections validated yet. Try a few queries per topic.', style: 'color:#666;'}}));
      renderMatchList();
      return;
    }}
    var table = el('div', {{style: 'display:grid;grid-template-columns:auto auto auto auto;gap:4px 12px;'}});
    table.appendChild(el('div', {{text: 'Section', style: 'font-weight:600;'}}));
    table.appendChild(el('div', {{text: '✓', style: 'font-weight:600;color:#2a7d2a;'}}));
    table.appendChild(el('div', {{text: '✗', style: 'font-weight:600;color:#a52a2a;'}}));
    table.appendChild(el('div', {{text: 'Chunks seen', style: 'font-weight:600;'}}));
    keys.sort().forEach(function(k) {{
      table.appendChild(el('div', {{text: k}}));
      table.appendChild(el('div', {{text: String(sections[k].correct || 0)}}));
      table.appendChild(el('div', {{text: String(sections[k].wrong || 0)}}));
      table.appendChild(el('div', {{text: String(sections[k].unique_chunks || 0)}}));
    }});
    coverage.appendChild(table);
    var hint = findUnvalidatedHint();
    if (hint) coverage.appendChild(el('div', {{text: hint, style: 'margin-top:6px;color:#555;font-style:italic;'}}));
    renderMatchList();
  }}

  function findUnvalidatedHint() {{
    // Sections present in the plot but absent (or barely present) in coverage.
    // Discover all sections from the current scatter traces' legendgroup/name.
    var plotted = new Set();
    (gd.data || []).forEach(function(trace) {{
      if (trace.name && trace.name !== 'Query') plotted.add(trace.name);
    }});
    var sections = lastCoverage.per_section || {{}};
    var unseen = [];
    var weak = [];
    plotted.forEach(function(name) {{
      var s = sections[name];
      if (!s || (s.correct + s.wrong) === 0) unseen.push(name);
      else if ((s.correct + s.wrong) < 2) weak.push(name);
    }});
    if (unseen.length > 0) {{
      return 'Not yet validated: ' + unseen.slice(0, 5).join(', ');
    }}
    if (weak.length > 0) {{
      return 'Thinly validated (< 2 events): ' + weak.slice(0, 5).join(', ');
    }}
    return '';
  }}

  function renderMatchList() {{
    matches.textContent = '';
    if (!lastQuery) return;
    lastQuery.top_k.forEach(function(m, i) {{
      var row = document.createElement('div');
      var score = (m.score != null) ? m.score.toFixed(3) : '?';
      var cov = (lastCoverage.per_chunk || {{}})[m.chunk_id];
      var icon = coverageIcon(cov);
      var counts = cov ? ' (' + cov.correct + '✓/' + cov.wrong + '✗)' : '';
      row.textContent = icon + ' ' + (i + 1) + '. ' + score + ' — ' + (m.section || '') + ' — ' + (m.text_preview || '') + counts;
      matches.appendChild(row);
    }});
  }}

  toggle.onclick = function() {{
    var showing = panel.style.display !== 'none';
    panel.style.display = showing ? 'none' : 'block';
    if (!showing) refreshCoverage();
  }};

  ask.onclick = async function() {{
    var q = input.value.trim();
    if (!q) return;
    status.textContent = 'Embedding + projecting...';
    matches.textContent = '';
    feedback.style.display = 'none';
    fbComment.style.display = 'none';
    fbStatus.textContent = '';
    fbTextarea.value = '';
    try {{
      var res = await fetch(embedUrl, {{method: 'POST', headers: authHeaders(), body: JSON.stringify({{text: q, kind: kind, top_k: 5}})}});
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var data = await res.json();
      var pos = data.position;
      var topK = data.top_k || [];
      lastQuery = {{query_id: data.query_id, text: q, kind: kind, top_k: topK}};
      var trace = kind === '3d'
        ? {{x:[pos[0]], y:[pos[1]], z:[pos[2]], mode:'markers+text', type:'scatter3d', marker:{{size:10, color:'red', symbol:'diamond'}}, text:['Your query'], textposition:'top center', name:'Query', hovertemplate:'%{{text}}<extra></extra>'}}
        : {{x:[pos[0]], y:[pos[1]], mode:'markers+text', type:'scatter', marker:{{size:16, color:'red', symbol:'star'}}, text:['Your query'], textposition:'top center', name:'Query', hovertemplate:'%{{text}}<extra></extra>'}};
      var existing = gd.data.findIndex(function(t) {{ return t.name === 'Query'; }});
      if (existing >= 0) Plotly.deleteTraces(gd, existing);
      Plotly.addTraces(gd, [trace]);
      status.textContent = 'Top ' + topK.length + ' matches (legend: ✓ validated correct, ✗ flagged wrong, ± mixed, ? never seen):';
      renderMatchList();
      feedback.style.display = 'block';
    }} catch (e) {{
      status.textContent = 'Overlay failed: ' + e.message;
    }}
  }};

  async function submitFeedback(verdict, comment) {{
    if (!lastQuery) return;
    fbStatus.textContent = 'Sending...';
    try {{
      var payload = {{
        query_id: lastQuery.query_id,
        query_text: lastQuery.text,
        kind: lastQuery.kind,
        verdict: verdict,
        top_k: lastQuery.top_k.map(function(m) {{ return {{chunk_id: m.chunk_id, section: m.section || '', score: m.score}}; }}),
      }};
      if (comment) payload.comment = comment;
      var res = await fetch(feedbackUrl, {{method: 'POST', headers: authHeaders(), body: JSON.stringify(payload)}});
      if (!res.ok) throw new Error('HTTP ' + res.status);
      fbStatus.textContent = 'Thanks. Recorded.';
      feedback.style.display = 'none';
      await refreshCoverage();
    }} catch (e) {{
      fbStatus.textContent = 'Feedback failed: ' + e.message;
    }}
  }}

  fbBtnOk.onclick = function() {{ submitFeedback('correct', null); }};
  fbBtnNo.onclick = function() {{
    fbComment.style.display = 'block';
    fbTextarea.focus();
  }};
  fbSend.onclick = function() {{
    var c = fbTextarea.value.trim();
    if (!c) {{ fbStatus.textContent = 'Comment required for wrong verdict.'; return; }}
    submitFeedback('wrong', c);
  }};
}})();
"""


def _render_one(
    coords: "np.ndarray",  # type: ignore[name-defined]
    records: list[dict[str, Any]],
    kind: str,
    out_path: Path,
    title: str,
    overlay_url: str | None,
    overlay_token: str | None,
) -> None:
    import pandas as pd
    import plotly.express as px

    assert kind in ("2d", "3d")
    is_3d = kind == "3d"

    sections = [r["section"] for r in records]
    hover_customdata = _build_hover_customdata(records)
    hovertemplate = (
        "<b>%{customdata[1]}</b><br>"
        "%{customdata[0]}<br>"
        "<i>%{customdata[2]}</i><br>"
        "<span style='color:#888;'>%{customdata[3]}</span>"
        "<extra></extra>"
    )

    frame_cols: dict[str, Any] = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "section": sections,
    }
    if is_3d:
        frame_cols["z"] = coords[:, 2]
    df = pd.DataFrame(frame_cols)

    if is_3d:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="section", title=title, opacity=0.85)
    else:
        fig = px.scatter(df, x="x", y="y", color="section", title=title, opacity=0.85)

    fig.update_traces(
        customdata=hover_customdata,
        hovertemplate=hovertemplate,
        marker=dict(size=6 if is_3d else 8),
    )
    fig.update_layout(
        legend=dict(title="KB section"),
        margin=dict(l=30, r=30, t=60, b=30),
    )

    post_script = _overlay_post_script(kind, overlay_url, overlay_token) if overlay_url else None
    html = fig.to_html(
        include_plotlyjs="inline",
        full_html=True,
        post_script=post_script,
    )
    out_path.write_text(html, encoding="utf-8")


def render(
    embeddings_path: Path,
    out_dir: Path,
    overlay_url: str | None = None,
    overlay_token: str | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_embeddings(embeddings_path)

    coords_2d, reducer_2d = _fit_umap(
        data.vectors, n_components=2, n_neighbors=UMAP_2D_N_NEIGHBORS, min_dist=UMAP_2D_MIN_DIST
    )
    out_2d = out_dir / "kb_viz_2d.html"
    _render_one(coords_2d, data.records, "2d", out_2d, TITLE_2D, overlay_url, overlay_token)

    written = {"2d_html": out_2d}
    _save_reducer(reducer_2d, out_dir / "umap_2d.joblib")
    written["2d_reducer"] = out_dir / "umap_2d.joblib"

    if PROJECTION_3D:
        coords_3d, reducer_3d = _fit_umap(
            data.vectors, n_components=3, n_neighbors=UMAP_3D_N_NEIGHBORS, min_dist=UMAP_3D_MIN_DIST
        )
        out_3d = out_dir / "kb_viz_3d.html"
        _render_one(coords_3d, data.records, "3d", out_3d, TITLE_3D, overlay_url, overlay_token)
        written["3d_html"] = out_3d
        _save_reducer(reducer_3d, out_dir / "umap_3d.joblib")
        written["3d_reducer"] = out_dir / "umap_3d.joblib"

    (out_dir / "viz_meta.json").write_text(
        json.dumps(
            {
                "source": str(embeddings_path),
                "count": len(data.records),
                "embedding_meta": data.meta,
                "umap": {
                    "2d": {
                        "n_neighbors": UMAP_2D_N_NEIGHBORS,
                        "min_dist": UMAP_2D_MIN_DIST,
                        "random_state": UMAP_RANDOM_STATE,
                    },
                    "3d": {
                        "n_neighbors": UMAP_3D_N_NEIGHBORS,
                        "min_dist": UMAP_3D_MIN_DIST,
                        "random_state": UMAP_RANDOM_STATE,
                    }
                    if PROJECTION_3D
                    else None,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    written["meta"] = out_dir / "viz_meta.json"
    return written


def _save_reducer(reducer: Any, path: Path) -> None:
    try:
        import joblib

        joblib.dump(reducer, path)
    except Exception as exc:
        sys.stderr.write(f"Warning: failed to save reducer {path}: {exc}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render KB embeddings to Plotly HTMLs.")
    parser.add_argument("--in", dest="in_path", default="rag_demo_system/results/embeddings.json")
    parser.add_argument("--out-dir", default="rag_demo_system/results")
    parser.add_argument("--overlay-url", default=None, help="If set, inject overlay button pointing at this endpoint.")
    parser.add_argument("--overlay-token", default=None, help="Bearer token to include in overlay requests.")
    args = parser.parse_args(argv)

    out = render(
        embeddings_path=Path(args.in_path),
        out_dir=Path(args.out_dir),
        overlay_url=args.overlay_url,
        overlay_token=args.overlay_token,
    )
    for k, p in out.items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
