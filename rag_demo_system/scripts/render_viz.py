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

HOVER_TEXT_MAX_CHARS: int = 80
MATCH_LIST_PREVIEW_CHARS: int = 180
TITLE_2D: str = "Micro Leasing KB · 2D Projection (UMAP)"
TITLE_3D: str = "Micro Leasing KB · 3D Projection (UMAP)"

_DEFAULT_SECTION = "Без раздела"

# Plotly Express creates one trace per unique color value. A KB with
# thousands of chunks where every heading is unique would produce
# thousands of traces, each carrying its own marker config — the emitted
# HTML grows to hundreds of MB and refuses to load in a browser. Cap the
# number of distinct color groups; overflow goes into "Other".
MAX_COLOR_GROUPS: int = 15
OTHER_GROUP_LABEL: str = "Other"


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
        # heading_path[0] is the doc-level root ("Knowledge Base") and is the
        # same for every chunk, so using it for coloring / sectioning makes
        # the whole plot one colour. Prefer heading_path[1] when present
        # (the actual topic e.g. "кто владеет компанией Микро Лизинг").
        if len(heading_path) >= 2 and heading_path[1]:
            section = heading_path[1]
        elif heading_path:
            section = heading_path[0]
        else:
            section = _DEFAULT_SECTION
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

    # Compute color_group: keep the top N most frequent sections as their
    # own Plotly color category, bucket everything else into OTHER_GROUP_LABEL.
    # Without this, a KB with 3000 unique headings would generate 3000
    # traces (hundreds of MB of HTML). See MAX_COLOR_GROUPS.
    from collections import Counter
    counts = Counter(r["section"] for r in records)
    top_sections = {name for name, _ in counts.most_common(MAX_COLOR_GROUPS)}
    for r in records:
        r["color_group"] = r["section"] if r["section"] in top_sections else OTHER_GROUP_LABEL

    meta = {k: v for k, v in payload.items() if k != "points"}
    meta["color_groups"] = sorted(top_sections) + ([OTHER_GROUP_LABEL] if len(counts) > MAX_COLOR_GROUPS else [])
    meta["distinct_sections"] = len(counts)
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


COVERAGE_POLL_MS: int = 15000
COVERAGE_VALIDATED_TRACE: str = "✓ Validated (shared)"
COVERAGE_FLAGGED_TRACE: str = "✗ Flagged wrong (shared)"
COVERAGE_USER_TRACE_PREFIX: str = "✓ by "
COVERAGE_WRONG_USER_TRACE_PREFIX: str = "✗ by "


def _json_for_script(value: Any) -> str:
    """JSON-encode a value safely for inlining into an HTML <script> block.

    Why not plain ``json.dumps``: json does not escape ``</``, so a hostile
    or careless string containing ``</script>`` would close the surrounding
    script tag and let subsequent characters be parsed as markup. Replacing
    ``</`` with ``<\\/`` within the JSON output neutralizes this without
    changing the semantic value of any string in JavaScript.
    """
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _overlay_post_script(
    kind: str,
    embed_url: str,
    token: str | None,
    chunk_coords: dict[str, list[float]],
    demo_snapshot: dict[str, Any] | None = None,
) -> str:
    """Inject the overlay UI: query input, feedback buttons, coverage panel.

    Uses DOM API only (createElement + textContent + appendChild) so response
    data from the server is never interpolated as HTML. No XSS surface.
    Coverage polls every COVERAGE_POLL_MS while the panel is open, so
    concurrent users see each other's validation within that interval.

    ``chunk_coords`` is a server-computed ``chunk_id -> [x, y, (z,) section]``
    map. Injecting it directly fixes a subtle correctness bug: Plotly Express
    splits the figure into one trace per color_group, but ``update_traces(
    customdata=...)`` sets the FULL customdata array on EVERY trace, so
    per-trace rows and per-trace x/y no longer line up. Reconstructing coords
    from ``trace.customdata[idx]`` therefore mis-attributes chunks on any
    multi-section KB. We bypass that entirely by feeding the JS a
    ground-truth dict.
    """
    # ensure_ascii=False keeps the UTF-8 glyphs (✓ ✗) intact in the emitted
    # JS literals so the smoke test and legend strings match the configured
    # constants. The output HTML is served as UTF-8 so this is safe.
    #
    # _json_for_script also escapes ``</`` as ``<\/`` so a hostile string
    # (e.g., an operator pasting a crafted --overlay-url) cannot close the
    # surrounding <script> block and inject markup. json.dumps does not do
    # this escaping by default.
    token_js = _json_for_script(token) if token else "null"
    url_js = _json_for_script(embed_url)
    kind_js = _json_for_script(kind)
    poll_ms_js = _json_for_script(COVERAGE_POLL_MS)
    validated_name_js = _json_for_script(COVERAGE_VALIDATED_TRACE)
    flagged_name_js = _json_for_script(COVERAGE_FLAGGED_TRACE)
    chunk_coords_js = _json_for_script(chunk_coords)
    demo_snapshot_js = _json_for_script(demo_snapshot) if demo_snapshot else "null"
    return f"""
(function() {{
  var embedUrl = {url_js};
  var token = {token_js};
  var kind = {kind_js};
  var pollMs = {poll_ms_js};
  var VALIDATED_NAME = {validated_name_js};
  var FLAGGED_NAME = {flagged_name_js};
  var INVESTIGATED_NAME = '★ Проверенные чанки';  // legacy, kept for filter cleanup
  var VERIFIED_OK_NAME = '✓ Подтверждено';
  var VERIFIED_BAD_NAME = '✗ Есть ошибка';
  var QUERY_ZONE_NAME = 'Top-5 этого запроса';
  var QUERY_LINKS_NAME = 'Связи запроса';
  var QUERY_STAR_NAME = 'Query';
  // Ground-truth chunk_id -> {{x, y, z, section}} map built from the source
  // records. Do NOT reconstruct this from Plotly's trace.customdata — the
  // Plotly Express split-by-color produces per-trace x/y arrays that do
  // NOT line up 1:1 with the shared customdata array, which silently
  // misplaces every chunk on any KB with more than one color group.
  var __KB_VIZ_CHUNK_COORDS__ = {chunk_coords_js};
  // Non-null when the HTML was rendered with a pre-baked snapshot of a
  // live query + coverage response. The overlay boots into a "demo" mode
  // where the visualisation is fully populated but the server is never
  // contacted — used to hand clients a self-contained preview file.
  var __KB_VIZ_DEMO__ = {demo_snapshot_js};
  var feedbackUrl = embedUrl.replace(/\\/overlay_query(\\?.*)?$/, '/feedback$1');
  var coverageUrl = embedUrl.replace(/\\/overlay_query(\\?.*)?$/, '/coverage$1');
  var profilesUrl = embedUrl.replace(/\\/overlay_query(\\?.*)?$/, '/profiles$1');
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

  // ---- User identity: ?user= > localStorage > prompt ----
  function currentUser() {{
    try {{
      var params = new URLSearchParams(window.location.search || '');
      var u = params.get('user');
      if (u && u.trim()) {{
        localStorage.setItem('kb_viz_user', u.trim());
        return u.trim();
      }}
    }} catch (e) {{ /* URLSearchParams not available: fall through */ }}
    try {{
      var stored = localStorage.getItem('kb_viz_user');
      if (stored && stored.trim()) return stored.trim();
    }} catch (e) {{ /* localStorage blocked */ }}
    return null;
  }}

  function setUser(name) {{
    if (!name) return null;
    name = String(name).trim().slice(0, 128);
    if (!name) return null;
    try {{ localStorage.setItem('kb_viz_user', name); }} catch (e) {{}}
    // Register with the server so it appears in other users' pickers.
    try {{
      fetch(profilesUrl, {{
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({{name: name}})
      }}).catch(function() {{}});
    }} catch (e) {{}}
    return name;
  }}

  async function fetchProfiles() {{
    try {{
      var headers = token ? {{Authorization: 'Bearer ' + token}} : {{}};
      var res = await fetch(profilesUrl, {{headers: headers}});
      if (!res.ok) return [];
      var data = await res.json();
      return (data.profiles || []).map(function(p) {{ return p.name; }});
    }} catch (e) {{ return []; }}
  }}

  // Lightweight inline picker: shows existing profiles as clickable buttons
  // plus an input for a new profile. Resolves with the chosen name (or null
  // if the user dismisses).
  function showProfilePicker(existingNames) {{
    return new Promise(function(resolve) {{
      var backdrop = el('div', {{style: 'position:fixed;inset:0;background:rgba(0,0,0,0.45);z-index:9999;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif;'}});
      var modal = el('div', {{style: 'background:#fff;border-radius:8px;padding:20px;max-width:420px;width:90%;box-shadow:0 8px 32px rgba(0,0,0,0.25);'}});
      var title = el('div', {{text: 'Who is using this viz?', style: 'font-size:16px;font-weight:600;margin-bottom:4px;'}});
      var sub = el('div', {{text: 'Pick an existing profile or create a new one. No password — this is a lightweight identifier so feedback is attributable across devices.', style: 'font-size:12px;color:#666;margin-bottom:12px;'}});
      modal.appendChild(title);
      modal.appendChild(sub);

      if (existingNames && existingNames.length > 0) {{
        var existingLabel = el('div', {{text: 'Existing profiles:', style: 'font-size:12px;color:#666;margin-bottom:6px;'}});
        modal.appendChild(existingLabel);
        var row = el('div', {{style: 'display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;'}});
        existingNames.forEach(function(n) {{
          var btn = el('button', {{text: n, style: 'padding:6px 10px;border:1px solid #ccc;border-radius:4px;background:#f8f8f8;cursor:pointer;'}});
          btn.onclick = function() {{
            document.body.removeChild(backdrop);
            resolve(n);
          }};
          row.appendChild(btn);
        }});
        modal.appendChild(row);
      }} else {{
        modal.appendChild(el('div', {{text: '(No profiles yet — be the first.)', style: 'font-size:12px;color:#888;margin-bottom:12px;'}}));
      }}

      var newLabel = el('div', {{text: 'Or create new:', style: 'font-size:12px;color:#666;margin-bottom:4px;'}});
      var newInput = el('input', {{type: 'text', placeholder: 'Your name or initials', style: 'width:100%;padding:8px;border:1px solid #ccc;border-radius:4px;box-sizing:border-box;'}});
      var btnRow = el('div', {{style: 'display:flex;justify-content:flex-end;gap:8px;margin-top:12px;'}});
      var cancelBtn = el('button', {{text: 'Cancel', style: 'padding:6px 12px;cursor:pointer;background:#f0f0f0;border:1px solid #ccc;border-radius:4px;'}});
      var createBtn = el('button', {{text: 'Create', style: 'padding:6px 12px;cursor:pointer;background:#1e66c8;color:#fff;border:none;border-radius:4px;'}});
      modal.appendChild(newLabel);
      modal.appendChild(newInput);
      btnRow.appendChild(cancelBtn);
      btnRow.appendChild(createBtn);
      modal.appendChild(btnRow);

      cancelBtn.onclick = function() {{ document.body.removeChild(backdrop); resolve(null); }};
      createBtn.onclick = function() {{
        var v = newInput.value.trim();
        if (!v) {{ newInput.focus(); return; }}
        document.body.removeChild(backdrop);
        resolve(v);
      }};
      newInput.addEventListener('keydown', function(ev) {{
        if (ev.key === 'Enter') createBtn.click();
        if (ev.key === 'Escape') cancelBtn.click();
      }});

      backdrop.appendChild(modal);
      document.body.appendChild(backdrop);
      setTimeout(function() {{ newInput.focus(); }}, 50);
    }});
  }}

  async function chooseUser() {{
    var names = await fetchProfiles();
    var picked = await showProfilePicker(names);
    return picked ? setUser(picked) : null;
  }}

  var user = currentUser();
  // If ?user= was present, also register it server-side so other devices see it.
  if (user) {{
    try {{
      fetch(profilesUrl, {{
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({{name: user}})
      }}).catch(function() {{}});
    }} catch (e) {{}}
  }}

  // ---- State ----
  var lastQuery = null;  // {{query_id, text, kind, top_k}}
  var lastCoverage = {{per_chunk: {{}}, per_section: {{}}, total_feedback: 0}};
  var pollTimer = null;
  // Cached chunk_id -> coords map, built once from the baseline traces so
  // per-chunk coverage overlays can be drawn without re-parsing hover data.
  var chunkCoords = null;

  // ---- UI scaffolding ----
  var bar = el('div', {{style: 'font-family:system-ui,sans-serif;font-size:13px;padding:10px;background:#fafafa;border-bottom:1px solid #ddd;'}});
  var toggle = el('button', {{text: 'Enable live overlay', style: 'padding:6px 10px;cursor:pointer;'}});
  var userBadge = el('span', {{style: 'margin-left:10px;color:#333;'}});
  var userChange = el('a', {{href: '#', text: '(change)', style: 'margin-left:6px;color:#357;cursor:pointer;text-decoration:underline;'}});
  var note = el('span', {{text: 'Experimental. Calls the GPU server.', style: 'margin-left:10px;color:#888;'}});
  var panel = el('div', {{style: 'display:none;margin-top:10px;'}});
  var input = el('input', {{type: 'text', placeholder: 'Ask a question (Russian)', style: 'width:60%;padding:6px;'}});
  var ask = el('button', {{text: 'Go', style: 'padding:6px 10px;margin-left:6px;cursor:pointer;'}});
  var filterBtn = el('button', {{text: 'Показать только проверенные', style: 'padding:6px 10px;margin-left:6px;cursor:pointer;background:#f8f8f8;border:1px solid #ccc;border-radius:3px;'}});
  var umapNote = el('div', {{
    text: 'Top-5 — ближайшие по смыслу в 1024-мерном пространстве. 3D — это проекция (UMAP), поэтому визуально ближайшие точки не всегда совпадают с top-5. Красные линии показывают, какие именно чанки были извлечены.',
    style: 'margin-top:6px;color:#666;font-size:11px;line-height:1.5;font-style:italic;max-width:780px;'
  }});
  var status = el('div', {{style: 'margin-top:6px;color:#666;'}});
  var matches = el('div', {{style: 'margin-top:8px;color:#222;'}});
  var fbStatus = el('div', {{style: 'margin-top:6px;color:#666;font-size:12px;'}});
  var coverage = el('div', {{style: 'margin-top:10px;padding:8px;background:#f4f8ff;border:1px solid #cfe0f5;font-size:12px;color:#234;'}});
  panel.appendChild(input);
  panel.appendChild(ask);
  panel.appendChild(filterBtn);
  panel.appendChild(umapNote);
  panel.appendChild(status);
  panel.appendChild(matches);
  panel.appendChild(fbStatus);
  panel.appendChild(coverage);
  bar.appendChild(toggle);
  bar.appendChild(userBadge);
  bar.appendChild(userChange);
  bar.appendChild(note);
  bar.appendChild(panel);
  gd.parentNode.insertBefore(bar, gd);

  function renderUserBadge() {{
    userBadge.textContent = user ? ('You: ' + user) : '(no user set)';
  }}
  renderUserBadge();
  userChange.onclick = async function(e) {{
    e.preventDefault();
    var u = await chooseUser();
    if (u) {{ user = u; renderUserBadge(); }}
  }};

  function authHeaders() {{
    var h = {{'Content-Type': 'application/json'}};
    if (token) h['Authorization'] = 'Bearer ' + token;
    return h;
  }}

  // ---- chunk_id -> coords lookup ----
  // The server emits a ground-truth {{chunk_id: [x, y, z, section]}} map
  // baked into the HTML at render time. Earlier versions rebuilt this by
  // walking Plotly traces, but that was silently wrong once Plotly Express
  // split the figure into per-color-group traces (the shared customdata
  // array does not line up with per-trace x/y). See the docstring of
  // _overlay_post_script() for the full story.
  function buildChunkCoords() {{
    if (chunkCoords !== null) return chunkCoords;
    chunkCoords = {{}};
    var raw = __KB_VIZ_CHUNK_COORDS__ || {{}};
    Object.keys(raw).forEach(function(cid) {{
      var row = raw[cid] || [];
      if (row.length < 3) return;
      var has_z = (typeof row[2] === 'number');
      chunkCoords[String(cid)] = {{
        x: row[0],
        y: row[1],
        z: has_z ? row[2] : undefined,
        section: row[has_z ? 3 : 2] || ''
      }};
    }});
    return chunkCoords;
  }}

  // Deterministic color per user label. Current user gets a distinct
  // "YOU_COLOR" regardless of their name hash so they can always spot
  // themselves at a glance.
  var YOU_COLOR = '#1e66c8';
  function stringHash(s) {{
    var h = 2166136261 >>> 0;
    for (var i = 0; i < s.length; i++) {{ h = Math.imul((h ^ s.charCodeAt(i)) >>> 0, 16777619) >>> 0; }}
    return h >>> 0;
  }}
  function colorForUser(name, isMe) {{
    if (isMe) return YOU_COLOR;
    // Golden-angle hue spacing gives good separation even for small sets.
    var hue = (stringHash(name) * 137.508) % 360;
    return 'hsl(' + Math.round(hue) + ', 55%, 45%)';
  }}

  function rebuildCoverageTraces() {{
    var coords = buildChunkCoords();
    var perUser = lastCoverage.per_user || {{}};
    var perChunk = lastCoverage.per_chunk || {{}};

    function makeTrace(name, color, symbol, size, xs, ys, zs, texts) {{
      if (kind === '3d') {{
        return {{
          type: 'scatter3d', mode: 'markers', name: name,
          x: xs, y: ys, z: zs,
          marker: {{size: size, color: color, symbol: symbol, line: {{width: 1, color: '#fff'}}}},
          text: texts,
          hovertemplate: '<b>%{{text}}</b><extra>' + name + '</extra>',
          showlegend: true
        }};
      }}
      return {{
        type: 'scatter', mode: 'markers', name: name,
        x: xs, y: ys,
        marker: {{size: size, color: color, symbol: symbol, line: {{width: 1, color: '#fff'}}}},
        text: texts,
        hovertemplate: '<b>%{{text}}</b><extra>' + name + '</extra>',
        showlegend: true
      }};
    }}

    var newTraces = [];

    // Verdict halos: two color-coded open-circle rings instead of one
    // grey diamond. "Net correct" (last_verdict==correct OR correct>wrong)
    // gets a green ring; "net wrong" gets a red ring. Relevance-only
    // chunks (no content vote) get no ring — their presence is already
    // implied by the per-user scatter traces. Far cleaner and more
    // informative than the previous undifferentiated diamonds.
    var okXs = [], okYs = [], okZs = [], okTexts = [];
    var badXs = [], badYs = [], badZs = [], badTexts = [];
    Object.keys(perChunk).forEach(function(cid) {{
      var c = coords[cid];
      if (!c) return;
      var cov = perChunk[cid] || {{}};
      var correct = cov.correct || 0;
      var wrong = cov.wrong || 0;
      // Relevance-only vote: no content signal, skip the halo.
      if (correct === 0 && wrong === 0) return;
      var verdict;
      if (correct > wrong) verdict = 'ok';
      else if (wrong > correct) verdict = 'bad';
      else verdict = (cov.last_verdict === 'wrong') ? 'bad' : 'ok';
      var label = cid + ' — ' + (c.section || '') +
        ' (' + correct + '✓ / ' + wrong + '✗)';
      if (verdict === 'ok') {{
        okXs.push(c.x); okYs.push(c.y);
        if (c.z !== undefined) okZs.push(c.z);
        okTexts.push(label);
      }} else {{
        badXs.push(c.x); badYs.push(c.y);
        if (c.z !== undefined) badZs.push(c.z);
        badTexts.push(label);
      }}
    }});
    // Ring fill: light tinted, not fully transparent, so the LEGEND icon
    // actually shows a colored circle instead of an empty outline — the
    // user reported "panel on the right has ✓ Подтверждено with no icon".
    // Also add an icon inside each ring (✓ or ✗) so the meaning is legible
    // even at a glance without reading the legend label.
    function pushHaloTrace(name, borderColor, fillColor, glyph, xs, ys, zs, texts) {{
      if (xs.length === 0) return;
      var labelTexts = xs.map(function() {{ return glyph; }});
      if (kind === '3d') {{
        newTraces.push({{
          type: 'scatter3d', mode: 'markers+text', name: name,
          x: xs, y: ys, z: zs,
          marker: {{size: 16, color: fillColor, symbol: 'circle', line: {{width: 3, color: borderColor}}}},
          text: labelTexts, textposition: 'middle center',
          textfont: {{size: 11, color: borderColor, family: 'system-ui, sans-serif'}},
          customdata: texts,
          hovertemplate: '<b>%{{customdata}}</b><extra>' + name + '</extra>',
          showlegend: true
        }});
      }} else {{
        newTraces.push({{
          type: 'scatter', mode: 'markers+text', name: name,
          x: xs, y: ys,
          marker: {{size: 26, color: fillColor, symbol: 'circle', line: {{width: 3, color: borderColor}}}},
          text: labelTexts, textposition: 'middle center',
          textfont: {{size: 14, color: borderColor, family: 'system-ui, sans-serif'}},
          customdata: texts,
          hovertemplate: '<b>%{{customdata}}</b><extra>' + name + '</extra>',
          showlegend: true
        }});
      }}
    }}
    pushHaloTrace(VERIFIED_OK_NAME,  '#2a7d2a', 'rgba(42, 125, 42, 0.18)',  '✓', okXs,  okYs,  okZs,  okTexts);
    pushHaloTrace(VERIFIED_BAD_NAME, '#b3261e', 'rgba(179, 38, 30, 0.18)',  '✗', badXs, badYs, badZs, badTexts);

    // Order users so the current user renders on top
    var keys = Object.keys(perUser).sort(function(a, b) {{
      if (user && a === user) return -1;
      if (user && b === user) return 1;
      return a.localeCompare(b);
    }});

    keys.forEach(function(userKey) {{
      var u = perUser[userKey] || {{}};
      var isMe = user && userKey === user;
      var color = colorForUser(userKey, isMe);
      var label = isMe ? 'you (' + userKey + ')' : userKey;

      // Hover text for every marker on this chunk shows the full
      // attribution across all users, not just the current user. That
      // way, hovering any colored ring (regardless of whose trace it
      // belongs to) tells you who else touched this chunk.
      function chunkHover(cid, c) {{
        var cov = perChunk[cid] || {{}};
        var line = cid + ' — ' + (c.section || '');
        var v = (cov.validated_by || []).join(', ');
        var f = (cov.flagged_by || []).join(', ');
        if (v) line += '<br>✓ by: ' + v;
        if (f) line += '<br>✗ by: ' + f;
        return line;
      }}

      var cXs = [], cYs = [], cZs = [], cTexts = [];
      (u.correct_chunks || []).forEach(function(cid) {{
        var c = coords[cid];
        if (!c) return;
        cXs.push(c.x); cYs.push(c.y); if (c.z !== undefined) cZs.push(c.z);
        cTexts.push(chunkHover(cid, c));
      }});
      if (cXs.length > 0) {{
        newTraces.push(makeTrace(
          '✓ by ' + label,
          color,
          'circle-open',
          kind === '3d' ? (isMe ? 9 : 7) : (isMe ? 15 : 12),
          cXs, cYs, cZs, cTexts
        ));
      }}

      var wXs = [], wYs = [], wZs = [], wTexts = [];
      (u.wrong_chunks || []).forEach(function(cid) {{
        var c = coords[cid];
        if (!c) return;
        wXs.push(c.x); wYs.push(c.y); if (c.z !== undefined) wZs.push(c.z);
        wTexts.push(chunkHover(cid, c));
      }});
      if (wXs.length > 0) {{
        newTraces.push(makeTrace(
          '✗ by ' + label,
          color,
          'x',
          kind === '3d' ? (isMe ? 7 : 5) : (isMe ? 13 : 10),
          wXs, wYs, wZs, wTexts
        ));
      }}
    }});

    // Drop previous overlay traces (current + legacy shared ones)
    var toDelete = [];
    (gd.data || []).forEach(function(t, i) {{
      if (!t || !t.name) return;
      if (t.name.indexOf('✓ by ') === 0 || t.name.indexOf('✗ by ') === 0) toDelete.push(i);
      else if (t.name === VALIDATED_NAME || t.name === FLAGGED_NAME) toDelete.push(i);
      else if (t.name === INVESTIGATED_NAME) toDelete.push(i);
      else if (t.name === VERIFIED_OK_NAME || t.name === VERIFIED_BAD_NAME) toDelete.push(i);
    }});
    toDelete.reverse().forEach(function(i) {{ Plotly.deleteTraces(gd, i); }});
    if (newTraces.length > 0) Plotly.addTraces(gd, newTraces);
    // Re-apply the "only-investigated" visibility filter after the trace
    // set has changed, otherwise newly-added traces come back visible even
    // when the user had the filter toggled on.
    applyVisibilityFilter();
  }}

  // ---- Visibility filter: show only investigated chunks ----
  // When enabled, dim the entire base Plotly Express trace set to
  // 'legendonly' so only the investigated halo + per-user colored rings
  // + query markers remain on screen. Re-applied every time the trace
  // stack changes (coverage refresh, new query) so it survives redraws.
  var onlyInvestigated = false;
  function isOverlayTraceName(name) {{
    if (!name) return false;
    if (name.indexOf('✓ by ') === 0 || name.indexOf('✗ by ') === 0) return true;
    return (
      name === VALIDATED_NAME || name === FLAGGED_NAME ||
      name === INVESTIGATED_NAME ||
      name === VERIFIED_OK_NAME || name === VERIFIED_BAD_NAME ||
      name === QUERY_STAR_NAME ||
      name === QUERY_ZONE_NAME || name === QUERY_LINKS_NAME
    );
  }}
  function applyVisibilityFilter() {{
    if (!gd.data || gd.data.length === 0) return;
    var vis = [];
    gd.data.forEach(function(t) {{
      if (isOverlayTraceName(t.name)) {{
        vis.push(t.visible === false ? false : true);
      }} else {{
        vis.push(onlyInvestigated ? 'legendonly' : true);
      }}
    }});
    try {{
      Plotly.restyle(gd, {{visible: vis}});
    }} catch (e) {{ /* plotly not ready yet */ }}
  }}
  filterBtn.onclick = function() {{
    onlyInvestigated = !onlyInvestigated;
    filterBtn.textContent = onlyInvestigated
      ? 'Показать все чанки'
      : 'Показать только проверенные';
    filterBtn.style.background = onlyInvestigated ? '#e7f0fb' : '#f8f8f8';
    filterBtn.style.borderColor = onlyInvestigated ? '#7aa4d4' : '#ccc';
    applyVisibilityFilter();
  }};

  // ---- Falling-star animation ----
  // Visual cue that the query has been "cast" at the plot: a red star
  // flies from the input row to the center of the plot in ~900ms, then
  // fades. Runs in parallel with the embed+search request so the user
  // has feedback while the server is thinking. Pure DOM + CSS, no Plotly.
  function animateFallingStar() {{
    try {{
      var inputRect = input.getBoundingClientRect();
      var plotRect = gd.getBoundingClientRect();
      var star = document.createElement('div');
      star.textContent = '★';
      star.setAttribute('aria-hidden', 'true');
      var startX = inputRect.left + Math.min(inputRect.width * 0.5, 220);
      var startY = inputRect.top + inputRect.height / 2 - 14;
      star.style.cssText = (
        'position:fixed;' +
        'left:' + startX + 'px;' +
        'top:' + startY + 'px;' +
        'font-size:28px;line-height:28px;' +
        'color:#e23c3c;' +
        'text-shadow:0 0 8px rgba(255,180,180,0.95), 0 0 2px #fff;' +
        'pointer-events:none;z-index:9998;' +
        'transition:left 900ms cubic-bezier(0.45,0.05,0.55,0.95),' +
        ' top 900ms cubic-bezier(0.45,0.05,0.55,0.95),' +
        ' transform 900ms cubic-bezier(0.45,0.05,0.55,0.95),' +
        ' opacity 900ms ease-out;' +
        'transform:rotate(0deg) scale(1);'
      );
      document.body.appendChild(star);
      var endX = plotRect.left + plotRect.width / 2 - 14;
      var endY = plotRect.top + plotRect.height / 2 - 14;
      requestAnimationFrame(function() {{
        star.style.left = endX + 'px';
        star.style.top = endY + 'px';
        star.style.transform = 'rotate(540deg) scale(0.45)';
        star.style.opacity = '0.25';
      }});
      setTimeout(function() {{
        if (star.parentNode) star.parentNode.removeChild(star);
      }}, 980);
    }} catch (e) {{ /* animation is best-effort, never block the fetch */ }}
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
    var header = el('div', {{text: 'Validation coverage (shared across all users): ' + lastCoverage.total_feedback + ' feedback events, ' + lastCoverage.unique_chunks_validated + ' unique chunks touched', style: 'font-weight:600;margin-bottom:4px;'}});
    coverage.appendChild(header);
    var sections = lastCoverage.per_section || {{}};
    var keys = Object.keys(sections);
    if (keys.length === 0) {{
      coverage.appendChild(el('div', {{text: 'No sections validated yet. Try a few queries per topic.', style: 'color:#666;'}}));
    }} else {{
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
    }}
    coverage.appendChild(el('div', {{text: 'Updates every ' + Math.round(pollMs / 1000) + 's while this panel is open.', style: 'margin-top:6px;color:#888;font-size:11px;'}}));
    renderMatchList();
    rebuildCoverageTraces();
  }}

  function findUnvalidatedHint() {{
    var plotted = new Set();
    (gd.data || []).forEach(function(trace) {{
      if (!trace.name) return;
      if (isOverlayTraceName(trace.name)) return;
      plotted.add(trace.name);
    }});
    var sections = lastCoverage.per_section || {{}};
    var unseen = [];
    var weak = [];
    plotted.forEach(function(name) {{
      var s = sections[name];
      if (!s || (s.correct + s.wrong) === 0) unseen.push(name);
      else if ((s.correct + s.wrong) < 2) weak.push(name);
    }});
    if (unseen.length > 0) return 'Not yet validated: ' + unseen.slice(0, 5).join(', ');
    if (weak.length > 0) return 'Thinly validated (< 2 events): ' + weak.slice(0, 5).join(', ');
    return '';
  }}

  // Step-through review flow. Shows ONE chunk at a time with full text
  // expanded, vote buttons that persist their selected state, and a Next
  // button that always advances. When all N chunks are reviewed the panel
  // collapses so the 3D graph is not pushed off-screen.
  //
  // State per query:
  //   currentChunkIdx — index of the chunk being reviewed (0..N-1, N=done)
  //   chunkStates[i]  — {{content: 'correct'|'wrong'|null, relevance: ...}}
  var currentChunkIdx = 0;
  var chunkStates = [];
  var matchesCollapsed = false;

  function resetReviewState() {{
    currentChunkIdx = 0;
    chunkStates = [];
    matchesCollapsed = false;
  }}

  function markSelected(btn, isSelected, selectedBg) {{
    if (isSelected) {{
      btn.style.borderWidth = '2px';
      btn.style.fontWeight = '700';
      btn.style.background = selectedBg;
      btn.style.boxShadow = 'inset 0 0 0 1px rgba(0,0,0,0.08)';
    }} else {{
      btn.style.borderWidth = '1px';
      btn.style.fontWeight = '400';
      btn.style.boxShadow = 'none';
    }}
  }}

  function renderMatchList() {{
    matches.textContent = '';
    if (!lastQuery) return;

    var total = lastQuery.top_k.length;
    // Ensure the state array is the right length.
    while (chunkStates.length < total) chunkStates.push({{content: null, relevance: null}});

    var instruction = el('div', {{style: 'margin-bottom:8px;padding:8px 10px;background:#fff8e1;border-left:3px solid #e0b84a;color:#4a3d14;font-size:12px;line-height:1.5;'}});
    instruction.appendChild(el('div', {{text: 'Как проверять:', style: 'font-weight:600;margin-bottom:4px;'}}));
    instruction.appendChild(el('div', {{text: '• Точность (главное) — верны ли факты в тексте чанка. ✗ требует комментарий.'}}));
    instruction.appendChild(el('div', {{text: '• Релевантность (по желанию) — подходит ли этот чанк как ответ на ваш вопрос.'}}));
    instruction.appendChild(el('div', {{text: '• Если не уверены — просто нажмите «Пропустить».', style: 'margin-top:4px;color:#6d5210;'}}));
    matches.appendChild(instruction);

    // Collapse toggle (works at any time: mid-review or after completion).
    var header = el('div', {{style: 'display:flex;align-items:center;gap:10px;margin-bottom:8px;'}});
    var progressLabel = el('span', {{style: 'font-weight:600;color:#234;'}});
    progressLabel.textContent = currentChunkIdx >= total
      ? 'Все ' + total + ' чанков просмотрены. Спасибо!'
      : 'Чанк ' + (currentChunkIdx + 1) + ' из ' + total;
    header.appendChild(progressLabel);
    var collapseBtn = el('a', {{href: '#', style: 'margin-left:auto;color:#357;cursor:pointer;text-decoration:underline;font-size:12px;'}});
    collapseBtn.textContent = matchesCollapsed ? 'Показать чанки ▼' : 'Скрыть чанки ▲';
    collapseBtn.onclick = function(e) {{
      e.preventDefault();
      matchesCollapsed = !matchesCollapsed;
      renderMatchList();
    }};
    header.appendChild(collapseBtn);
    matches.appendChild(header);

    if (matchesCollapsed) return;

    // All done: show a summary of what the user voted and stop here.
    if (currentChunkIdx >= total) {{
      var summary = el('div', {{style: 'padding:10px;background:#eef7ee;border:1px solid #b5d8b5;border-radius:4px;color:#234;font-size:12px;'}});
      var contentVotes = chunkStates.filter(function(s) {{ return s.content; }}).length;
      var relVotes = chunkStates.filter(function(s) {{ return s.relevance; }}).length;
      summary.textContent = 'Записано голосов: ' + contentVotes + ' по точности, ' + relVotes + ' по релевантности. ' +
        'Задайте другой вопрос, чтобы продолжить.';
      matches.appendChild(summary);
      return;
    }}

    var m = lastQuery.top_k[currentChunkIdx];
    var state = chunkStates[currentChunkIdx];
    var cov = (lastCoverage.per_chunk || {{}})[m.chunk_id];
    var icon = coverageIcon(cov);
    var counts = cov ? cov.correct + '✓ / ' + cov.wrong + '✗' : 'ещё не проверялось';
    var score = (m.score != null) ? m.score.toFixed(3) : '?';

    var card = el('div', {{style: 'padding:12px 14px;background:#fff;border:1px solid #d0d0d0;border-radius:6px;font-size:13px;box-shadow:0 1px 3px rgba(0,0,0,0.04);'}});

    var head = el('div', {{style: 'display:flex;align-items:center;gap:10px;margin-bottom:8px;'}});
    head.appendChild(el('span', {{text: icon + ' #' + (currentChunkIdx + 1), style: 'font-weight:700;font-size:14px;'}}));
    head.appendChild(el('span', {{text: score, style: 'color:#888;font-size:11px;'}}));
    head.appendChild(el('span', {{text: m.section || '(без раздела)', style: 'color:#223;flex:1;font-weight:500;'}}));
    head.appendChild(el('span', {{text: counts, style: 'color:#555;font-size:11px;'}}));
    card.appendChild(head);

    var full = m.text_full || m.text_preview || '';
    var textBox = el('div', {{text: full, style: 'color:#222;margin:0 0 12px 0;line-height:1.5;white-space:pre-wrap;background:#fafafa;padding:10px;border-radius:4px;border:1px solid #eee;max-height:360px;overflow:auto;'}});
    card.appendChild(textBox);

    // Dedup badge: when this chunk represents a group of same-section
    // near-duplicates, show how many were folded in and let the reviewer
    // expand the list. Nothing is lost — the hidden chunks are still in
    // Qdrant and written to the query log — this just surfaces that the
    // top-K they see has been cleaned up.
    var hidden = m.hidden_duplicates || [];
    if (hidden.length > 0) {{
      var dupBox = el('div', {{style: 'margin:0 0 10px 0;padding:6px 10px;background:#fff8e1;border:1px dashed #e0b84a;border-radius:4px;font-size:12px;color:#4a3d14;'}});
      var dupToggle = el('a', {{
        href: '#',
        text: '+' + hidden.length + ' похожих из раздела «' + (m.section || 'без раздела') + '» (показать)',
        style: 'color:#357;cursor:pointer;text-decoration:underline;'
      }});
      var dupList = el('div', {{style: 'display:none;margin-top:6px;font-size:11px;line-height:1.5;color:#333;'}});
      hidden.forEach(function(h) {{
        var row = el('div', {{style: 'padding:4px 0;border-top:1px solid #f0e4bf;'}});
        row.appendChild(el('div', {{
          text: h.chunk_id + ' · score ' + (h.score != null ? h.score.toFixed(3) : '?'),
          style: 'color:#555;font-family:monospace;font-size:10px;'
        }}));
        row.appendChild(el('div', {{
          text: h.text_preview || '(нет превью)',
          style: 'color:#222;'
        }}));
        dupList.appendChild(row);
      }});
      var expanded = false;
      dupToggle.onclick = function(e) {{
        e.preventDefault();
        expanded = !expanded;
        dupList.style.display = expanded ? 'block' : 'none';
        dupToggle.textContent = (expanded ? '−' : '+') + hidden.length +
          ' похожих из раздела «' + (m.section || 'без раздела') + '» ' +
          (expanded ? '(скрыть)' : '(показать)');
      }};
      dupBox.appendChild(dupToggle);
      dupBox.appendChild(dupList);
      card.appendChild(dupBox);
    }}

    // --- Content accuracy buttons ---
    var contentRow = el('div', {{style: 'display:flex;gap:6px;align-items:center;margin-bottom:6px;'}});
    contentRow.appendChild(el('span', {{text: 'Точность:', style: 'font-size:12px;color:#555;min-width:110px;'}}));
    var ok = el('button', {{text: '✓ Верно', style: 'padding:6px 12px;cursor:pointer;background:#e7f7e7;border:1px solid #7bc97b;border-radius:3px;font-size:12px;'}});
    var no = el('button', {{text: '✗ Ошибка', style: 'padding:6px 12px;cursor:pointer;background:#fbeaea;border:1px solid #d98080;border-radius:3px;font-size:12px;'}});
    markSelected(ok, state.content === 'correct', '#bfe4bf');
    markSelected(no, state.content === 'wrong', '#f2c4c4');
    contentRow.appendChild(ok);
    contentRow.appendChild(no);
    card.appendChild(contentRow);

    // Comment textarea, visible when wrong is selected (or being chosen)
    var commentBox = el('div', {{style: 'display:' + (state.content === 'wrong' ? 'block' : 'none') + ';margin:6px 0 10px 0;'}});
    var ta = el('textarea', {{placeholder: 'Что именно неверно? (обязательно для ✗)', style: 'width:100%;min-height:60px;padding:8px;box-sizing:border-box;font-size:12px;'}});
    if (state.contentComment) ta.value = state.contentComment;
    commentBox.appendChild(ta);
    card.appendChild(commentBox);

    // --- Relevance buttons (optional) ---
    var relRow = el('div', {{style: 'display:flex;gap:6px;align-items:center;margin-bottom:12px;'}});
    relRow.appendChild(el('span', {{text: 'Релевантность:', style: 'font-size:12px;color:#555;min-width:110px;'}}));
    var rel = el('button', {{text: '◯ Подходит', style: 'padding:6px 12px;cursor:pointer;background:#eef3fb;border:1px solid #7aa4d4;border-radius:3px;font-size:12px;'}});
    var notRel = el('button', {{text: '⊘ Не подходит', style: 'padding:6px 12px;cursor:pointer;background:#f5f0f8;border:1px solid #a989c2;border-radius:3px;font-size:12px;'}});
    markSelected(rel, state.relevance === 'relevant', '#c8d8ef');
    markSelected(notRel, state.relevance === 'not_relevant', '#dcc9e5');
    relRow.appendChild(rel);
    relRow.appendChild(notRel);
    card.appendChild(relRow);

    // --- Status + navigation ---
    var footer = el('div', {{style: 'display:flex;align-items:center;gap:10px;'}});
    var rowStatus = el('span', {{style: 'color:#666;font-size:11px;flex:1;'}});
    var skipBtn = el('button', {{text: 'Пропустить →', style: 'padding:6px 14px;cursor:pointer;background:#f0f0f0;border:1px solid #bbb;border-radius:3px;font-size:12px;'}});
    var nextBtn = el('button', {{text: 'Дальше →', style: 'padding:6px 14px;cursor:pointer;background:#1e66c8;border:1px solid #1e66c8;color:#fff;border-radius:3px;font-size:12px;display:' + ((state.content || state.relevance) ? 'inline-block' : 'none') + ';'}});
    footer.appendChild(rowStatus);
    footer.appendChild(skipBtn);
    footer.appendChild(nextBtn);
    card.appendChild(footer);

    matches.appendChild(card);

    // Wire up vote buttons — each records the signal and re-renders for
    // the highlight state. Local state is only mutated after the server
    // confirms the POST, otherwise a failed /feedback silently "records"
    // a vote that was never persisted. Buttons are disabled while a
    // submission is in flight so the user can't double-click.
    function setInFlight(on) {{
      ok.disabled = on; no.disabled = on;
      rel.disabled = on; notRel.disabled = on;
      nextBtn.disabled = on; skipBtn.disabled = on;
    }}
    ok.onclick = async function() {{
      setInFlight(true);
      var okResult = await submitChunkFeedback(m, 'content', 'correct', null, rowStatus);
      setInFlight(false);
      if (!okResult) return;
      state.content = 'correct';
      state.contentComment = null;
      renderMatchList();
    }};
    no.onclick = function() {{
      // No submission yet — the wrong verdict requires a comment, which
      // is collected in the textarea. State flips locally so the comment
      // box opens, but nothing has been logged server-side yet.
      state.content = 'wrong';
      renderMatchList();
      setTimeout(function() {{ ta.focus(); }}, 10);
    }};
    // When the comment textarea loses focus and has content + we're in
    // 'wrong' state, submit the wrong verdict. Only record the comment
    // as persisted (state.contentComment) if the POST actually succeeded.
    ta.addEventListener('blur', async function() {{
      var c = ta.value.trim();
      if (state.content === 'wrong' && c && state.contentComment !== c) {{
        setInFlight(true);
        var okResult = await submitChunkFeedback(m, 'content', 'wrong', c, rowStatus);
        setInFlight(false);
        if (okResult) state.contentComment = c;
      }}
    }});
    rel.onclick = async function() {{
      setInFlight(true);
      var okResult = await submitChunkFeedback(m, 'relevance', 'relevant', null, rowStatus);
      setInFlight(false);
      if (!okResult) return;
      state.relevance = 'relevant';
      renderMatchList();
    }};
    notRel.onclick = async function() {{
      setInFlight(true);
      var okResult = await submitChunkFeedback(m, 'relevance', 'not_relevant', null, rowStatus);
      setInFlight(false);
      if (!okResult) return;
      state.relevance = 'not_relevant';
      renderMatchList();
    }};
    skipBtn.onclick = function() {{ currentChunkIdx++; renderMatchList(); }};
    nextBtn.onclick = async function() {{
      if (state.content === 'wrong') {{
        var c = ta.value.trim();
        if (!c) {{ rowStatus.textContent = 'Нужен комментарий к ошибке.'; ta.focus(); return; }}
        if (state.contentComment !== c) {{
          setInFlight(true);
          var okResult = await submitChunkFeedback(m, 'content', 'wrong', c, rowStatus);
          setInFlight(false);
          if (!okResult) return;  // stay on this chunk so the user retries
          state.contentComment = c;
        }}
      }}
      currentChunkIdx++;
      renderMatchList();
    }};
  }}

  function startPolling() {{
    stopPolling();
    pollTimer = setInterval(function() {{
      if (!document.hidden && panel.style.display !== 'none') refreshCoverage();
    }}, pollMs);
  }}
  function stopPolling() {{
    if (pollTimer) {{ clearInterval(pollTimer); pollTimer = null; }}
  }}

  toggle.onclick = async function() {{
    var showing = panel.style.display !== 'none';
    panel.style.display = showing ? 'none' : 'block';
    if (!showing) {{
      if (!user) {{
        var picked = await chooseUser();
        if (picked) {{ user = picked; renderUserBadge(); }}
      }}
      refreshCoverage();
      startPolling();
    }} else {{
      stopPolling();
    }}
  }};

  // Enter key in the query input triggers the same action as Go click.
  input.addEventListener('keydown', function(ev) {{
    if (ev.key === 'Enter' && !ev.shiftKey) {{
      ev.preventDefault();
      ask.click();
    }}
  }});

  ask.onclick = async function() {{
    var q = input.value.trim();
    if (!q) return;
    status.textContent = 'Поиск в KB...';
    matches.textContent = '';
    fbStatus.textContent = '';
    // Fire the visual cue immediately so the user sees motion while the
    // embed+qdrant call is in flight. Animation lasts ~900 ms and is
    // independent of the fetch result.
    animateFallingStar();
    try {{
      var body = {{text: q, kind: kind, top_k: 5}};
      if (user) body.client_id = user;
      var res = await fetch(embedUrl, {{method: 'POST', headers: authHeaders(), body: JSON.stringify(body)}});
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var data = await res.json();
      var topK = data.top_k || [];
      lastQuery = {{query_id: data.query_id, text: q, kind: kind, top_k: topK}};
      resetReviewState();

      // Star position: use centroid of the top-K chunks' UMAP coordinates
      // instead of the raw query-vector transform. UMAP.transform() of a
      // query vector embedded with "query: " prefix often lands outside
      // the passage cloud (e5-large puts query/passage in offset
      // sub-regions). The centroid is visually honest — "your question
      // retrieved these specific points, here's the center of them" —
      // and the high-dim top-K is computed correctly upstream regardless.
      var coords = buildChunkCoords();
      var cx = 0, cy = 0, cz = 0, n = 0;
      var chunkXs = [], chunkYs = [], chunkZs = [], chunkLabels = [], chunkHovers = [];
      topK.forEach(function(m, idx) {{
        var c = coords[m.chunk_id];
        if (!c) return;
        cx += c.x; cy += c.y; if (c.z !== undefined) cz += c.z;
        n += 1;
        chunkXs.push(c.x); chunkYs.push(c.y);
        if (c.z !== undefined) chunkZs.push(c.z);
        chunkLabels.push(String(idx + 1));
        var score = (m.score != null) ? ' · score ' + m.score.toFixed(3) : '';
        chunkHovers.push('#' + (idx + 1) + ' · ' + m.chunk_id + ' — ' + (m.section || '') + score);
      }});
      var pos;
      if (n > 0) {{
        pos = [cx / n, cy / n, kind === '3d' ? cz / n : undefined];
      }} else {{
        // Fallback to the server-reported projection (rare: all top-K
        // chunks missing from coords, e.g., KB re-indexed mid-session).
        pos = data.position;
      }}

      // Tether lines from star → each retrieved chunk. Without this the
      // plot doesn't show WHICH chunks belong to the current query, and
      // the user is left wondering why the chunks aren't always visually
      // closest to the star (answer: UMAP is a lossy projection of the
      // 1024-dim cosine space; the "closest in 3D" is not necessarily
      // "closest in meaning"). A line makes the retrieval set obvious.
      var linkTrace = null;
      if (chunkXs.length > 0) {{
        var lx = [], ly = [], lz = [];
        for (var i = 0; i < chunkXs.length; i++) {{
          lx.push(pos[0], chunkXs[i], null);
          ly.push(pos[1], chunkYs[i], null);
          if (kind === '3d') lz.push(pos[2], chunkZs[i], null);
        }}
        if (kind === '3d') {{
          linkTrace = {{
            type: 'scatter3d', mode: 'lines', name: QUERY_LINKS_NAME,
            x: lx, y: ly, z: lz,
            line: {{color: 'rgba(226, 60, 60, 0.55)', width: 3}},
            hoverinfo: 'skip', showlegend: true
          }};
        }} else {{
          linkTrace = {{
            type: 'scatter', mode: 'lines', name: QUERY_LINKS_NAME,
            x: lx, y: ly,
            line: {{color: 'rgba(226, 60, 60, 0.55)', width: 2}},
            hoverinfo: 'skip', showlegend: true
          }};
        }}
      }}

      // Numbered bubbles 1..K on each retrieved chunk so the user can
      // see exactly which 5 landed in the top-K (without having to read
      // the match list below).
      var chunkMarkerTrace = null;
      if (chunkXs.length > 0) {{
        if (kind === '3d') {{
          chunkMarkerTrace = {{
            type: 'scatter3d', mode: 'markers+text', name: QUERY_ZONE_NAME,
            x: chunkXs, y: chunkYs, z: chunkZs,
            marker: {{size: 20, color: '#fff4f4', symbol: 'circle', line: {{width: 3, color: '#e23c3c'}}}},
            text: chunkLabels, textposition: 'middle center',
            textfont: {{size: 13, color: '#b11616', family: 'system-ui, sans-serif'}},
            customdata: chunkHovers,
            hovertemplate: '%{{customdata}}<extra>' + QUERY_ZONE_NAME + '</extra>',
            showlegend: true
          }};
        }} else {{
          chunkMarkerTrace = {{
            type: 'scatter', mode: 'markers+text', name: QUERY_ZONE_NAME,
            x: chunkXs, y: chunkYs,
            marker: {{size: 32, color: '#fff4f4', symbol: 'circle', line: {{width: 3, color: '#e23c3c'}}}},
            text: chunkLabels, textposition: 'middle center',
            textfont: {{size: 16, color: '#b11616', family: 'system-ui, sans-serif'}},
            customdata: chunkHovers,
            hovertemplate: '%{{customdata}}<extra>' + QUERY_ZONE_NAME + '</extra>',
            showlegend: true
          }};
        }}
      }}

      // Compact star. Big enough to spot, small enough not to swallow
      // the numbered bubbles when the centroid sits inside a dense cluster.
      var starTrace = kind === '3d'
        ? {{
            x:[pos[0]], y:[pos[1]], z:[pos[2]],
            mode:'markers+text', type:'scatter3d',
            marker:{{size:14, color:'#e23c3c', symbol:'diamond', line:{{width:3, color:'#fff'}}}},
            text:['★ Ваш запрос'], textposition:'top center',
            textfont:{{size:13, color:'#c22020'}},
            name: QUERY_STAR_NAME,
            hovertemplate:'%{{text}}<extra></extra>'
          }}
        : {{
            x:[pos[0]], y:[pos[1]],
            mode:'markers+text', type:'scatter',
            marker:{{size:20, color:'#e23c3c', symbol:'star', line:{{width:2, color:'#fff'}}}},
            text:['★ Ваш запрос'], textposition:'top center',
            textfont:{{size:13, color:'#c22020'}},
            name: QUERY_STAR_NAME,
            hovertemplate:'%{{text}}<extra></extra>'
          }};

      // Replace old query layer atomically so indices stay sane.
      var toDelete = [];
      gd.data.forEach(function(t, i) {{
        if (!t || !t.name) return;
        if (
          t.name === QUERY_STAR_NAME ||
          t.name === QUERY_ZONE_NAME ||
          t.name === QUERY_LINKS_NAME
        ) toDelete.push(i);
      }});
      toDelete.reverse().forEach(function(i) {{ Plotly.deleteTraces(gd, i); }});
      var addList = [];
      if (linkTrace) addList.push(linkTrace);      // drawn first so it sits under markers
      if (chunkMarkerTrace) addList.push(chunkMarkerTrace);
      addList.push(starTrace);
      Plotly.addTraces(gd, addList);
      applyVisibilityFilter();
      status.textContent = 'Найдено ' + topK.length + ' чанков. Проверяйте по одному.';
      renderMatchList();
    }} catch (e) {{
      status.textContent = 'Overlay failed: ' + e.message;
    }}
  }};

  // Per-chunk feedback: a single click on a row's ✓/✗ sends one event
  // for that specific chunk. The feedback payload still carries the
  // same top_k shape (with one entry) so the coverage endpoint and the
  // JSONL-backed aggregation keep working unchanged.
  // signalType: 'content' | 'relevance'
  // content verdicts: 'correct' | 'wrong'
  // relevance verdicts: 'relevant' | 'not_relevant'
  //
  // Returns true on 2xx response (vote persisted), false otherwise. The
  // caller must check the return value and only mutate local review
  // state after a confirmed success — earlier versions always updated
  // state, which caused a 401/400/network failure to look like a
  // successful vote while nothing was logged server-side.
  async function submitChunkFeedback(match, signalType, verdict, comment, rowStatus) {{
    if (!lastQuery || !match) return false;
    rowStatus.textContent = 'Отправка...';
    try {{
      var payload = {{
        query_id: lastQuery.query_id,
        query_text: lastQuery.text,
        kind: lastQuery.kind,
        signal_type: signalType,
        verdict: verdict,
        top_k: [{{chunk_id: match.chunk_id, section: match.section || '', score: match.score}}],
      }};
      if (comment) payload.comment = comment;
      if (user) payload.client_id = user;
      var res = await fetch(feedbackUrl, {{method: 'POST', headers: authHeaders(), body: JSON.stringify(payload)}});
      if (!res.ok) {{
        var errText = '';
        try {{ errText = (await res.json()).detail || ''; }} catch (e) {{ /* body not json */ }}
        throw new Error('HTTP ' + res.status + (errText ? ' — ' + errText : ''));
      }}
      var labels = {{
        correct: '✓ записано',
        wrong: '✗ записано',
        relevant: '◯ подходит',
        not_relevant: '⊘ не подходит'
      }};
      rowStatus.textContent = labels[verdict] || 'записано';
      if (signalType === 'content') await refreshCoverage();
      return true;
    }} catch (e) {{
      rowStatus.textContent = 'Не записано: ' + e.message;
      return false;
    }}
  }}

  window.addEventListener('beforeunload', stopPolling);

  // ---- Demo bootstrap ----
  // When the HTML is rendered with a live snapshot baked in, pre-populate
  // lastQuery + lastCoverage from the snapshot, draw the full overlay
  // (tether lines, numbered top-K, verdict halos, per-user rings) with
  // no network calls, and disable the Go / vote buttons so the client
  // sees exactly what an interactive session looks like without needing
  // the server to be running.
  if (__KB_VIZ_DEMO__) {{
    // Normalize shape: accept either the new {{queries: [...]}} bundle or
    // the legacy single-snapshot {{top_k, query_text, query_id, ...}}.
    var demoQueries = (__KB_VIZ_DEMO__.queries && __KB_VIZ_DEMO__.queries.length > 0)
      ? __KB_VIZ_DEMO__.queries
      : (__KB_VIZ_DEMO__.top_k
          ? [{{
              query_id: __KB_VIZ_DEMO__.query_id || 'demo',
              query_text: __KB_VIZ_DEMO__.query_text || '',
              top_k: __KB_VIZ_DEMO__.top_k
            }}]
          : []);

    if (demoQueries.length > 0) {{
      // Pretend the panel is open and the user is already "logged in".
      panel.style.display = 'block';
      toggle.textContent = 'Demo · overlay preview';
      toggle.disabled = true;
      if (__KB_VIZ_DEMO__.user) {{
        user = __KB_VIZ_DEMO__.user;
        renderUserBadge();
      }}

      // Demo banner — replace the experimental label at the top of the bar.
      note.textContent = 'Демо-режим — данные заморожены, сервер отключён.';
      note.style.color = '#b3261e';
      note.style.fontWeight = '600';

      // Seed coverage once from the snapshot (drives verdict halos + per-user rings).
      lastCoverage = __KB_VIZ_DEMO__.coverage || {{per_chunk: {{}}, per_section: {{}}, per_user: {{}}, total_feedback: 0, unique_chunks_validated: 0}};

      // Helper: swap the query layer (tether lines, numbered top-K,
      // centroid star) for a given snapshot. Called on initial render
      // and every time the pill picker selects a different query.
      function _buildDemoQueryTraces(topK) {{
        var coords = buildChunkCoords();
        var cx = 0, cy = 0, cz = 0, n = 0;
        var xs = [], ys = [], zs = [], labels = [], hovers = [];
        topK.forEach(function(m, idx) {{
          var c = coords[m.chunk_id];
          if (!c) return;
          cx += c.x; cy += c.y; if (c.z !== undefined) cz += c.z;
          n += 1;
          xs.push(c.x); ys.push(c.y);
          if (c.z !== undefined) zs.push(c.z);
          labels.push(String(idx + 1));
          var s = (m.score != null) ? ' · score ' + m.score.toFixed(3) : '';
          hovers.push('#' + (idx + 1) + ' · ' + m.chunk_id + ' — ' + (m.section || '') + s);
        }});
        if (n === 0) return [];
        var pos = [cx / n, cy / n, kind === '3d' ? cz / n : undefined];
        var lx = [], ly = [], lz = [];
        for (var i = 0; i < xs.length; i++) {{
          lx.push(pos[0], xs[i], null);
          ly.push(pos[1], ys[i], null);
          if (kind === '3d') lz.push(pos[2], zs[i], null);
        }}
        var traces = [];
        if (kind === '3d') {{
          traces.push({{type:'scatter3d', mode:'lines', name: QUERY_LINKS_NAME,
            x: lx, y: ly, z: lz,
            line: {{color: 'rgba(226, 60, 60, 0.55)', width: 3}},
            hoverinfo: 'skip', showlegend: true}});
          traces.push({{type:'scatter3d', mode:'markers+text', name: QUERY_ZONE_NAME,
            x: xs, y: ys, z: zs,
            marker: {{size: 20, color: '#fff4f4', symbol: 'circle', line: {{width: 3, color: '#e23c3c'}}}},
            text: labels, textposition: 'middle center',
            textfont: {{size: 13, color: '#b11616', family: 'system-ui, sans-serif'}},
            customdata: hovers,
            hovertemplate: '%{{customdata}}<extra>' + QUERY_ZONE_NAME + '</extra>',
            showlegend: true}});
          traces.push({{x:[pos[0]], y:[pos[1]], z:[pos[2]],
            mode:'markers+text', type:'scatter3d',
            marker:{{size:14, color:'#e23c3c', symbol:'diamond', line:{{width:3, color:'#fff'}}}},
            text:['★ Ваш запрос'], textposition:'top center',
            textfont:{{size:13, color:'#c22020'}},
            name: QUERY_STAR_NAME, hovertemplate:'%{{text}}<extra></extra>'}});
        }} else {{
          traces.push({{type:'scatter', mode:'lines', name: QUERY_LINKS_NAME,
            x: lx, y: ly,
            line: {{color: 'rgba(226, 60, 60, 0.55)', width: 2}},
            hoverinfo: 'skip', showlegend: true}});
          traces.push({{type:'scatter', mode:'markers+text', name: QUERY_ZONE_NAME,
            x: xs, y: ys,
            marker: {{size: 32, color: '#fff4f4', symbol: 'circle', line: {{width: 3, color: '#e23c3c'}}}},
            text: labels, textposition: 'middle center',
            textfont: {{size: 16, color: '#b11616', family: 'system-ui, sans-serif'}},
            customdata: hovers,
            hovertemplate: '%{{customdata}}<extra>' + QUERY_ZONE_NAME + '</extra>',
            showlegend: true}});
          traces.push({{x:[pos[0]], y:[pos[1]],
            mode:'markers+text', type:'scatter',
            marker:{{size:20, color:'#e23c3c', symbol:'star', line:{{width:2, color:'#fff'}}}},
            text:['★ Ваш запрос'], textposition:'top center',
            textfont:{{size:13, color:'#c22020'}},
            name: QUERY_STAR_NAME, hovertemplate:'%{{text}}<extra></extra>'}});
        }}
        return traces;
      }}

      function applyDemoSnapshot(snap) {{
        lastQuery = {{query_id: snap.query_id || 'demo', text: snap.query_text || '', kind: kind, top_k: snap.top_k || []}};
        input.value = snap.query_text || '';
        resetReviewState();

        // Remove old query layer; add the new.
        var toDelete = [];
        gd.data.forEach(function(t, i) {{
          if (!t || !t.name) return;
          if (t.name === QUERY_STAR_NAME || t.name === QUERY_ZONE_NAME || t.name === QUERY_LINKS_NAME) {{
            toDelete.push(i);
          }}
        }});
        toDelete.reverse().forEach(function(i) {{ Plotly.deleteTraces(gd, i); }});
        var newTraces = _buildDemoQueryTraces(lastQuery.top_k);
        if (newTraces.length > 0) Plotly.addTraces(gd, newTraces);

        applyVisibilityFilter();
        status.textContent = 'Демо: показано ' + (lastQuery.top_k || []).length + ' чанков для примера запроса.';
        renderMatchList();
      }}

      // Query picker: one pill per baked query, highlighted when active.
      // Inserted between the input row and the UMAP note. Skipped when
      // only one query is baked in (nothing to switch between).
      if (demoQueries.length > 1) {{
        var pickerRow = el('div', {{style: 'margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;'}});
        pickerRow.appendChild(el('span', {{text: 'Пример вопроса:', style: 'font-size:12px;color:#555;margin-right:4px;'}}));
        var pills = [];
        demoQueries.forEach(function(snap, idx) {{
          var pill = el('button', {{
            text: (idx + 1) + '. ' + (snap.query_text || '(пустой запрос)'),
            style: 'padding:5px 10px;font-size:12px;cursor:pointer;border:1px solid #ccc;border-radius:14px;background:#fff;color:#234;max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
          }});
          pill.title = snap.query_text || '';
          pill.onclick = function() {{
            pills.forEach(function(p) {{
              p.style.background = '#fff';
              p.style.color = '#234';
              p.style.borderColor = '#ccc';
              p.style.fontWeight = '400';
            }});
            pill.style.background = '#1e66c8';
            pill.style.color = '#fff';
            pill.style.borderColor = '#1e66c8';
            pill.style.fontWeight = '600';
            applyDemoSnapshot(snap);
          }};
          pills.push(pill);
          pickerRow.appendChild(pill);
        }});
        // Insert the picker right after the input/ask/filter row — the
        // UI order becomes: input + Go + filter → picker → UMAP note → status.
        panel.insertBefore(pickerRow, umapNote);
        // Auto-activate the first pill.
        pills[0].click();
      }} else {{
        // Single query — just bootstrap directly.
        applyDemoSnapshot(demoQueries[0]);
      }}

      // Verdict halos + per-user rings are computed from coverage, which
      // is the same across all queries, so only rebuild once.
      rebuildCoverageTraces();
      applyVisibilityFilter();

      // Disable live-only actions: Go, Enter, vote buttons.
      ask.textContent = 'Демо · сервер отключён';
      ask.disabled = true;
      ask.style.background = '#eee';
      ask.style.color = '#888';
      ask.onclick = function() {{ /* no-op in demo mode */ }};
      input.readOnly = true;
      input.style.background = '#f8f8f8';
      submitChunkFeedback = async function(match, signalType, verdict, comment, rowStatus) {{
        if (rowStatus) rowStatus.textContent = 'Демо — голоса не сохраняются.';
        return false;
      }};
    }}
  }}
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
    demo_snapshot: dict[str, Any] | None = None,
) -> None:
    import pandas as pd
    import plotly.express as px

    assert kind in ("2d", "3d")
    is_3d = kind == "3d"

    color_groups = [r.get("color_group") or r["section"] for r in records]
    hover_customdata = _build_hover_customdata(records)
    hovertemplate = (
        "<b>%{customdata[1]}</b><br>"
        "%{customdata[0]}<br>"
        "<span style='color:#888;font-size:11px;'>%{customdata[3]}</span>"
        "<extra></extra>"
    )

    # Plotly Express splits one trace per unique color value. If we set
    # customdata via fig.update_traces() AFTER the split, the FULL array
    # ends up on every trace while per-trace x/y only hold that group's
    # rows, so hover shows the wrong chunk on any multi-section KB. The
    # correct path is to pass per-row customdata columns through the
    # dataframe and let px route them alongside x/y via custom_data=...
    frame_cols: dict[str, Any] = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "color_group": color_groups,
        "cd_text": [row[0] for row in hover_customdata],
        "cd_section": [row[1] for row in hover_customdata],
        "cd_doc": [row[2] for row in hover_customdata],
        "cd_chunk_id": [row[3] for row in hover_customdata],
    }
    if is_3d:
        frame_cols["z"] = coords[:, 2]
    df = pd.DataFrame(frame_cols)

    cd_cols = ["cd_text", "cd_section", "cd_doc", "cd_chunk_id"]
    if is_3d:
        fig = px.scatter_3d(
            df, x="x", y="y", z="z", color="color_group",
            custom_data=cd_cols, title=title, opacity=0.85,
        )
    else:
        fig = px.scatter(
            df, x="x", y="y", color="color_group",
            custom_data=cd_cols, title=title, opacity=0.85,
        )

    fig.update_traces(
        hovertemplate=hovertemplate,
        marker=dict(size=6 if is_3d else 8),
    )
    fig.update_layout(
        legend=dict(title="KB section"),
        margin=dict(l=30, r=30, t=60, b=30),
    )

    # Ground-truth chunk_id -> coord map for the overlay JS. Format:
    #   3D -> {chunk_id: [x, y, z, section]}
    #   2D -> {chunk_id: [x, y, section]}
    # Numpy floats are not JSON-serializable without coercion.
    chunk_coords: dict[str, list[Any]] = {}
    for idx, r in enumerate(records):
        cid = str(r["chunk_id"])
        section = r.get("section") or _DEFAULT_SECTION
        if is_3d:
            chunk_coords[cid] = [
                float(coords[idx, 0]), float(coords[idx, 1]), float(coords[idx, 2]), section,
            ]
        else:
            chunk_coords[cid] = [
                float(coords[idx, 0]), float(coords[idx, 1]), section,
            ]

    post_script = (
        _overlay_post_script(kind, overlay_url, overlay_token, chunk_coords, demo_snapshot)
        if overlay_url
        else None
    )
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
    demo_snapshot: dict[str, Any] | None = None,
    html_suffix: str = "",
) -> dict[str, Path]:
    """Render 2D/3D plots.

    ``demo_snapshot`` bakes a live query + coverage response into the
    emitted HTML so the overlay runs in read-only demo mode without a
    server. ``html_suffix`` appends to the output filenames (e.g. "_demo")
    so demo renders don't clobber the live-facing ``kb_viz_{2d,3d}.html``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_embeddings(embeddings_path)

    coords_2d, reducer_2d = _fit_umap(
        data.vectors, n_components=2, n_neighbors=UMAP_2D_N_NEIGHBORS, min_dist=UMAP_2D_MIN_DIST
    )
    out_2d = out_dir / f"kb_viz_2d{html_suffix}.html"
    _render_one(
        coords_2d, data.records, "2d", out_2d, TITLE_2D,
        overlay_url, overlay_token, demo_snapshot,
    )

    written = {"2d_html": out_2d}
    _save_reducer(reducer_2d, out_dir / "umap_2d.joblib")
    written["2d_reducer"] = out_dir / "umap_2d.joblib"

    if PROJECTION_3D:
        coords_3d, reducer_3d = _fit_umap(
            data.vectors, n_components=3, n_neighbors=UMAP_3D_N_NEIGHBORS, min_dist=UMAP_3D_MIN_DIST
        )
        out_3d = out_dir / f"kb_viz_3d{html_suffix}.html"
        _render_one(
            coords_3d, data.records, "3d", out_3d, TITLE_3D,
            overlay_url, overlay_token, demo_snapshot,
        )
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
    parser.add_argument(
        "--demo-snapshot",
        default=None,
        help=(
            "Path to a JSON file with {query_id, query_text, top_k, coverage, [user]} "
            "captured from a live server. Emits read-only demo HTMLs (vote buttons "
            "disabled, server never contacted). Implies --overlay-url; a placeholder "
            "is used if --overlay-url is not provided."
        ),
    )
    parser.add_argument(
        "--html-suffix",
        default="",
        help="Append this to output filenames (e.g. '_demo'). Keeps demo renders "
             "from overwriting the live kb_viz_{2d,3d}.html.",
    )
    args = parser.parse_args(argv)

    demo_snapshot = None
    if args.demo_snapshot:
        demo_snapshot = json.loads(Path(args.demo_snapshot).read_text(encoding="utf-8"))

    overlay_url = args.overlay_url
    # Demo mode needs an overlay script block to exist in the HTML;
    # the URL inside it is never called but the JS won't emit otherwise.
    if demo_snapshot and not overlay_url:
        overlay_url = "https://demo.invalid/overlay_query"

    out = render(
        embeddings_path=Path(args.in_path),
        out_dir=Path(args.out_dir),
        overlay_url=overlay_url,
        overlay_token=args.overlay_token,
        demo_snapshot=demo_snapshot,
        html_suffix=args.html_suffix,
    )
    for k, p in out.items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
