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


def _overlay_post_script(kind: str, embed_url: str, token: str | None) -> str:
    """Inject the overlay UI: query input, feedback buttons, coverage panel.

    Uses DOM API only (createElement + textContent + appendChild) so response
    data from the server is never interpolated as HTML. No XSS surface.
    Coverage polls every COVERAGE_POLL_MS while the panel is open, so
    concurrent users see each other's validation within that interval.
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
    return f"""
(function() {{
  var embedUrl = {url_js};
  var token = {token_js};
  var kind = {kind_js};
  var pollMs = {poll_ms_js};
  var VALIDATED_NAME = {validated_name_js};
  var FLAGGED_NAME = {flagged_name_js};
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
  var ask = el('button', {{text: 'Project', style: 'padding:6px 10px;margin-left:6px;cursor:pointer;'}});
  var status = el('div', {{style: 'margin-top:6px;color:#666;'}});
  var matches = el('div', {{style: 'margin-top:8px;color:#222;'}});
  var fbStatus = el('div', {{style: 'margin-top:6px;color:#666;font-size:12px;'}});
  var coverage = el('div', {{style: 'margin-top:10px;padding:8px;background:#f4f8ff;border:1px solid #cfe0f5;font-size:12px;color:#234;'}});
  panel.appendChild(input);
  panel.appendChild(ask);
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

  // ---- Build chunk_id -> coords map from baseline traces once ----
  function buildChunkCoords() {{
    if (chunkCoords !== null) return chunkCoords;
    chunkCoords = {{}};
    (gd.data || []).forEach(function(trace) {{
      if (!trace || !trace.customdata) return;
      if (trace.name === VALIDATED_NAME || trace.name === FLAGGED_NAME || trace.name === 'Query') return;
      var xs = trace.x || [], ys = trace.y || [], zs = trace.z || [];
      trace.customdata.forEach(function(cd, idx) {{
        // customdata layout comes from _build_hover_customdata:
        //   [text_preview, section, doc_name, chunk_id]
        if (!cd || cd.length < 4) return;
        var cid = String(cd[3]);
        chunkCoords[cid] = {{x: xs[idx], y: ys[idx], z: zs ? zs[idx] : undefined, section: cd[1]}};
      }});
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

      var cXs = [], cYs = [], cZs = [], cTexts = [];
      (u.correct_chunks || []).forEach(function(cid) {{
        var c = coords[cid];
        if (!c) return;
        cXs.push(c.x); cYs.push(c.y); if (c.z !== undefined) cZs.push(c.z);
        var cov = perChunk[cid] || {{}};
        cTexts.push(cid + ' — ' + (c.section || '') + ' — ' + (cov.correct || 0) + '✓/' + (cov.wrong || 0) + '✗');
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
        var cov = perChunk[cid] || {{}};
        wTexts.push(cid + ' — ' + (c.section || '') + ' — ' + (cov.correct || 0) + '✓/' + (cov.wrong || 0) + '✗');
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
    }});
    toDelete.reverse().forEach(function(i) {{ Plotly.deleteTraces(gd, i); }});
    if (newTraces.length > 0) Plotly.addTraces(gd, newTraces);
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
      if (trace.name === 'Query' || trace.name === VALIDATED_NAME || trace.name === FLAGGED_NAME) return;
      if (trace.name.indexOf('✓ by ') === 0 || trace.name.indexOf('✗ by ') === 0) return;
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

  // Per-chunk feedback: each row collects two orthogonal signals.
  //   • Точность (content):  "is this chunk's text factually correct?"
  //     — mandatory signal, drives the coverage icons.
  //   • Релевантность (relevance): "is this chunk a good answer to the
  //     query?" — optional secondary signal, logged for later analysis,
  //     does not change coverage.
  // Clients are encouraged to skip chunks they can't confidently judge.
  function renderMatchList() {{
    matches.textContent = '';
    if (!lastQuery) return;

    var instruction = el('div', {{style: 'margin-bottom:8px;padding:8px 10px;background:#fff8e1;border-left:3px solid #e0b84a;color:#4a3d14;font-size:12px;line-height:1.5;'}});
    instruction.appendChild(el('div', {{
      text: 'Как проверять:',
      style: 'font-weight:600;margin-bottom:4px;'
    }}));
    instruction.appendChild(el('div', {{
      text: '• Точность (главное) — верны ли факты в тексте чанка: цифры, телефоны, условия. ✓ если всё верно, ✗ если есть ошибка (нужен комментарий).'
    }}));
    instruction.appendChild(el('div', {{
      text: '• Релевантность (по желанию) — подходит ли этот чанк как ответ на ваш вопрос. Не обязательно, но полезно для будущего анализа поиска.'
    }}));
    instruction.appendChild(el('div', {{
      text: '• Пропустите чанк, если не уверены. Не голосуйте просто так.',
      style: 'margin-top:4px;color:#6d5210;'
    }}));
    matches.appendChild(instruction);

    lastQuery.top_k.forEach(function(m, i) {{
      var cov = (lastCoverage.per_chunk || {{}})[m.chunk_id];
      var icon = coverageIcon(cov);
      var counts = cov ? cov.correct + '✓ / ' + cov.wrong + '✗' : 'ещё не проверялось';
      var score = (m.score != null) ? m.score.toFixed(3) : '?';

      var row = el('div', {{
        style: 'margin-bottom:6px;padding:8px 10px;background:#fff;border:1px solid #e0e0e0;border-radius:4px;font-size:13px;'
      }});

      // Header line with rank, score, section, coverage counts
      var head = el('div', {{
        style: 'display:flex;align-items:center;gap:8px;margin-bottom:4px;'
      }});
      head.appendChild(el('span', {{text: icon + ' #' + (i + 1), style: 'font-weight:600;min-width:38px;'}}));
      head.appendChild(el('span', {{text: score, style: 'color:#888;font-size:11px;min-width:42px;'}}));
      head.appendChild(el('span', {{text: m.section || '(без раздела)', style: 'color:#223;flex:1;'}}));
      head.appendChild(el('span', {{text: counts, style: 'color:#555;font-size:11px;'}}));
      row.appendChild(head);

      // Preview text with click-to-expand
      var preview = m.text_preview || '';
      var full = m.text_full || preview;
      var textBox = el('div', {{
        text: preview,
        style: 'color:#333;margin:4px 0 6px 0;line-height:1.45;max-height:3.8em;overflow:hidden;white-space:pre-wrap;cursor:pointer;'
      }});
      var expanded = false;
      textBox.addEventListener('click', function() {{
        expanded = !expanded;
        textBox.textContent = expanded ? full : preview;
        textBox.style.maxHeight = expanded ? 'none' : '3.8em';
      }});
      row.appendChild(textBox);

      // --- Content accuracy buttons (primary signal) ---
      var contentRow = el('div', {{style: 'display:flex;gap:6px;align-items:center;margin-top:4px;'}});
      contentRow.appendChild(el('span', {{text: 'Точность:', style: 'font-size:11px;color:#555;min-width:80px;'}}));
      var ok = el('button', {{
        text: '✓ Верно', style: 'padding:4px 10px;cursor:pointer;background:#e7f7e7;border:1px solid #7bc97b;border-radius:3px;font-size:12px;'
      }});
      var no = el('button', {{
        text: '✗ Ошибка', style: 'padding:4px 10px;cursor:pointer;background:#fbeaea;border:1px solid #d98080;border-radius:3px;font-size:12px;'
      }});
      var rowStatus = el('span', {{style: 'color:#666;font-size:11px;margin-left:4px;'}});
      contentRow.appendChild(ok);
      contentRow.appendChild(no);
      contentRow.appendChild(rowStatus);
      row.appendChild(contentRow);

      // --- Relevance buttons (optional secondary signal) ---
      var relRow = el('div', {{style: 'display:flex;gap:6px;align-items:center;margin-top:4px;'}});
      relRow.appendChild(el('span', {{text: 'Релевантность:', style: 'font-size:11px;color:#555;min-width:110px;'}}));
      var rel = el('button', {{
        text: '◯ Подходит', style: 'padding:4px 10px;cursor:pointer;background:#eef3fb;border:1px solid #7aa4d4;border-radius:3px;font-size:12px;'
      }});
      var notRel = el('button', {{
        text: '⊘ Не подходит', style: 'padding:4px 10px;cursor:pointer;background:#f5f0f8;border:1px solid #a989c2;border-radius:3px;font-size:12px;'
      }});
      var relStatus = el('span', {{style: 'color:#666;font-size:11px;margin-left:4px;'}});
      relRow.appendChild(rel);
      relRow.appendChild(notRel);
      relRow.appendChild(relStatus);
      row.appendChild(relRow);

      // Lazy comment input, appears only on ✗ Ошибка
      var commentBox = el('div', {{style: 'display:none;margin-top:6px;'}});
      var ta = el('textarea', {{
        placeholder: 'Что именно неверно в этом чанке? (обязательно)',
        style: 'width:100%;min-height:50px;padding:6px;box-sizing:border-box;font-size:12px;'
      }});
      var send = el('button', {{
        text: 'Отправить', style: 'padding:4px 10px;cursor:pointer;margin-top:4px;font-size:12px;'
      }});
      commentBox.appendChild(ta);
      commentBox.appendChild(send);
      row.appendChild(commentBox);

      ok.onclick = function() {{
        submitChunkFeedback(m, 'content', 'correct', null, rowStatus);
      }};
      no.onclick = function() {{
        commentBox.style.display = 'block';
        ta.focus();
      }};
      send.onclick = function() {{
        var c = ta.value.trim();
        if (!c) {{ rowStatus.textContent = 'Нужен комментарий.'; return; }}
        submitChunkFeedback(m, 'content', 'wrong', c, rowStatus);
        commentBox.style.display = 'none';
      }};
      rel.onclick = function() {{
        submitChunkFeedback(m, 'relevance', 'relevant', null, relStatus);
      }};
      notRel.onclick = function() {{
        submitChunkFeedback(m, 'relevance', 'not_relevant', null, relStatus);
      }};

      matches.appendChild(row);
    }});
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

  ask.onclick = async function() {{
    var q = input.value.trim();
    if (!q) return;
    status.textContent = 'Embedding + projecting...';
    matches.textContent = '';
    fbStatus.textContent = '';
    try {{
      var body = {{text: q, kind: kind, top_k: 5}};
      if (user) body.client_id = user;
      var res = await fetch(embedUrl, {{method: 'POST', headers: authHeaders(), body: JSON.stringify(body)}});
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
      status.textContent = 'Top ' + topK.length + ' matches. (icons: ✓ validated correct, ✗ flagged wrong, ± mixed, ? unvalidated — cumulative across all users)';
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
  async function submitChunkFeedback(match, signalType, verdict, comment, rowStatus) {{
    if (!lastQuery || !match) return;
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
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var labels = {{
        correct: '✓ записано',
        wrong: '✗ записано',
        relevant: '◯ подходит',
        not_relevant: '⊘ не подходит'
      }};
      rowStatus.textContent = labels[verdict] || 'записано';
      if (signalType === 'content') await refreshCoverage();
    }} catch (e) {{
      rowStatus.textContent = 'Ошибка: ' + e.message;
    }}
  }}

  window.addEventListener('beforeunload', stopPolling);
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
    # Intentionally compact: Plotly tooltips clip off-screen on very wide
    # content, so the hover shows just enough to identify the chunk
    # (section + short preview + id). Full text lives in the match list
    # below the plot, which supports click-to-expand.
    hovertemplate = (
        "<b>%{customdata[1]}</b><br>"
        "%{customdata[0]}<br>"
        "<span style='color:#888;font-size:11px;'>%{customdata[3]}</span>"
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
