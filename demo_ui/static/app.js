const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  session: null,
  runs: [],
  selectedRunId: null,
  logPoll: null,
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = `${res.status}`;
    try {
      const data = await res.json();
      msg = data?.error || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function setBadge(status, text) {
  const badge = $("#sessionBadge");
  const dot = badge.querySelector(".dot");
  dot.classList.remove("good", "danger", "warn");
  if (status === "ready") dot.classList.add("good");
  else if (status === "running") dot.classList.add("warn");
  else if (status === "error") dot.classList.add("danger");
  else dot.classList.add("warn");
  $("#sessionBadgeText").textContent = text;
}

function formatStatus(run) {
  if (!run) return { text: "—", dot: "warn" };
  const map = {
    queued: ["Queued", "warn"],
    running: ["Running", "warn"],
    success: ["Success", "good"],
    failed: ["Failed", "danger"],
    stopped: ["Stopped", "danger"],
  };
  const [text, dot] = map[run.status] || [run.status, "warn"];
  return { text, dot };
}

function mountTabs() {
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      ["pipeline", "logs", "metrics", "json"].forEach((name) => {
        const el = $(`#tab_${name}`);
        el.style.display = name === tab ? "block" : "none";
      });
    });
  });
}

async function startSession() {
  const resp = await api("/api/session", {
    method: "POST",
    body: JSON.stringify({ name: "" }),
  });
  state.session = resp.session;
  $("#sessionId").textContent = state.session.id.slice(0, 8);
  setBadge("ready", "Session active");
  $("#btnStop").disabled = true;
  $("#lastTask").textContent = "—";
  $("#lastStatus").textContent = "—";
  await refreshRuns();
}

async function refreshRuns() {
  if (!state.session) return;
  const resp = await api(`/api/runs?session_id=${encodeURIComponent(state.session.id)}`);
  state.runs = resp.runs;
  renderRunsList();
}

function renderRunsList() {
  const list = $("#runsList");
  list.innerHTML = "";
  if (!state.runs.length) {
    list.innerHTML = `<div class="help">No runs yet.</div>`;
    return;
  }
  for (const run of state.runs.slice(0, 20)) {
    const { text } = formatStatus(run);
    const row = document.createElement("div");
    row.className = "item";
    row.style.cursor = "pointer";
    row.innerHTML = `
      <span class="k">${run.task}</span>
      <span class="v">${text}</span>
    `;
    row.addEventListener("click", async () => {
      state.selectedRunId = run.id;
      await loadLog(run.id, "#logView");
    });
    list.appendChild(row);
  }
}

async function loadLog(runId, targetSelector = "#liveLog") {
  const resp = await api(`/api/log?run_id=${encodeURIComponent(runId)}&limit=400`);
  const view = $(targetSelector);
  view.textContent = resp.lines.join("\n") || "(empty)";
  view.scrollTop = view.scrollHeight;
}

async function runTask(task) {
  if (!state.session) {
    await startSession();
  }
  setBadge("running", "Running task…");
  const resp = await api("/api/run", {
    method: "POST",
    body: JSON.stringify({ session_id: state.session.id, task }),
  });
  const run = resp.run;
  state.selectedRunId = run.id;
  $("#lastTask").textContent = task;
  $("#lastStatus").textContent = "Running";
  $("#btnStop").disabled = false;

  if (state.logPoll) clearInterval(state.logPoll);
  state.logPoll = setInterval(async () => {
    try {
      await refreshRuns();
      const current = state.runs.find((r) => r.id === run.id);
      if (current) {
        const st = formatStatus(current).text;
        $("#lastStatus").textContent = st;
        await loadLog(run.id, "#liveLog");
        if (["success", "failed", "stopped"].includes(current.status)) {
          clearInterval(state.logPoll);
          state.logPoll = null;
          $("#btnStop").disabled = true;
          setBadge(current.status === "success" ? "ready" : "error", `Last: ${st}`);
        }
      }
    } catch {
      // ignore transient errors
    }
  }, 1200);
}

async function stopCurrent() {
  if (!state.selectedRunId) return;
  await api("/api/stop", { method: "POST", body: JSON.stringify({ run_id: state.selectedRunId }) });
  $("#btnStop").disabled = true;
}

function prettyJson(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

async function refreshJsonFiles() {
  const kind = $("#jsonKind").value;
  const resp = await api(`/api/files?kind=${encodeURIComponent(kind)}`);
  const sel = $("#jsonFile");
  sel.innerHTML = "";
  for (const name of resp.files) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  }
}

async function loadSelectedJson() {
  const kind = $("#jsonKind").value;
  const name = $("#jsonFile").value;
  if (!name) return;
  const resp = await api(`/api/file?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(name)}`);
  $("#jsonView").textContent = prettyJson(resp.text);
}

function svgBarChart({ title, data, maxBars = 8 }) {
  const items = [...data].slice(0, maxBars);
  const max = Math.max(1, ...items.map((d) => d.value));
  const w = 1000;
  const h = 260;
  const pad = 18;
  const barH = 18;
  const gap = 10;
  const startY = 44;
  const barAreaW = w - 2 * pad - 220;
  const labelW = 210;
  const rowsH = items.length * (barH + gap);
  const viewH = Math.max(h, startY + rowsH + pad);

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const bars = items
    .map((d, i) => {
      const y = startY + i * (barH + gap);
      const bw = Math.round((d.value / max) * barAreaW);
      return `
        <text x="${pad}" y="${y + 13}" fill="rgba(242,244,248,0.78)" font-size="12" font-family="var(--sans)">${esc(d.label)}</text>
        <rect x="${pad + labelW}" y="${y}" width="${bw}" height="${barH}" rx="8" fill="rgba(0,123,255,0.65)"></rect>
        <text x="${pad + labelW + bw + 8}" y="${y + 13}" fill="rgba(242,244,248,0.9)" font-size="12" font-weight="800" font-family="var(--sans)">${d.value}</text>
      `;
    })
    .join("");

  return `
  <svg viewBox="0 0 ${w} ${viewH}" width="100%" height="100%" preserveAspectRatio="none">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#0a84ff" stop-opacity="0.9"/>
        <stop offset="1" stop-color="#00c853" stop-opacity="0.65"/>
      </linearGradient>
    </defs>
    <rect x="0" y="0" width="${w}" height="${viewH}" fill="rgba(255,255,255,0.01)"/>
    <text x="${pad}" y="24" fill="rgba(242,244,248,0.92)" font-size="13" font-weight="900" font-family="var(--sans)">${esc(title)}</text>
    ${bars.replaceAll('rgba(0,123,255,0.65)', 'url(#g)')}
  </svg>`;
}

function svgDonut({ title, parts }) {
  const w = 520;
  const h = 260;
  const cx = 130;
  const cy = 140;
  const r = 74;
  const stroke = 18;
  const total = parts.reduce((a, b) => a + b.value, 0) || 1;

  let offset = 0;
  const circles = parts
    .map((p) => {
      const pct = p.value / total;
      const dash = 2 * Math.PI * r;
      const len = dash * pct;
      const gap = dash - len;
      const c = `
        <circle cx="${cx}" cy="${cy}" r="${r}" fill="transparent" stroke="${p.color}" stroke-width="${stroke}"
          stroke-dasharray="${len} ${gap}" stroke-dashoffset="${-offset}" stroke-linecap="round"></circle>
      `;
      offset += len;
      return c;
    })
    .join("");

  const legend = parts
    .map((p, i) => {
      const y = 70 + i * 26;
      return `
        <circle cx="310" cy="${y}" r="6" fill="${p.color}"></circle>
        <text x="324" y="${y + 4}" fill="rgba(242,244,248,0.86)" font-size="12" font-weight="800" font-family="var(--sans)">${p.label}: ${p.value}</text>
      `;
    })
    .join("");

  return `
  <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
    <text x="16" y="24" fill="rgba(242,244,248,0.92)" font-size="13" font-weight="900" font-family="var(--sans)">${title}</text>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="transparent" stroke="rgba(255,255,255,0.08)" stroke-width="${stroke}"></circle>
    ${circles}
    <text x="${cx}" y="${cy}" text-anchor="middle" fill="rgba(242,244,248,0.92)" font-size="22" font-weight="900" font-family="var(--sans)">${Math.round((parts[0]?.value || 0) / total * 100)}%</text>
    <text x="${cx}" y="${cy + 22}" text-anchor="middle" fill="rgba(170,176,187,0.9)" font-size="11" font-family="var(--sans)">resolved</text>
    ${legend}
  </svg>`;
}

async function computeMetrics() {
  // Metrics are computed client-side from repo outputs via /api/file endpoints.
  const intents = await safeLoadJson("insights_global", "global_top_intents.json");
  const perCall = await safeListAndLoadMany("insights_per_call", 200);

  const topIntents = Array.isArray(intents)
    ? intents.slice(0, 8).map((i) => ({ label: i.intent, value: i.count || 0 }))
    : [];

  // Resolution rate
  const resolutionCounts = { resolved: 0, partially_resolved: 0, unresolved: 0 };
  const qualityCounts = new Map();
  const emotionCounts = new Map();
  for (const item of perCall) {
    const status = item?.resolution_status;
    if (status && status in resolutionCounts) resolutionCounts[status] += 1;
    const flags = item?.quality_flags || [];
    for (const f of flags) qualityCounts.set(f, (qualityCounts.get(f) || 0) + 1);
    const emo = item?.emotions?.client;
    if (emo) emotionCounts.set(emo, (emotionCounts.get(emo) || 0) + 1);
  }

  const qualityTop = Array.from(qualityCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([label, value]) => ({ label, value }));

  const emoTop = Array.from(emotionCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([label, value]) => ({ label, value }));

  $("#chartTopIntents").innerHTML = svgBarChart({
    title: "Top reasons (global intents)",
    data: topIntents.length ? topIntents : [{ label: "No data yet", value: 0 }],
  });

  const resolved = resolutionCounts.resolved;
  const total = resolved + resolutionCounts.partially_resolved + resolutionCounts.unresolved;
  $("#chartResolution").innerHTML = svgDonut({
    title: "Solved vs Unresolved",
    parts: [
      { label: "resolved", value: resolved, color: "rgba(0,200,83,0.9)" },
      { label: "partial", value: resolutionCounts.partially_resolved, color: "rgba(255,179,0,0.85)" },
      { label: "unresolved", value: resolutionCounts.unresolved, color: "rgba(211,47,47,0.9)" },
    ],
  });

  $("#chartQuality").innerHTML = svgBarChart({
    title: "Quality flags (top)",
    data: qualityTop.length ? qualityTop : [{ label: "No data yet", value: 0 }],
  });

  $("#chartEmotion").innerHTML = svgBarChart({
    title: "Client emotion (top)",
    data: emoTop.length ? emoTop : [{ label: "No data yet", value: 0 }],
  });
}

async function safeLoadJson(kind, name) {
  try {
    const resp = await api(`/api/file?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(name)}`);
    return JSON.parse(resp.text);
  } catch {
    return null;
  }
}

async function safeListAndLoadMany(kind, limit = 200) {
  try {
    const resp = await api(`/api/files?kind=${encodeURIComponent(kind)}`);
    const names = resp.files.slice(0, limit);
    const out = [];
    for (const n of names) {
      try {
        const f = await api(`/api/file?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(n)}`);
        out.push(JSON.parse(f.text));
      } catch {}
    }
    return out;
  } catch {
    return [];
  }
}

async function showSettings() {
  const resp = await api("/api/env");
  const env = resp.env || {};
  const rows = Object.entries(env)
    .map(([k, v]) => `<div class="item"><span class="k">${k}</span><span class="v">${v}</span></div>`)
    .join("");
  showModal("Settings", `
    <div class="help">Values are read from environment variables. Set them in your shell or .env before running tasks.</div>
    <div class="kv" style="margin-top:12px;">${rows}</div>
  `);
}

async function showFeedback() {
  showModal("Submit Feedback", `
    <div class="help">Feedback is stored locally in <code>demo_ui/.state/feedback.jsonl</code> (gitignored).</div>
    <div style="margin-top:12px;">
      <textarea id="feedbackText" placeholder="What worked? What felt confusing? What should be improved?"></textarea>
      <div class="actions" style="margin-top:10px; justify-content:flex-end;">
        <button class="btn ghost" id="btnCloseModal">Cancel</button>
        <button class="btn primary" id="btnSendFeedback">Send</button>
      </div>
    </div>
  `);
  $("#btnSendFeedback").addEventListener("click", async () => {
    const message = $("#feedbackText").value.trim();
    if (!message) return;
    await api("/api/feedback", {
      method: "POST",
      body: JSON.stringify({ session_id: state.session?.id || null, message }),
    });
    closeModal();
  });
  $("#btnCloseModal").addEventListener("click", closeModal);
}

function showModal(title, html) {
  const modal = $("#modal");
  modal.style.display = "block";
  modal.innerHTML = `
    <div style="position:fixed; inset:0; background:rgba(0,0,0,0.55); z-index:9999; display:flex; align-items:center; justify-content:center; padding:16px;">
      <div class="card" style="width:min(760px, 96vw); background:var(--card);">
        <div class="row" style="margin-bottom:10px;">
          <div style="font-weight:900; letter-spacing:0.3px;">${title}</div>
          <button class="btn ghost" id="btnModalX">Close</button>
        </div>
        ${html}
      </div>
    </div>
  `;
  $("#btnModalX").addEventListener("click", closeModal);
}

function closeModal() {
  const modal = $("#modal");
  modal.style.display = "none";
  modal.innerHTML = "";
}

async function main() {
  mountTabs();
  setBadge("idle", "No session");

  $("#btnNewSession").addEventListener("click", () => startSession().catch(alertError));
  $("#btnStop").addEventListener("click", () => stopCurrent().catch(alertError));
  $("#btnRefreshRuns").addEventListener("click", () => refreshRuns().catch(alertError));
  $("#btnRefreshMetrics").addEventListener("click", () => computeMetrics().catch(alertError));
  $("#btnSettings").addEventListener("click", () => showSettings().catch(alertError));
  $("#btnFeedback").addEventListener("click", () => showFeedback().catch(alertError));

  $$("#tab_pipeline [data-task]").forEach((btn) => {
    btn.addEventListener("click", () => runTask(btn.dataset.task).catch(alertError));
  });

  $("#jsonKind").addEventListener("change", async () => {
    await refreshJsonFiles();
    await loadSelectedJson();
  });
  $("#btnLoadJson").addEventListener("click", () => loadSelectedJson().catch(alertError));

  await refreshJsonFiles();
  await computeMetrics();
}

function alertError(err) {
  console.error(err);
  showModal("Error", `<div class="help">${String(err.message || err)}</div>`);
}

main().catch(alertError);

