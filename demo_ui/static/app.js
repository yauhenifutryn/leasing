const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  session: null,
  runs: [],
  selectedRunId: null,
  logPoll: null,
  audio: [],
  audioSelected: new Set(),
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
  state.audioSelected = new Set();
  $("#sessionId").textContent = state.session.id.slice(0, 8);
  setBadge("ready", "Session active");
  $("#btnStop").disabled = true;
  $("#lastTask").textContent = "—";
  $("#lastStatus").textContent = "—";
  $("#btnSetSessionAudio").disabled = true;
  renderAudioList();
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

function formatBytes(n) {
  if (!Number.isFinite(n)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  const s = i === 0 ? String(Math.round(v)) : v.toFixed(1);
  return `${s} ${units[i]}`;
}

async function refreshAudio() {
  const resp = await api("/api/audio");
  state.audio = resp.files || [];
  renderAudioList();
}

function renderAudioList() {
  const list = $("#audioList");
  list.innerHTML = "";
  if (!state.audio.length) {
    list.innerHTML = `<div class="help" style="padding:8px;">No audio files in <code>audio/</code>.</div>`;
    $("#btnSetSessionAudio").disabled = true;
    return;
  }
  for (const f of state.audio) {
    const row = document.createElement("div");
    row.className = "item";
    const checked = state.audioSelected.has(f.name);
    row.innerHTML = `
      <div class="left">
        <input class="chk" type="checkbox" ${checked ? "checked" : ""} />
        <div class="name" title="${f.name}">${f.name}</div>
      </div>
      <div class="meta">${formatBytes(f.size)}</div>
    `;
    row.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) state.audioSelected.add(f.name);
      else state.audioSelected.delete(f.name);
      $("#btnSetSessionAudio").disabled = state.audioSelected.size === 0;
    });
    list.appendChild(row);
  }
  $("#btnSetSessionAudio").disabled = state.audioSelected.size === 0;
}

async function setSessionAudio() {
  if (!state.session) await startSession();
  const files = Array.from(state.audioSelected);
  await api("/api/session/audio", {
    method: "POST",
    body: JSON.stringify({ session_id: state.session.id, files }),
  });
  showModal("Audio selection saved", `<div class="help">Selected <b>${files.length}</b> file(s) for this session.</div>`);
}

async function uploadAudioFiles() {
  const input = $("#audioUpload");
  const files = Array.from(input.files || []);
  if (!files.length) return;

  const toB64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Failed to read file"));
      reader.onload = () => {
        const bytes = new Uint8Array(reader.result);
        let binary = "";
        const chunk = 0x8000;
        for (let i = 0; i < bytes.length; i += chunk) {
          binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
        }
        resolve(btoa(binary));
      };
      reader.readAsArrayBuffer(file);
    });

  $("#btnUploadAudio").disabled = true;
  try {
    const payload = [];
    for (const f of files) {
      payload.push({ name: f.name, data_base64: await toB64(f) });
    }
    await api("/api/audio/upload", { method: "POST", body: JSON.stringify({ files: payload }) });
    input.value = "";
    await refreshAudio();
  } finally {
    $("#btnUploadAudio").disabled = false;
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
  const w = 1100;
  const pad = 18;
  const gap = 12;
  const startY = 54;
  const labelW = 340;
  const barAreaW = w - 2 * pad - labelW - 80;

  const wrapLabel = (text, maxChars = 34) => {
    const s = String(text || "").trim();
    if (!s) return [""];
    const words = s.split(/\s+/);
    const lines = [];
    let cur = "";
    for (const word of words) {
      const next = (cur ? `${cur} ${word}` : word).trim();
      if (next.length > maxChars && cur) {
        lines.push(cur);
        cur = word;
      } else {
        cur = next;
      }
    }
    if (cur) lines.push(cur);
    if (lines.length <= 2) return lines;
    const firstTwo = lines.slice(0, 2);
    firstTwo[1] = firstTwo[1].replace(/\.*$/, "") + "…";
    return firstTwo;
  };

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  let y = startY;
  const rows = items.map((d) => {
    const lines = wrapLabel(d.label, 34);
    const rowH = Math.max(28, lines.length * 14 + 8);
    const barH = 16;
    const barY = y + Math.floor((rowH - barH) / 2);
    const bw = Math.round((d.value / max) * barAreaW);
    const texts = lines
      .map((line, idx) => {
        const ty = y + 14 + idx * 14;
        return `<text x="${pad}" y="${ty}" fill="rgba(242,244,248,0.78)" font-size="12" font-family="var(--sans)">${esc(line)}</text>`;
      })
      .join("");
    const out = `
      <title>${esc(d.label)}</title>
      ${texts}
      <rect x="${pad + labelW}" y="${barY}" width="${bw}" height="${barH}" rx="8" fill="rgba(0,123,255,0.65)"></rect>
      <text x="${pad + labelW + bw + 10}" y="${barY + 13}" fill="rgba(242,244,248,0.92)" font-size="12" font-weight="900" font-family="var(--sans)">${d.value}</text>
    `;
    y += rowH + gap;
    return out;
  });

  const viewH = Math.max(360, y + pad);
  return `
  <svg viewBox="0 0 ${w} ${viewH}" width="100%" height="${viewH}" preserveAspectRatio="xMinYMin meet">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#0a84ff" stop-opacity="0.9"/>
        <stop offset="1" stop-color="#00c853" stop-opacity="0.65"/>
      </linearGradient>
    </defs>
    <rect x="0" y="0" width="${w}" height="${viewH}" fill="rgba(255,255,255,0.01)"/>
    <text x="${pad}" y="30" fill="rgba(242,244,248,0.92)" font-size="14" font-weight="900" font-family="var(--sans)">${esc(title)}</text>
    ${rows.join("").replaceAll('rgba(0,123,255,0.65)', 'url(#g)')}
  </svg>`;
}

function svgDonut({ title, parts }) {
  const w = 640;
  const h = 360;
  const cx = 170;
  const cy = 200;
  const r = 96;
  const stroke = 22;
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
      const y = 92 + i * 30;
      return `
        <circle cx="420" cy="${y}" r="7" fill="${p.color}"></circle>
        <text x="436" y="${y + 4}" fill="rgba(242,244,248,0.86)" font-size="13" font-weight="900" font-family="var(--sans)">${p.label}: ${p.value}</text>
      `;
    })
    .join("");

  return `
  <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
    <text x="16" y="24" fill="rgba(242,244,248,0.92)" font-size="13" font-weight="900" font-family="var(--sans)">${title}</text>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="transparent" stroke="rgba(255,255,255,0.08)" stroke-width="${stroke}"></circle>
    ${circles}
    <text x="${cx}" y="${cy}" text-anchor="middle" fill="rgba(242,244,248,0.92)" font-size="26" font-weight="900" font-family="var(--sans)">${Math.round((parts[0]?.value || 0) / total * 100)}%</text>
    <text x="${cx}" y="${cy + 26}" text-anchor="middle" fill="rgba(170,176,187,0.9)" font-size="12" font-family="var(--sans)">resolved</text>
    ${legend}
  </svg>`;
}

async function computeMetrics() {
  const resp = await api("/api/metrics");
  const metrics = resp.metrics || {};

  const topIntents = metrics.top_reasons?.length ? metrics.top_reasons.slice(0, 10) : [{ label: "No data yet", value: 0 }];

  $("#chartTopIntents").innerHTML = svgBarChart({
    title: "Top reasons (global intents)",
    data: topIntents,
    maxBars: 10,
  });

  const rc = metrics.resolution || { resolved: 0, partial: 0, unresolved: 0, unknown: 0 };
  const resolved = rc.resolved || 0;
  $("#chartResolution").innerHTML = svgDonut({
    title: "Solved vs Unresolved",
    parts: [
      { label: "resolved", value: resolved, color: "rgba(0,200,83,0.9)" },
      { label: "partial", value: rc.partial || 0, color: "rgba(255,179,0,0.85)" },
      { label: "unresolved", value: rc.unresolved || 0, color: "rgba(211,47,47,0.9)" },
    ],
  });

  const unresolvedTop = metrics.unresolved_reasons_top?.length
    ? metrics.unresolved_reasons_top
    : [{ label: "No data yet", value: 0 }];
  $("#chartUnresolved").innerHTML = svgBarChart({
    title: "Unresolved drivers (top)",
    data: unresolvedTop,
    maxBars: 8,
  });

  const qualityTop = metrics.quality_flags_top?.length ? metrics.quality_flags_top : [{ label: "No data yet", value: 0 }];
  $("#chartQuality").innerHTML = svgBarChart({
    title: "Quality flags (top)",
    data: qualityTop,
  });

  const emoTop = metrics.emotions_top?.length ? metrics.emotions_top : [{ label: "No data yet", value: 0 }];
  $("#chartEmotion").innerHTML = svgBarChart({
    title: "Client emotion (top)",
    data: emoTop,
  });
}

async function showSettings() {
  const resp = await api("/api/env");
  const env = resp.env || {};
  const rows = Object.entries(env)
    .map(([k, v]) => {
      const val = v === "" ? "<span class=\"muted\">(empty)</span>" : String(v);
      return `<div class="item"><span class="k">${k}</span><span class="v">${val}</span></div>`;
    })
    .join("");
  showModal("Settings", `
    <div class="help">Loaded from <code>.env</code> (if present) and/or exported env vars. Tokens are never displayed.</div>
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
  $("#btnRefreshAudio").addEventListener("click", () => refreshAudio().catch(alertError));
  $("#btnUploadAudio").addEventListener("click", () => uploadAudioFiles().catch(alertError));
  $("#btnSetSessionAudio").addEventListener("click", () => setSessionAudio().catch(alertError));

  $$("#tab_pipeline [data-task]").forEach((btn) => {
    btn.addEventListener("click", () => runTask(btn.dataset.task).catch(alertError));
  });

  $("#jsonKind").addEventListener("change", async () => {
    await refreshJsonFiles();
    await loadSelectedJson();
  });
  $("#btnLoadJson").addEventListener("click", () => loadSelectedJson().catch(alertError));

  await refreshJsonFiles();
  await refreshAudio();
  await computeMetrics();
}

function alertError(err) {
  console.error(err);
  showModal("Error", `<div class="help">${String(err.message || err)}</div>`);
}

main().catch(alertError);
