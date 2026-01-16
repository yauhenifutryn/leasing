const API_BASE = "http://127.0.0.1:8000";

const $ = (sel) => document.querySelector(sel);

let sessionId = null;
let chunksCollapsed = false;
const STREAMING = true;

function setStatus(text, level = "warn") {
  const badge = $("#statusBadge");
  const dot = badge.querySelector(".dot");
  dot.classList.remove("good", "danger", "warn");
  dot.classList.add(level);
  $("#statusText").textContent = text;
}

function renderMessage(role, text) {
  const row = document.createElement("div");
  row.className = "message";
  row.innerHTML = `<div class="role">${role}</div><div class="text">${text}</div>`;
  $("#chatWindow").appendChild(row);
  $("#chatWindow").scrollTop = $("#chatWindow").scrollHeight;
  return row.querySelector(".text");
}

function renderChunks(chunks) {
  const panel = $("#chunksPanel");
  panel.innerHTML = "";
  if (!chunks || chunks.length === 0) {
    panel.innerHTML = `<div class="help">Нет использованных фрагментов</div>`;
    return;
  }
  for (const c of chunks) {
    const title = (c.heading_path || []).join(" / ") || c.doc_name || "Без раздела";
    const el = document.createElement("div");
    el.className = "chunk";
    el.innerHTML = `
      <div class="meta">${title}</div>
      <div class="meta">chunk_id: ${c.chunk_id}</div>
      <div class="text">${c.snippet}</div>
    `;
    panel.appendChild(el);
  }
}

async function api(path, opts = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json();
  if (!res.ok || data?.ok === false) {
    throw new Error(data?.error || res.status);
  }
  return data;
}

async function sendMessage() {
  const input = $("#chatInput");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  renderMessage("user", message);
  setStatus("Thinking", "warn");

  const payload = { message, session_id: sessionId, stream: STREAMING };
  if (!STREAMING) {
    const resp = await api("/api/chat", { method: "POST", body: JSON.stringify(payload) });
    sessionId = resp.session_id;
    renderMessage("agent", resp.answer);
    renderChunks(resp.used_knowledge || []);
    setStatus("Idle", "good");
    return;
  }

  const agentTextEl = renderMessage("agent", "");
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) {
    throw new Error("Stream failed");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const jsonText = line.slice(5).trim();
      if (!jsonText) continue;
      const evt = JSON.parse(jsonText);
      if (evt.type === "delta") {
        answer += evt.text || "";
        agentTextEl.textContent = answer;
      } else if (evt.type === "final") {
        sessionId = evt.session_id;
        agentTextEl.textContent = evt.answer || answer;
        renderChunks(evt.used_knowledge || []);
        setStatus("Idle", "good");
      }
    }
  }
}

async function indexKb() {
  setStatus("Indexing", "warn");
  await api("/api/index", { method: "POST", body: JSON.stringify({ rebuild: false }) });
  setStatus("Indexed", "good");
}

async function refreshLogs() {
  const resp = await api("/api/logs?limit=200");
  const items = resp.items || [];
  const logPanel = $("#logPanel");
  logPanel.textContent = items.map((i) => JSON.stringify(i)).join("\n") || "(empty)";
}

function toggleChunks() {
  chunksCollapsed = !chunksCollapsed;
  $("#chunksPanel").style.display = chunksCollapsed ? "none" : "block";
  $("#btnToggleChunks").textContent = chunksCollapsed ? "Раскрыть" : "Свернуть";
}

async function sendConsent() {
  $("#chatInput").value = "Подтверждаю согласие";
  await sendMessage();
}

$("#btnSend").addEventListener("click", () => sendMessage().catch(alert));
$("#chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage().catch(alert);
});
$("#btnIndex").addEventListener("click", () => indexKb().catch(alert));
$("#btnConsent").addEventListener("click", () => sendConsent().catch(alert));
$("#btnClear").addEventListener("click", () => {
  $("#chatWindow").innerHTML = "";
  $("#chunksPanel").innerHTML = "";
});
$("#btnRefreshLogs").addEventListener("click", () => refreshLogs().catch(alert));
$("#btnToggleChunks").addEventListener("click", toggleChunks);

setStatus("Idle", "good");
refreshLogs().catch(() => {});
