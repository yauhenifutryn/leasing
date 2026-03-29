const API_BASE = window.__RAG_API_BASE__ || window.location.origin;
const WS_BASE = API_BASE.replace(/^http/, "ws");
const $ = (sel) => document.querySelector(sel);

let sessionId = null;
let chunksCollapsed = false;
const STREAMING = true;
let activeTimer = null;
let requestStartMs = null;
let consentState = "needed";
let voiceFast = false;
let selectedBackend = "our_rag";
let selectedVoiceProvider = "local";
let selectedBrainModel = "Qwen/Qwen3.5-35B-A3B-FP8";
let selectedSttProvider = "sensevoice";
let selectedTtsProvider = "cosyvoice";
let voiceSocket = null;
let voiceConnected = false;
let talking = false;
let audioCtx = null;
let recorderNode = null;
let playCtx = null;
let nextPlayTime = 0;

function formatMs(ms) {
  return `${(ms / 1000).toFixed(1)}s`;
}

function updateTimer(elapsedMs) {
  const el = $("#responseTimer");
  if (!el) return;
  el.textContent = formatMs(elapsedMs);
}

function startTimer() {
  stopTimer();
  requestStartMs = performance.now();
  updateTimer(0);
  activeTimer = setInterval(() => updateTimer(performance.now() - requestStartMs), 120);
}

function stopTimer() {
  if (activeTimer) {
    clearInterval(activeTimer);
    activeTimer = null;
  }
  if (requestStartMs === null) return null;
  const elapsedMs = performance.now() - requestStartMs;
  updateTimer(elapsedMs);
  requestStartMs = null;
  return elapsedMs;
}

function setSessionId(id) {
  if (!id) return;
  sessionId = id;
  localStorage.setItem("rag_session_id", id);
}

function setConsentState(state) {
  if (!state) return;
  consentState = state;
  localStorage.setItem("rag_consent", state);
  const granted = state === "granted";
  $("#chatInput").disabled = !granted;
  $("#btnSend").disabled = !granted;
  $("#btnConsent").disabled = granted;
  $("#chatInput").placeholder = granted ? "Введите вопрос..." : "Сначала подтвердите согласие...";
}

function setStatus(text, level = "warn") {
  const badge = $("#statusBadge");
  const dot = badge.querySelector(".dot");
  dot.classList.remove("good", "danger", "warn");
  dot.classList.add(level);
  $("#statusText").textContent = text;
}

function setVoiceStatus(text, level = "warn") {
  const badge = $("#voiceStatus");
  const dot = badge.querySelector(".dot");
  dot.classList.remove("good", "danger", "warn");
  dot.classList.add(level);
  $("#voiceText").textContent = text;
}

function buildSessionUpdate() {
  return JSON.stringify({
    type: "session.update",
    backend: selectedBackend,
    voice_provider: selectedVoiceProvider,
    brain_model: selectedBrainModel,
    stt_provider: selectedSttProvider,
    tts_provider: selectedTtsProvider,
  });
}

function localVoiceReady(services) {
  return Boolean(services.sensevoice?.available || services.whisper?.available) && Boolean(services.cosyvoice?.available);
}

function voiceProviderReady(provider, services) {
  if (provider === "yandex_realtime") return Boolean(services.yandex_realtime?.available);
  if (provider === "yandex_speechkit") return Boolean(services.yandex_speechkit?.available);
  if (provider === "oss_russian") return Boolean(services.vosk?.available) && Boolean(services.vosk_tts?.available);
  return localVoiceReady(services);
}

function setTimePill(metaEl, elapsedMs) {
  if (!metaEl || elapsedMs === null) return;
  metaEl.innerHTML = `<span class="time-pill">${formatMs(elapsedMs)}</span>`;
}

function setIncompletePill(metaEl) {
  if (!metaEl) return;
  metaEl.innerHTML += ` <span class="warn-pill">Ответ неполный</span>`;
}

function renderMessage(role, text, opts = {}) {
  const row = document.createElement("div");
  row.className = "message";
  if (opts.pending) row.classList.add("pending");

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role;

  const textEl = document.createElement("div");
  textEl.className = "text";
  if (opts.pending) {
    const pendingText = opts.pendingText || "Агент думает...";
    textEl.innerHTML = `<span class="spinner"></span><span class="pending-text">${pendingText}</span>`;
  } else {
    textEl.textContent = text;
  }

  const metaEl = document.createElement("div");
  metaEl.className = "meta";

  row.appendChild(roleEl);
  row.appendChild(textEl);
  row.appendChild(metaEl);
  $("#chatWindow").appendChild(row);
  $("#chatWindow").scrollTop = $("#chatWindow").scrollHeight;
  return { row, textEl, metaEl };
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

function toggleChunks() {
  chunksCollapsed = !chunksCollapsed;
  $("#chunksPanel").style.display = chunksCollapsed ? "none" : "block";
  $("#btnToggleChunks").textContent = chunksCollapsed ? "Раскрыть" : "Свернуть";
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

async function refreshCapabilities() {
  try {
    const backendData = await api("/api/backends");
    const voiceData = await api("/api/voice/status");
    const selector = $("#backendSelect");
    const backends = backendData.backends || {};
    selector.innerHTML = "";
    for (const key of ["our_rag", "dify_rag"]) {
      if (!(key in backends)) continue;
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = key;
      const status = backends[key];
      if (!status.available) opt.textContent += " (off)";
      selector.appendChild(opt);
    }
    selector.value = selectedBackend;
    const voiceServices = voiceData.services || {};
    const ready = voiceProviderReady(selectedVoiceProvider, voiceServices);
    $("#btnVoiceConnect").disabled = !ready;
    setVoiceStatus(ready ? "Voice services detected" : "Voice services not configured", ready ? "good" : "warn");
  } catch (err) {
    setVoiceStatus("Voice status unavailable", "warn");
  }
}

function initSession() {
  const storedSession = localStorage.getItem("rag_session_id");
  const storedConsent = localStorage.getItem("rag_consent");
  const storedFast = localStorage.getItem("rag_voice_fast");
  const storedBackend = localStorage.getItem("rag_backend");
  const storedVoiceProvider = localStorage.getItem("rag_voice_provider");
  const storedBrainModel = localStorage.getItem("rag_brain_model");
  const storedSttProvider = localStorage.getItem("rag_stt_provider");
  const storedTtsProvider = localStorage.getItem("rag_tts_provider");
  if (storedSession) sessionId = storedSession;
  if (storedConsent) consentState = storedConsent;
  if (storedFast) voiceFast = storedFast === "true";
  if (storedBackend) selectedBackend = storedBackend;
  if (storedVoiceProvider) selectedVoiceProvider = storedVoiceProvider;
  if (storedBrainModel) selectedBrainModel = storedBrainModel;
  if (storedSttProvider) selectedSttProvider = storedSttProvider;
  if (storedTtsProvider) selectedTtsProvider = storedTtsProvider;
  setConsentState(consentState || "needed");
  $("#fastToggle").checked = voiceFast;
  $("#backendSelect").value = selectedBackend;
  $("#voiceProviderSelect").value = selectedVoiceProvider;
  $("#brainModelSelect").value = selectedBrainModel;
  $("#sttProviderSelect").value = selectedSttProvider;
  $("#ttsProviderSelect").value = selectedTtsProvider;
}

async function sendMessage(opts = {}) {
  const input = $("#chatInput");
  const message = input.value.trim();
  if (!message) return;
  if (!opts.bypassConsent && consentState !== "granted") {
    setStatus("Need consent", "warn");
    return;
  }
  input.value = "";
  renderMessage("user", message);
  setStatus("Thinking", "warn");
  startTimer();
  const agentMsg = renderMessage("agent", "", { pending: true });

  const payload = {
    message,
    session_id: sessionId,
    stream: STREAMING,
    backend: selectedBackend,
  };
  if (voiceFast) payload.mode = "voice_fast";

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok || !res.body) throw new Error("Stream failed");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let answer = "";
    let sawDelta = false;
    let sawFinal = false;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const evt = JSON.parse(line.slice(5).trim());
        if (evt.type === "delta") {
          if (!sawDelta) {
            sawDelta = true;
            agentMsg.row.classList.remove("pending");
            setStatus(`Streaming (${evt.backend || selectedBackend})`, "warn");
          }
          answer += evt.text || "";
          agentMsg.textEl.textContent = answer;
        } else if (evt.type === "final") {
          sawFinal = true;
          setSessionId(evt.session_id);
          setConsentState(evt.consent || consentState);
          agentMsg.row.classList.remove("pending");
          agentMsg.textEl.textContent = evt.answer || answer;
          setTimePill(agentMsg.metaEl, stopTimer());
          if (evt.incomplete) setIncompletePill(agentMsg.metaEl);
          renderChunks(evt.used_knowledge || []);
          setStatus(`Idle (${evt.backend || selectedBackend})`, "good");
        }
      }
    }
    if (!sawFinal) {
      agentMsg.row.classList.remove("pending");
      agentMsg.textEl.textContent = answer || "Ответ не получен полностью.";
      setTimePill(agentMsg.metaEl, stopTimer());
      setIncompletePill(agentMsg.metaEl);
      setStatus("Idle", "good");
    }
  } catch (err) {
    stopTimer();
    setStatus("Error", "danger");
    agentMsg.row.classList.remove("pending");
    agentMsg.textEl.textContent = "Ошибка запроса. Проверьте соединение и повторите.";
    throw err;
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
  $("#logPanel").textContent = items.map((i) => JSON.stringify(i)).join("\n") || "(empty)";
}

async function sendConsent() {
  $("#chatInput").value = "Подтверждаю согласие";
  setConsentState("pending");
  await sendMessage({ bypassConsent: true });
}

function b64ToInt16(b64) {
  const bin = atob(b64);
  const buf = new Int16Array(bin.length / 2);
  for (let i = 0; i < buf.length; i++) {
    buf[i] = (bin.charCodeAt(i * 2 + 1) << 8) | bin.charCodeAt(i * 2);
  }
  return buf;
}

function playPcm(int16, sampleRate = 24000) {
  if (!playCtx) {
    playCtx = new AudioContext({ sampleRate });
    nextPlayTime = playCtx.currentTime + 0.05;
  }
  const buffer = playCtx.createBuffer(1, int16.length, sampleRate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < int16.length; i++) channel[i] = int16[i] / 0x8000;
  const src = playCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(playCtx.destination);
  src.start(nextPlayTime);
  nextPlayTime += buffer.duration;
}

let voiceT0 = 0; // timestamp when user stops talking (commit sent)
let voiceFirstText = false;
let voiceFirstAudio = false;

function handleVoiceEvent(evt) {
  if (evt.type === "session.ready") {
    setSessionId(evt.session_id);
    setVoiceStatus(`Voice connected (${evt.voice_provider || "local"})`, "good");
    $("#btnVoiceTalk").disabled = false;
  }
  if (evt.type === "session.updated") {
    setVoiceStatus(`Stack: ${evt.stack_id || evt.voice_provider || "local"}`, "good");
  }
  if (evt.type === "conversation.item.input_audio_transcription.completed") {
    $("#transcript").textContent = evt.transcription || evt.transcript || "";
    const sttMs = Date.now() - voiceT0;
    setVoiceStatus(`STT: ${sttMs}ms`, "warn");
  }
  if (evt.type === "assistant_response") {
    // Session management only (barge-in state). Text display handled by output_text.delta.
  }
  if (evt.type === "response.output_text.delta") {
    // Append for streaming (multiple sentences arrive as separate deltas)
    $("#assistantVoice").textContent += evt.delta || "";
    if (!voiceFirstText) {
      voiceFirstText = true;
      const ttftMs = Date.now() - voiceT0;
      setVoiceStatus(`First text: ${ttftMs}ms`, "warn");
    }
  }
  if (evt.type === "response.output_audio.delta" && evt.delta) {
    playPcm(b64ToInt16(evt.delta), evt.sample_rate_hz || 24000);
    if (!voiceFirstAudio) {
      voiceFirstAudio = true;
      const ttfaMs = Date.now() - voiceT0;
      setVoiceStatus(`First audio: ${ttfaMs}ms`, "good");
    }
  }
  if (evt.type === "interrupt") {
    $("#assistantVoice").textContent = "";
    setVoiceStatus("Assistant interrupted", "warn");
  }
  if (evt.type === "response.done") {
    renderChunks(evt.used_knowledge || []);
    const totalMs = Date.now() - voiceT0;
    setVoiceStatus(`Done: ${totalMs}ms total`, "good");
  }
  if (evt.type === "warning") {
    setVoiceStatus(evt.message || "Voice warning", "warn");
  }
  if (evt.type === "error") {
    setVoiceStatus(evt.error || "Voice error", "danger");
  }
}

async function initAudio() {
  if (audioCtx) return;
  audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule("/recorder-worklet.js");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioCtx.createMediaStreamSource(stream);
  recorderNode = new AudioWorkletNode(audioCtx, "recorder-worklet");
  recorderNode.port.onmessage = (e) => {
    if (!talking || !voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) return;
    const chunk = e.data;
    const b64 = btoa(String.fromCharCode(...new Uint8Array(chunk.buffer)));
    voiceSocket.send(JSON.stringify({ type: "input_audio_buffer.append", audio: b64 }));
  };
  source.connect(recorderNode);
}

async function connectVoice() {
  if (voiceConnected) return;
  voiceSocket = new WebSocket(`${WS_BASE}/ws/voice`);
  voiceSocket.onopen = async () => {
    voiceConnected = true;
    setVoiceStatus("Voice websocket connected", "good");
    await initAudio();
    voiceSocket.send(buildSessionUpdate());
  };
  voiceSocket.onmessage = (msg) => handleVoiceEvent(JSON.parse(msg.data));
  voiceSocket.onerror = () => setVoiceStatus("Voice websocket error", "danger");
  voiceSocket.onclose = () => {
    voiceConnected = false;
    $("#btnVoiceTalk").disabled = true;
    setVoiceStatus("Voice disconnected", "warn");
  };
}

function startTalking() {
  if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) return;
  talking = true;
  $("#transcript").textContent = "";
  $("#assistantVoice").textContent = "";
  $("#btnVoiceTalk").classList.add("talking");
  setVoiceStatus("Listening...", "warn");
}

function stopTalking() {
  if (!voiceSocket || !talking) return;
  talking = false;
  $("#btnVoiceTalk").classList.remove("talking");
  setVoiceStatus("Processing...", "warn");
  voiceT0 = Date.now();
  voiceFirstText = false;
  voiceFirstAudio = false;
  $("#assistantVoice").textContent = "";
  voiceSocket.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
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
$("#fastToggle").addEventListener("change", (e) => {
  voiceFast = Boolean(e.target.checked);
  localStorage.setItem("rag_voice_fast", voiceFast ? "true" : "false");
});
$("#backendSelect").addEventListener("change", (e) => {
  selectedBackend = e.target.value;
  localStorage.setItem("rag_backend", selectedBackend);
  if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
    voiceSocket.send(buildSessionUpdate());
  }
});
$("#voiceProviderSelect").addEventListener("change", (e) => {
  selectedVoiceProvider = e.target.value;
  localStorage.setItem("rag_voice_provider", selectedVoiceProvider);
  refreshCapabilities().catch(() => {});
  if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
    voiceSocket.send(buildSessionUpdate());
  }
});
$("#brainModelSelect").addEventListener("change", (e) => {
  selectedBrainModel = e.target.value;
  localStorage.setItem("rag_brain_model", selectedBrainModel);
  if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
    voiceSocket.send(buildSessionUpdate());
  }
});
$("#sttProviderSelect").addEventListener("change", (e) => {
  selectedSttProvider = e.target.value;
  localStorage.setItem("rag_stt_provider", selectedSttProvider);
  if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
    voiceSocket.send(buildSessionUpdate());
  }
});
$("#ttsProviderSelect").addEventListener("change", (e) => {
  selectedTtsProvider = e.target.value;
  localStorage.setItem("rag_tts_provider", selectedTtsProvider);
  if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
    voiceSocket.send(buildSessionUpdate());
  }
});
$("#btnVoiceConnect").addEventListener("click", () => connectVoice().catch(alert));
$("#btnVoiceTalk").addEventListener("mousedown", startTalking);
$("#btnVoiceTalk").addEventListener("mouseup", stopTalking);
$("#btnVoiceTalk").addEventListener("mouseleave", stopTalking);
$("#btnVoiceTalk").addEventListener("touchstart", (e) => {
  e.preventDefault();
  startTalking();
});
$("#btnVoiceTalk").addEventListener("touchend", (e) => {
  e.preventDefault();
  stopTalking();
});

setStatus("Idle", "good");
refreshLogs().catch(() => {});
initSession();
refreshCapabilities().catch(() => {});
