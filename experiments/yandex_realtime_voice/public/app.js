const statusEl = document.getElementById("status");
const connectBtn = document.getElementById("connect");
const talkBtn = document.getElementById("talk");
const transcriptEl = document.getElementById("transcript");
const assistantEl = document.getElementById("assistant");

let ws;
let audioCtx;
let workletNode;
let playCtx;
let nextPlayTime = 0;
let connected = false;
let talking = false;

function setStatus(text) {
  statusEl.textContent = text;
}

function b64ToInt16(b64) {
  const bin = atob(b64);
  const buf = new Int16Array(bin.length / 2);
  for (let i = 0; i < buf.length; i++) {
    buf[i] = (bin.charCodeAt(i * 2 + 1) << 8) | bin.charCodeAt(i * 2);
  }
  return buf;
}

function playPcm(int16) {
  if (!playCtx) {
    playCtx = new AudioContext({ sampleRate: 24000 });
    nextPlayTime = playCtx.currentTime + 0.05;
  }
  const buffer = playCtx.createBuffer(1, int16.length, 24000);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < int16.length; i++) {
    channel[i] = int16[i] / 0x8000;
  }
  const src = playCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(playCtx.destination);
  src.start(nextPlayTime);
  nextPlayTime += buffer.duration;
}

function handleServerEvent(evt) {
  if (evt.type === "conversation.item.input_audio_transcription.completed") {
    transcriptEl.textContent = evt.transcription || "";
  }
  if (evt.type === "response.output_text.delta") {
    assistantEl.textContent += evt.delta || "";
  }
  if (evt.type === "response.output_audio.delta") {
    playPcm(b64ToInt16(evt.delta));
  }
  if (evt.type === "input_audio_buffer.speech_started") {
    if (playCtx) {
      playCtx.close();
      playCtx = null;
      nextPlayTime = 0;
    }
    assistantEl.textContent = "";
  }
  if (evt.type === "response.done") {
    assistantEl.textContent += "\n";
  }
  if (evt.type === "error") {
    setStatus("Error");
  }
}

async function initAudio() {
  audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule("/recorder-worklet.js");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioCtx.createMediaStreamSource(stream);
  workletNode = new AudioWorkletNode(audioCtx, "recorder-worklet");
  workletNode.port.onmessage = (e) => {
    if (!talking || !ws || ws.readyState !== WebSocket.OPEN) return;
    const chunk = e.data;
    const b64 = btoa(String.fromCharCode(...new Uint8Array(chunk.buffer)));
    ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio: b64 }));
  };
  source.connect(workletNode);
}

connectBtn.onclick = async () => {
  if (connected) return;
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = async () => {
    connected = true;
    setStatus("Connected");
    talkBtn.disabled = false;
    await initAudio();
  };
  ws.onmessage = (msg) => {
    const evt = JSON.parse(msg.data);
    handleServerEvent(evt);
  };
  ws.onerror = () => setStatus("Error");
  ws.onclose = () => {
    setStatus("Disconnected");
    connected = false;
    talkBtn.disabled = true;
  };
};

function startTalking() {
  if (!ws) return;
  talking = true;
  assistantEl.textContent = "";
  transcriptEl.textContent = "";
}

function stopTalking() {
  if (!ws || !talking) return;
  talking = false;
  ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
  ws.send(JSON.stringify({ type: "response.create" }));
}

talkBtn.onmousedown = startTalking;
talkBtn.onmouseup = stopTalking;
talkBtn.onmouseleave = stopTalking;

talkBtn.ontouchstart = (e) => {
  e.preventDefault();
  startTalking();
};

talkBtn.ontouchend = (e) => {
  e.preventDefault();
  stopTalking();
};
