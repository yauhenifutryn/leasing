# Yandex Realtime Voice Assistant Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a minimal realtime voice assistant prototype using Yandex AI Studio Realtime API with streaming audio in and out, plus a minimal demo UI under `experiments/yandex_realtime_voice/`.

**Architecture:** Browser captures mic audio, downsamples to PCM16 24 kHz and streams to backend over WebSocket. Backend relays to Yandex Realtime API WebSocket, receives text and audio deltas, forwards to browser for UI and playback. RAG is enabled via file search tool if `YC_VECTOR_STORE_ID` is set.

**Tech Stack:** Node.js 20, `ws` WebSocket library, minimal static HTML/JS UI, Web Audio API.

---

### Task 1: Scaffold the experiment folder

**Files:**
- Create: `experiments/yandex_realtime_voice/README.md`
- Create: `experiments/yandex_realtime_voice/package.json`
- Create: `experiments/yandex_realtime_voice/.env.example`
- Create: `experiments/yandex_realtime_voice/server.js`
- Create: `experiments/yandex_realtime_voice/server/config.js`
- Create: `experiments/yandex_realtime_voice/server/validate.js`
- Create: `experiments/yandex_realtime_voice/server/__tests__/config.test.js`
- Create: `experiments/yandex_realtime_voice/server/__tests__/validate.test.js`
- Create: `experiments/yandex_realtime_voice/public/index.html`
- Create: `experiments/yandex_realtime_voice/public/app.js`
- Create: `experiments/yandex_realtime_voice/public/styles.css`
- Create: `experiments/yandex_realtime_voice/public/recorder-worklet.js`

**Step 1: Write failing test for config loader**

```js
// server/__tests__/config.test.js
import test from "node:test";
import assert from "node:assert/strict";
import { loadConfig } from "../config.js";

const baseEnv = {
  YC_FOLDER_ID: "folder",
  YC_API_KEY: "key",
};

test("loadConfig throws when missing required env", () => {
  assert.throws(() => loadConfig({}), /YC_FOLDER_ID/);
});

test("loadConfig returns defaults", () => {
  const cfg = loadConfig(baseEnv);
  assert.equal(cfg.model, `gpt://${baseEnv.YC_FOLDER_ID}/speech-realtime-250923`);
  assert.equal(cfg.voice, "ermil");
});
```

**Step 2: Run test to verify it fails**

Run: `node --test server/__tests__/config.test.js`
Expected: FAIL because `loadConfig` does not exist.

**Step 3: Implement minimal config loader**

```js
// server/config.js
export function loadConfig(env) {
  const required = ["YC_FOLDER_ID"];
  for (const key of required) {
    if (!env[key]) throw new Error(`Missing ${key}`);
  }
  const apiKey = env.YC_API_KEY;
  const iamToken = env.YC_IAM_TOKEN;
  if (!apiKey && !iamToken) throw new Error("Missing YC_API_KEY or YC_IAM_TOKEN");

  return {
    folderId: env.YC_FOLDER_ID,
    apiKey,
    iamToken,
    model: env.YC_MODEL || `gpt://${env.YC_FOLDER_ID}/speech-realtime-250923`,
    wsUrl:
      env.YC_REALTIME_WS_URL ||
      `wss://ai.api.cloud.yandex.net/v1/realtime/openai?model=${encodeURIComponent(
        `gpt://${env.YC_FOLDER_ID}/speech-realtime-250923`
      )}`,
    voice: env.YC_VOICE || "ermil",
    vectorStoreId: env.YC_VECTOR_STORE_ID || "",
  };
}
```

**Step 4: Run test to verify it passes**

Run: `node --test server/__tests__/config.test.js`
Expected: PASS

**Step 5: Commit**

```bash
git add experiments/yandex_realtime_voice/server/config.js \
  experiments/yandex_realtime_voice/server/__tests__/config.test.js

git commit -m "test: add config loader for yandex realtime demo"
```

---

### Task 2: Validate incoming client events

**Files:**
- Modify: `experiments/yandex_realtime_voice/server/validate.js`
- Modify: `experiments/yandex_realtime_voice/server/__tests__/validate.test.js`

**Step 1: Write failing test for event validation**

```js
// server/__tests__/validate.test.js
import test from "node:test";
import assert from "node:assert/strict";
import { validateClientEvent } from "../validate.js";

test("rejects unknown event type", () => {
  assert.equal(validateClientEvent({ type: "nope" }).ok, false);
});

test("accepts input_audio_buffer.append with audio", () => {
  const res = validateClientEvent({ type: "input_audio_buffer.append", audio: "abc" });
  assert.equal(res.ok, true);
});
```

**Step 2: Run test to verify it fails**

Run: `node --test server/__tests__/validate.test.js`
Expected: FAIL because `validateClientEvent` does not exist.

**Step 3: Implement minimal validation**

```js
// server/validate.js
const ALLOWED = new Set([
  "input_audio_buffer.append",
  "input_audio_buffer.commit",
  "response.create",
  "session.update",
]);

export function validateClientEvent(evt) {
  if (!evt || typeof evt !== "object") return { ok: false, error: "invalid" };
  if (!ALLOWED.has(evt.type)) return { ok: false, error: "type" };
  if (evt.type === "input_audio_buffer.append" && !evt.audio) {
    return { ok: false, error: "audio" };
  }
  return { ok: true };
}
```

**Step 4: Run test to verify it passes**

Run: `node --test server/__tests__/validate.test.js`
Expected: PASS

**Step 5: Commit**

```bash
git add experiments/yandex_realtime_voice/server/validate.js \
  experiments/yandex_realtime_voice/server/__tests__/validate.test.js

git commit -m "test: add validation for client events"
```

---

### Task 3: Implement backend relay server

**Files:**
- Create/Modify: `experiments/yandex_realtime_voice/server.js`

**Step 1: Write a failing smoke test**

```js
// server/__tests__/server.test.js
import test from "node:test";
import assert from "node:assert/strict";
import { buildSessionUpdate } from "../server.js";

test("buildSessionUpdate sets tools only when vector store provided", () => {
  const noTools = buildSessionUpdate({ voice: "ermil", vectorStoreId: "" });
  assert.equal(Array.isArray(noTools.session.tools), true);
  assert.equal(noTools.session.tools.length, 0);

  const withTools = buildSessionUpdate({ voice: "ermil", vectorStoreId: "vs" });
  assert.equal(withTools.session.tools.length, 1);
});
```

**Step 2: Run test to verify it fails**

Run: `node --test server/__tests__/server.test.js`
Expected: FAIL because `buildSessionUpdate` does not exist.

**Step 3: Implement relay server**

```js
// server.js
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import WebSocket, { WebSocketServer } from "ws";
import { loadConfig } from "./server/config.js";
import { validateClientEvent } from "./server/validate.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const cfg = loadConfig(process.env);

export function buildSessionUpdate({ voice, vectorStoreId }) {
  const session = {
    voice,
    modalities: ["text", "audio"],
    input_audio_format: "pcm16",
    output_audio_format: "pcm16",
    tool_choice: vectorStoreId ? "auto" : "none",
    tools: vectorStoreId
      ? [{
          type: "file_search",
          vector_store_ids: [vectorStoreId],
          max_num_results: 3,
        }]
      : [],
  };
  return { type: "session.update", session };
}

const app = express();
app.use(express.static(path.join(__dirname, "public")));

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });

wss.on("connection", (client) => {
  const headers = cfg.apiKey
    ? { Authorization: `Api-Key ${cfg.apiKey}` }
    : { Authorization: `Bearer ${cfg.iamToken}` };

  const upstream = new WebSocket(cfg.wsUrl, { headers });

  upstream.on("open", () => {
    upstream.send(JSON.stringify(buildSessionUpdate({
      voice: cfg.voice,
      vectorStoreId: cfg.vectorStoreId,
    })));
  });

  upstream.on("message", (data) => {
    client.send(data.toString());
  });

  upstream.on("close", () => client.close());
  upstream.on("error", (err) => {
    client.send(JSON.stringify({ type: "error", error: err.message }));
  });

  client.on("message", (data) => {
    let evt;
    try {
      evt = JSON.parse(data.toString());
    } catch {
      return;
    }
    const ok = validateClientEvent(evt);
    if (!ok.ok) return;
    upstream.send(JSON.stringify(evt));
  });

  client.on("close", () => upstream.close());
});

server.listen(process.env.PORT || 8787, () => {
  console.log(`Server listening on http://localhost:${process.env.PORT || 8787}`);
});
```

**Step 4: Run tests to verify they pass**

Run: `node --test server/__tests__/server.test.js`
Expected: PASS

**Step 5: Commit**

```bash
git add experiments/yandex_realtime_voice/server.js \
  experiments/yandex_realtime_voice/server/__tests__/server.test.js

git commit -m "feat: add websocket relay server for yandex realtime"
```

---

### Task 4: Implement frontend UI with audio streaming

**Files:**
- Modify: `experiments/yandex_realtime_voice/public/index.html`
- Modify: `experiments/yandex_realtime_voice/public/styles.css`
- Modify: `experiments/yandex_realtime_voice/public/app.js`
- Modify: `experiments/yandex_realtime_voice/public/recorder-worklet.js`

**Step 1: Create minimal HTML UI**

```html
<!-- public/index.html -->
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Yandex Realtime Voice Demo</title>
  <link rel="stylesheet" href="/styles.css" />
</head>
<body>
  <main class="app">
    <header class="header">
      <h1>Realtime Voice Demo</h1>
      <div id="status" class="status">Disconnected</div>
    </header>
    <section class="controls">
      <button id="connect">Connect</button>
      <button id="talk" disabled>Hold to Talk</button>
    </section>
    <section class="panel">
      <h2>Transcript</h2>
      <pre id="transcript"></pre>
    </section>
    <section class="panel">
      <h2>Assistant</h2>
      <pre id="assistant"></pre>
    </section>
  </main>
  <script src="/app.js"></script>
</body>
</html>
```

**Step 2: Add a minimal style sheet**

```css
/* public/styles.css */
:root { color-scheme: light; font-family: system-ui, sans-serif; }
body { margin: 0; padding: 24px; background: #f6f7fb; }
.app { max-width: 720px; margin: 0 auto; display: grid; gap: 16px; }
.header { display: flex; justify-content: space-between; align-items: center; }
.status { padding: 4px 8px; border-radius: 999px; background: #ddd; }
.controls { display: flex; gap: 12px; }
button { padding: 10px 16px; font-size: 14px; }
.panel { background: #fff; padding: 12px; border-radius: 8px; }
pre { white-space: pre-wrap; }
```

**Step 3: Implement audio capture worklet**

```js
// public/recorder-worklet.js
class RecorderWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 24000;
    this.buffer = [];
    this.port.onmessage = (e) => {
      if (e.data === "reset") this.buffer = [];
    };
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    const srcRate = sampleRate;
    const ratio = srcRate / this.targetSampleRate;
    const outLength = Math.floor(input.length / ratio);
    const out = new Int16Array(outLength);

    for (let i = 0; i < outLength; i++) {
      const srcIndex = Math.floor(i * ratio);
      const s = Math.max(-1, Math.min(1, input[srcIndex]));
      out[i] = s * 0x7fff;
    }

    this.port.postMessage(out);
    return true;
  }
}

registerProcessor("recorder-worklet", RecorderWorklet);
```

**Step 4: Implement UI logic**

```js
// public/app.js
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
  if (evt.type === "response.done") {
    assistantEl.textContent += "\n";
  }
}

async function initAudio() {
  audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule("/recorder-worklet.js");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioCtx.createMediaStreamSource(stream);
  workletNode = new AudioWorkletNode(audioCtx, "recorder-worklet");
  workletNode.port.onmessage = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
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
  ws.onclose = () => setStatus("Disconnected");
};

// push-to-talk
let talking = false;
talkBtn.onmousedown = () => {
  if (!ws) return;
  talking = true;
  assistantEl.textContent = "";
  transcriptEl.textContent = "";
};

talkBtn.onmouseup = () => {
  if (!ws || !talking) return;
  talking = false;
  ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
  ws.send(JSON.stringify({ type: "response.create" }));
};
```

**Step 5: Manual UI Trinity gates**

Tools are missing, so use manual checklist:
- Gate 1: architecture review (component layout, spacing scale, semantic tags).
- Gate 2: Vercel design checklist.
- Gate 3: Accessibility review (focus states, hit targets, contrast).

**Step 6: Commit**

```bash
git add experiments/yandex_realtime_voice/public

git commit -m "feat: add minimal realtime voice demo UI"
```

---

### Task 5: README and env example

**Files:**
- Modify: `experiments/yandex_realtime_voice/README.md`
- Modify: `experiments/yandex_realtime_voice/.env.example`
- Modify: `experiments/yandex_realtime_voice/package.json`

**Step 1: Write README**

Include:
- Required env vars.
- How to run `npm install` and `npm run dev`.
- Notes on headset use to avoid feedback.
- Notes on vector store optional config.

**Step 2: Add package.json**

```json
{
  "name": "yandex-realtime-voice",
  "type": "module",
  "private": true,
  "scripts": {
    "dev": "node server.js",
    "test": "node --test server/__tests__"
  },
  "dependencies": {
    "express": "^4.19.2",
    "ws": "^8.18.0",
    "dotenv": "^16.4.5"
  }
}
```

**Step 3: .env.example**

```
YC_FOLDER_ID=
YC_API_KEY=
YC_IAM_TOKEN=
YC_VECTOR_STORE_ID=
YC_VOICE=ermil
YC_MODEL=gpt://<folder_ID>/speech-realtime-250923
YC_REALTIME_WS_URL=
PORT=8787
```

**Step 4: Commit**

```bash
git add experiments/yandex_realtime_voice/README.md \
  experiments/yandex_realtime_voice/.env.example \
  experiments/yandex_realtime_voice/package.json

git commit -m "docs: add README and env example for yandex realtime demo"
```

---

### Task 6: Verification

Run:

```bash
npm install
npm test
npm run dev
```

Manual check:
- Connect.
- Hold to talk and release.
- See transcript and assistant text.
- Hear audio response.

---

**Notes**
- Realtime API WebSocket endpoint and event types are defined in Yandex docs and should be used as written.
- The demo uses the openai-compat Realtime WS URL unless overridden.
- Audio format must be PCM16 mono 24 kHz base64.
