import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import WebSocket, { WebSocketServer } from "ws";
import { buildSessionUpdate } from "./server/session.js";
import { loadConfig } from "./server/config.js";
import { validateClientEvent } from "./server/validate.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const cfg = loadConfig(process.env);

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
    upstream.send(
      JSON.stringify(
        buildSessionUpdate({
          voice: cfg.voice,
          vectorStoreId: cfg.vectorStoreId,
        })
      )
    );
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

const port = process.env.PORT || 8787;
server.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
});
