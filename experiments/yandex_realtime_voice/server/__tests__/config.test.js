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
