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
