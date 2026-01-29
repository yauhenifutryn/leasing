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
