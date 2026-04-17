# Spec 4: Turn-Taking Control (Silence + Stop)

**Cluster:** D + G merged — silence handling and stop-command detection
**Depends on:** Spec 2 (SessionAgent emits `is_stop_request`)
**Blocks:** —

## Context

Two transcript patterns caused user friction:

1. **Mid-sentence cut-off:** at 18:03:32-37 Sergey says *"Хорошо, Ксения,
   подождите, я всё-таки хочу понять,"* pauses, says *"Отечественные."*. Bot
   responded to the pause, then got barge-in on the second fragment. Result:
   confused state.

2. **Stop-command ignored:** client's explicit "стоп", "подожди",
   "Ты невозможно прервать" (18:12:49) produced no detectable effect on the
   bot's behavior. Only acoustic barge-in (VAD detecting any sound) cancels
   TTS; there is no semantic understanding of stop.

## Problem

1. `VAD_SILENCE_MS=500` (env default) ends speech detection too aggressively on
   natural pauses inside a client's turn.
2. No "pre-response hold" window. As soon as VAD calls `speech_end`, bot's
   entire pipeline kicks off. No second chance to detect continuation.
3. Stop intent is only acoustic, never semantic. Client's quiet word "стоп" may
   not exceed the RMS 300 floor (Silero echo residue filter). Even when it
   does, the bot just starts listening; it doesn't go quiet and stay silent.

## Goals

- Bump `VAD_SILENCE_MS` default to 700 (from 500).
- Introduce `PRE_RESPONSE_HOLD_MS` gate (default 300): after `speech_end`, wait
  this window; if new speech appears, cancel and re-buffer into the same
  utterance.
- Add `listen_mode` session state, entered on `is_stop_request=true` from
  SessionAgent: TTS immediately canceled, bot goes quiet, thresholds relaxed
  for sensitive capture.
- Auto-exit `listen_mode` on first new utterance OR 3-second timeout with
  gentle *"Слушаю Вас."* prompt.
- All thresholds exposed as env vars for live tuning.

## Non-goals

- Turn-end detection via a separate LLM call (deferred; SessionAgent already
  gives semantic signals)
- Phonetic stop-word dictionary (rejected; semantic detection via
  SessionAgent is more robust)
- Adaptive silence based on client speech rate (future optimization)

## Design

### Env vars

| Var | Default | Current | Purpose |
|---|---|---|---|
| `VAD_SILENCE_MS` | **700** | 500 | Silence duration before speech_end fires |
| `PRE_RESPONSE_HOLD_MS` | **300** | 0 (new) | Delay between speech_end and LLM start |
| `LISTEN_MODE_TIMEOUT_SEC` | **3.0** | n/a (new) | Max time in listen_mode before prompt |
| `LISTEN_MODE_VAD_RMS` | **180** | 300 | Lower RMS floor during listen_mode |
| `LISTEN_MODE_MIN_SPEECH_MS` | **300** | 800 normal / 400 barge-in | Shorter capture |

### Pre-response hold

After VAD returns `speech_audio` (line `app.py:2588`), instead of immediately
calling `_jambonz_process_utterance`:

```python
# Hold window: if client resumes within PRE_RESPONSE_HOLD_MS, buffer into same utterance
pending = speech_audio
hold_start = asyncio.get_event_loop().time()
while asyncio.get_event_loop().time() - hold_start < PRE_RESPONSE_HOLD_MS / 1000:
    try:
        msg = await asyncio.wait_for(websocket.receive(), timeout=0.05)
    except asyncio.TimeoutError:
        continue
    if "bytes" in msg and msg["bytes"]:
        # Feed to VAD; if it reopens speech, append and extend hold window
        more = vad.feed(msg["bytes"])
        if vad.is_speaking:
            # Continuation detected; break out of hold, buffer more audio
            pending += vad.pull_buffer()
            hold_start = asyncio.get_event_loop().time()  # reset hold window
            # continue outer while to capture rest of speech
```

(Pseudo-code — actual implementation adjusts to existing async flow.)

### Listen mode state

**New fields on `ChatSession`:**

```python
listen_mode: bool = False
listen_mode_until: float = 0.0
listen_mode_original_rms: int = 0  # to restore after exit
listen_mode_original_min_speech: int = 0
```

**Entry (from SessionAgent output):**

```python
if session_agent_result.get("is_stop_request"):
    session.interrupted = True
    session.assistant_speaking = False
    # Cancel any in-flight TTS immediately
    await websocket.send_text(json.dumps({"type": "killAudio"}))
    session.listen_mode = True
    session.listen_mode_until = time.time() + LISTEN_MODE_TIMEOUT_SEC
    session.listen_mode_original_rms = BARGE_IN_RMS_FLOOR
    # Override thresholds for this session duration
    BARGE_IN_RMS_FLOOR = LISTEN_MODE_VAD_RMS
    print(f"[listen_mode] entered, timeout={LISTEN_MODE_TIMEOUT_SEC}s")
    return  # no LLM response generated; bot goes silent
```

**Exit (two paths):**

1. **New utterance detected while `listen_mode`:**
   - Normal utterance processing kicks in.
   - On `assistant_speaking = True` (response starts), reset thresholds.
   - `session.listen_mode = False`.

2. **Timeout (no utterance within `LISTEN_MODE_TIMEOUT_SEC`):**
   - Background task (`asyncio.create_task`) waits for
     `session.listen_mode_until - time.time()` seconds, then checks.
   - If still in listen_mode, emit gentle prompt: `"Слушаю Вас."`.
   - Reset thresholds, `listen_mode = False`.

### Integration with SessionAgent

Spec 2's SessionAgent output contract already contains `is_stop_request`.
Spec 4 consumes it:

```python
# In the utterance handler, after session agent returns:
sa_result = await _run_session_agent(...)
if sa_result.get("is_stop_request"):
    await _enter_listen_mode(session, websocket)
    return
# ... else continue normal flow
```

### Barge-in interaction

Existing barge-in at `app.py:2543` stays untouched. It handles acoustic
interruption during TTS. Listen mode is a complementary layer that kicks in
AFTER STT confirms the semantic content is a stop request. Flow:

```
Client speaks during TTS → acoustic barge-in → TTS cancels → STT runs
                                                             ↓
                                                  SessionAgent: is_stop_request?
                                                             ↓  (yes)
                                                      enter listen_mode
                                                      (no bot response)
```

## Files to change

- `rag_demo_system/backend/app.py` (pre-response hold loop, listen_mode state,
  SessionAgent `is_stop_request` integration)
- `rag_demo_system/backend/session.py` (new fields on ChatSession)
- `rag_demo_system/.env.example` (new env vars)
- `rag_demo_system/backend/config.py` (expose env vars)

## Testing

**Unit — pre-response hold**
1. Simulate speech_end → immediate new audio within 200ms → assert same
   utterance buffer extended, hold_start reset, no LLM call yet.
2. Simulate speech_end → no new audio for 350ms → assert LLM call fires after
   300ms.

**Unit — listen_mode state**
1. Mock SessionAgent returns `is_stop_request=true` → assert
   `listen_mode=True`, `killAudio` sent, no LLM call.
2. After listen_mode entry, feed quiet new utterance (RMS 200) → assert it is
   captured (RMS floor lowered to 180).
3. No speech for 3.1s → assert "Слушаю Вас." TTS fires, listen_mode off.

**Integration — transcript replay**
1. Find transcript turn where Sergey says "Ксения, стоп" (18:05:52).
2. Replay. Assert bot's next action is listen_mode entry, not another
   clarify_client_type question.

**Metrics**
- `listen_mode_entries_per_session` (count).
- `listen_mode_timeouts` (count).
- `pre_response_hold_extends` (count — how often client did continue within
  hold window).
- `mid_turn_cutoffs_prevented` — implicit metric from hold extends.

## Risks

| Risk | Mitigation |
|---|---|
| 700ms silence feels laggy to clients who want fast back-and-forth | Env-tunable; can roll back to 500 for A/B |
| Over-enters listen_mode on false stop (SessionAgent misclassifies) | Listen mode timeout (3s) + gentle "Слушаю Вас." auto-recovers |
| Pre-response hold extends forever on chattery background noise | VAD still has its own end-detection — if VAD re-closes, hold window expires normally |
| Killing TTS mid-word sounds abrupt | Silero already supports clean cutoff; existing barge-in does this |

## Rollback

Env var flags:
- `PRE_RESPONSE_HOLD_MS=0` disables pre-response hold.
- `VAD_SILENCE_MS=500` restores original silence window.
- Listen mode: branch check `if not STOP_COMMAND_ENABLED: skip`.

All three surfaces as simple env edits + restart.
