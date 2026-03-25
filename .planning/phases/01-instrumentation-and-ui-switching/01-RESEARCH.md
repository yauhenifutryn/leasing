# Phase 1: Instrumentation and UI Switching - Research

**Researched:** 2026-03-25
**Domain:** Python/FastAPI backend instrumentation, VoiceSession dataclass, vanilla JS frontend selector wiring
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INST-01 | Voice session logs 6 timing milestones per turn: speech_stopped, stt_done, retrieval_done, llm_first_token, tts_first_chunk, playback_started | Timestamps must be captured at call sites in `app.py` WebSocket handler; wall-clock `time.time()` is sufficient |
| INST-02 | Each voice turn emits a structured JSON log line with question_id, stack_id, all timestamps, and derived latencies | `StateStore.log()` already writes JSONL to `.state/logs.jsonl`; extend event dict with new fields |
| INST-03 | Primary KPI (playback_started - speech_stopped) is computed and logged for every turn | Arithmetic in the log call; no library needed |
| SWITCH-01 | UI exposes selectors for RAG backend, brain model, STT provider, and TTS provider | Brain model and separate STT/TTS selectors do not exist yet; must be added to `index.html` and `app.js` |
| SWITCH-02 | Switching any selector updates the active configuration without restarting the backend | WebSocket `session.update` already propagates backend and voice_provider; brain model selector must route through a new `session.update` field or a new HTTP endpoint |
| SWITCH-03 | Active stack_id is captured in every log line automatically | stack_id = deterministic string derived from active selector values; computed in `VoiceSession` or at log time |
</phase_requirements>

---

## Summary

Phase 1 is a pure instrumentation and wiring phase. There are no new external services and no model downloads. Every requirement maps directly to an edit in one of three files: `backend/voice_session.py`, `backend/app.py`, or `frontend/{index.html,app.js}`.

The existing logging infrastructure (`StateStore.log`) already writes append-only JSONL and the WebSocket `session.update` event already propagates `backend` and `voice_provider` without restart. The work is to extend these two mechanisms: add timestamp capture at the six milestone points in the WebSocket handler, and add two missing UI selectors (brain model, and separate STT/TTS controls) with corresponding server-side state in `VoiceSession`.

The VoiceSession dataclass is the natural home for the mutable active configuration. `stack_id` is a derived string, not stored separately: it is computed from the active fields at the moment a log line is written.

**Primary recommendation:** Extend `VoiceSession` with `brain_model`, `stt_provider`, and `tts_provider` fields. Add timestamp capture at six points in the WebSocket handler in `app.py`. Derive `stack_id` at log time from those four fields. Wire four selectors in the HTML/JS. No new endpoints are required.

---

## Standard Stack

### Core (already in project, no new installs needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | pinned in requirements.txt | WebSocket server, HTTP endpoints | Already running |
| Python stdlib `time` | built-in | Wall-clock timestamps for milestones | Zero dependency, sufficient precision for ms-level latency |
| Python stdlib `uuid` | built-in | `question_id` generation per voice turn | Already used in `app.py` |
| Python stdlib `json` | built-in | JSONL log serialization | `StateStore.log` already uses it |

### No New Packages Needed

All six requirements are achievable with the current venv. Do not add dependencies for this phase.

---

## Architecture Patterns

### Current State: What Exists

```
rag_demo_system/
├── backend/
│   ├── app.py              # FastAPI app, WebSocket handler /ws/voice
│   ├── voice_session.py    # VoiceSession dataclass (mutable per-connection state)
│   ├── voice_adapters.py   # transcribe_audio(), synthesize_audio_with_provider()
│   ├── state.py            # StateStore: .log(event_dict) -> appends JSONL
│   └── settings.py         # Settings loaded from app.yaml + env vars
├── frontend/
│   ├── index.html          # UI: has backendSelect, voiceProviderSelect
│   └── app.js              # JS: selectedBackend, selectedVoiceProvider, session.update over WS
└── tests/
    ├── test_voice_session.py
    └── test_voice_adapters_official.py
```

**Existing selector coverage:**
- `backendSelect` (our_rag / dify_rag): HTML select, wired to `session.update`
- `voiceProviderSelect` (local / yandex_speechkit / oss_russian / yandex_realtime): HTML select, wired to `session.update`

**Missing:**
- Brain model selector (Qwen3-30B-A3B / Qwen3.5-35B-A3B): not in HTML, not in VoiceSession
- Separate STT provider selector: not in HTML, not independently tracked (currently inferred from voice provider by `_voice_pipeline()`)
- Separate TTS provider selector: same gap

### Pattern 1: VoiceSession as Configuration Container

VoiceSession is a `@dataclass` with `session_id`, `backend`, `voice_provider`, and behavioral flags. Extend it with three new fields for the selectors that are missing:

```python
# Source: rag_demo_system/backend/voice_session.py (current + proposed additions)
@dataclass
class VoiceSession:
    session_id: str
    backend: str = "our_rag"
    voice_provider: str = "local"
    brain_model: str = "Qwen/Qwen3-30B-A3B"   # NEW
    stt_provider: str = "sensevoice"             # NEW
    tts_provider: str = "cosyvoice"             # NEW
    assistant_speaking: bool = False
    interrupted: bool = False
    active_task_id: str | None = None
    last_user_message: str = ""

    @property
    def stack_id(self) -> str:
        """Derived, not stored. Computed at log time."""
        return f"{self.backend}__{self.brain_model.split('/')[-1]}__{self.stt_provider}__{self.tts_provider}"
```

`stack_id` must never be user-supplied: it is always computed from the four active fields so benchmark logs are automatically tagged without manual annotation.

### Pattern 2: Timestamp Capture in the WebSocket Handler

The voice pipeline in `app.py` handles `input_audio_buffer.commit` sequentially. Six milestones map to specific points in that block:

```
speech_stopped    -> time.time() immediately when input_audio_buffer.commit is received
stt_done          -> time.time() after transcribe_audio() returns
retrieval_done    -> time.time() after the /api/chat call completes (retrieval happens inside chat)
llm_first_token   -> already tracked internally as timings["llm_ttfb_ms"]; extract absolute ts
tts_first_chunk   -> time.time() after synthesize_audio_with_provider() returns (first/only chunk)
playback_started  -> time.time() after the response.output_audio.delta message is sent to client
```

For this phase, `llm_first_token` and `tts_first_chunk` may be approximations if the pipeline is not streaming within the voice path (the current `/ws/voice` handler calls `chat()` with `stream=False`). That is acceptable: the six fields exist in every log line; their values are best-effort wall-clock timestamps.

```python
# Source: rag_demo_system/backend/app.py, inside input_audio_buffer.commit block
import uuid as _uuid

question_id = str(_uuid.uuid4())
t_speech_stopped = time.time()

# ... STT ...
t_stt_done = time.time()

# ... chat() call ...
t_retrieval_done = time.time()         # after chat() returns; retrieval is inside

# llm_first_token: use timings["llm_ttfb_ms"] relative offset if available
# For non-streaming voice path, llm_first_token = t_retrieval_done (conservative)
t_llm_first_token = t_retrieval_done   # will improve in streaming path

# ... synthesize_audio_with_provider() ...
t_tts_first_chunk = time.time()

# ... websocket.send_json(response.output_audio.delta) ...
t_playback_started = time.time()

primary_kpi_ms = (t_playback_started - t_speech_stopped) * 1000

state.log({
    "event": "voice_turn",
    "question_id": question_id,
    "stack_id": session.stack_id,
    "speech_stopped": t_speech_stopped,
    "stt_done": t_stt_done,
    "retrieval_done": t_retrieval_done,
    "llm_first_token": t_llm_first_token,
    "tts_first_chunk": t_tts_first_chunk,
    "playback_started": t_playback_started,
    "primary_kpi_ms": primary_kpi_ms,
    "session_id": session.session_id,
    "backend": session.backend,
    "brain_model": session.brain_model,
    "stt_provider": session.stt_provider,
    "tts_provider": session.tts_provider,
    "transcript": text,
})
```

### Pattern 3: session.update Extension (No Restart Required)

The existing WebSocket `session.update` handler reads `event.get("backend")` and `event.get("voice_provider")`. Extend it to also read `brain_model`, `stt_provider`, and `tts_provider`:

```python
# Source: rag_demo_system/backend/app.py, session.update branch
if event_type == "session.update":
    session.backend = _selected_backend(event.get("backend"))
    session.voice_provider = normalize_voice_provider(event.get("voice_provider"))
    if "brain_model" in event:
        session.brain_model = _valid_brain_model(event["brain_model"])
    if "stt_provider" in event:
        session.stt_provider = event["stt_provider"].strip().lower()
    if "tts_provider" in event:
        session.tts_provider = event["tts_provider"].strip().lower()
    # ... existing yandex_realtime relay setup ...
    await websocket.send_json({
        "type": "session.updated",
        "session_id": session_id,
        "backend": session.backend,
        "voice_provider": session.voice_provider,
        "brain_model": session.brain_model,
        "stt_provider": session.stt_provider,
        "tts_provider": session.tts_provider,
        "stack_id": session.stack_id,
    })
```

`_valid_brain_model()` is a small guard function that accepts only the two known model strings and falls back to the default.

### Pattern 4: UI Selector Wiring (Vanilla JS, No Build Step)

The frontend is plain HTML + vanilla JS. The existing `backendSelect` and `voiceProviderSelect` are the reference implementation. New selectors follow the same pattern:

HTML additions in the Voice section of `index.html`:
```html
<div class="toggle-row">
  <span class="toggle-label">Brain model</span>
  <select id="brainModelSelect">
    <option value="Qwen/Qwen3-30B-A3B">Qwen3-30B-A3B</option>
    <option value="Qwen/Qwen3.5-35B-A3B">Qwen3.5-35B-A3B</option>
  </select>
</div>
<div class="toggle-row">
  <span class="toggle-label">STT provider</span>
  <select id="sttProviderSelect">
    <option value="sensevoice">sensevoice</option>
    <option value="whisper">whisper</option>
    <option value="vosk">vosk</option>
    <option value="yandex_speechkit">yandex_speechkit</option>
  </select>
</div>
<div class="toggle-row">
  <span class="toggle-label">TTS provider</span>
  <select id="ttsProviderSelect">
    <option value="cosyvoice">cosyvoice</option>
    <option value="vosk_tts">vosk_tts</option>
    <option value="yandex_speechkit">yandex_speechkit</option>
  </select>
</div>
```

JS additions follow the existing `backendSelect` change listener pattern. On change, send `session.update` over the WebSocket if connected, and persist to localStorage.

### Pattern 5: _voice_pipeline() Refactor

Currently `_voice_pipeline()` in `app.py` maps `voice_provider` to `(stt_name, tts_name)`. Once STT and TTS are independently tracked in `VoiceSession`, the WebSocket handler should bypass `_voice_pipeline()` and read `session.stt_provider` / `session.tts_provider` directly. The old `_voice_pipeline()` can be kept for the non-WebSocket `/api/voice/chat` endpoint but should not be used in the `/ws/voice` handler after this phase.

### Recommended File Touch List

| File | Change |
|------|--------|
| `backend/voice_session.py` | Add `brain_model`, `stt_provider`, `tts_provider` fields + `stack_id` property |
| `backend/app.py` | (1) Extend `session.update` handler to read new fields; (2) Capture 6 timestamps in `input_audio_buffer.commit` block; (3) Call `state.log()` with full voice_turn event; (4) Pass `session.stt_provider`/`session.tts_provider` directly instead of `_voice_pipeline()` |
| `frontend/index.html` | Add 3 new `<select>` elements in Voice section |
| `frontend/app.js` | Add 3 JS variables, 3 change listeners, persist to localStorage, include in `session.update` payload |
| `tests/test_voice_session.py` | Add tests for new fields, `stack_id` computation |
| `tests/test_instrumentation.py` | New file: unit tests for log event structure |

### Anti-Patterns to Avoid

- **Storing `stack_id` as a mutable field**: it must be a `@property` computed from live fields. If stored, it can drift out of sync.
- **Capturing timestamps with `time.perf_counter()`**: use `time.time()` for cross-process timestamps in log files. `perf_counter()` is monotonic within one process but not comparable across processes or log analysis tools.
- **Making brain model changes restart the backend**: the selector is purely advisory for this phase. The brain model field records which model the operator intended to use; the actual loaded model is determined by `RAG_LLM_MODEL` env var on the server. This phase only records intent in the log line; Phase 3 adds the actual model-switching mechanism.
- **Splitting `voice_provider` selector into STT+TTS without removing the old field**: the old `voice_provider` field is still used for `yandex_realtime` relay detection. Keep it but treat it as the legacy provider mode; the new `stt_provider`/`tts_provider` fields shadow it in the WebSocket handler.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| JSONL logging | Custom file writer | `StateStore.log()` already exists; add fields to the dict |
| Timestamp precision | Custom timer class | `time.time()` returns float seconds with sub-millisecond resolution on all target platforms |
| stack_id uniqueness | UUID or hash | Deterministic string from field values: readable in logs without a lookup table |
| Selector persistence | Server-side session storage | `localStorage` already used for `backend` and `voice_provider`; same pattern for new selectors |

---

## Common Pitfalls

### Pitfall 1: llm_first_token Approximation in Non-Streaming Voice Path

**What goes wrong:** The `/ws/voice` handler calls `chat()` with `stream=False`. The LLM first-token timestamp is computed inside `chat()` as `first_token_at` but not returned in the response dict. The outer handler has no way to know when the first token arrived.

**Why it happens:** The non-streaming path accumulates the full answer before returning. The internal `first_token_at` is a local variable inside a generator that never runs in non-streaming mode.

**How to avoid:** For this phase, set `llm_first_token = retrieval_done` (conservative approximation). Add a `TODO` comment. Phase 3 (Benchmark Framework) can switch the voice path to streaming if finer resolution is needed.

**Warning signs:** Logs showing `llm_first_token == retrieval_done` identically are the expected approximation, not a bug.

### Pitfall 2: Selector State Diverges Between Frontend and Backend

**What goes wrong:** The user changes a selector, the WebSocket is not connected yet, and the selector value is never sent to the backend. All log lines use the default values.

**Why it happens:** The `session.update` event is only sent over the WebSocket. If the user sets selectors before clicking "Connect", those values sit in JS variables but never reach the server.

**How to avoid:** On `connectVoice()`, after the socket opens, always send a `session.update` message with all current selector values (the existing code already does this for `backend` and `voice_provider`; extend to include the new fields). This is the synchronization point.

**Warning signs:** All log lines in a benchmark run show the same default `stack_id` despite the user having changed selectors.

### Pitfall 3: question_id Collision in Replays

**What goes wrong:** Replaying a benchmark by re-sending the same audio generates different `question_id` values, making it impossible to correlate repeated runs of the same question.

**Why it happens:** `question_id` is generated with `uuid.uuid4()` at the start of each `input_audio_buffer.commit` handler.

**How to avoid:** For Phase 1, this is acceptable. `question_id` is per-turn, not per-question-text. The benchmark runner in Phase 3 will maintain a fixed question set with stable `question_text` field for correlation. For now, `question_id` just ensures each turn log line is uniquely addressable.

### Pitfall 4: tts_first_chunk Timestamp After Full Synthesis

**What goes wrong:** The current TTS path (`synthesize_audio_with_provider`) is a blocking HTTP POST that returns the full audio. Recording the timestamp after it returns gives "full synthesis done", not "first chunk available".

**Why it happens:** There is no streaming TTS path in this phase.

**How to avoid:** Document that `tts_first_chunk` = "TTS synthesis complete" for this phase's blocking implementation. The field name is forward-compatible: when Phase 2 adds Qwen3-TTS streaming, this timestamp can be sharpened. The log schema does not change.

### Pitfall 5: playback_started Before Audio Actually Plays

**What goes wrong:** `playback_started` is recorded server-side when the `response.output_audio.delta` message is sent. Actual playback starts client-side after the message is received and decoded.

**Why it happens:** The server has no visibility into the AudioContext playback schedule. True "audio starts playing" is a browser-side event.

**How to avoid:** Accept the server-side timestamp as a proxy. It is consistent across all runs and captures the meaningful pipeline boundary (when audio leaves the server). The ~10-50ms network delay is constant and does not distort comparison across stacks.

---

## Code Examples

### Correct `stack_id` Property

```python
# Source: design derived from existing VoiceSession pattern
@property
def stack_id(self) -> str:
    brain = self.brain_model.split("/")[-1]   # strips "Qwen/" prefix
    return f"{self.backend}__{brain}__{self.stt_provider}__{self.tts_provider}"
```

Example output: `our_rag__Qwen3-30B-A3B__sensevoice__cosyvoice`

### Correct voice_turn Log Event Structure

```json
{
  "event": "voice_turn",
  "question_id": "3f7a1c2d-...",
  "stack_id": "our_rag__Qwen3-30B-A3B__sensevoice__cosyvoice",
  "session_id": "ws-session-uuid",
  "backend": "our_rag",
  "brain_model": "Qwen/Qwen3-30B-A3B",
  "stt_provider": "sensevoice",
  "tts_provider": "cosyvoice",
  "transcript": "Какой аванс по лизингу?",
  "speech_stopped": 1711364400.123,
  "stt_done": 1711364400.456,
  "retrieval_done": 1711364401.234,
  "llm_first_token": 1711364401.234,
  "tts_first_chunk": 1711364402.100,
  "playback_started": 1711364402.105,
  "primary_kpi_ms": 1982.0
}
```

All timestamps are Unix epoch floats (seconds). `primary_kpi_ms` is `(playback_started - speech_stopped) * 1000`.

### Frontend session.update Payload (Extended)

```javascript
// Source: pattern from existing connectVoice() in app.js
voiceSocket.send(JSON.stringify({
    type: "session.update",
    backend: selectedBackend,
    voice_provider: selectedVoiceProvider,
    brain_model: selectedBrainModel,
    stt_provider: selectedSttProvider,
    tts_provider: selectedTtsProvider,
}));
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Voice provider as monolithic selector (local/yandex/oss/realtime) | Separate STT + TTS selectors for split pipeline | Phase 1 | Enables independent STT and TTS benchmarking |
| No benchmark tagging | `stack_id` auto-tagged on every log line | Phase 1 | No manual annotation during benchmark runs |
| Wall-clock timings only in chat path | Six explicit milestone timestamps in voice path | Phase 1 | Enables per-segment latency breakdown |

---

## Open Questions

1. **`brain_model` selector: record-only vs. actual model switch**
   - What we know: In Phase 1, the backend always uses `RAG_LLM_MODEL` env var. The UI selector records operator intent in log lines only.
   - What's unclear: Should switching the brain model selector also update `RAG_LLM_BASE_URL`/`RAG_LLM_MODEL` live? That requires vLLM reload or a second vLLM process.
   - Recommendation: Phase 1 is record-only. Log the selected model name. Phase 3 adds the actual switching mechanism. Document this clearly in the UI tooltip.

2. **STT/TTS selector and `yandex_realtime` interaction**
   - What we know: `yandex_realtime` bypasses the split STT+TTS pipeline entirely. The relay runs end-to-end on Yandex's side.
   - What's unclear: If `voice_provider = yandex_realtime`, what should `stt_provider` and `tts_provider` show?
   - Recommendation: When `voice_provider == "yandex_realtime"`, set `stt_provider = "yandex_realtime"` and `tts_provider = "yandex_realtime"` automatically. `stack_id` then reads `our_rag__Qwen3-30B-A3B__yandex_realtime__yandex_realtime`, which is correct for benchmark analysis.

---

## Environment Availability

Step 2.6: SKIPPED. This phase makes no calls to external services, installs no packages, and has no runtime dependencies beyond the existing venv and the already-running FastAPI backend. All changes are to Python source files and static HTML/JS.

---

## Validation Architecture

No `.planning/config.json` found; `workflow.nyquist_validation` key is absent; treating as enabled.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.3.4 |
| Config file | none (uses default discovery) |
| Quick run command | `python3 -m pytest rag_demo_system/tests/test_voice_session.py rag_demo_system/tests/test_instrumentation.py -x -q` |
| Full suite command | `python3 -m pytest rag_demo_system/tests/ -x -q` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INST-01 | VoiceSession exposes `stt_provider`, `tts_provider`, `brain_model` fields | unit | `pytest rag_demo_system/tests/test_voice_session.py -x -q` | Partial (exists, needs new cases) |
| INST-02 | `state.log` receives a dict with `question_id`, `stack_id`, all 6 timestamp keys | unit | `pytest rag_demo_system/tests/test_instrumentation.py -x -q` | No - Wave 0 gap |
| INST-03 | `primary_kpi_ms` equals `(playback_started - speech_stopped) * 1000` | unit | `pytest rag_demo_system/tests/test_instrumentation.py -x -q` | No - Wave 0 gap |
| SWITCH-01 | HTML has `brainModelSelect`, `sttProviderSelect`, `ttsProviderSelect` elements | manual | open browser, verify selectors visible | N/A |
| SWITCH-02 | `session.update` with `brain_model`/`stt_provider`/`tts_provider` updates `VoiceSession` fields | unit | `pytest rag_demo_system/tests/test_voice_session.py -x -q` | Partial (needs new case) |
| SWITCH-03 | `stack_id` property returns correct composite string from current field values | unit | `pytest rag_demo_system/tests/test_voice_session.py::test_stack_id_composition -x -q` | No - Wave 0 gap |

### Sampling Rate

- Per task commit: `python3 -m pytest rag_demo_system/tests/test_voice_session.py -x -q`
- Per wave merge: `python3 -m pytest rag_demo_system/tests/ -x -q`
- Phase gate: full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `rag_demo_system/tests/test_instrumentation.py` - covers INST-02, INST-03, SWITCH-03 (log event structure and stack_id tests)
- New test cases in existing `test_voice_session.py`:
  - [ ] `test_stack_id_composition` - covers SWITCH-03
  - [ ] `test_session_update_sets_brain_model` - covers SWITCH-02
  - [ ] `test_session_update_sets_stt_tts_providers` - covers SWITCH-02

*(Framework and conftest already exist; no new install needed)*

---

## Sources

### Primary (HIGH confidence)

- Direct code inspection: `rag_demo_system/backend/app.py` (full file read, 827 lines)
- Direct code inspection: `rag_demo_system/backend/voice_session.py`
- Direct code inspection: `rag_demo_system/backend/voice_adapters.py`
- Direct code inspection: `rag_demo_system/backend/state.py`
- Direct code inspection: `rag_demo_system/frontend/index.html` and `app.js`
- Direct code inspection: `rag_demo_system/tests/` (24 existing test files)
- `python3 -m pytest rag_demo_system/tests/test_voice_session.py` - 4 passed, confirming test infrastructure works

### Secondary (MEDIUM confidence)

- `.planning/REQUIREMENTS.md` - phase requirement definitions
- `.planning/STATE.md` - project decisions and blocked concerns
- `.planning/PROJECT.md` - architecture constraints

### Tertiary (LOW confidence)

None. All research findings are grounded in direct code inspection.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - zero new dependencies; all tools already in venv
- Architecture: HIGH - derived from direct inspection of all relevant source files
- Pitfalls: HIGH - each pitfall is traceable to a specific code path in app.py or voice_session.py

**Research date:** 2026-03-25
**Valid until:** 2026-05-25 (stable internal codebase; no external API dependencies)
