# Turn-Taking Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the bot from cutting clients off mid-sentence and make it respect semantic stop requests, while preserving fast replies when the client is actually done. Bump `VAD_SILENCE_MS` to 700, add a 300 ms pre-response hold, and introduce `listen_mode` activated by SessionAgent's `is_stop_request` flag.

**Architecture:** Two new env vars drive VAD silence + pre-response hold. `ChatSession` gets `listen_mode` boolean + deadline. The existing VAD loop in `app.py` gains a hold-window that can cancel itself when new speech arrives, and a branch that lowers RMS floor + min-speech length when `listen_mode` is active. SessionAgent's `is_stop_request` from the session-agent-profile plan triggers entry into listen_mode and cancels in-flight TTS.

**Tech Stack:** Python 3.12, asyncio, existing VAD + WebSocket infrastructure.

**Spec:** `docs/superpowers/specs/2026-04-16-turn-taking-control-design.md`

**Depends on:** session-agent-profile plan Task 3 (`is_stop_request` must be emitted by SessionAgent).

---

### Task 1: Add turn-taking env vars to settings

**Files:**
- Modify: `rag_demo_system/backend/config.py`
- Modify: `rag_demo_system/.env.example`

- [ ] **Step 1: Write failing test**

Create `rag_demo_system/tests/test_turn_taking_config.py`:

```python
"""Turn-taking env var defaults and overrides."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_turn_taking_defaults(monkeypatch) -> None:
    for k in ("VAD_SILENCE_MS", "PRE_RESPONSE_HOLD_MS",
              "LISTEN_MODE_TIMEOUT_SEC", "LISTEN_MODE_VAD_RMS",
              "LISTEN_MODE_MIN_SPEECH_MS"):
        monkeypatch.delenv(k, raising=False)
    from backend.config import Settings

    s = Settings()
    assert s.vad_silence_ms == 700
    assert s.pre_response_hold_ms == 300
    assert s.listen_mode_timeout_sec == 3.0
    assert s.listen_mode_vad_rms == 180
    assert s.listen_mode_min_speech_ms == 300


def test_turn_taking_env_override(monkeypatch) -> None:
    monkeypatch.setenv("VAD_SILENCE_MS", "500")
    monkeypatch.setenv("PRE_RESPONSE_HOLD_MS", "0")
    from backend.config import Settings
    s = Settings()
    assert s.vad_silence_ms == 500
    assert s.pre_response_hold_ms == 0
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_turn_taking_config.py -v`
Expected: FAIL — fields missing.

- [ ] **Step 3: Add fields to `Settings`**

In `rag_demo_system/backend/config.py`:

```python
vad_silence_ms: int = Field(default=700, description="Silence duration (ms) before VAD declares speech_end.")
pre_response_hold_ms: int = Field(default=300, description="Hold (ms) after speech_end before LLM start; extended if new speech arrives.")
listen_mode_timeout_sec: float = Field(default=3.0, description="Auto-exit listen_mode after this many seconds of silence.")
listen_mode_vad_rms: int = Field(default=180, description="Lower RMS floor during listen_mode to catch quiet speech.")
listen_mode_min_speech_ms: int = Field(default=300, description="Shorter min-speech during listen_mode.")
```

Preserve existing `VAD_SILENCE_MS` env-var convention if different from 500; the default we set here (700) will supersede.

- [ ] **Step 4: Run test, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_turn_taking_config.py -v`
Expected: PASS.

- [ ] **Step 5: Append to `.env.example`**

```
# Turn-taking control
VAD_SILENCE_MS=700
PRE_RESPONSE_HOLD_MS=300
LISTEN_MODE_TIMEOUT_SEC=3.0
LISTEN_MODE_VAD_RMS=180
LISTEN_MODE_MIN_SPEECH_MS=300
```

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/config.py rag_demo_system/.env.example rag_demo_system/tests/test_turn_taking_config.py
git commit -m "feat(config): add turn-taking env vars for silence + listen_mode"
```

---

### Task 2: `ChatSession.listen_mode` state fields

**Files:**
- Modify: wherever `ChatSession` lives (same as session-agent-profile plan Task 2)

- [ ] **Step 1: Write failing test**

Create `rag_demo_system/tests/test_listen_mode_state.py`:

```python
"""ChatSession listen_mode fields: default false, deadline optional."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_listen_mode_defaults() -> None:
    from backend.app import ChatSession  # adjust import if ChatSession elsewhere

    cs = ChatSession(session_id="test")
    assert cs.listen_mode is False
    assert cs.listen_mode_until == 0.0


def test_listen_mode_can_be_set() -> None:
    from backend.app import ChatSession

    cs = ChatSession(session_id="test")
    cs.listen_mode = True
    cs.listen_mode_until = 1234567890.0
    assert cs.listen_mode is True
    assert cs.listen_mode_until == 1234567890.0
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_listen_mode_state.py -v`
Expected: FAIL.

- [ ] **Step 3: Add fields to `ChatSession`**

In the file defining `ChatSession` (match the pattern used for the `client_profile` addition):

```python
listen_mode: bool = False
listen_mode_until: float = 0.0
```

If `ChatSession` is not a dataclass, use `__init__`:

```python
self.listen_mode = False
self.listen_mode_until = 0.0
```

- [ ] **Step 4: Run test, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_listen_mode_state.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(session): add listen_mode state fields to ChatSession"
```

---

### Task 3: Plumb `VAD_SILENCE_MS` through VAD initialization

**Files:**
- Modify: `rag_demo_system/backend/app.py` — where VAD is instantiated (search: `SileroVAD` or `vad = `)

- [ ] **Step 1: Locate VAD instantiation**

Run: `cd rag_demo_system && grep -n "SileroVAD\|vad = \|silence_ms" backend/app.py | head -20`

- [ ] **Step 2: Pass `settings.vad_silence_ms` into VAD constructor**

Replace any hardcoded silence_ms argument with `silence_ms=settings.vad_silence_ms`. If VAD is instantiated multiple times (RTC + WebSocket paths), update all.

- [ ] **Step 3: Smoke-test locally**

Run the backend with `VAD_SILENCE_MS=700` set; observe `[Jambonz:xxxx] VAD: speech_end` firing after ~700 ms of silence (not 500 ms).

- [ ] **Step 4: Commit**

```bash
git add rag_demo_system/backend/app.py
git commit -m "feat(vad): honor VAD_SILENCE_MS env (default 700ms from 500ms)"
```

---

### Task 4: Pre-response hold window

**Files:**
- Modify: `rag_demo_system/backend/app.py` — around line 2588 (after `speech_audio = vad.feed(pcm_16k)`)

- [ ] **Step 1: Write failing test (unit, mock-based)**

Create `rag_demo_system/tests/test_pre_response_hold.py`:

```python
"""Pre-response hold: delay LLM start, extend window if new speech arrives."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_hold_expires_without_new_speech() -> None:
    """Silent hold window: returns after PRE_RESPONSE_HOLD_MS, no extension."""
    from backend.app import _pre_response_hold

    extended = _run(_pre_response_hold(hold_ms=100, poll_fn=None))
    assert extended is False


def test_hold_extended_on_new_speech() -> None:
    """poll_fn returns True (speech detected) within window → extension."""
    from backend.app import _pre_response_hold

    calls = {"n": 0}

    async def fake_poll() -> bool:
        calls["n"] += 1
        return calls["n"] == 1  # first poll detects speech

    extended = _run(_pre_response_hold(hold_ms=200, poll_fn=fake_poll))
    assert extended is True
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_pre_response_hold.py -v`
Expected: FAIL — helper missing.

- [ ] **Step 3: Add `_pre_response_hold` helper**

In `rag_demo_system/backend/app.py`, add:

```python
async def _pre_response_hold(hold_ms: int, poll_fn=None) -> bool:
    """Wait up to `hold_ms` after speech_end. If `poll_fn` returns True (new speech),
    return True immediately (caller extends buffer and re-enters hold).

    poll_fn is an awaitable that returns bool; None disables polling (simple sleep).
    Returns True if caller should extend/continue capture; False if timer expired.
    """
    import time
    deadline = time.monotonic() + hold_ms / 1000.0
    if poll_fn is None:
        await asyncio.sleep(hold_ms / 1000.0)
        return False
    while time.monotonic() < deadline:
        try:
            result = await asyncio.wait_for(poll_fn(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
        if result:
            return True
    return False
```

- [ ] **Step 4: Run test, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_pre_response_hold.py -v`
Expected: PASS.

- [ ] **Step 5: Integrate into the VAD loop**

In `_stream_voice_response` (or whichever handler owns VAD speech-end), locate (around line 2632):

```python
session.assistant_speaking = True
session.interrupted = False
asyncio.create_task(_jambonz_process_utterance(
    websocket, session, session_id, speech_24k,
))
```

Insert the hold gate BEFORE the `create_task` call:

```python
# Pre-response hold: if client resumes speaking within the hold window, cancel
# this utterance and continue capturing. Keeps natural mid-sentence pauses from
# causing premature LLM firing.
hold_ms = settings.pre_response_hold_ms
if hold_ms > 0:
    extended = await _pre_response_hold(
        hold_ms=hold_ms,
        poll_fn=lambda: _peek_for_new_speech(vad),
    )
    if extended:
        # Stay in the VAD loop; do not fire LLM yet. Let VAD re-open speech.
        continue

session.assistant_speaking = True
# ... existing code
```

Where `_peek_for_new_speech` is a small helper:

```python
async def _peek_for_new_speech(vad) -> bool:
    """Cheap check: has VAD re-detected speech since last speech_end?"""
    return getattr(vad, "is_speaking", False)
```

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/app.py rag_demo_system/tests/test_pre_response_hold.py
git commit -m "feat(turn-taking): add PRE_RESPONSE_HOLD_MS gate before LLM kickoff"
```

---

### Task 5: `listen_mode` entry on `is_stop_request`

**Files:**
- Modify: `rag_demo_system/backend/app.py` — where SessionAgent result is consumed

- [ ] **Step 1: Write failing test**

Append to `rag_demo_system/tests/test_listen_mode_state.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock


def test_enter_listen_mode_cancels_tts_and_sets_flag() -> None:
    from backend.app import _enter_listen_mode, ChatSession

    cs = ChatSession(session_id="test")
    cs.assistant_speaking = True
    fake_ws = AsyncMock()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_enter_listen_mode(cs, fake_ws, timeout_sec=3.0))
    finally:
        loop.close()

    assert cs.listen_mode is True
    assert cs.listen_mode_until > 0
    assert cs.assistant_speaking is False
    assert cs.interrupted is True
    # TTS killAudio message sent
    sent = [c.args[0] for c in fake_ws.send_text.await_args_list]
    assert any("killAudio" in s for s in sent)
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_listen_mode_state.py::test_enter_listen_mode_cancels_tts_and_sets_flag -v`
Expected: FAIL.

- [ ] **Step 3: Add `_enter_listen_mode` helper**

```python
async def _enter_listen_mode(session, websocket, timeout_sec: float) -> None:
    """Cancel in-flight TTS, set listen_mode flag, start auto-exit timer."""
    import json, time

    session.listen_mode = True
    session.listen_mode_until = time.time() + timeout_sec
    session.interrupted = True
    session.assistant_speaking = False

    try:
        await websocket.send_text(json.dumps({"type": "killAudio"}))
    except Exception as exc:  # noqa: BLE001
        print(f"[listen_mode] killAudio send failed: {exc}", flush=True)

    # Background auto-exit: after timeout_sec, if still in listen_mode, prompt "Слушаю Вас."
    asyncio.create_task(_listen_mode_auto_exit(session, websocket, timeout_sec))


async def _listen_mode_auto_exit(session, websocket, timeout_sec: float) -> None:
    await asyncio.sleep(timeout_sec)
    if session.listen_mode and session.listen_mode_until <= time.time() + 0.1:
        session.listen_mode = False
        try:
            await _speak_text(websocket, session, "Слушаю Вас.")
        except Exception as exc:  # noqa: BLE001
            print(f"[listen_mode] auto-exit prompt failed: {exc}", flush=True)
```

Ensure `time` is imported at top of `app.py`.

- [ ] **Step 4: Wire `is_stop_request` into SessionAgent handler**

Find where `_run_session_agent` result is consumed (in `_stream_voice_response`). Replace the earlier short-circuit stub from session-agent-profile plan Task 4 (`if sa.get("is_stop_request"): ... return`) with:

```python
if sa.get("is_stop_request"):
    await _enter_listen_mode(
        session, websocket,
        timeout_sec=settings.listen_mode_timeout_sec,
    )
    print(f"[SessionAgent] is_stop_request -> listen_mode for {settings.listen_mode_timeout_sec}s", flush=True)
    return
```

- [ ] **Step 5: Run full test suite**

Run: `cd rag_demo_system && python -m pytest tests/ -v -x`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/app.py rag_demo_system/tests/test_listen_mode_state.py
git commit -m "feat(turn-taking): enter listen_mode on SessionAgent is_stop_request"
```

---

### Task 6: Lower RMS floor during listen_mode

**Files:**
- Modify: `rag_demo_system/backend/app.py` — barge-in RMS check around line 2543

- [ ] **Step 1: Replace hardcoded RMS floor**

Find the block (around `app.py:2529-2548`):

```python
if _prob >= 0.40 and _frame_rms >= 300:
```

Replace the literal `300` with a dynamic lookup:

```python
_rms_floor = settings.listen_mode_vad_rms if session.listen_mode else 300
if _prob >= 0.40 and _frame_rms >= _rms_floor:
```

- [ ] **Step 2: Similarly lower min-speech during listen_mode**

Find the min-speech check (around `app.py:2602-2610`):

```python
_min_bytes = 12800 if _was_bi else 25600
```

Replace with:

```python
if session.listen_mode:
    _min_bytes = int(settings.listen_mode_min_speech_ms * 32)  # 16-bit mono 16kHz → 32 bytes/ms
elif _was_bi:
    _min_bytes = 12800
else:
    _min_bytes = 25600
```

- [ ] **Step 3: Exit listen_mode on first new utterance**

When `session.assistant_speaking = True` is next set (i.e. bot is about to respond), reset listen_mode:

```python
# Immediately before `session.assistant_speaking = True` (line 2630):
if session.listen_mode:
    print(f"[listen_mode] exiting (user utterance received)", flush=True)
    session.listen_mode = False
    session.listen_mode_until = 0.0
```

- [ ] **Step 4: Smoke test on dev server (observational)**

Deploy. Make a SIP call. Say: *"Ксения, стоп."*. Expected: TTS cancels, bot goes silent. Wait ≥ 1 second, speak very quietly: *"да."*. Expected: VAD still catches it (RMS floor 180 instead of 300). Bot responds normally.

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/app.py
git commit -m "feat(turn-taking): dynamic RMS floor + shorter min-speech during listen_mode"
```

---

## Self-review

**Spec coverage:**
- Env vars (5 new) → Task 1 ✓
- ChatSession fields → Task 2 ✓
- `VAD_SILENCE_MS` plumbed → Task 3 ✓
- Pre-response hold helper + integration → Task 4 ✓
- Listen_mode entry on stop request → Task 5 ✓
- Auto-exit on timeout + "Слушаю Вас." prompt → Task 5 ✓
- Dynamic RMS floor + min-speech → Task 6 ✓
- Exit on new utterance → Task 6 ✓

**Placeholders:** none.

**Type consistency:**
- `_pre_response_hold(hold_ms: int, poll_fn: Optional[Callable[[], Awaitable[bool]]]) -> bool`.
- `_enter_listen_mode(session, websocket, timeout_sec: float) -> None`.
- `_listen_mode_auto_exit(session, websocket, timeout_sec: float) -> None`.
- `ChatSession.listen_mode: bool`, `listen_mode_until: float`.
- Used consistently in all tasks.
