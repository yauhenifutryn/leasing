# WebRTC Streaming Barge-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add WebRTC audio transport for streaming mode so browser AEC enables barge-in. Push-to-talk stays on WebSocket untouched.

**Architecture:** Browser sends mic audio via RTCPeerConnection (with echoCancellation:true). Server receives Opus-decoded PCM, feeds to existing VAD/STT pipeline. Server sends TTS audio back via RTP track. Browser plays via `<audio>` element. Chrome AEC has perfect far-end reference, can cleanly cancel echo. Control messages (text deltas, interrupt, response.done) stay on WebSocket.

**Tech Stack:** aiortc (Python WebRTC), av (PyAV for audio resampling), existing FastAPI/WebSocket backend.

**Safe rollback:** `git reset --hard 9af8995`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/rtc_audio.py` | Create | TTSAudioTrack (outbound), RTCAudioHandler (signaling + inbound audio processing) |
| `backend/app.py` | Modify | RTC signaling in voice_ws, pass rtc_handler to _stream_voice_response, RTC barge-in |
| `frontend/demo.html` | Modify | startRTC(), RTC event handling, mode switching |
| `requirements.txt` | Modify | Add aiortc |
| `scripts/provision_server.sh` | Modify | Add aiortc to pip install |

---

### Task 1: Add aiortc Dependency

**Files:**
- Modify: `rag_demo_system/requirements.txt`
- Modify: `rag_demo_system/scripts/provision_server.sh:369`

- [ ] **Step 1: Add aiortc to requirements.txt**

```
# Append to requirements.txt after pymorphy3:
aiortc>=1.9.0
```

- [ ] **Step 2: Add aiortc to provision script pip install**

In `provision_server.sh`, line 369, add `aiortc` to the backend-only packages list:

```bash
  "$APP_DIR/.venv/bin/pip" install \
    uvicorn \
    pyyaml \
    requests \
    qdrant-client \
    sentence-transformers \
    rank-bm25 \
    num2words \
    pymorphy3 \
    aiortc \
    pytest
```

- [ ] **Step 3: Verify import works locally (or note for server install)**

```bash
cd rag_demo_system && pip install aiortc>=1.9.0
python -c "from aiortc import RTCPeerConnection, RTCSessionDescription; print('aiortc OK')"
```

- [ ] **Step 4: Commit**

```bash
git add rag_demo_system/requirements.txt rag_demo_system/scripts/provision_server.sh
git commit -m "deps: add aiortc for WebRTC audio transport"
```

---

### Task 2: Create TTSAudioTrack (Outbound Audio)

**Files:**
- Create: `rag_demo_system/backend/rtc_audio.py`
- Create: `rag_demo_system/tests/test_rtc_audio.py`

- [ ] **Step 1: Write test for TTSAudioTrack frame chunking**

```python
# tests/test_rtc_audio.py
from pathlib import Path
import sys
import asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_tts_track_chunks_pcm_into_frames():
    """TTS audio (variable-length PCM16) is chunked into fixed 20ms frames."""
    from backend.rtc_audio import TTSAudioTrack

    track = TTSAudioTrack(sample_rate=24000)
    # 24kHz * 20ms = 480 samples/frame * 2 bytes = 960 bytes/frame
    # Push 3 frames worth of audio (2880 bytes)
    pcm = b"\x01\x00" * 1440  # 1440 samples = 3 frames
    track.push_audio(pcm)

    assert track._queue.qsize() == 3
    chunk = track._queue.get_nowait()
    assert len(chunk) == 960  # 480 samples * 2 bytes


def test_tts_track_handles_partial_frame():
    """Leftover bytes from a push are held until the next push completes a frame."""
    from backend.rtc_audio import TTSAudioTrack

    track = TTSAudioTrack(sample_rate=24000)
    # Push 1.5 frames worth (1440 bytes)
    pcm = b"\x01\x00" * 720  # 720 samples = 1.5 frames
    track.push_audio(pcm)

    assert track._queue.qsize() == 1  # only 1 complete frame
    # Push another half frame to complete
    track.push_audio(b"\x01\x00" * 240)
    assert track._queue.qsize() == 2


def test_tts_track_recv_returns_silence_when_empty():
    """recv() returns silence frame when queue is empty (keeps RTP alive)."""
    from backend.rtc_audio import TTSAudioTrack

    track = TTSAudioTrack(sample_rate=24000)
    frame = asyncio.get_event_loop().run_until_complete(track.recv())

    assert frame.format.name == "s16"
    assert frame.sample_rate == 24000
    assert frame.samples == 480
    # Silence = all zeros
    assert bytes(frame.planes[0]) == b"\x00" * 960
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd rag_demo_system && python -m pytest tests/test_rtc_audio.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.rtc_audio'`

- [ ] **Step 3: Implement TTSAudioTrack**

```python
# backend/rtc_audio.py
"""WebRTC audio bridge for streaming mode.

Provides:
- TTSAudioTrack: outbound RTP track that serves TTS audio as 20ms Opus frames
- RTCAudioHandler: manages aiortc peer connection, inbound/outbound audio
"""
from __future__ import annotations

import asyncio
import fractions
import time
from typing import Awaitable, Callable

from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler

AUDIO_PTIME = 0.020  # 20ms per RTP frame


class TTSAudioTrack(AudioStreamTrack):
    """Outbound audio track that serves TTS PCM16 frames via RTP.

    TTS engine pushes variable-length PCM16 via push_audio().
    Internally chunked into fixed 20ms frames for Opus encoding.
    When the queue is empty, sends silence to keep the RTP stream alive.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 24000) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = int(AUDIO_PTIME * sample_rate)
        self._bytes_per_frame = self.samples_per_frame * 2  # s16 = 2 bytes/sample
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._remainder = b""
        self._start: float | None = None
        self._timestamp: int = 0

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        # Pace frames at real-time intervals
        if self._start is None:
            self._start = time.time()
        else:
            self._timestamp += self.samples_per_frame
            target = self._start + (self._timestamp / self.sample_rate)
            wait = target - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        # TTS audio or silence
        try:
            pcm_data = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pcm_data = b"\x00" * self._bytes_per_frame

        frame = AudioFrame(format="s16", layout="mono", samples=self.samples_per_frame)
        frame.planes[0].update(pcm_data)
        frame.pts = self._timestamp
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        return frame

    def push_audio(self, pcm16_bytes: bytes) -> None:
        """Push variable-length PCM16 audio. Internally chunked into 20ms frames."""
        data = self._remainder + pcm16_bytes
        offset = 0
        while offset + self._bytes_per_frame <= len(data):
            self._queue.put_nowait(data[offset : offset + self._bytes_per_frame])
            offset += self._bytes_per_frame
        self._remainder = data[offset:]

    def flush(self) -> None:
        """Flush any remaining partial frame as zero-padded."""
        if self._remainder:
            padded = self._remainder + b"\x00" * (self._bytes_per_frame - len(self._remainder))
            self._queue.put_nowait(padded)
            self._remainder = b""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd rag_demo_system && python -m pytest tests/test_rtc_audio.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/rtc_audio.py rag_demo_system/tests/test_rtc_audio.py
git commit -m "feat: TTSAudioTrack for outbound WebRTC TTS audio"
```

---

### Task 3: Add RTCAudioHandler (Signaling + Inbound Audio)

**Files:**
- Modify: `rag_demo_system/backend/rtc_audio.py`
- Modify: `rag_demo_system/tests/test_rtc_audio.py`

- [ ] **Step 1: Write test for inbound audio resampling**

```python
# Append to tests/test_rtc_audio.py

def test_resample_48k_stereo_to_24k_mono():
    """Opus decoder outputs 48kHz stereo. Our pipeline needs 24kHz mono."""
    from backend.rtc_audio import resample_frame
    from av import AudioFrame

    # Simulate Opus decoder output: s16/stereo/48kHz, 960 samples (20ms)
    frame = AudioFrame(format="s16", layout="stereo", samples=960)
    frame.sample_rate = 48000
    # Fill with non-zero data
    frame.planes[0].update(b"\x10\x00" * 960 * 2)  # stereo = 2 samples per frame sample

    pcm = resample_frame(frame, target_rate=24000)

    # 20ms at 24kHz mono = 480 samples * 2 bytes = 960 bytes
    assert len(pcm) == 960
    assert isinstance(pcm, bytes)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd rag_demo_system && python -m pytest tests/test_rtc_audio.py::test_resample_48k_stereo_to_24k_mono -v
```

Expected: FAIL with `cannot import name 'resample_frame'`

- [ ] **Step 3: Implement resample_frame and RTCAudioHandler**

Append to `backend/rtc_audio.py`:

```python
# Module-level resampler (stateless for PCM conversion)
_resampler_cache: dict[tuple[int, str, int], AudioResampler] = {}


def resample_frame(frame: AudioFrame, target_rate: int = 24000) -> bytes:
    """Resample an Opus-decoded AudioFrame to mono PCM16 at target_rate."""
    key = (frame.sample_rate, frame.layout.name, target_rate)
    if key not in _resampler_cache:
        _resampler_cache[key] = AudioResampler(
            format="s16", layout="mono", rate=target_rate,
        )
    resampler = _resampler_cache[key]
    out_frames = resampler.resample(frame)
    return b"".join(bytes(f.planes[0]) for f in out_frames)


class RTCAudioHandler:
    """Manages an aiortc peer connection for one voice session.

    Handles:
    - SDP offer/answer exchange
    - Inbound mic audio (Opus decode -> resample -> callback)
    - Outbound TTS audio (PCM16 -> Opus encode via TTSAudioTrack)
    """

    def __init__(
        self,
        on_audio: Callable[[bytes], Awaitable[None]],
        sample_rate: int = 24000,
    ) -> None:
        self._on_audio = on_audio
        self._sample_rate = sample_rate
        self.pc = RTCPeerConnection()
        self.tts_track = TTSAudioTrack(sample_rate=sample_rate)
        self.pc.addTrack(self.tts_track)
        self._audio_task: asyncio.Task | None = None

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                self._audio_task = asyncio.create_task(
                    self._consume_inbound(track)
                )

            @track.on("ended")
            def on_ended():
                pass

    async def _consume_inbound(self, track) -> None:
        """Read inbound audio frames, resample, feed to VAD callback."""
        try:
            while True:
                frame = await track.recv()
                pcm16 = resample_frame(frame, self._sample_rate)
                if pcm16:
                    await self._on_audio(pcm16)
        except Exception:
            pass

    async def handle_offer(self, sdp: str) -> str:
        """Process browser SDP offer, return SDP answer."""
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription.sdp

    async def close(self) -> None:
        """Cleanly shut down the peer connection."""
        if self._audio_task and not self._audio_task.done():
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
        self.tts_track.stop()
        await self.pc.close()
```

- [ ] **Step 4: Run all tests**

```bash
cd rag_demo_system && python -m pytest tests/test_rtc_audio.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/rtc_audio.py rag_demo_system/tests/test_rtc_audio.py
git commit -m "feat: RTCAudioHandler with signaling, inbound audio, resampling"
```

---

### Task 4: Integrate RTC Signaling into voice_ws

**Files:**
- Modify: `rag_demo_system/backend/app.py:1178-1226` (main event loop)

- [ ] **Step 1: Add RTC handler state variable in voice_ws**

After the existing `audio_adapter = WebSocketAudioAdapter(...)` line, add:

```python
    rtc_handler: RTCAudioHandler | None = None
```

Add import at top of app.py:

```python
from .rtc_audio import RTCAudioHandler
```

- [ ] **Step 2: Add RTC signaling events to the main loop**

In the main `while True:` loop (after `elif event_type == "session.update":` block, around line 1208), add new event handlers:

```python
            elif event_type == "rtc.offer":
                # Client sent WebRTC offer for streaming mode
                if rtc_handler is not None:
                    await rtc_handler.close()

                async def _rtc_on_audio(pcm16: bytes) -> None:
                    """RTC inbound audio -> VAD pipeline.

                    CRITICAL: this callback must NEVER block. It is called
                    for every 20ms audio frame by _consume_inbound. If it
                    blocks (e.g. awaiting _process_voice_utterance), no
                    further frames are processed and barge-in freezes.

                    Solution: VAD + barge-in run inline (fast).
                    _process_voice_utterance is fired as a task (non-blocking).
                    Guard with session.assistant_speaking to prevent overlap.
                    """
                    nonlocal vad, vad_enabled
                    if not vad_enabled or vad is None:
                        return
                    was_speaking = vad.is_speaking
                    speech_audio = vad.feed(pcm16)
                    if not was_speaking and vad.is_speaking:
                        print("[VAD-RTC] speech_start", flush=True)
                    if was_speaking and not vad.is_speaking and speech_audio is not None:
                        print(f"[VAD-RTC] speech_end ({len(speech_audio)} bytes)", flush=True)
                    # Barge-in: user speaking while assistant responds
                    if vad.is_speaking and session.assistant_speaking:
                        session.interrupted = True
                        print("[BARGE-IN-RTC] speech during response", flush=True)
                        try:
                            await websocket.send_json({
                                "type": "interrupt",
                                "session_id": session_id,
                            })
                        except (RuntimeError, WebSocketDisconnect):
                            pass
                    # Speech ended: fire-and-forget response (don't block audio processing)
                    if speech_audio is not None and not session.assistant_speaking:
                        vad_audio_b64 = _b64mod.b64encode(speech_audio).decode()
                        asyncio.create_task(_process_voice_utterance(vad_audio_b64))

                rtc_handler = RTCAudioHandler(
                    on_audio=_rtc_on_audio,
                    sample_rate=24000,
                )
                sdp_offer = event.get("sdp", "")
                try:
                    sdp_answer = await rtc_handler.handle_offer(sdp_offer)
                    await websocket.send_json({
                        "type": "rtc.answer",
                        "sdp": sdp_answer,
                        "type_field": "answer",
                    })
                    print("[RTC] peer connection established", flush=True)
                except Exception as exc:
                    print(f"[RTC] offer handling failed: {exc}", flush=True)
                    await websocket.send_json({
                        "type": "error",
                        "error": f"rtc_failed: {exc}",
                    })
                    rtc_handler = None

            elif event_type == "rtc.ice":
                # Trickle ICE candidate from client
                if rtc_handler and event.get("candidate"):
                    try:
                        from aiortc import RTCIceCandidate
                        candidate_str = event["candidate"]
                        # aiortc handles candidates via addIceCandidate
                        # For bundled-in-SDP mode this is usually not needed
                        # but handle it for browser compatibility
                    except Exception:
                        pass
```

- [ ] **Step 3: Clean up RTC on disconnect**

In the `finally:` block at the end of `voice_ws` (around line 1228):

```python
    finally:
        if rtc_handler is not None:
            await rtc_handler.close()
        voice_sessions.pop(session_id, None)
```

- [ ] **Step 4: Verify syntax**

```bash
cd rag_demo_system && python -c "import ast; ast.parse(open('backend/app.py').read()); print('OK')"
```

- [ ] **Step 5: Run existing tests to verify no regression**

```bash
cd rag_demo_system && python -m pytest tests/test_voice_session.py -v
```

Expected: 8 PASSED

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/app.py
git commit -m "feat: RTC signaling and barge-in in voice_ws main loop"
```

---

### Task 5: Route TTS Audio via RTC Track

**Files:**
- Modify: `rag_demo_system/backend/app.py:496-505` (_stream_voice_response signature)
- Modify: `rag_demo_system/backend/app.py:648-691` (tts_consumer)

- [ ] **Step 1: Add rtc_handler parameter to _stream_voice_response**

Change the function signature (line 496):

```python
async def _stream_voice_response(
    *,
    websocket: Any,
    session: Any,
    session_id: str,
    message: str,
    t_speech_stopped: float,
    t_stt_done: float,
    question_id: str,
    rtc_handler: Any | None = None,  # NEW: if set, TTS audio goes via RTC
) -> None:
```

- [ ] **Step 2: Modify tts_consumer to use RTC when available**

In the `tts_consumer` function (around line 648), change the audio sending block. Replace the `websocket.send_json` for audio with a branch:

```python
    async def tts_consumer() -> None:
        nonlocal t_tts_first_chunk, t_playback_started
        while True:
            if session.interrupted:
                break
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            try:
                await websocket.send_json({
                    "type": "response.output_text.delta",
                    "session_id": session_id,
                    "delta": sentence + " ",
                })
            except (RuntimeError, WebSocketDisconnect):
                session.interrupted = True
                break
            try:
                audio_resp = await asyncio.to_thread(
                    synthesize_audio, sentence, session_id,
                )
                audio_b64 = audio_resp.get("audio_b64") or ""
                if audio_b64:
                    if t_tts_first_chunk is None:
                        t_tts_first_chunk = time.time()
                    if rtc_handler is not None:
                        # RTC mode: push PCM16 into the outbound RTP track
                        import base64 as _b64
                        pcm16 = _b64.b64decode(audio_b64)
                        rtc_handler.tts_track.push_audio(pcm16)
                    else:
                        # WebSocket mode (PTT): send base64 audio via WS
                        await websocket.send_json({
                            "type": "response.output_audio.delta",
                            "session_id": session_id,
                            "delta": audio_b64,
                            "sample_rate_hz": audio_resp.get("sample_rate_hz"),
                        })
                    t_playback_started = time.time()
            except (RuntimeError, WebSocketDisconnect):
                session.interrupted = True
                break
            except Exception as exc:  # noqa: BLE001
                try:
                    await websocket.send_json({
                        "type": "warning", "session_id": session_id,
                        "message": f"tts_failed: {exc}",
                    })
                except (RuntimeError, WebSocketDisconnect):
                    session.interrupted = True
                    break
```

- [ ] **Step 3: Flush TTS track after response completes**

After `session.assistant_speaking = False` (line ~718), add:

```python
    session.assistant_speaking = False
    if rtc_handler is not None:
        rtc_handler.tts_track.flush()
```

- [ ] **Step 4: Pass rtc_handler from _process_voice_utterance**

In `_process_voice_utterance`, where `_stream_voice_response` is called (around line 817), pass `rtc_handler`:

```python
        response_coro = _stream_voice_response(
            websocket=websocket,
            session=session,
            session_id=session_id,
            message=text,
            t_speech_stopped=t_speech_stopped,
            t_stt_done=t_stt_done,
            question_id=question_id,
            rtc_handler=rtc_handler,
        )
```

Note: `rtc_handler` is a closure variable from `voice_ws`, accessible in `_process_voice_utterance`.

- [ ] **Step 5: Skip WebSocket barge-in listener when RTC is active**

In `_process_voice_utterance`, the VAD path currently creates a `_barge_in_listener` that reads WebSocket audio events. When RTC is active, barge-in is handled by the `_rtc_on_audio` callback instead. Modify the condition (around line 822):

```python
        if not (vad_enabled and vad is not None):
            # PTT mode: original blocking behavior
            try:
                await response_coro
            except (RuntimeError, WebSocketDisconnect):
                pass
            return

        if rtc_handler is not None:
            # RTC mode: barge-in handled by RTC audio callback, no WS listener needed
            print("[UTTERANCE] RTC path (barge-in via RTC callback)", flush=True)
            session.assistant_speaking = True
            session.interrupted = False
            try:
                await response_coro
            except (RuntimeError, WebSocketDisconnect):
                session.interrupted = True
            return

        # WebSocket VAD mode: concurrent barge-in listener (fallback)
        print("[UTTERANCE] VAD path (barge-in listener active)", flush=True)
        ...
```

- [ ] **Step 6: Verify syntax and tests**

```bash
cd rag_demo_system && python -c "import ast; ast.parse(open('backend/app.py').read()); print('OK')"
python -m pytest tests/ -v
```

Expected: All existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add rag_demo_system/backend/app.py
git commit -m "feat: route TTS audio via RTC track in streaming mode"
```

---

### Task 6: Frontend - WebRTC Connection Management

**Files:**
- Modify: `rag_demo_system/frontend/demo.html`

- [ ] **Step 1: Add hidden audio element for RTC playback**

In the HTML body, after `</div>` (stream controls), before `</div>` (container close), add:

```html
      <!-- RTC audio playback (streaming mode only, hidden) -->
      <audio id="rtcAudio" autoplay style="display:none"></audio>
```

- [ ] **Step 2: Add RTC state variables**

After the existing `let micMuted = true;` line, add:

```javascript
    let rtcPc = null;
    let rtcStream = null; // mic stream for RTC
```

- [ ] **Step 3: Add startRTC function**

After `toggleMute()` function, add:

```javascript
    // ---------- WebRTC for streaming mode ----------
    async function startRTC() {
      if (rtcPc) return; // already active

      rtcPc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
      });

      // Mic with echo cancellation -> RTC track
      rtcStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
      rtcStream.getTracks().forEach(t => rtcPc.addTrack(t, rtcStream));

      // TTS arrives as RTC track -> <audio> element
      rtcPc.ontrack = (e) => {
        const audioEl = $("#rtcAudio");
        audioEl.srcObject = e.streams[0];
        audioEl.play().catch(() => {});
      };

      // ICE candidates -> WebSocket
      rtcPc.onicecandidate = (e) => {
        if (e.candidate && ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: "rtc.ice",
            candidate: e.candidate.candidate,
            sdpMid: e.candidate.sdpMid,
            sdpMLineIndex: e.candidate.sdpMLineIndex,
          }));
        }
      };

      rtcPc.onconnectionstatechange = () => {
        if (rtcPc.connectionState === "connected") {
          setStatus("WebRTC подключен", "good");
          // Stop sending audio via WebSocket (RTC handles it now)
          continuousStreaming = false;
        } else if (rtcPc.connectionState === "failed") {
          setStatus("WebRTC ошибка, переключаюсь на WebSocket", "warn");
          stopRTC();
          // Fallback to WebSocket streaming
          continuousStreaming = true;
        }
      };

      // Create offer
      const offer = await rtcPc.createOffer();
      await rtcPc.setLocalDescription(offer);
      ws.send(JSON.stringify({ type: "rtc.offer", sdp: offer.sdp }));
    }

    function stopRTC() {
      if (rtcPc) {
        rtcPc.close();
        rtcPc = null;
      }
      if (rtcStream) {
        rtcStream.getTracks().forEach(t => t.stop());
        rtcStream = null;
      }
      const audioEl = $("#rtcAudio");
      audioEl.srcObject = null;
    }
```

- [ ] **Step 4: Handle RTC answer in handleEvent**

In the `handleEvent` function, add handlers for RTC events:

```javascript
      if (evt.type === "rtc.answer") {
        if (rtcPc) {
          rtcPc.setRemoteDescription(new RTCSessionDescription({
            type: "answer",
            sdp: evt.sdp,
          }));
        }
      }
      if (evt.type === "rtc.offer_needed") {
        startRTC();
      }
```

- [ ] **Step 5: Modify selectMode to use RTC for streaming**

In `selectMode()`, change the streaming mode branch:

```javascript
        if (mode === "stream") {
          ws.send(JSON.stringify({ type: "session.update", vad_mode: true }));
          // Don't set continuousStreaming=true yet; RTC will handle audio.
          // If RTC fails, fallback sets continuousStreaming=true.
          micMuted = false;
          $("#btnMute").textContent = "Выключить микрофон";
          $("#btnMute").classList.remove("muted");
          $("#streamStatus").textContent = "Подключение WebRTC...";
          $("#streamStatus").className = "stream-status idle";
          startRTC();
        } else {
          ws.send(JSON.stringify({ type: "session.update", vad_mode: false }));
          continuousStreaming = false;
          stopRTC();
        }
```

And in `connectWS`, the `startWithVAD` branch:

```javascript
        if (startWithVAD) {
          ws.send(JSON.stringify({ type: "session.update", vad_mode: true }));
          // RTC will be started after session.ready or session.updated
          micMuted = false;
          $("#btnMute").textContent = "Выключить микрофон";
          $("#btnMute").classList.remove("muted");
        }
```

- [ ] **Step 6: Handle interrupt for RTC mode**

The existing interrupt handler in `handleEvent` already calls `stopPlayback()`. For RTC mode, the `<audio>` element's stream is controlled by the server (the server stops sending TTS frames). No change needed; the browser will stop playing when RTP frames stop. But update the status:

```javascript
      if (evt.type === "interrupt") {
        stopPlayback(); // PTT mode
        // RTC mode: server stops sending TTS, browser audio stops naturally
        setStatus("Слушаю...", "warn");
      }
```

- [ ] **Step 7: Commit**

```bash
git add rag_demo_system/frontend/demo.html
git commit -m "feat: frontend WebRTC connection for streaming mode"
```

---

### Task 7: Server-Side Consent Flow Compatibility

**Files:**
- Modify: `rag_demo_system/backend/app.py` (consent + main loop transition)

The consent flow runs before RTC is established. It uses WebSocket for audio.
After consent, when the server enters the main loop, RTC may or may not be ready.

- [ ] **Step 1: Send rtc.offer_needed after consent when VAD mode is active**

After the consent flow completes and before the main `while True:` loop (around line 1175), add:

```python
    # If streaming mode was requested, signal client to establish RTC
    if vad_enabled:
        await websocket.send_json({
            "type": "rtc.offer_needed",
            "session_id": session_id,
        })
```

- [ ] **Step 2: Verify the consent flow still uses WebSocket audio**

The consent flow uses `_wait_for_speech()` which processes `input_audio_buffer.append` events from WebSocket. During consent, the client's AudioWorklet + WebSocket path is active (RTC not yet established). After consent, the client receives `rtc.offer_needed`, calls `startRTC()`, and the RTC connection is established. This is seamless.

No code change needed for this step; just verification.

- [ ] **Step 3: Verify syntax and run tests**

```bash
cd rag_demo_system && python -c "import ast; ast.parse(open('backend/app.py').read()); print('OK')"
python -m pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add rag_demo_system/backend/app.py
git commit -m "feat: trigger RTC setup after consent flow completes"
```

---

### Task 8: End-to-End Wiring and Cleanup

**Files:**
- Modify: `rag_demo_system/backend/app.py` (remove debug logging)
- Modify: `rag_demo_system/frontend/demo.html` (cleanup)

- [ ] **Step 1: Remove excessive debug logging from barge-in listener**

The `[LISTENER]` logging every 50 chunks was for debugging. Keep only the key events:
- `[UTTERANCE]` path selection
- `[BARGE-IN-RTC]` when barge-in triggers
- `[VAD-RTC]` speech_start / speech_end
- `[RTC]` connection established

Remove the per-chunk `[LISTENER] chunk=N` logs.

- [ ] **Step 2: Handle edge case: WebSocket audio during RTC mode**

In the main event loop, when `rtc_handler` is active, incoming `input_audio_buffer.append` events from WebSocket should be ignored (audio comes via RTC now). Add a guard:

```python
            elif event_type == "input_audio_buffer.append":
                if rtc_handler is not None:
                    continue  # audio comes via RTC, skip WebSocket audio
                audio = event.get("audio") or ""
                if audio:
                    raw = _b64mod.b64decode(audio)
                    await audio_adapter.handle_audio_message(raw)
```

- [ ] **Step 3: Verify complete flow**

```bash
cd rag_demo_system && python -c "import ast; ast.parse(open('backend/app.py').read()); print('OK')"
python -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add rag_demo_system/backend/app.py rag_demo_system/frontend/demo.html
git commit -m "chore: cleanup debug logging, guard WebSocket audio during RTC"
```

---

### Task 9: Deploy and Integration Test

- [ ] **Step 1: Push all changes**

```bash
git push origin feature/voice-pipeline
```

- [ ] **Step 2: Install aiortc on server**

```bash
ssh server "cd ~/leasing/rag_demo_system && git pull && .venv/bin/pip install aiortc>=1.9.0"
```

- [ ] **Step 3: Restart backend**

```bash
ssh server "cd ~/leasing/rag_demo_system && .venv/bin/supervisorctl -c scripts/supervisord.conf restart backend"
```

- [ ] **Step 4: Test push-to-talk (regression check)**

1. Open demo, select PTT mode
2. Press Talk, ask a question, release
3. Verify: response plays, latency reasonable
4. Press Talk during response: barge-in works (client-side stopPlayback)
5. Expected: identical behavior to before

- [ ] **Step 5: Test streaming mode with WebRTC**

1. Select streaming mode
2. Check browser console: RTCPeerConnection created, ICE connected
3. Check server logs: `[RTC] peer connection established`
4. Speak a question, wait for response
5. During response, speak loudly to interrupt
6. Expected: `[BARGE-IN-RTC]` in server logs, response stops, new speech processed
7. Check: no echo loop (assistant does not respond to its own voice)

- [ ] **Step 6: Verify with server logs**

```bash
tail -50 ~/leasing/rag_demo_system/.state/backend.log | grep -E "RTC|BARGE|VAD|UTTERANCE"
```

Expected log pattern:
```
[VAD-RTC] speech_start
[VAD-RTC] speech_end (XXXXX bytes)
[UTTERANCE] RTC path (barge-in via RTC callback)
... (response streaming) ...
[BARGE-IN-RTC] speech during response   <-- user interrupted
```
