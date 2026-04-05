# WebRTC Streaming Mode with Barge-In

**Date:** 2026-04-05
**Branch:** feature/voice-pipeline
**Safe rollback commit:** 9af8995

## Problem

Streaming mode (VAD-based continuous conversation) cannot support barge-in (user interrupting the assistant mid-speech) because the browser microphone picks up TTS audio from speakers. Without echo cancellation, the VAD either detects the assistant's own voice as user speech (feedback loop) or, with browser echoCancellation enabled, the AEC over-suppresses and blanks the entire mic signal.

Every production voice assistant (ElevenLabs, OpenAI Realtime, Vapi, LiveKit) solved this by switching audio transport to WebRTC, which gives the browser a precise far-end reference signal for its AEC.

## Scope

- Add WebRTC audio transport for **streaming mode only**
- Push-to-talk stays on WebSocket (unchanged, working)
- Backend VAD/STT/LLM/TTS pipeline internals unchanged
- Control messages (text deltas, response.done, interrupt) stay on WebSocket

## Architecture

```
Streaming mode (WebRTC):
  Browser getUserMedia(echoCancellation:true)
    -> RTCPeerConnection audio track -> [RTP/UDP] -> aiortc server
    -> Opus decode -> PCM16 -> VAD -> STT -> LLM -> TTS
    -> PCM16 -> Opus encode -> RTP track -> Browser <audio> element
    -> Chrome AEC has perfect reference -> mic signal is clean

Push-to-talk (WebSocket, unchanged):
  Browser AudioWorklet -> base64 -> WebSocket -> server
  Server TTS -> base64 -> WebSocket -> AudioContext playback

Control messages (both modes):
  WebSocket for: session.init, session.update, response.cancel,
  response.done, interrupt, text deltas, transcription events
```

## Signaling Flow

When user selects streaming mode:

1. Client sends `{type: "session.update", vad_mode: true}` via WebSocket
2. Server responds `{type: "rtc.offer_needed"}`
3. Client creates RTCPeerConnection with mic track (echoCancellation:true)
4. Client sends `{type: "rtc.offer", sdp: "..."}` via WebSocket
5. Server creates aiortc RTCPeerConnection, adds TTS output track, creates SDP answer
6. Server sends `{type: "rtc.answer", sdp: "..."}` via WebSocket
7. ICE candidates exchanged via WebSocket: `{type: "rtc.ice", candidate: "..."}`
8. RTP audio flows. WebSocket stays open for control messages.

When user switches to PTT or disconnects:
- RTC peer connection closed
- Falls back to WebSocket-only audio

## Server-Side: New File backend/rtc_audio.py

```python
class RTCAudioHandler:
    """Bridges aiortc RTC tracks with the existing VAD/STT pipeline."""

    def __init__(self, on_audio: Callable[[bytes], Awaitable[None]]):
        self.pc = RTCPeerConnection()
        self._tts_track = AudioStreamTrack()  # outbound TTS audio
        self.pc.addTrack(self._tts_track)
        self._on_audio = on_audio

    async def handle_inbound_audio(self, frame: AudioFrame):
        """Called when browser sends mic audio via RTP."""
        pcm16 = frame.to_ndarray().tobytes()
        await self._on_audio(pcm16)  # feeds existing VAD pipeline

    async def send_tts_audio(self, pcm16: bytes, sample_rate: int):
        """Push TTS audio into the outbound RTP track."""
        frame = AudioFrame.from_ndarray(pcm16_to_ndarray(pcm16))
        self._tts_track.queue_frame(frame)
```

## Server-Side: Changes to app.py

### voice_ws handler (main event loop)

New event types handled in the main WebSocket loop:
- `rtc.offer`: create aiortc peer connection, set remote SDP, return answer
- `rtc.ice`: add ICE candidate to peer connection

### _stream_voice_response

When RTC is active:
- `tts_consumer` calls `rtc_handler.send_tts_audio()` instead of `websocket.send_json(audio_delta)`
- Text deltas and `response.done` still go via WebSocket

### Barge-in with RTC

When RTC is active, the audio flow is fundamentally different from WebSocket:
- Inbound mic audio arrives via aiortc track callback (not WebSocket events)
- The `RTCAudioHandler.handle_inbound_audio` callback feeds PCM16 to the same VAD
- Barge-in detection moves INTO the RTC audio callback: when VAD detects speech during `session.assistant_speaking`, set `session.interrupted = True` and send `{type: "interrupt"}` via WebSocket
- The current `_barge_in_listener` (which reads WebSocket audio events) is NOT used in RTC mode
- The RTC audio callback runs independently of the LLM producer (no event loop blocking issue), so barge-in detection is real-time

### What does NOT change

- Push-to-talk path (all WebSocket)
- Consent flow (`_wait_for_speech` reads WebSocket events, runs before mode selection)
- VAD module (vad.py)
- STT, LLM, TTS pipeline internals
- Session/state management
- Text normalization, router, consent detection

## Frontend: Changes to demo.html

### New: RTC connection management

```javascript
let rtcPc = null;

async function startRTC() {
    rtcPc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
    });

    // Mic with AEC -> RTC track
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
    stream.getTracks().forEach(t => rtcPc.addTrack(t, stream));

    // TTS arrives as RTC track -> <audio> element
    rtcPc.ontrack = (e) => {
        document.getElementById("rtcAudio").srcObject = e.streams[0];
    };

    // ICE -> WebSocket
    rtcPc.onicecandidate = (e) => {
        if (e.candidate) ws.send(JSON.stringify({
            type: "rtc.ice", candidate: e.candidate
        }));
    };

    const offer = await rtcPc.createOffer();
    await rtcPc.setLocalDescription(offer);
    ws.send(JSON.stringify({ type: "rtc.offer", sdp: offer.sdp }));
}
```

### Mode selection changes

- `selectMode("stream")` calls `startRTC()`
- `selectMode("ptt")` closes RTC if open, reverts to WebSocket audio
- New hidden `<audio id="rtcAudio" autoplay>` element in HTML

### What stays unchanged

- PTT button, AudioWorklet, base64 WebSocket audio path
- `playPcm()` function (used only in PTT mode)
- Transcript display, status bar, download, mode tabs UI

## Dependencies

```
# Added to requirements.txt
aiortc>=1.9.0
```

Transitive: av (PyAV), cffi, cryptography (aiortc bundles libsrtp).

Provision script: `pip install aiortc` in the main venv.

## STUN/TURN

- Development: `stun:stun.l.google.com:19302` (free, public)
- Production: self-hosted coturn or commercial TURN for NAT traversal
- Current demo uses ngrok (public IP), so STUN alone is sufficient

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| aiortc Opus latency | Benchmark; Opus encode is ~1ms/frame |
| ICE fails behind NAT | ngrok gives public IP; TURN for production |
| Audio quality change | Opus 48kHz is perceptually lossless; A/B test |
| Streaming mode breaks | PTT untouched; immediate fallback |
| aiortc + uvicorn workers | Single worker handles both WS and RTC for a voice session |
| Consent flow affected | Consent runs before mode selection; uses WebSocket; unaffected |

## Success Criteria

1. Streaming mode: user can interrupt the assistant mid-speech, assistant stops, processes the interruption
2. No echo feedback loop (VAD does not detect TTS as speech)
3. Push-to-talk works identically to current behavior
4. Consent flow works identically
5. Latency comparable to current WebSocket path (< 200ms additional)
6. Works on Chrome desktop (primary target); Safari/Firefox stretch goals
