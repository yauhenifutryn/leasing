#!/usr/bin/env python3
"""Minimal standalone AudioSocket test server.

Tests the SIP adapter code without starting the full app (no RAG, no LLM,
no GPU dependencies). Runs on any Mac/Linux with Python 3.10+ and scipy.

What it does:
    1. Starts an AudioSocket TCP server on port 9092
    2. Starts a FastAPI app on port 8000 with /ws/sip-monitor endpoint
    3. When a client connects: reads UUID, logs caller, feeds audio to VAD
    4. On speech detection: logs it, sends back a canned TTS tone
    5. Broadcasts all events to the monitor page

Usage:
    cd rag_demo_system
    python tests/sip_test_server.py

    Then in another terminal:
    python tests/test_sip_client.py

    And open http://localhost:8000/sip_monitor.html in your browser.
"""
from __future__ import annotations

import asyncio
import base64
import math
import os
import struct
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.sip_audio import (
    SIPAudioAdapter,
    build_audio_frame,
    resample_24k_to_8k,
    query_caller_id_ami,
)

# ---------------------------------------------------------------------------
# Minimal VAD stub (no torch dependency)
# ---------------------------------------------------------------------------

class SimpleEnergyVAD:
    """Energy-based VAD for testing. No torch required.

    Detects speech by comparing RMS energy to a threshold.
    """

    def __init__(self, sample_rate: int = 16000, silence_ms: int = 500, threshold: float = 500.0):
        self.sample_rate = sample_rate
        self.silence_ms = silence_ms
        self.threshold = threshold
        self._is_speaking = False
        self._speech_buffer = b""
        self._silence_samples = 0
        self._silence_threshold = int(sample_rate * silence_ms / 1000)

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def feed(self, pcm16_bytes: bytes) -> bytes | None:
        n_samples = len(pcm16_bytes) // 2
        if n_samples == 0:
            return None
        samples = struct.unpack(f"<{n_samples}h", pcm16_bytes)
        rms = math.sqrt(sum(s * s for s in samples) / n_samples)

        if rms >= self.threshold:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_buffer = pcm16_bytes
            else:
                self._speech_buffer += pcm16_bytes
            self._silence_samples = 0
            return None

        if self._is_speaking:
            self._speech_buffer += pcm16_bytes
            self._silence_samples += n_samples
            if self._silence_samples >= self._silence_threshold:
                audio = self._speech_buffer
                self._speech_buffer = b""
                self._is_speaking = False
                self._silence_samples = 0
                return audio

        return None

    def reset(self):
        self._is_speaking = False
        self._speech_buffer = b""
        self._silence_samples = 0


# ---------------------------------------------------------------------------
# Monitor broadcast (standalone, no FastAPI WebSocket)
# ---------------------------------------------------------------------------

monitor_clients: set = set()


async def broadcast(event: dict[str, Any]) -> None:
    for ws in list(monitor_clients):
        try:
            await ws.send_json(event)
        except Exception:
            monitor_clients.discard(ws)
    # Also print to terminal
    etype = event.get("type", "")
    call_id = event.get("call_id", "")[:8]
    detail = ""
    if "text" in event:
        detail = f" {event['text'][:60]}"
    elif "digit" in event:
        detail = f" {event['digit']}"
    elif "phone" in event:
        detail = f" {event['phone']}"
    elif "event" in event:
        detail = f" {event['event']}"
    print(f"  [MONITOR] {etype} [{call_id}]{detail}", flush=True)


# ---------------------------------------------------------------------------
# Generate canned TTS response (sine tone at 24kHz, then resample to 8k)
# ---------------------------------------------------------------------------

def generate_response_audio(duration_ms: int = 2000) -> bytes:
    """Generate a 24kHz sine tone, resample to 8kHz, return AudioSocket frames."""
    sample_rate = 24000
    n_samples = int(sample_rate * duration_ms / 1000)
    freq = 440
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        value = int(8000 * math.sin(2 * math.pi * freq * t))
        samples.append(max(-32768, min(32767, value)))
    pcm_24k = struct.pack(f"<{n_samples}h", *samples)

    # Resample to 8kHz
    pcm_8k = resample_24k_to_8k(pcm_24k)

    # Split into 20ms frames (160 samples at 8kHz)
    frame_size = 320  # 160 samples * 2 bytes
    frames = b""
    for i in range(0, len(pcm_8k), frame_size):
        chunk = pcm_8k[i:i + frame_size]
        if len(chunk) == frame_size:
            frames += build_audio_frame(chunk)
    return frames


# ---------------------------------------------------------------------------
# SIP call handler (standalone version)
# ---------------------------------------------------------------------------

async def handle_sip_call(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle one AudioSocket connection."""
    adapter = SIPAudioAdapter(reader, writer)
    session_id = ""

    try:
        # 1. Read UUID
        first = await adapter.read_next()
        if first is None or first.get("type") != "uuid":
            print("[SIP] No UUID frame, closing", flush=True)
            await adapter.close()
            return
        session_id = first["uuid"]
        print(f"\n[SIP:{session_id[:8]}] === CALL CONNECTED ===", flush=True)

        await broadcast({"type": "sip.call.start", "call_id": session_id, "phone": "test-client"})

        # 2. Create VAD (energy-based, no torch)
        vad = SimpleEnergyVAD(sample_rate=16000, silence_ms=500, threshold=500.0)

        # 3. Send intro audio
        print(f"[SIP:{session_id[:8]}] Sending intro TTS (2s tone)...", flush=True)
        intro_audio = generate_response_audio(2000)
        writer.write(intro_audio)
        await writer.drain()
        await broadcast({"type": "sip.tts.start", "call_id": session_id, "text": "[intro greeting]"})

        # 4. Audio loop
        speech_count = 0
        frame_count = 0
        t_start = time.time()

        while True:
            frame = await adapter.read_next()
            if frame is None or frame["type"] == "hangup":
                duration = time.time() - t_start
                print(f"[SIP:{session_id[:8]}] === HANGUP === (duration: {duration:.1f}s, frames: {frame_count}, speeches: {speech_count})", flush=True)
                await broadcast({"type": "sip.call.end", "call_id": session_id})
                break

            if frame["type"] == "dtmf":
                print(f"[SIP:{session_id[:8]}] DTMF: {frame['digit']}", flush=True)
                await broadcast({"type": "sip.dtmf", "call_id": session_id, "digit": frame["digit"]})
                continue

            if frame["type"] == "error":
                print(f"[SIP:{session_id[:8]}] ERROR: {frame.get('message')}", flush=True)
                break

            if frame["type"] != "audio":
                continue

            frame_count += 1
            pcm_16k = frame["pcm16"]

            # VAD
            was_speaking = vad.is_speaking
            speech_audio = vad.feed(pcm_16k)

            if not was_speaking and vad.is_speaking:
                print(f"[SIP:{session_id[:8]}] VAD: speech_start (frame {frame_count})", flush=True)
                await broadcast({"type": "sip.vad.speech", "call_id": session_id, "event": "start"})

            if speech_audio is not None:
                speech_count += 1
                audio_ms = len(speech_audio) / (16000 * 2) * 1000
                print(f"[SIP:{session_id[:8]}] VAD: speech_end ({len(speech_audio)} bytes, {audio_ms:.0f}ms)", flush=True)
                await broadcast({"type": "sip.vad.speech", "call_id": session_id, "event": "end"})

                # Mock STT
                mock_text = "Сколько стоит лизинг на Тигуан?"
                print(f"[SIP:{session_id[:8]}] STT: {mock_text}", flush=True)
                await broadcast({"type": "sip.stt.result", "call_id": session_id, "text": mock_text})

                # Mock LLM response
                mock_response = "По предварительному расчёту, ежемесячный платёж составит примерно 1250 рублей."
                print(f"[SIP:{session_id[:8]}] LLM: {mock_response}", flush=True)
                await broadcast({"type": "sip.llm.sentence", "call_id": session_id, "text": mock_response})

                # Send TTS response audio
                print(f"[SIP:{session_id[:8]}] Sending TTS response (1.5s tone)...", flush=True)
                response_audio = generate_response_audio(1500)
                writer.write(response_audio)
                await writer.drain()
                await broadcast({"type": "sip.tts.start", "call_id": session_id, "text": mock_response})

    except Exception as exc:
        import traceback
        print(f"[SIP:{session_id[:8]}] ERROR: {exc}\n{traceback.format_exc()}", flush=True)
    finally:
        await adapter.close()
        print(f"[SIP:{session_id[:8]}] Cleaned up", flush=True)


# ---------------------------------------------------------------------------
# FastAPI app for monitor page
# ---------------------------------------------------------------------------

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="SIP Test Server")
FRONTEND_DIR = ROOT / "frontend"


@app.websocket("/ws/sip-monitor")
async def sip_monitor_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    monitor_clients.add(websocket)
    print("[MONITOR] Client connected", flush=True)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        monitor_clients.discard(websocket)
        print("[MONITOR] Client disconnected", flush=True)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ---------------------------------------------------------------------------
# Main: start both AudioSocket TCP server and FastAPI HTTP server
# ---------------------------------------------------------------------------

async def main():
    # Start AudioSocket TCP server
    sip_server = await asyncio.start_server(
        handle_sip_call,
        host="127.0.0.1",
        port=9092,
    )
    print("=" * 60, flush=True)
    print("SIP TEST SERVER", flush=True)
    print("=" * 60, flush=True)
    print(f"AudioSocket:  127.0.0.1:9092 (waiting for connections)", flush=True)
    print(f"Monitor page: http://localhost:8000/sip_monitor.html", flush=True)
    print(f"", flush=True)
    print(f"To test, run in another terminal:", flush=True)
    print(f"  cd rag_demo_system && python tests/test_sip_client.py", flush=True)
    print("=" * 60, flush=True)

    # Start FastAPI with uvicorn
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
