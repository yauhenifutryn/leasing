#!/usr/bin/env python3
"""Interactive AudioSocket test client for local SIP testing.

Connects directly to the AudioSocket TCP server (bypassing Asterisk),
sends audio frames, and plays back responses. Used for local development
testing on macOS where Asterisk is not available.

Usage:
    python tests/test_sip_client.py [--host 127.0.0.1] [--port 9092]

What it does:
    1. Connects TCP to the AudioSocket server
    2. Sends a UUID frame (simulates Asterisk)
    3. Sends 3 seconds of silence (lets the bot play its intro TTS)
    4. Sends a pre-generated speech-like audio burst (triggers VAD)
    5. Sends silence again (VAD detects speech end, triggers STT)
    6. Reads response frames (TTS audio from the bot)
    7. Optionally sends DTMF digits
    8. Sends hangup frame

Watch the terminal running the app server and the sip_monitor.html page
in your browser to see events flowing through the pipeline.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import struct
import sys
import time
import uuid

# AudioSocket frame types
FRAME_UUID = 0x01
FRAME_AUDIO = 0x10
FRAME_HANGUP = 0x00
FRAME_DTMF = 0x03


def build_frame(frame_type: int, payload: bytes) -> bytes:
    """Build an AudioSocket wire frame."""
    return struct.pack("!BH", frame_type, len(payload)) + payload


def build_uuid_frame(call_uuid: str | None = None) -> bytes:
    """Build a UUID frame (first frame in AudioSocket protocol)."""
    uid = call_uuid or str(uuid.uuid4())
    return build_frame(FRAME_UUID, uid.encode("ascii"))


def build_silence_frame(n_samples: int = 160) -> bytes:
    """Build a 20ms silence frame at 8kHz (160 samples)."""
    pcm = struct.pack(f"<{n_samples}h", *([0] * n_samples))
    return build_frame(FRAME_AUDIO, pcm)


def build_tone_frame(freq: int = 440, n_samples: int = 160, amplitude: int = 16000) -> bytes:
    """Build a 20ms tone frame at 8kHz. Simulates speech for VAD trigger."""
    samples = []
    for i in range(n_samples):
        t = i / 8000.0
        value = int(amplitude * math.sin(2 * math.pi * freq * t))
        samples.append(max(-32768, min(32767, value)))
    pcm = struct.pack(f"<{n_samples}h", *samples)
    return build_frame(FRAME_AUDIO, pcm)


def build_dtmf_frame(digit: str) -> bytes:
    """Build a DTMF frame."""
    return build_frame(FRAME_DTMF, digit.encode("ascii"))


def build_hangup_frame() -> bytes:
    """Build a hangup frame."""
    return build_frame(FRAME_HANGUP, b"")


async def read_responses(reader: asyncio.StreamReader, duration: float = 5.0) -> list[dict]:
    """Read response frames from the server for a given duration."""
    frames = []
    deadline = time.time() + duration
    while time.time() < deadline:
        try:
            header = await asyncio.wait_for(reader.readexactly(3), timeout=0.5)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            continue
        frame_type = header[0]
        length = struct.unpack("!H", header[1:3])[0]
        if length > 0:
            try:
                payload = await asyncio.wait_for(reader.readexactly(length), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                break
        else:
            payload = b""
        frames.append({"type": frame_type, "length": length, "payload_size": len(payload)})
        if frame_type == FRAME_AUDIO:
            pass  # audio response, counted
        elif frame_type == FRAME_HANGUP:
            print("  Server sent hangup")
            break
    return frames


async def run_test(host: str, port: int) -> None:
    """Run the full test sequence."""
    call_uuid = str(uuid.uuid4())
    print(f"Connecting to {host}:{port}...")

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=5.0,
        )
    except (ConnectionRefusedError, OSError) as exc:
        print(f"FAILED: Cannot connect to {host}:{port}: {exc}")
        print("  Make sure the app is running with SIP_ENABLED=true")
        sys.exit(1)

    print(f"Connected. Call UUID: {call_uuid[:8]}...")

    # Step 1: Send UUID frame
    print("\n[1] Sending UUID frame...")
    writer.write(build_uuid_frame(call_uuid))
    await writer.drain()
    print("  Sent. Server should log: [SIP:...] Connected")

    # Step 2: Send silence for 3 seconds (let bot play intro TTS)
    print("\n[2] Sending 3s of silence (bot will play intro)...")
    silence_frame = build_silence_frame()
    frames_per_second = 50  # 20ms per frame = 50 frames/sec
    for i in range(3 * frames_per_second):
        writer.write(silence_frame)
        if i % 50 == 0:
            await writer.drain()
        await asyncio.sleep(0.02)  # 20ms pacing
    await writer.drain()
    print("  Done. Reading intro TTS response...")

    # Read any TTS response frames from the intro
    intro_frames = await read_responses(reader, duration=3.0)
    audio_count = sum(1 for f in intro_frames if f["type"] == FRAME_AUDIO)
    print(f"  Received {len(intro_frames)} frames ({audio_count} audio)")

    # Step 3: Send speech-like tone burst (triggers VAD)
    print("\n[3] Sending 2s of tone (simulates speech, triggers VAD)...")
    tone_frame = build_tone_frame(freq=300, amplitude=20000)
    for i in range(2 * frames_per_second):
        writer.write(tone_frame)
        if i % 50 == 0:
            await writer.drain()
        await asyncio.sleep(0.02)
    await writer.drain()
    print("  Done. Server should log: VAD: speech_start")

    # Step 4: Send silence (VAD detects speech end, triggers STT -> LLM -> TTS)
    print("\n[4] Sending 1s of silence (VAD speech end -> STT -> LLM -> TTS)...")
    for i in range(1 * frames_per_second):
        writer.write(silence_frame)
        if i % 50 == 0:
            await writer.drain()
        await asyncio.sleep(0.02)
    await writer.drain()
    print("  Done. Server should log: VAD: speech_end, STT:, then stream response")

    # Step 5: Read LLM/TTS response
    print("\n[5] Reading response (LLM -> TTS audio)...")
    response_frames = await read_responses(reader, duration=8.0)
    audio_count = sum(1 for f in response_frames if f["type"] == FRAME_AUDIO)
    print(f"  Received {len(response_frames)} frames ({audio_count} audio)")

    # Step 6: Send DTMF digits
    print("\n[6] Sending DTMF digits: 1, 2, 3, 4...")
    for digit in ["1", "2", "3", "4"]:
        writer.write(build_dtmf_frame(digit))
        await writer.drain()
        await asyncio.sleep(0.2)
    print("  Done. Server should log: DTMF: 1, DTMF: 2, DTMF: 3, DTMF: 4")

    # Step 7: Hangup
    print("\n[7] Sending hangup...")
    writer.write(build_hangup_frame())
    await writer.drain()
    print("  Done. Server should log: Hangup, Cleaned up")

    await asyncio.sleep(1.0)
    writer.close()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nCheck:")
    print("  1. App terminal: look for [SIP:...] log messages")
    print("  2. Browser: open http://localhost:8000/sip_monitor.html")
    print("     - Should show call start, VAD events, STT, bot response, DTMF, call end")
    print(f"\nTotal response frames: {audio_count} audio")
    if audio_count > 0:
        print("  AudioSocket bidirectional audio: WORKING")
    else:
        print("  WARNING: No audio frames received. Check server logs for errors.")


def main():
    parser = argparse.ArgumentParser(description="AudioSocket test client")
    parser.add_argument("--host", default="127.0.0.1", help="AudioSocket server host")
    parser.add_argument("--port", type=int, default=9092, help="AudioSocket server port")
    args = parser.parse_args()
    asyncio.run(run_test(args.host, args.port))


if __name__ == "__main__":
    main()
