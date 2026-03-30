"""Tests for transport-agnostic audio input adapters."""
import asyncio
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pcm_chunk(n: int = 512) -> bytes:
    return struct.pack(f"<{n}h", *([0] * n))


class TestWebSocketAdapter:
    def test_receives_audio_and_forwards_to_callback(self):
        from backend.audio_input import WebSocketAudioAdapter

        received = []

        async def on_chunk(chunk: bytes) -> None:
            received.append(chunk)

        async def _run():
            adapter = WebSocketAudioAdapter(on_chunk=on_chunk)
            chunk = _pcm_chunk(512)
            await adapter.handle_audio_message(chunk)
            assert len(received) == 1
            assert received[0] == chunk

        asyncio.run(_run())

    def test_mute_stops_forwarding(self):
        from backend.audio_input import WebSocketAudioAdapter

        received = []

        async def on_chunk(chunk: bytes) -> None:
            received.append(chunk)

        async def _run():
            adapter = WebSocketAudioAdapter(on_chunk=on_chunk)
            adapter.muted = True
            await adapter.handle_audio_message(_pcm_chunk())
            assert len(received) == 0

        asyncio.run(_run())

    def test_unmute_resumes_forwarding(self):
        from backend.audio_input import WebSocketAudioAdapter

        received = []

        async def on_chunk(chunk: bytes) -> None:
            received.append(chunk)

        async def _run():
            adapter = WebSocketAudioAdapter(on_chunk=on_chunk)
            adapter.muted = True
            await adapter.handle_audio_message(_pcm_chunk())
            adapter.muted = False
            await adapter.handle_audio_message(_pcm_chunk())
            assert len(received) == 1

        asyncio.run(_run())
