"""Transport-agnostic audio input adapters.

Decouples the audio source (WebSocket, SIP gateway, etc.) from the
VAD/STT processing pipeline. Each adapter normalizes incoming audio
into PCM16 chunks and forwards them via an async callback.
"""
from __future__ import annotations

from typing import Awaitable, Callable


class WebSocketAudioAdapter:
    """Receives base64-decoded PCM16 audio from a WebSocket client."""

    def __init__(self, on_chunk: Callable[[bytes], Awaitable[None]]) -> None:
        self._on_chunk = on_chunk
        self.muted = False

    async def handle_audio_message(self, pcm16_bytes: bytes) -> None:
        """Called for each audio chunk from the WebSocket."""
        if self.muted:
            return
        await self._on_chunk(pcm16_bytes)
