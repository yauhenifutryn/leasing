from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from .session import ClientProfile


@dataclass
class VoiceSession:
    session_id: str
    backend: str = "our_rag"
    brain_model: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    stt_provider: str = "whisper"
    tts_provider: str = "silero_tts"
    assistant_speaking: bool = False
    # Fix 33: token-based ownership of the assistant_speaking flag.
    # Each TTS call stamps a unique token on entry; only the holder of the
    # current token is allowed to reset `assistant_speaking=False` on exit.
    # This fixes the intro-vs-readback race where a late-finishing
    # `_jambonz_send_tts` (intro) wiped the True set by a concurrent
    # `_emit_plain_assistant_response` (readback), breaking barge-in.
    tts_speaker_token: str | None = None
    interrupted: bool = False
    active_task_id: str | None = None
    last_user_message: str = ""
    client_name: str = ""
    turn_count: int = 0
    tool_calls_this_turn: list = field(default_factory=list)

    # Full cumulative history of tool calls across the whole session (preserved
    # even after reset_turn_state clears tool_calls_this_turn).
    tool_calls_history: list = field(default_factory=list)

    # Circuit breaker: track repeated identical calc attempts that fail upstream.
    last_calc_signature: str = ""
    consecutive_calc_failures: int = 0

    # SIP telephony fields (defaults preserve existing WebSocket/RTC behavior)
    transport: str = "websocket"          # "websocket" | "rtc" | "jambonz"
    client_phone: str | None = None       # from SIP caller ID, None for browser
    call_id: str | None = None            # Jambonz call ID or RTC session ID

    # Client profile: incrementally populated, gates calculator invocation.
    client_profile: ClientProfile = field(default_factory=ClientProfile)

    # Turn-taking: listen_mode entered on semantic stop request.
    listen_mode: bool = False
    listen_mode_until: float = 0.0
    listen_mode_task: Optional["asyncio.Task[None]"] = None  # auto-exit background task, see listen_mode.py

    def reset_turn_state(self) -> None:
        """Clear per-turn scratch state at the start of each user turn.

        Name `tool_calls_this_turn` was historically cumulative in practice
        (never reset), which caused stuck-in-calculator loops: once a calc
        succeeded, classifier hints like `recalculate` plus non-empty
        tool_calls_this_turn made the orchestrator bypass all gates on every
        subsequent turn. Clearing per turn fixes that; `tool_calls_history`
        preserves historical context for prompts that need it.
        """
        if self.tool_calls_this_turn:
            self.tool_calls_history.extend(self.tool_calls_this_turn)
            self.tool_calls_this_turn = []

    @property
    def stack_id(self) -> str:
        brain = self.brain_model.split("/")[-1]
        return f"{self.backend}__{brain}__{self.stt_provider}__{self.tts_provider}"

    def on_audio_chunk(self, _audio_b64: str) -> list[dict[str, Any]]:
        if not self.assistant_speaking:
            return []
        self.assistant_speaking = False
        self.interrupted = True
        return [
            {
                "type": "interrupt",
                "session_id": self.session_id,
                "task_id": self.active_task_id,
                "backend": self.backend,
            }
        ]

    def on_transcript_final(self, text: str) -> list[dict[str, Any]]:
        self.last_user_message = text
        self.interrupted = False
        return [
            {
                "type": "dispatch_message",
                "session_id": self.session_id,
                "backend": self.backend,
                "message": text,
                "voice_fast": True,
            }
        ]

    def on_provider_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        conversation_ref = response.get("conversation_ref") or {}
        self.active_task_id = conversation_ref.get("task_id")
        self.assistant_speaking = bool(response.get("answer"))
        return [
            {
                "type": "assistant_response",
                "session_id": self.session_id,
                "backend": response.get("backend", self.backend),
                "answer": response.get("answer", ""),
                "can_barge_in": bool(response.get("can_barge_in", True)),
            }
        ]
