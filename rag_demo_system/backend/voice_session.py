from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VoiceSession:
    session_id: str
    backend: str = "our_rag"
    brain_model: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    stt_provider: str = "whisper"
    tts_provider: str = "silero_tts"
    assistant_speaking: bool = False
    interrupted: bool = False
    active_task_id: str | None = None
    last_user_message: str = ""
    client_name: str = ""
    turn_count: int = 0
    tool_calls_this_turn: list = field(default_factory=list)

    # SIP telephony fields (defaults preserve existing WebSocket/RTC behavior)
    transport: str = "websocket"          # "websocket" | "rtc" | "jambonz"
    client_phone: str | None = None       # from SIP caller ID, None for browser
    call_id: str | None = None            # AudioSocket UUID or RTC session ID

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
