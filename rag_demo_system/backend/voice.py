from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class VoiceStatus:
    enabled: bool
    reason: str


def get_voice_status(enabled: bool, api_key_env: str, voice_id_env: str) -> VoiceStatus:
    api_key = os.getenv(api_key_env)
    voice_id = os.getenv(voice_id_env)
    if not enabled:
        return VoiceStatus(enabled=False, reason="voice_disabled")
    if not api_key:
        return VoiceStatus(enabled=False, reason="missing_elevenlabs_api_key")
    if not voice_id:
        return VoiceStatus(enabled=False, reason="missing_voice_id")
    return VoiceStatus(enabled=True, reason="ok")


def tts_stream(text: str, api_key: str, voice_id: str, tts_url_template: str) -> bytes:
    url = tts_url_template.format(voice_id=voice_id)
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.content

# NOTE: Realtime STT via ElevenLabs WebSocket should be implemented here once
# endpoint format is confirmed. This demo leaves a TODO placeholder.
