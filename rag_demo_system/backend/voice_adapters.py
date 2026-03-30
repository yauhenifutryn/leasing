from __future__ import annotations

import base64
import os
from typing import Any

import requests


def _service_status(name: str, base_url: str | None) -> dict[str, Any]:
    if not base_url:
        return {"name": name, "available": False, "healthy": False, "reason": "not_configured"}
    health_url = base_url.rstrip("/") + "/health"
    try:
        resp = requests.get(health_url, timeout=2)
        return {
            "name": name,
            "available": True,
            "healthy": resp.ok,
            "reason": "ok" if resp.ok else f"http_{resp.status_code}",
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "available": True, "healthy": False, "reason": str(exc)}


def build_voice_statuses() -> dict[str, dict[str, Any]]:
    return {
        "whisper": _service_status("whisper", os.getenv("WHISPER_BASE_URL")),
        "silero_tts": _service_status("silero_tts", os.getenv("SILERO_TTS_BASE_URL")),
    }


def build_llm_status(base_url: str, model: str) -> dict[str, dict[str, Any]]:
    if not base_url or not model:
        return {"qwen": {"name": "qwen", "available": False, "healthy": False, "reason": "not_configured"}}
    url = base_url.rstrip("/") + "/models"
    try:
        resp = requests.get(url, timeout=2)
        return {
            "qwen": {
                "name": "qwen",
                "available": True,
                "healthy": resp.ok,
                "reason": "ok" if resp.ok else f"http_{resp.status_code}",
                "model": model,
            }
        }
    except Exception as exc:  # noqa: BLE001
        return {"qwen": {"name": "qwen", "available": True, "healthy": False, "reason": str(exc), "model": model}}


def transcribe_audio(audio_b64: str, session_id: str) -> dict[str, Any]:
    """Transcribe audio using the Whisper STT service."""
    base_url = os.getenv("WHISPER_BASE_URL")
    if not base_url:
        raise RuntimeError("WHISPER_BASE_URL is not configured")
    resp = requests.post(
        base_url.rstrip("/") + "/transcribe",
        json={"audio_b64": audio_b64, "session_id": session_id, "language": "ru", "sample_rate_hz": 24000},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("text"):
        raise RuntimeError("Whisper returned empty transcription")
    data.setdefault("provider", "whisper")
    return data


def synthesize_audio(text: str, session_id: str) -> dict[str, Any]:
    """Synthesize speech using the Silero TTS service."""
    base_url = os.getenv("SILERO_TTS_BASE_URL")
    if not base_url:
        raise RuntimeError("SILERO_TTS_BASE_URL is not configured")
    resp = requests.post(
        base_url.rstrip("/") + "/speak",
        json={"text": text, "session_id": session_id, "language": "ru"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    data.setdefault("provider", "silero_tts")
    data.setdefault("session_id", session_id)
    return data
