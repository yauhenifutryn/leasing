from __future__ import annotations

import base64
import os
from typing import Any
from urllib.parse import urlencode

import requests

from .audio_utils import resample_pcm16_mono


def build_status(env: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = env or os.environ
    if not cfg.get("YC_FOLDER_ID"):
        return {
            "name": "yandex_speechkit",
            "available": False,
            "healthy": False,
            "reason": "missing_folder_id",
        }
    if not (cfg.get("YC_API_KEY") or cfg.get("YC_IAM_TOKEN")):
        return {
            "name": "yandex_speechkit",
            "available": False,
            "healthy": False,
            "reason": "missing_credentials",
        }
    return {
        "name": "yandex_speechkit",
        "available": True,
        "healthy": True,
        "reason": "ok",
    }


def _auth_headers(env: dict[str, str] | None = None) -> dict[str, str]:
    cfg = env or os.environ
    folder_id = (cfg.get("YC_FOLDER_ID") or "").strip()
    api_key = (cfg.get("YC_API_KEY") or "").strip()
    iam_token = (cfg.get("YC_IAM_TOKEN") or "").strip()
    if not folder_id:
        raise RuntimeError("YC_FOLDER_ID is required for Yandex SpeechKit")
    if not api_key and not iam_token:
        raise RuntimeError("YC_API_KEY or YC_IAM_TOKEN is required for Yandex SpeechKit")
    auth = f"Api-Key {api_key}" if api_key else f"Bearer {iam_token}"
    return {
        "Authorization": auth,
        "x-folder-id": folder_id,
    }


def transcribe_audio(audio_b64: str, sample_rate_hz: int = 24000, env: dict[str, str] | None = None) -> dict[str, Any]:
    payload = resample_pcm16_mono(base64.b64decode(audio_b64), sample_rate_hz, 48000)
    params = urlencode(
        {
            "lang": "ru-RU",
            "topic": "general",
            "format": "lpcm",
            "sampleRateHertz": 48000,
        }
    )
    resp = requests.post(
        f"https://stt.api.cloud.yandex.net/speech/v1/stt:recognize?{params}",
        headers=_auth_headers(env),
        data=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "text": str(data.get("result") or "").strip(),
        "provider": "yandex_speechkit",
    }


def synthesize_audio(text: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = env or os.environ
    voice = (cfg.get("YC_VOICE") or "ermil").strip()
    resp = requests.post(
        "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize",
        headers=_auth_headers(cfg),
        data={
            "text": text,
            "lang": "ru-RU",
            "voice": voice,
            "format": "lpcm",
            "sampleRateHertz": "48000",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return {
        "audio_b64": base64.b64encode(resp.content).decode("ascii"),
        "sample_rate_hz": 48000,
        "provider": "yandex_speechkit",
    }
