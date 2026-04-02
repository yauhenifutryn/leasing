from __future__ import annotations

import base64
import io
import os
import wave
from typing import Any

import requests

from .yandex_speechkit import (
    build_status as build_yandex_speechkit_status,
    synthesize_audio as synthesize_with_yandex_speechkit,
    transcribe_audio as transcribe_with_yandex_speechkit,
)
from .tts_normalize import normalize_for_tts
from .yandex_realtime import build_status as build_yandex_realtime_status

# STT providers that hard-fail when their BASE_URL is unset (no fallback allowed)
_HARD_FAIL_STT: frozenset[str] = frozenset({"qwen3_asr", "voxtral"})


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
        "sensevoice": _service_status("sensevoice", os.getenv("SENSEVOICE_BASE_URL")),
        "whisper": _service_status("whisper", os.getenv("WHISPER_BASE_URL")),
        "cosyvoice": _service_status("cosyvoice", os.getenv("COSYVOICE_BASE_URL")),
        "vosk": _service_status("vosk", os.getenv("VOSK_BASE_URL")),
        "vosk_tts": _service_status("vosk_tts", os.getenv("VOSK_TTS_BASE_URL")),
        "yandex_speechkit": build_yandex_speechkit_status(),
        "yandex_realtime": build_yandex_realtime_status(),
        "qwen3_asr": _service_status("qwen3_asr", os.getenv("QWEN3_ASR_BASE_URL")),
        "qwen3_tts": _service_status("qwen3_tts", os.getenv("QWEN3_TTS_BASE_URL")),
        "silero_tts": _service_status("silero_tts", os.getenv("SILERO_TTS_BASE_URL")),
        "voxtral": _service_status("voxtral", os.getenv("VOXTRAL_BASE_URL")),
        "qwen3_omni": _service_status("qwen3_omni", os.getenv("QWEN3_OMNI_BASE_URL")),
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


def _pcm16_b64_to_wav_bytes(audio_b64: str, sample_rate_hz: int = 24000) -> bytes:
    pcm_bytes = base64.b64decode(audio_b64)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def _parse_sensevoice_response(payload: Any) -> str:
    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return str(first.get("text", "")).strip()
        if "text" in payload:
            return str(payload.get("text", "")).strip()
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return str(first.get("text", "")).strip()
    return ""


def _extract_pcm16_audio(audio_bytes: bytes) -> tuple[bytes, int]:
    if audio_bytes[:4] != b"RIFF":
        return audio_bytes, int(os.getenv("COSYVOICE_SAMPLE_RATE_HZ", "22050"))
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        sample_rate_hz = wav_file.getframerate()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2 or channels != 1:
        raise RuntimeError("CosyVoice audio must be mono PCM16 WAV")
    return pcm_bytes, sample_rate_hz


def _sensevoice_api_style() -> str:
    return (os.getenv("SENSEVOICE_API_STYLE") or "compat").strip().lower()


def _cosyvoice_api_style() -> str:
    return (os.getenv("COSYVOICE_API_STYLE") or "compat").strip().lower()


def transcribe_audio(audio_b64: str, session_id: str, preferred: str = "sensevoice") -> dict[str, Any]:
    # Hard-fail path for new STT providers: no silent fallback allowed.
    if preferred in _HARD_FAIL_STT:
        base_url = os.getenv(f"{preferred.upper()}_BASE_URL")
        if not base_url:
            raise RuntimeError(
                f"{preferred} service unavailable: {preferred.upper()}_BASE_URL not set"
            )
        resp = requests.post(
            base_url.rstrip("/") + "/transcribe",
            json={"audio_b64": audio_b64, "session_id": session_id, "language": "ru", "sample_rate_hz": 24000},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("text"):
            data.setdefault("provider", preferred)
            return data
        raise RuntimeError(f"{preferred} returned empty transcription")

    order = [preferred]
    for fallback in ("sensevoice", "whisper"):
        if fallback not in order:
            order.append(fallback)
    print(f"[stt] preferred={preferred} order={order} audio_len={len(audio_b64)}", flush=True)
    for name in order:
        if name == "yandex_speechkit":
            data = transcribe_with_yandex_speechkit(audio_b64, sample_rate_hz=24000)
            if data.get("text"):
                return data
            continue
        base_url = os.getenv(f"{name.upper()}_BASE_URL")
        if not base_url:
            print(f"[stt] {name}: no BASE_URL, skip", flush=True)
            continue
        try:
            if name == "sensevoice" and _sensevoice_api_style() == "official":
                wav_bytes = _pcm16_b64_to_wav_bytes(audio_b64)
                resp = requests.post(
                    base_url.rstrip("/") + "/api/v1/asr",
                    files=[("files", (f"{session_id}.wav", wav_bytes, "audio/wav"))],
                    data={"keys": session_id, "lang": "auto"},
                    timeout=60,
                )
                resp.raise_for_status()
                text = _parse_sensevoice_response(resp.json())
                if text:
                    return {"text": text, "provider": name}
                continue
            resp = requests.post(
                base_url.rstrip("/") + "/transcribe",
                json={"audio_b64": audio_b64, "session_id": session_id, "language": "ru", "sample_rate_hz": 24000},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("text"):
                data.setdefault("provider", name)
                return data
        except Exception as exc:  # noqa: BLE001
            print(f"[stt] {name}: error: {exc}", flush=True)
            continue
    raise RuntimeError("No STT service configured")


def synthesize_audio(text: str, session_id: str, preferred: str = "cosyvoice") -> dict[str, Any]:
    return synthesize_audio_with_provider(text, session_id, preferred=preferred)


def synthesize_audio_with_provider(text: str, session_id: str, preferred: str = "cosyvoice") -> dict[str, Any]:
    text = normalize_for_tts(text)
    if preferred == "yandex_speechkit":
        data = synthesize_with_yandex_speechkit(text)
        data.setdefault("session_id", session_id)
        return data
    if preferred == "qwen3_tts":
        base_url = os.getenv("QWEN3_TTS_BASE_URL")
        if not base_url:
            raise RuntimeError("Qwen3-TTS service unavailable: QWEN3_TTS_BASE_URL not set")
        resp = requests.post(
            base_url.rstrip("/") + "/speak",
            json={"text": text, "session_id": session_id, "language": "ru"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        data.setdefault("provider", "qwen3_tts")
        data.setdefault("session_id", session_id)
        return data
    if preferred == "silero_tts":
        base_url = os.getenv("SILERO_TTS_BASE_URL")
        if not base_url:
            raise RuntimeError("Silero TTS service unavailable: SILERO_TTS_BASE_URL not set")
        resp = requests.post(
            base_url.rstrip("/") + "/speak",
            json={"text": text, "session_id": session_id, "language": "ru"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        data.setdefault("provider", "silero_tts")
        data.setdefault("session_id", session_id)
        return data
    if preferred == "vosk_tts":
        base_url = os.getenv("VOSK_TTS_BASE_URL")
        if not base_url:
            raise RuntimeError("VOSK_TTS_BASE_URL is not configured")
        resp = requests.post(
            base_url.rstrip("/") + "/speak",
            json={"text": text, "session_id": session_id, "language": "ru"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        data.setdefault("provider", "vosk_tts")
        data.setdefault("session_id", session_id)
        return data
    base_url = os.getenv("COSYVOICE_BASE_URL")
    if not base_url:
        raise RuntimeError("COSYVOICE_BASE_URL is not configured")
    if _cosyvoice_api_style() == "official":
        resp = requests.post(
            base_url.rstrip("/") + "/inference_sft",
            data={
                "tts_text": text,
                "spk_id": os.getenv("COSYVOICE_SPK_ID", "中文女"),
            },
            timeout=60,
        )
        resp.raise_for_status()
        pcm_bytes, sample_rate_hz = _extract_pcm16_audio(resp.content)
        return {
            "audio_b64": base64.b64encode(pcm_bytes).decode("ascii"),
            "sample_rate_hz": sample_rate_hz,
            "provider": "cosyvoice",
            "session_id": session_id,
        }
    resp = requests.post(
        base_url.rstrip("/") + "/speak",
        json={"text": text, "session_id": session_id, "language": "ru"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    data.setdefault("provider", "cosyvoice")
    data.setdefault("session_id", session_id)
    return data
