from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import quote

import websockets


DEFAULT_PROMPT = (
    "Вы — голосовой помощник.\n"
    "Если доступен file_search, сначала используйте его.\n"
    "Отвечайте кратко и по делу.\n"
)


def normalize_voice_provider(value: str | None) -> str:
    provider = (value or "local").strip().lower()
    if provider == "yandex":
        return "yandex_realtime"
    if provider not in {"local", "yandex_realtime", "yandex_speechkit", "oss_russian", "qwen3_omni"}:
        return "local"
    return provider


def build_status(env: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = env or os.environ
    if not cfg.get("YC_FOLDER_ID"):
        return {
            "name": "yandex_realtime",
            "available": False,
            "healthy": False,
            "reason": "missing_folder_id",
        }
    if not (cfg.get("YC_API_KEY") or cfg.get("YC_IAM_TOKEN")):
        return {
            "name": "yandex_realtime",
            "available": False,
            "healthy": False,
            "reason": "missing_credentials",
        }
    return {
        "name": "yandex_realtime",
        "available": True,
        "healthy": True,
        "reason": "ok",
    }


def build_connection_settings(env: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = env or os.environ
    folder_id = (cfg.get("YC_FOLDER_ID") or "").strip()
    if not folder_id:
        raise RuntimeError("YC_FOLDER_ID is required for Yandex Realtime")
    api_key = (cfg.get("YC_API_KEY") or "").strip()
    iam_token = (cfg.get("YC_IAM_TOKEN") or "").strip()
    if not api_key and not iam_token:
        raise RuntimeError("YC_API_KEY or YC_IAM_TOKEN is required for Yandex Realtime")

    model = (cfg.get("YC_MODEL") or f"gpt://{folder_id}/speech-realtime-250923").strip()
    url = (
        cfg.get("YC_REALTIME_WS_URL")
        or "wss://rest-assistant.api.cloud.yandex.net/v1/realtime/openai"
        f"?model={quote(model, safe='')}"
    )
    auth = f"Api-Key {api_key}" if api_key else f"Bearer {iam_token}"
    return {
        "url": url,
        "headers": {
            "Authorization": auth,
            "OpenAI-Project": folder_id,
        },
    }


def build_session_update(env: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = env or os.environ
    voice = (cfg.get("YC_VOICE") or "ermil").strip()
    prompt = (cfg.get("YANDEX_REALTIME_PROMPT") or DEFAULT_PROMPT).strip()
    search_index_id = (cfg.get("YANDEX_AI_SEARCH_INDEX_ID") or cfg.get("YC_VECTOR_STORE_ID") or "").strip()
    turn_detection = None
    return {
        "type": "session.update",
        "session": {
            "instructions": prompt,
            "output_modalities": ["audio"],
            "turn_detection": turn_detection,
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                        "channels": 1,
                    },
                    "turn_detection": turn_detection,
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 44100,
                    },
                    "voice": voice,
                },
            },
            "tools": [
                {
                    "type": "function",
                    "name": "file_search",
                    "description": search_index_id,
                    "parameters": "{}",
                }
            ]
            if search_index_id
            else [],
        },
    }


def normalize_server_event(event: dict[str, Any]) -> dict[str, Any]:
    event_type = event.get("type")
    if event_type == "conversation.item.input_audio_transcription.completed":
        transcript = event.get("transcription") or event.get("transcript")
        if transcript:
            event = dict(event)
            event.setdefault("transcription", transcript)
    return event


class YandexRealtimeRelay:
    def __init__(self, client: Any, env: dict[str, str] | None = None) -> None:
        self._client = client
        self._env = env or os.environ
        self._upstream: Any | None = None
        self._forward_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        if self._upstream is not None:
            return
        settings = build_connection_settings(self._env)
        self._upstream = await websockets.connect(
            settings["url"],
            additional_headers=settings["headers"],
        )
        self._forward_task = asyncio.create_task(self._forward_upstream())
        await self._upstream.send(json.dumps(build_session_update(self._env)))

    async def send_event(self, event: dict[str, Any]) -> None:
        await self.connect()
        assert self._upstream is not None
        await self._upstream.send(json.dumps(event))

    async def refresh_session(self) -> None:
        await self.connect()
        assert self._upstream is not None
        await self._upstream.send(json.dumps(build_session_update(self._env)))

    async def close(self) -> None:
        if self._upstream is not None:
            await self._upstream.close()
            self._upstream = None
        if self._forward_task is not None:
            try:
                await self._forward_task
            except Exception:
                pass
            self._forward_task = None

    async def _forward_upstream(self) -> None:
        try:
            assert self._upstream is not None
            async for raw in self._upstream:
                message = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                try:
                    payload = normalize_server_event(json.loads(message))
                except json.JSONDecodeError:
                    payload = {
                        "type": "warning",
                        "message": "Yandex Realtime returned a non-JSON event",
                    }
                await self._client.send_json(payload)
        except Exception as exc:
            try:
                await self._client.send_json({"type": "error", "error": f"yandex_relay_failed: {exc}"})
            except Exception:
                pass
