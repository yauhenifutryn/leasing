from __future__ import annotations

import json
from typing import Any

import requests

from .rag_backends import ProviderResponse, map_dify_retriever_resources


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _iter_sse_payloads(lines: Any) -> Any:
    for raw_line in lines:
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.split("data:", 1)[1].strip()
        if not payload or payload == "[DONE]":
            continue
        yield json.loads(payload)


def chat_once(
    *,
    base_url: str,
    api_key: str,
    query: str,
    user: str,
    inputs: dict[str, Any] | None = None,
    conversation_id: str | None = None,
    timeout_sec: int = 60,
) -> ProviderResponse:
    url = base_url.rstrip("/") + "/chat-messages"
    payload: dict[str, Any] = {
        "inputs": inputs or {},
        "query": query,
        "response_mode": "streaming",
        "user": user,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    resp = requests.post(
        url,
        headers=_auth_headers(api_key),
        json=payload,
        timeout=timeout_sec,
        stream=True,
    )
    resp.raise_for_status()

    answer_parts: list[str] = []
    conversation_ref: dict[str, Any] = {}
    used_knowledge: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    incomplete = False

    for event in _iter_sse_payloads(resp.iter_lines(decode_unicode=True)):
        event_type = event.get("event")
        if event_type == "message":
            answer_parts.append(event.get("answer") or "")
            if event.get("conversation_id"):
                conversation_ref["conversation_id"] = event["conversation_id"]
            if event.get("task_id"):
                conversation_ref["task_id"] = event["task_id"]
        elif event_type == "message_end":
            metadata = event.get("metadata") or {}
            resources = metadata.get("retriever_resources") or []
            used_knowledge = map_dify_retriever_resources(resources)
            citations = used_knowledge
        elif event_type == "error":
            incomplete = True

    return ProviderResponse(
        answer="".join(answer_parts),
        backend="dify_rag",
        conversation_ref=conversation_ref,
        used_knowledge=used_knowledge,
        citations=citations,
        timings={},
        incomplete=incomplete,
        can_barge_in=True,
    )


def stop_generation(
    *,
    base_url: str,
    api_key: str,
    task_id: str,
    user: str,
    timeout_sec: int = 15,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + f"/chat-messages/{task_id}/stop"
    resp = requests.post(
        url,
        headers=_auth_headers(api_key),
        json={"user": user},
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    return resp.json()
