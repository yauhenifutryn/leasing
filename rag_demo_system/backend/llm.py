from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

from .llm_stream import iter_openai_stream_text, iter_openai_stream_events

@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any]


# Process-level capability flag. Once any backend rejects response_format
# with 4xx, future calls in this process skip the param entirely. Avoids
# paying a 4xx round-trip on every classifier turn after the first failure
# AND keeps the bot routing correctly even if the deployed vLLM is rebuilt
# with a binary that no longer supports response_format=json_schema.
# Codex adversarial review 2026-04-28: without this, every classifier call
# would 4xx and exception-fallback to neutral CONVERSATION intent, silently
# breaking calculator / SMS / change-param routing.
_response_format_supported: bool = True


def call_openai_compatible(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    response_format: dict[str, Any] | None = None,
) -> LLMResponse:
    global _response_format_supported
    if not base_url:
        raise ValueError("RAG_LLM_BASE_URL is not set")
    if not model:
        raise ValueError("RAG_LLM_MODEL is not set")
    url = base_url.rstrip("/") + "/chat/completions"

    def _build_payload(include_response_format: bool) -> dict[str, Any]:
        p: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Disable Qwen3 thinking mode: prevents <think>...</think> wrapper
            # that wastes tokens on reasoning before the actual answer.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_format is not None and include_response_format:
            p["response_format"] = response_format
        return p

    use_rf = response_format is not None and _response_format_supported
    resp = requests.post(url, json=_build_payload(use_rf), timeout=timeout_sec)

    if (
        use_rf
        and resp.status_code in (400, 422)
        and "response_format" in (resp.text or "")
    ):
        # Backend rejected response_format. Latch off for this process and
        # retry once without it. Log loudly so operators see the downgrade.
        print(
            f"[LLM] CAPABILITY-DOWNGRADE: backend at {url} rejected "
            f"response_format ({resp.status_code}); retrying without it. "
            f"Future calls in this process will skip response_format. "
            f"Body excerpt: {(resp.text or '')[:200]!r}",
            flush=True,
        )
        _response_format_supported = False
        resp = requests.post(url, json=_build_payload(False), timeout=timeout_sec)

    resp.raise_for_status()
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    return LLMResponse(text=text.strip(), raw=data)


def iter_openai_compatible_stream(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> Any:
    if not base_url:
        raise ValueError("RAG_LLM_BASE_URL is not set")
    if not model:
        raise ValueError("RAG_LLM_MODEL is not set")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Disable Qwen3 thinking mode for streaming too
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(url, json=payload, timeout=timeout_sec, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        for chunk in iter_openai_stream_text([line]):
            yield chunk


def iter_openai_compatible_stream_events(
    base_url: str,
    model: str,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 120,
    timeout_sec: int = 60,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    if not base_url:
        raise ValueError("RAG_LLM_BASE_URL is not set")
    if not model:
        raise ValueError("RAG_LLM_MODEL is not set")
    url = base_url.rstrip("/") + "/chat/completions"

    if messages is not None:
        msg_list = messages
    else:
        msg_list = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ]

    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": msg_list,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    resp = requests.post(url, json=payload, timeout=timeout_sec, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        for event in iter_openai_stream_events([line]):
            yield event
