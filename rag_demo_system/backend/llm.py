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


def call_openai_compatible(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> LLMResponse:
    if not base_url:
        raise ValueError("RAG_LLM_BASE_URL is not set")
    if not model:
        raise ValueError("RAG_LLM_MODEL is not set")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(url, json=payload, timeout=timeout_sec)
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
    }
    resp = requests.post(url, json=payload, timeout=timeout_sec, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        for event in iter_openai_stream_events([line]):
            yield event
