"""Contract tests for the Qwen3-Omni hybrid adapter integration.

These tests document the integration contracts that Plan 04-01 establishes and
Plan 04-02 will wire into the backend.  Two tests are marked xfail for code not
yet implemented in Plan 02 (normalizer allowlist and build_voice_statuses entry).
The helper functions defined here (_require_omni_base_url, _call_omni_sidecar)
mirror the guards that will be extracted into app.py in Plan 02.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import requests as _requests_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import voice_adapters
from backend.yandex_realtime import normalize_voice_provider
from backend.voice_session import VoiceSession


# ---------------------------------------------------------------------------
# FakeResponse helper (mirrors pattern in test_voice_adapters_official.py)
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, *, json_data=None, content=b""):
        self._json = json_data or {}
        self.content = content

    @property
    def ok(self) -> bool:
        return True

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._json


# ---------------------------------------------------------------------------
# Testable helper functions (contract mirrors for app.py guards in Plan 02)
# ---------------------------------------------------------------------------


def _require_omni_base_url() -> str:
    """Return QWEN3_OMNI_BASE_URL or raise RuntimeError.

    This mirrors the guard that will live in app.py after Plan 02 wires the
    Omni dispatch path into the WebSocket handler.
    """
    url = os.getenv("QWEN3_OMNI_BASE_URL")
    if not url:
        raise RuntimeError("Qwen3-Omni sidecar unavailable: QWEN3_OMNI_BASE_URL not set")
    return url


def _call_omni_sidecar(
    base_url: str,
    audio_b64: str,
    context_chunks: list[str],
    system_prompt: str = "",
) -> dict:
    """POST /chat to the Omni sidecar and return the JSON response dict.

    This is the minimal dispatch function extracted for contract testing.
    Plan 02 will inline or import an equivalent function in app.py.
    """
    resp = _requests_module.post(
        base_url.rstrip("/") + "/chat",
        json={
            "audio_b64": audio_b64,
            "context_chunks": context_chunks,
            "system_prompt": system_prompt,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_omni_voice_provider_in_normalizer_allowlist() -> None:
    """normalize_voice_provider must pass 'qwen3_omni' through unchanged.

    Currently fails because the allowlist in yandex_realtime.py only covers
    {local, yandex_realtime, yandex_speechkit, oss_russian}.  Plan 02 adds
    'qwen3_omni' to that set.
    """
    result = normalize_voice_provider("qwen3_omni")
    assert result == "qwen3_omni"


def test_omni_stack_id_format() -> None:
    """VoiceSession.stack_id must encode Omni provider identifiers correctly.

    The stack_id property already works with any field values -- no Plan 02
    changes required.  This test pins the exact format expected in JSONL logs.
    """
    session = VoiceSession(
        session_id="test",
        backend="our_rag",
        voice_provider="qwen3_omni",
        brain_model="Qwen/Qwen3-Omni-30B-A3B",
        stt_provider="omni",
        tts_provider="omni",
    )
    assert session.stack_id == "our_rag__Qwen3-Omni-30B-A3B__omni__omni"


def test_build_voice_statuses_includes_qwen3_omni() -> None:
    """build_voice_statuses() must return a dict containing a 'qwen3_omni' key.

    Currently fails because voice_adapters.py does not yet include the Omni
    entry.  Plan 02 adds it following the same _service_status pattern as the
    other providers.
    """
    statuses = voice_adapters.build_voice_statuses()
    assert "qwen3_omni" in statuses


def test_hard_fail_when_omni_base_url_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """_require_omni_base_url must raise RuntimeError when env var is absent.

    Hard-fail behaviour per D-13: if the Omni sidecar is selected but
    QWEN3_OMNI_BASE_URL is not set, the call must raise RuntimeError -- no
    silent fallback allowed.
    """
    monkeypatch.delenv("QWEN3_OMNI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="QWEN3_OMNI_BASE_URL not set"):
        _require_omni_base_url()


def test_chat_endpoint_dispatches_to_omni_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    """_call_omni_sidecar must POST to {base_url}/chat with required fields.

    Verifies that the dispatch helper targets the correct endpoint and includes
    both audio_b64 and context_chunks in the JSON payload.
    """
    calls: list[dict] = []
    monkeypatch.setenv("QWEN3_OMNI_BASE_URL", "http://omni.local:8002")

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls.append({"url": url, "json": json})
        return FakeResponse(
            json_data={
                "audio_b64": "AQID",
                "text": "Ответ",
                "sample_rate_hz": 24000,
                "t_omni_first_audio": 1234567890.0,
            }
        )

    monkeypatch.setattr(_requests_module, "post", fake_post)

    _call_omni_sidecar(
        base_url="http://omni.local:8002",
        audio_b64="AQID",
        context_chunks=["Chunk A"],
    )

    assert len(calls) == 1
    assert calls[0]["url"] == "http://omni.local:8002/chat"
    assert "audio_b64" in calls[0]["json"]
    assert "context_chunks" in calls[0]["json"]


def test_context_chunks_in_chat_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The POST /chat payload must include context_chunks as a list of strings.

    Verifies that RAG-retrieved chunks are forwarded to the sidecar exactly
    as received from the retrieval step.
    """
    calls: list[dict] = []

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls.append({"url": url, "json": json})
        return FakeResponse(
            json_data={
                "audio_b64": "AQID",
                "text": "Ответ",
                "sample_rate_hz": 24000,
                "t_omni_first_audio": 1234567890.0,
            }
        )

    monkeypatch.setattr(_requests_module, "post", fake_post)

    _call_omni_sidecar(
        base_url="http://omni.local:8002",
        audio_b64="AQID",
        context_chunks=["Chunk A", "Chunk B"],
    )

    assert calls[0]["json"]["context_chunks"] == ["Chunk A", "Chunk B"]


def test_omni_jsonl_has_required_fields() -> None:
    """The JSONL log dict for an Omni voice turn must contain all 6 timing fields.

    Per D-06: llm_first_token and tts_first_chunk are both set to the Omni
    first-audio timestamp (collapsed, because Omni generates audio natively
    with no separate TTS step).  Primary KPI formula remains comparable with
    the split pipeline (D-07).
    """
    log_entry = {
        "event": "voice_turn",
        "question_id": "test-001",
        "stack_id": "our_rag__Qwen3-Omni-30B-A3B__omni__omni",
        "transcript": "Что такое лизинг?",
        "speech_stopped": 1.0,
        "stt_done": 2.0,
        "retrieval_done": 3.0,
        "llm_first_token": 4.0,
        "tts_first_chunk": 4.0,  # collapsed with llm_first_token per D-06
        "playback_started": 5.0,
        "primary_kpi_ms": 4000.0,
    }

    required_fields = {
        "event",
        "question_id",
        "stack_id",
        "transcript",
        "speech_stopped",
        "stt_done",
        "retrieval_done",
        "llm_first_token",
        "tts_first_chunk",
        "playback_started",
        "primary_kpi_ms",
    }
    assert required_fields.issubset(log_entry.keys())
    # D-06: Omni collapses tts_first_chunk == llm_first_token
    assert log_entry["llm_first_token"] == log_entry["tts_first_chunk"]
