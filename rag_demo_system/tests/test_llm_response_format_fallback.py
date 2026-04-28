"""Capability-downgrade tests for call_openai_compatible.

Codex adversarial review 2026-04-28: when the backend rejects
response_format with 4xx, the call must transparently retry without
the param so classifier routing stays alive even on a vLLM build that
doesn't support response_format=json_schema. Verifies that:
  1. A successful first call uses response_format and returns parsed text.
  2. A 4xx-on-response_format triggers a retry without the param and
     succeeds.
  3. After the first downgrade, subsequent calls in the same process
     skip response_format entirely (no second 4xx round-trip).
  4. Non-response_format errors still raise (e.g. 500, network).
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from backend import llm as llm_mod


def _ok_response(content: str = '{"intent":"TOOL"}') -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "choices": [{"message": {"content": content}}],
    }
    r.raise_for_status = MagicMock()
    return r


def _bad_request_response(body: str) -> MagicMock:
    r = MagicMock()
    r.status_code = 400
    r.text = body

    def _raise():
        from requests.exceptions import HTTPError
        raise HTTPError(f"400 Client Error: {body}")

    r.raise_for_status = _raise
    return r


@pytest.fixture(autouse=True)
def _reset_capability_cache():
    """Each test starts with an empty per-backend capability cache —
    every (base_url, model) pair defaults to 'supported' until proven
    otherwise by a 4xx response."""
    llm_mod._response_format_supported.clear()
    yield
    llm_mod._response_format_supported.clear()


def test_response_format_supported_path_uses_param() -> None:
    """Happy path: 200 with response_format → param is in the request body."""
    with patch.object(llm_mod.requests, "post", return_value=_ok_response()) as post:
        result = llm_mod.call_openai_compatible(
            base_url="http://x/v1",
            model="m",
            system_prompt="s",
            user_prompt="u",
            temperature=0.0,
            max_tokens=10,
            timeout_sec=4,
            response_format={"type": "json_schema", "json_schema": {"name": "X", "schema": {}}},
        )
    assert result.text == '{"intent":"TOOL"}'
    assert post.call_count == 1
    sent = post.call_args.kwargs["json"]
    assert "response_format" in sent
    # Cache stays empty (or has a True for this key) — happy path.
    assert llm_mod._rf_supported("http://x/v1", "m") is True


def test_4xx_on_response_format_triggers_downgrade_and_retry() -> None:
    """Codex finding: backend rejects response_format → retry once without
    it, succeed, latch capability off for this process."""
    bad = _bad_request_response("invalid response_format: not supported in this build")
    good = _ok_response('{"intent":"RAG"}')
    with patch.object(llm_mod.requests, "post", side_effect=[bad, good]) as post:
        result = llm_mod.call_openai_compatible(
            base_url="http://x/v1",
            model="m",
            system_prompt="s",
            user_prompt="u",
            temperature=0.0,
            max_tokens=10,
            timeout_sec=4,
            response_format={"type": "json_schema", "json_schema": {"name": "X", "schema": {}}},
        )
    assert result.text == '{"intent":"RAG"}'
    assert post.call_count == 2
    # First call had response_format, second did not.
    first_payload = post.call_args_list[0].kwargs["json"]
    second_payload = post.call_args_list[1].kwargs["json"]
    assert "response_format" in first_payload
    assert "response_format" not in second_payload
    # Capability marked unsupported for THIS (base_url, model).
    assert llm_mod._rf_supported("http://x/v1", "m") is False


def test_subsequent_call_after_downgrade_skips_response_format() -> None:
    """Once marked unsupported for a backend+model, no second 4xx
    round-trip — go straight to plain for that key."""
    llm_mod._rf_mark_unsupported("http://x/v1", "m")
    with patch.object(llm_mod.requests, "post", return_value=_ok_response()) as post:
        llm_mod.call_openai_compatible(
            base_url="http://x/v1",
            model="m",
            system_prompt="s",
            user_prompt="u",
            temperature=0.0,
            max_tokens=10,
            timeout_sec=4,
            response_format={"type": "json_schema", "json_schema": {"name": "X", "schema": {}}},
        )
    assert post.call_count == 1
    sent = post.call_args.kwargs["json"]
    assert "response_format" not in sent


def test_downgrade_is_per_backend_not_global() -> None:
    """Codex finding (medium): a 4xx from one backend must NOT disable
    response_format for OTHER backends. Critical because app.py sends
    response_format to the SessionAgent classifier (port 8788) AND the
    first-utterance name extractor (potentially a different endpoint).
    A reject from a weaker endpoint must NOT remove schema enforcement
    from the production classifier vLLM that supports it."""
    # Backend A rejects response_format.
    bad = _bad_request_response("response_format not supported on this build")
    # Backend B accepts it normally.
    good_b = _ok_response('{"intent":"TOOL"}')

    with patch.object(llm_mod.requests, "post", side_effect=[bad, _ok_response(), good_b]):
        # Call A — gets 4xx, downgrade for A only.
        llm_mod.call_openai_compatible(
            base_url="http://A/v1", model="mA",
            system_prompt="s", user_prompt="u",
            temperature=0.0, max_tokens=10, timeout_sec=4,
            response_format={"type": "json_schema", "json_schema": {"name": "X", "schema": {}}},
        )
        # Call B — should still try response_format (not affected by A's downgrade).
        llm_mod.call_openai_compatible(
            base_url="http://B/v1", model="mB",
            system_prompt="s", user_prompt="u",
            temperature=0.0, max_tokens=10, timeout_sec=4,
            response_format={"type": "json_schema", "json_schema": {"name": "X", "schema": {}}},
        )

    assert llm_mod._rf_supported("http://A/v1", "mA") is False, (
        "backend A must be marked unsupported"
    )
    assert llm_mod._rf_supported("http://B/v1", "mB") is True, (
        "backend B unaffected — different (base_url, model)"
    )


def test_downgrade_per_model_not_just_per_url() -> None:
    """Same base_url but different model — different cache key. A schema
    that one model rejects (e.g. older variant) must not block the
    next model from trying it."""
    llm_mod._rf_mark_unsupported("http://x/v1", "old_model")
    assert llm_mod._rf_supported("http://x/v1", "old_model") is False
    assert llm_mod._rf_supported("http://x/v1", "new_model") is True


def test_unrelated_4xx_still_raises() -> None:
    """A 400 that isn't about response_format must propagate — don't mask
    real bugs (auth failure, malformed prompt, etc.)."""
    bad_other = _bad_request_response("model 'foo' not found")
    with patch.object(llm_mod.requests, "post", return_value=bad_other):
        with pytest.raises(Exception):
            llm_mod.call_openai_compatible(
                base_url="http://x/v1",
                model="m",
                system_prompt="s",
                user_prompt="u",
                temperature=0.0,
                max_tokens=10,
                timeout_sec=4,
                response_format={"type": "json_schema", "json_schema": {"name": "X", "schema": {}}},
            )
    # Cache must NOT mark unsupported — the 4xx was about something else.
    assert llm_mod._rf_supported("http://x/v1", "m") is True


def test_no_response_format_supplied_skips_capability_logic() -> None:
    """Backwards-compat: if caller doesn't pass response_format, no retry
    logic kicks in and the capability flag is irrelevant."""
    with patch.object(llm_mod.requests, "post", return_value=_ok_response()) as post:
        llm_mod.call_openai_compatible(
            base_url="http://x/v1",
            model="m",
            system_prompt="s",
            user_prompt="u",
            temperature=0.0,
            max_tokens=10,
            timeout_sec=4,
        )
    assert post.call_count == 1
    sent = post.call_args.kwargs["json"]
    assert "response_format" not in sent
