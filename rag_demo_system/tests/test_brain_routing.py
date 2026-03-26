"""
Unit tests for brain_model routing through ChatRequest and effective_model resolution.

These tests verify that:
1. ChatRequest accepts brain_model=None (backward compatible default)
2. ChatRequest accepts a non-default brain_model string without error
3. effective_model resolves to brain_model when brain_model is set and fast=True
4. effective_model falls back to settings.llm.fast_model when brain_model is None and fast=True
5. effective_model falls back to settings.llm.model when brain_model is None and fast=False
"""
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_app_module():
    """
    Load backend.app with heavy infrastructure dependencies stubbed out so that
    ChatRequest (a pure pydantic model) can be instantiated in unit tests
    without a running Qdrant, rank_bm25, etc.
    """
    import importlib
    import importlib.util

    # Stub all modules that require installed packages or running services
    _stubs = [
        "rank_bm25",
        "qdrant_client",
        "qdrant_client.models",
        "qdrant_client.http",
        "qdrant_client.http.models",
        "sentence_transformers",
        "torch",
        "backend.engine",
        "backend.dify_client",
        "backend.rag_backends",
        "backend.router",
        "backend.voice_adapters",
        "backend.yandex_realtime",
    ]
    mocks = {name: MagicMock() for name in _stubs}

    with patch.dict("sys.modules", mocks):
        # Force re-import if module was previously partially loaded
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("backend.app") or mod_name == "backend.app":
                del sys.modules[mod_name]
        spec = importlib.util.find_spec("backend.app")
        assert spec is not None, "backend.app module is missing"
        mod = importlib.import_module("backend.app")
    return mod


def _resolve_effective_model(brain_model, fast, fast_model, base_model):
    """
    Replicate the effective_model resolution logic from chat() in app.py:
    effective_model = payload.brain_model or (fast_model if fast and fast_model else base_model)
    """
    return brain_model or (fast_model if fast and fast_model else base_model)


def test_chat_request_brain_model_field() -> None:
    """ChatRequest must accept brain_model=None (backward-compatible default)."""
    app = _load_app_module()
    req = app.ChatRequest(message="hello", session_id="s1")
    assert req.brain_model is None


def test_chat_request_accepts_non_default_brain_model() -> None:
    """ChatRequest must accept brain_model='Qwen/Qwen3.5-35B-A3B' without error."""
    app = _load_app_module()
    req = app.ChatRequest(
        message="hello",
        session_id="s1",
        brain_model="Qwen/Qwen3.5-35B-A3B",
    )
    assert req.brain_model == "Qwen/Qwen3.5-35B-A3B"


def test_effective_model_prefers_brain_model() -> None:
    """effective_model must resolve to brain_model when brain_model is set, even if fast=True."""
    effective = _resolve_effective_model(
        brain_model="Qwen/Qwen3.5-35B-A3B",
        fast=True,
        fast_model="Qwen/Qwen3-30B-A3B",
        base_model="Qwen/Qwen3-30B-A3B",
    )
    assert effective == "Qwen/Qwen3.5-35B-A3B"


def test_effective_model_falls_back_to_fast_model() -> None:
    """When brain_model is None and fast=True, effective_model must use fast_model."""
    effective = _resolve_effective_model(
        brain_model=None,
        fast=True,
        fast_model="Qwen/Qwen3-30B-A3B",
        base_model="fallback-model",
    )
    assert effective == "Qwen/Qwen3-30B-A3B"


def test_effective_model_falls_back_to_base_model() -> None:
    """When brain_model is None and fast=False, effective_model must use base settings.llm.model."""
    effective = _resolve_effective_model(
        brain_model=None,
        fast=False,
        fast_model="Qwen/Qwen3-30B-A3B",
        base_model="base-default-model",
    )
    assert effective == "base-default-model"
