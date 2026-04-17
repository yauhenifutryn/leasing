"""SessionAgent configuration: env-driven base_url and model with fallback."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _reload_settings():
    """Reload settings module fresh so env changes take effect."""
    import importlib
    from backend import settings as _settings_mod
    importlib.reload(_settings_mod)
    return _settings_mod


def test_session_agent_defaults_from_env(monkeypatch, tmp_path) -> None:
    """When SESSIONAGENT_BASE_URL is set, LLMConfig reflects it."""
    monkeypatch.setenv("SESSIONAGENT_BASE_URL", "http://127.0.0.1:8788/v1")
    monkeypatch.setenv("SESSIONAGENT_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8")

    # Write a minimal yaml config for load_settings
    yaml_path = tmp_path / "app.yaml"
    yaml_path.write_text(
        "app:\n  name: test\n  language: ru\n"
        "  system_prompt_path: config/system_prompt_ru_v2.txt\n"
        "  kb_markdown_path: config/kb.md\n"
        "  strict_refusal_text: 'no'\n"
        "  memory_turns: 4\n"
        "embedding:\n  model_name: test\n  batch_size: 1\n  device: cpu\n"
        "  max_characters: 1000\n  chunk_size_tokens: 100\n  chunk_overlap_tokens: 10\n"
        "qdrant:\n  url: http://localhost:6333\n  collection: x\n"
        "retrieval:\n  vector_top_k: 1\n  bm25_top_k: 1\n  final_top_n: 1\n"
        "  score_threshold: 0\n  min_rerank_score: 0\n  context_max_tokens: 100\n"
        "llm:\n  provider: p\n  base_url: http://x\n  model: m\n  temperature: 0\n"
        "  max_tokens: 100\n  fast_base_url: http://x\n  fast_model: fm\n"
        "  fast_max_tokens: 50\n  timeout_sec: 10\n"
        "  concise_sentences_min: 2\n  concise_sentences_max: 5\n"
        "  expand_triggers: []\n"
        "reranker:\n  enabled: false\n  model_name: x\n  device: cpu\n  batch_size: 1\n"
        "  allow_no_rerank: true\n"
        "query_rewrite:\n  abbreviations_path: config/abbr.yaml\n"
        "voice:\n  enabled: false\n  provider: x\n  api_key_env: x\n  voice_id_env: x\n"
        "  stt_ws_url: ''\n  tts_stream_url: ''\n  sample_rate_hz: 16000\n"
        "tools:\n  calculator_api_base_url: ''\n  calculator_api_token: ''\n"
        "  sms_api_login: ''\n  sms_api_password: ''\n  sms_sender_name: ''\n"
        "  crm_webhook_url: ''\n  crm_webhook_token: ''\n"
        "jambonz:\n  enabled: false\n"
    )

    mod = _reload_settings()
    s = mod.load_settings(yaml_path)
    assert s.llm.session_agent_base_url == "http://127.0.0.1:8788/v1"
    assert s.llm.session_agent_model == "Qwen/Qwen3-4B-Instruct-2507-FP8"


def test_session_agent_env_override(monkeypatch, tmp_path) -> None:
    """Explicit override of env var takes precedence over yaml default."""
    monkeypatch.setenv("SESSIONAGENT_BASE_URL", "http://1.2.3.4:9999/v1")
    monkeypatch.setenv("SESSIONAGENT_MODEL", "my/custom-model")

    yaml_path = tmp_path / "app.yaml"
    yaml_path.write_text(
        "app:\n  name: test\n  language: ru\n"
        "  system_prompt_path: config/system_prompt_ru_v2.txt\n"
        "  kb_markdown_path: config/kb.md\n"
        "  strict_refusal_text: 'no'\n  memory_turns: 4\n"
        "embedding:\n  model_name: test\n  batch_size: 1\n  device: cpu\n"
        "  max_characters: 1000\n  chunk_size_tokens: 100\n  chunk_overlap_tokens: 10\n"
        "qdrant:\n  url: http://localhost:6333\n  collection: x\n"
        "retrieval:\n  vector_top_k: 1\n  bm25_top_k: 1\n  final_top_n: 1\n"
        "  score_threshold: 0\n  min_rerank_score: 0\n  context_max_tokens: 100\n"
        "llm:\n  provider: p\n  base_url: http://x\n  model: m\n  temperature: 0\n"
        "  max_tokens: 100\n  fast_base_url: http://x\n  fast_model: fm\n"
        "  fast_max_tokens: 50\n  timeout_sec: 10\n"
        "  concise_sentences_min: 2\n  concise_sentences_max: 5\n  expand_triggers: []\n"
        "  session_agent_base_url: 'yaml-default'\n"
        "  session_agent_model: 'yaml-model'\n"
        "reranker:\n  enabled: false\n  model_name: x\n  device: cpu\n  batch_size: 1\n"
        "  allow_no_rerank: true\n"
        "query_rewrite:\n  abbreviations_path: config/abbr.yaml\n"
        "voice:\n  enabled: false\n  provider: x\n  api_key_env: x\n  voice_id_env: x\n"
        "  stt_ws_url: ''\n  tts_stream_url: ''\n  sample_rate_hz: 16000\n"
        "tools:\n  calculator_api_base_url: ''\n  calculator_api_token: ''\n"
        "  sms_api_login: ''\n  sms_api_password: ''\n  sms_sender_name: ''\n"
        "  crm_webhook_url: ''\n  crm_webhook_token: ''\n"
        "jambonz:\n  enabled: false\n"
    )
    mod = _reload_settings()
    s = mod.load_settings(yaml_path)
    # Env wins over yaml
    assert s.llm.session_agent_base_url == "http://1.2.3.4:9999/v1"
    assert s.llm.session_agent_model == "my/custom-model"
