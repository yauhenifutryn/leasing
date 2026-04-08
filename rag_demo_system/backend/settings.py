from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    name: str
    language: str
    system_prompt_path: Path
    kb_markdown_path: Path
    strict_refusal_text: str
    memory_turns: int


@dataclass
class EmbeddingConfig:
    model_name: str
    batch_size: int
    device: str
    max_characters: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int


@dataclass
class QdrantConfig:
    url: str
    collection: str


@dataclass
class RetrievalConfig:
    vector_top_k: int
    bm25_top_k: int
    final_top_n: int
    score_threshold: float
    min_rerank_score: float
    context_max_tokens: int
    fast_vector_top_k: int
    fast_bm25_top_k: int
    fast_final_top_n: int
    fast_context_max_tokens: int
    voice_vector_top_k: int
    voice_bm25_top_k: int
    voice_final_top_n: int
    voice_context_max_tokens: int


@dataclass
class LLMConfig:
    provider: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    fast_base_url: str
    fast_model: str
    fast_max_tokens: int
    timeout_sec: int
    concise_sentences_min: int
    concise_sentences_max: int
    expand_triggers: list[str]


@dataclass
class RerankerConfig:
    enabled: bool
    model_name: str
    device: str
    batch_size: int
    allow_no_rerank: bool


@dataclass
class QueryRewriteConfig:
    abbreviations_path: Path


@dataclass
class ToolsConfig:
    calculator_api_base_url: str
    calculator_api_token: str
    sms_api_login: str
    sms_api_password: str
    sms_sender_name: str
    crm_webhook_url: str
    crm_webhook_token: str


@dataclass
class VoiceConfig:
    enabled: bool
    provider: str
    api_key_env: str
    voice_id_env: str
    stt_ws_url: str
    tts_stream_url: str
    sample_rate_hz: int


@dataclass
class Settings:
    app: AppConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    reranker: RerankerConfig
    query_rewrite: QueryRewriteConfig
    voice: VoiceConfig
    tools: ToolsConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "rag_demo_system" / "config" / "app.yaml"
DEFAULT_ENV = REPO_ROOT / "rag_demo_system" / ".env"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return (REPO_ROOT / "rag_demo_system" / value).resolve()
    return path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings(path: Path | None = None) -> Settings:
    _load_env_file(DEFAULT_ENV)
    cfg_path = path or DEFAULT_CONFIG
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    app = payload.get("app", {})
    embedding = payload.get("embedding", {})
    qdrant = payload.get("qdrant", {})
    retrieval = payload.get("retrieval", {})
    llm = payload.get("llm", {})
    reranker = payload.get("reranker", {})
    query_rewrite = payload.get("query_rewrite", {})
    voice = payload.get("voice", {})
    tools_cfg = payload.get("tools", {})

    return Settings(
        app=AppConfig(
            name=app.get("name", "RAG Demo"),
            language=app.get("language", "ru"),
            system_prompt_path=_resolve_path(app.get("system_prompt_path")),
            kb_markdown_path=_resolve_path(app.get("kb_markdown_path")),
            strict_refusal_text=app.get("strict_refusal_text", ""),
            memory_turns=int(app.get("memory_turns", 4)),
        ),
        embedding=EmbeddingConfig(
            model_name=embedding.get("model_name", "intfloat/multilingual-e5-large"),
            batch_size=int(embedding.get("batch_size", 16)),
            device=os.getenv("RAG_EMBEDDING_DEVICE", embedding.get("device", "cpu")),
            max_characters=int(embedding.get("max_characters", 6000)),
            chunk_size_tokens=int(embedding.get("chunk_size_tokens", 700)),
            chunk_overlap_tokens=int(embedding.get("chunk_overlap_tokens", 120)),
        ),
        qdrant=QdrantConfig(
            url=qdrant.get("url", "http://localhost:6333"),
            collection=qdrant.get("collection", "micro_leasing_kb"),
        ),
        retrieval=RetrievalConfig(
            vector_top_k=int(retrieval.get("vector_top_k", 8)),
            bm25_top_k=int(retrieval.get("bm25_top_k", 8)),
            final_top_n=int(retrieval.get("final_top_n", 6)),
            score_threshold=float(retrieval.get("score_threshold", 0.35)),
            min_rerank_score=float(retrieval.get("min_rerank_score", 0.10)),
            context_max_tokens=int(retrieval.get("context_max_tokens", 1800)),
            fast_vector_top_k=int(retrieval.get("fast_vector_top_k", 4)),
            fast_bm25_top_k=int(retrieval.get("fast_bm25_top_k", 4)),
            fast_final_top_n=int(retrieval.get("fast_final_top_n", 3)),
            fast_context_max_tokens=int(retrieval.get("fast_context_max_tokens", 900)),
            voice_vector_top_k=int(retrieval.get("voice_vector_top_k", 3)),
            voice_bm25_top_k=int(retrieval.get("voice_bm25_top_k", 1)),
            voice_final_top_n=int(retrieval.get("voice_final_top_n", 2)),
            voice_context_max_tokens=int(retrieval.get("voice_context_max_tokens", 500)),
        ),
        llm=LLMConfig(
            provider=llm.get("provider", "openai_compatible"),
            base_url=os.getenv("RAG_LLM_BASE_URL", llm.get("base_url", "")),
            model=os.getenv("RAG_LLM_MODEL", llm.get("model", "")),
            temperature=float(llm.get("temperature", 0.1)),
            max_tokens=int(llm.get("max_tokens", 420)),
            fast_base_url=os.getenv("RAG_LLM_FAST_BASE_URL", llm.get("fast_base_url", "")),
            fast_model=os.getenv("RAG_LLM_FAST_MODEL", llm.get("fast_model", "")),
            fast_max_tokens=int(os.getenv("RAG_LLM_FAST_MAX_TOKENS", llm.get("fast_max_tokens", 220))),
            timeout_sec=int(llm.get("timeout_sec", 60)),
            concise_sentences_min=int(llm.get("concise_sentences_min", 3)),
            concise_sentences_max=int(llm.get("concise_sentences_max", 6)),
            expand_triggers=list(llm.get("expand_triggers", [])),
        ),
        reranker=RerankerConfig(
            enabled=bool(reranker.get("enabled", True)),
            model_name=reranker.get("model_name", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
            device=os.getenv("RAG_RERANKER_DEVICE", reranker.get("device", "cpu")),
            batch_size=int(reranker.get("batch_size", 16)),
            allow_no_rerank=bool(reranker.get("allow_no_rerank", False)),
        ),
        query_rewrite=QueryRewriteConfig(
            abbreviations_path=_resolve_path(query_rewrite.get("abbreviations_path", "config/abbreviations.yaml")),
        ),
        voice=VoiceConfig(
            enabled=bool(voice.get("enabled", False)),
            provider=voice.get("provider", "elevenlabs"),
            api_key_env=voice.get("api_key_env", "ELEVENLABS_API_KEY"),
            voice_id_env=voice.get("voice_id_env", "ELEVENLABS_VOICE_ID"),
            stt_ws_url=voice.get("stt_ws_url", ""),
            tts_stream_url=voice.get("tts_stream_url", ""),
            sample_rate_hz=int(voice.get("sample_rate_hz", 16000)),
        ),
        tools=ToolsConfig(
            calculator_api_base_url=os.getenv("CALCULATOR_API_BASE_URL", tools_cfg.get("calculator_api_base_url", "")),
            calculator_api_token=os.getenv("CALCULATOR_API_TOKEN", tools_cfg.get("calculator_api_token", "")),
            sms_api_login=os.getenv("SMS_API_LOGIN", tools_cfg.get("sms_api_login", "")),
            sms_api_password=os.getenv("SMS_API_PASSWORD", tools_cfg.get("sms_api_password", "")),
            sms_sender_name=os.getenv("SMS_SENDER_NAME", tools_cfg.get("sms_sender_name", "MikroLizing")),
            crm_webhook_url=os.getenv("CRM_WEBHOOK_URL", tools_cfg.get("crm_webhook_url", "")),
            crm_webhook_token=os.getenv("CRM_WEBHOOK_TOKEN", tools_cfg.get("crm_webhook_token", "")),
        ),
    )
