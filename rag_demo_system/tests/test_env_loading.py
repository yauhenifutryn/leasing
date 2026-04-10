from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import settings as settings_module


def test_load_settings_reads_dotenv(monkeypatch):
    env_path = settings_module.REPO_ROOT / "rag_demo_system" / ".env"
    original = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    original_base = settings_module.os.environ.get("RAG_LLM_BASE_URL")
    original_model = settings_module.os.environ.get("RAG_LLM_MODEL")
    try:
        monkeypatch.delenv("RAG_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
        env_path.write_text(
            "RAG_LLM_BASE_URL=http://example.local/v1\n"
            "RAG_LLM_MODEL=test-model\n",
            encoding="utf-8",
        )

        loaded = settings_module.load_settings()

        assert loaded.llm.base_url == "http://example.local/v1"
        assert loaded.llm.model == "test-model"
    finally:
        if original is None:
            try:
                env_path.unlink()
            except FileNotFoundError:
                pass
        else:
            env_path.write_text(original, encoding="utf-8")
        if original_base is None:
            settings_module.os.environ.pop("RAG_LLM_BASE_URL", None)
        else:
            settings_module.os.environ["RAG_LLM_BASE_URL"] = original_base
        if original_model is None:
            settings_module.os.environ.pop("RAG_LLM_MODEL", None)
        else:
            settings_module.os.environ["RAG_LLM_MODEL"] = original_model


def test_load_settings_reads_device_overrides(monkeypatch):
    original_embed = settings_module.os.environ.get("RAG_EMBEDDING_DEVICE")
    original_rerank = settings_module.os.environ.get("RAG_RERANKER_DEVICE")
    try:
        monkeypatch.setenv("RAG_EMBEDDING_DEVICE", "cuda")
        monkeypatch.setenv("RAG_RERANKER_DEVICE", "cuda")

        loaded = settings_module.load_settings()

        assert loaded.embedding.device == "cuda"
        assert loaded.reranker.device == "cuda"
    finally:
        if original_embed is None:
            settings_module.os.environ.pop("RAG_EMBEDDING_DEVICE", None)
        else:
            settings_module.os.environ["RAG_EMBEDDING_DEVICE"] = original_embed
        if original_rerank is None:
            settings_module.os.environ.pop("RAG_RERANKER_DEVICE", None)
        else:
            settings_module.os.environ["RAG_RERANKER_DEVICE"] = original_rerank


def test_load_settings_reads_fast_llm_env(monkeypatch):
    env_path = settings_module.REPO_ROOT / "rag_demo_system" / ".env"
    original = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    original_fast_base = settings_module.os.environ.get("RAG_LLM_FAST_BASE_URL")
    original_fast_model = settings_module.os.environ.get("RAG_LLM_FAST_MODEL")
    try:
        monkeypatch.delenv("RAG_LLM_FAST_BASE_URL", raising=False)
        monkeypatch.delenv("RAG_LLM_FAST_MODEL", raising=False)
        env_path.write_text(
            "RAG_LLM_FAST_BASE_URL=http://fast.local/v1\n"
            "RAG_LLM_FAST_MODEL=fast-model\n",
            encoding="utf-8",
        )

        loaded = settings_module.load_settings()

        assert loaded.llm.fast_base_url == "http://fast.local/v1"
        assert loaded.llm.fast_model == "fast-model"
    finally:
        if original is None:
            try:
                env_path.unlink()
            except FileNotFoundError:
                pass
        else:
            env_path.write_text(original, encoding="utf-8")
        if original_fast_base is None:
            settings_module.os.environ.pop("RAG_LLM_FAST_BASE_URL", None)
        else:
            settings_module.os.environ["RAG_LLM_FAST_BASE_URL"] = original_fast_base
        if original_fast_model is None:
            settings_module.os.environ.pop("RAG_LLM_FAST_MODEL", None)
        else:
            settings_module.os.environ["RAG_LLM_FAST_MODEL"] = original_fast_model


class TestSIPSettings:
    def test_sip_config_defaults(self):
        from backend.settings import load_settings
        s = load_settings()
        assert hasattr(s, "sip")
        assert s.sip.enabled is False
        assert s.sip.audiosocket_host == "127.0.0.1"
        assert s.sip.audiosocket_port == 9092
        assert s.sip.ami_host == "127.0.0.1"
        assert s.sip.ami_port == 5038
        assert s.sip.ami_username == "voicebot"
        assert s.sip.ami_secret == ""

    def test_sip_config_from_env(self, monkeypatch):
        monkeypatch.setenv("SIP_ENABLED", "true")
        monkeypatch.setenv("AUDIOSOCKET_PORT", "9999")
        monkeypatch.setenv("AMI_SECRET", "secret123")
        from backend.settings import load_settings
        s = load_settings()
        assert s.sip.enabled is True
        assert s.sip.audiosocket_port == 9999
        assert s.sip.ami_secret == "secret123"
