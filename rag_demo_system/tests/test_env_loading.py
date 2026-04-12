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


class TestJambonzSettings:
    def test_jambonz_config_defaults(self):
        from backend.settings import load_settings
        s = load_settings()
        assert hasattr(s, "jambonz")
        assert s.jambonz.enabled is False
        assert s.jambonz.api_base_url == "http://127.0.0.1:3000"
        assert s.jambonz.account_sid == ""
        assert s.jambonz.app_sid == ""
        assert s.jambonz.sip_realm == ""
        assert s.jambonz.sip_user == "test"
        assert s.jambonz.sip_password == ""

    def test_jambonz_config_from_env(self, monkeypatch):
        monkeypatch.setenv("JAMBONZ_ENABLED", "true")
        monkeypatch.setenv("JAMBONZ_ACCOUNT_SID", "acc-999")
        monkeypatch.setenv("JAMBONZ_SIP_PASSWORD", "secret456")
        from backend.settings import load_settings
        s = load_settings()
        assert s.jambonz.enabled is True
        assert s.jambonz.account_sid == "acc-999"
        assert s.jambonz.sip_password == "secret456"


def test_jambonz_config_loads_from_env(monkeypatch):
    """JambonzConfig should load from JAMBONZ_* env vars."""
    monkeypatch.setenv("JAMBONZ_ENABLED", "true")
    monkeypatch.setenv("JAMBONZ_API_BASE_URL", "http://localhost:3000")
    monkeypatch.setenv("JAMBONZ_ACCOUNT_SID", "acc-123")
    monkeypatch.setenv("JAMBONZ_APP_SID", "app-456")
    monkeypatch.setenv("JAMBONZ_SIP_REALM", "voice.example.com")
    monkeypatch.setenv("JAMBONZ_SIP_USER", "test")
    monkeypatch.setenv("JAMBONZ_SIP_PASSWORD", "secret123")

    import importlib
    import backend.settings as mod
    importlib.reload(mod)
    s = mod.load_settings()

    assert s.jambonz.enabled is True
    assert s.jambonz.api_base_url == "http://localhost:3000"
    assert s.jambonz.account_sid == "acc-123"
    assert s.jambonz.sip_user == "test"
    assert s.jambonz.sip_password == "secret123"
