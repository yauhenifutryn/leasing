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
