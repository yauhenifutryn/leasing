import importlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_module():
    spec = importlib.util.find_spec("backend.rag_backends")
    assert spec is not None, "backend.rag_backends module is missing"
    return importlib.import_module("backend.rag_backends")


def test_resolve_backend_defaults_to_our_rag() -> None:
    rag_backends = _load_module()
    provider = rag_backends.resolve_backend(None, {"our_rag": object()})

    assert provider is not None


def test_resolve_backend_uses_requested_provider() -> None:
    rag_backends = _load_module()
    requested = object()

    provider = rag_backends.resolve_backend(
        "dify_rag",
        {
            "our_rag": object(),
            "dify_rag": requested,
        },
    )

    assert provider is requested


def test_map_dify_retriever_resources_to_used_knowledge() -> None:
    rag_backends = _load_module()
    payload = [
        {
            "document_name": "kb_faq_ru_structured.md",
            "segment_position": 7,
            "content": "Требования к лизингу грузового транспорта включают...",
            "score": 0.91,
            "metadata": {"heading_path": ["Грузовой транспорт", "Требования"]},
        }
    ]

    used = rag_backends.map_dify_retriever_resources(payload)

    assert used == [
        {
            "chunk_id": "kb_faq_ru_structured.md:7",
            "doc_name": "kb_faq_ru_structured.md",
            "heading_path": ["Грузовой транспорт", "Требования"],
            "snippet": "Требования к лизингу грузового транспорта включают...",
            "score": 0.91,
        }
    ]


def test_build_backend_status_reports_all_services() -> None:
    rag_backends = _load_module()
    statuses = rag_backends.build_backend_status(
        launch_mode="supervisor",
        rag_statuses={
            "our_rag": {"available": True, "healthy": True},
            "dify_rag": {"available": False, "healthy": False},
        },
        voice_statuses={
            "sensevoice": {"available": True, "healthy": True},
            "whisper": {"available": True, "healthy": False},
            "cosyvoice": {"available": True, "healthy": True},
        },
        llm_status={"qwen": {"available": True, "healthy": True}},
    )

    assert statuses["ok"] is True
    assert statuses["launch_mode"] == "supervisor"
    assert set(statuses["backends"]) == {
        "our_rag",
        "dify_rag",
        "sensevoice",
        "whisper",
        "cosyvoice",
        "qwen",
    }
