from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_module():
    import importlib
    import importlib.util

    spec = importlib.util.find_spec("backend.yandex_realtime")
    assert spec is not None, "backend.yandex_realtime module is missing"
    return importlib.import_module("backend.yandex_realtime")


def test_build_status_reports_configured_when_folder_and_api_key_present(monkeypatch) -> None:
    yandex = _load_module()
    monkeypatch.setenv("YC_FOLDER_ID", "folder-1")
    monkeypatch.setenv("YC_API_KEY", "key-1")
    monkeypatch.delenv("YC_IAM_TOKEN", raising=False)

    status = yandex.build_status()

    assert status == {
        "name": "yandex_realtime",
        "available": True,
        "healthy": True,
        "reason": "ok",
    }


def test_build_status_reports_not_configured_without_credentials(monkeypatch) -> None:
    yandex = _load_module()
    monkeypatch.setenv("YC_FOLDER_ID", "folder-1")
    monkeypatch.delenv("YC_API_KEY", raising=False)
    monkeypatch.delenv("YC_IAM_TOKEN", raising=False)

    status = yandex.build_status()

    assert status == {
        "name": "yandex_realtime",
        "available": False,
        "healthy": False,
        "reason": "missing_credentials",
    }


def test_build_session_update_uses_yandex_function_tool_shape(monkeypatch) -> None:
    yandex = _load_module()
    monkeypatch.setenv("YC_FOLDER_ID", "folder-1")
    monkeypatch.setenv("YC_VOICE", "ermil")
    monkeypatch.setenv("YANDEX_AI_SEARCH_INDEX_ID", "search-7")

    payload = yandex.build_session_update()

    assert payload["type"] == "session.update"
    assert payload["session"]["output_modalities"] == ["audio"]
    assert payload["session"]["turn_detection"] is None
    assert payload["session"]["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
        "channels": 1,
    }
    assert payload["session"]["audio"]["output"]["format"] == {
        "type": "audio/pcm",
        "rate": 44100,
    }
    assert payload["session"]["tools"] == [
        {
            "type": "function",
            "name": "file_search",
            "description": "search-7",
            "parameters": "{}",
        }
    ]


def test_build_connection_settings_uses_rest_assistant_endpoint(monkeypatch) -> None:
    yandex = _load_module()
    monkeypatch.setenv("YC_FOLDER_ID", "folder-1")
    monkeypatch.setenv("YC_API_KEY", "key-1")
    monkeypatch.setenv("YC_MODEL", "gpt://folder-1/speech-realtime-250923")

    settings = yandex.build_connection_settings()

    assert settings["url"] == (
        "wss://rest-assistant.api.cloud.yandex.net/v1/realtime/openai"
        "?model=gpt%3A%2F%2Ffolder-1%2Fspeech-realtime-250923"
    )
    assert settings["headers"] == {
        "Authorization": "Api-Key key-1",
        "OpenAI-Project": "folder-1",
    }
