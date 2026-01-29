import builtins
import importlib
from pathlib import Path


def test_utils_falls_back_when_orjson_missing(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "orjson":
            raise ImportError("no orjson")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import scripts.utils as utils

    importlib.reload(utils)

    payload = {"a": 1, "b": [2, 3]}
    out_path = tmp_path / "sample.json"
    utils.write_json(out_path, payload)
    loaded = utils.read_json(out_path)
    assert loaded == payload
