from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_avoids_hardcoded_localhost_api_base() -> None:
    app_js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    assert "127.0.0.1:8000" not in app_js
    assert "window.location.origin" in app_js


def test_backend_mounts_frontend_static_files() -> None:
    app_py = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")

    assert "StaticFiles" in app_py
    assert "app.mount(\"/\", StaticFiles(directory=FRONTEND_DIR, html=True), name=\"frontend\")" in app_py
