"""Section 7 Phase C.5 — KB_LAYOUT env var resolves to legacy or topical KB file.

Default `legacy` keeps current behavior (kb_faq_ru_v2.md).
`topical` resolves to kb_topics_ru.md (the Phase C output).
Invalid values fall back to legacy with a warning.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import settings as settings_module


def _isolate_kb_layout(monkeypatch):
    """Strip KB_LAYOUT from env so each test starts clean."""
    monkeypatch.delenv("KB_LAYOUT", raising=False)


def test_kb_layout_unset_uses_legacy_v2_md(monkeypatch):
    _isolate_kb_layout(monkeypatch)
    loaded = settings_module.load_settings()
    assert loaded.app.kb_markdown_path.name == "kb_faq_ru_v2.md"


def test_kb_layout_legacy_explicit_uses_v2_md(monkeypatch):
    monkeypatch.setenv("KB_LAYOUT", "legacy")
    loaded = settings_module.load_settings()
    assert loaded.app.kb_markdown_path.name == "kb_faq_ru_v2.md"


def test_kb_layout_topical_resolves_to_topics_md(monkeypatch):
    monkeypatch.setenv("KB_LAYOUT", "topical")
    loaded = settings_module.load_settings()
    assert loaded.app.kb_markdown_path.name == "kb_topics_ru.md"
    # Path should be under knowledge_base/ regardless of legacy config value
    assert loaded.app.kb_markdown_path.parent.name == "knowledge_base"


def test_kb_layout_topical_uppercase_normalized(monkeypatch):
    monkeypatch.setenv("KB_LAYOUT", "TOPICAL")
    loaded = settings_module.load_settings()
    assert loaded.app.kb_markdown_path.name == "kb_topics_ru.md"


def test_kb_layout_invalid_falls_back_to_legacy(monkeypatch):
    monkeypatch.setenv("KB_LAYOUT", "garbage_value")
    loaded = settings_module.load_settings()
    # Invalid -> legacy fallback (no crash)
    assert loaded.app.kb_markdown_path.name == "kb_faq_ru_v2.md"


def test_kb_layout_topical_path_resolves_against_repo_root(monkeypatch):
    """Verify the resolved topical path is under the repo's knowledge_base/, not relative."""
    monkeypatch.setenv("KB_LAYOUT", "topical")
    loaded = settings_module.load_settings()
    p = loaded.app.kb_markdown_path
    assert p.is_absolute()
    # Path should end in knowledge_base/kb_topics_ru.md
    assert str(p).endswith("knowledge_base/kb_topics_ru.md")
