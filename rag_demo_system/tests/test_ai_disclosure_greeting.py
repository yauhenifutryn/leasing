"""Bug 18 — AI disclosure on call open.

Live client review (2026-04-29 evening): the bot must disclose that it
is an AI assistant on every call so the caller knows they may need to
double-check anything important with a human specialist.

The intro_text constant lives inline in app.py's Jambonz handler (after
the DTMF consent gate succeeds). This test asserts the canonical AI
disclosure markers are present in that file so a future cleanup can't
silently drop them.
"""
from pathlib import Path


APP_PY = Path(__file__).resolve().parents[1] / "backend" / "app.py"


def _app_text() -> str:
    return APP_PY.read_text(encoding="utf-8")


def test_intro_discloses_ai_basis() -> None:
    text = _app_text().lower()
    # Either "на основе искусственного интеллекта" or the abbreviated
    # "на основе ИИ" / "ии-ассистент" satisfies the disclosure.
    has_full = "искусственного интеллекта" in text
    has_short = "на основе ии" in text
    assert has_full or has_short, (
        "Bug 18: intro must disclose AI basis (искусственного интеллекта "
        "or на основе ИИ) so caller knows they're talking to an AI."
    )


def test_intro_warns_about_possible_errors() -> None:
    text = _app_text().lower()
    # Disclosure includes a "I can be wrong, double-check with a human"
    # caveat so callers aren't blindsided when the bot misses something.
    assert "могу ошибаться" in text, (
        "Bug 18: intro must include 'я могу ошибаться' so caller knows "
        "to verify critical info with a human specialist."
    )
    assert "уточняйте у специалиста" in text or "уточните у специалиста" in text
