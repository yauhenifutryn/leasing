"""Prompt-content guards.

Active prompt is `config/system_prompt_ru_v2.txt` (see config/app.yaml).
The legacy v1 file is kept around for chat-only paths but the voice path
must not regress on Batch 3 client-feedback rules.
"""
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
PROMPT_V1 = CONFIG_DIR / "system_prompt_ru.txt"
PROMPT_V2 = CONFIG_DIR / "system_prompt_ru_v2.txt"


def _v2_text() -> str:
    return PROMPT_V2.read_text(encoding="utf-8")


def test_system_prompt_forbids_reasoning_output():
    prompt_text = PROMPT_V1.read_text(encoding="utf-8")
    # v1 prompt uses singular imperative "Не выводи рассуждения".
    assert "Не выводи рассуждения" in prompt_text


def test_system_prompt_requires_final_marker() -> None:
    prompt_text = PROMPT_V1.read_text(encoding="utf-8")
    assert "FINAL:" in prompt_text


# ── Bug 24: schedule type framing — output side uses lay phrasing ─────
# Banking jargon "аннуитет"/"линейный" are still accepted on input
# (utterance grounding) but the bot must SPEAK only the lay phrasing.

def test_v2_prompt_does_not_speak_annuity_or_linear() -> None:
    text = _v2_text().lower()
    assert "аннуитет" not in text, (
        "Bug 24: prompt must not instruct the bot to say 'аннуитет' — "
        "use 'равными платежами' instead."
    )
    assert "линейн" not in text, (
        "Bug 24: prompt must not instruct the bot to say 'линейный' — "
        "use 'с уменьшением суммы к концу срока' instead."
    )


def test_v2_prompt_uses_lay_schedule_phrasing() -> None:
    text = _v2_text()
    assert "равными платежами" in text
    assert "с уменьшением суммы к концу срока" in text
