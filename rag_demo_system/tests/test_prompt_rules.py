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


# ── Bug 13: gendered persona — neutral self-claims, feminine grammar ──
# Source: live call 15:34:53 — bot replied "у меня есть пол, я женщина".
# Right behavior: stay neutral about personhood claims, but keep feminine
# grammatical agreement on first-person verbs/adjectives.

def test_v2_prompt_blocks_gendered_self_claims() -> None:
    text = _v2_text().lower()
    # No personhood assertion that the bot IS a woman.
    assert "вы — женщина" not in text
    assert "вы женщина" not in text
    # Old "if asked your gender, you ARE a woman" rule must be gone.
    assert "если клиент спрашивает ваш пол, вы женщина" not in text


def test_v2_prompt_keeps_feminine_grammar_forms() -> None:
    text = _v2_text()
    # The feminine-form instructions stay — they're about grammar, not
    # personhood. рада / готова / смогла / могла are listed in the prompt
    # as the required first-person forms.
    for form in ("рада", "готова", "смогла", "могла"):
        assert form in text, f"Bug 13: feminine grammar form '{form}' must stay"


# ── Bug 16: Belarus-only scope ─────────────────────────────────────────
# Live call ccf0139a 14:37:12: caller asked "Что по лизингу в России?"
# and the bot answered with generic Russian-context info. Active prompt
# must restrict the consultation scope to Belarus and gracefully
# redirect questions about other countries.

def test_v2_prompt_restricts_scope_to_belarus() -> None:
    text = _v2_text().lower()
    assert "только по лизингу в беларуси" in text, (
        "Bug 16: prompt must explicitly scope consultation to Belarus."
    )


def test_v2_prompt_handles_other_country_questions() -> None:
    text = _v2_text().lower()
    # The prompt must mention at least one non-Belarus country in the
    # redirect rule so the bot has an explicit pattern for the case.
    assert "россии" in text or "россия" in text or "казахстан" in text
