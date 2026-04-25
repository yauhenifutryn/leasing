from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.rag_skip import should_skip_rag


def test_pure_name_capture_is_skipped():
    assert should_skip_rag("Привет, я Вадим.", {"name": "Вадим"}, {}) is True


def test_name_plus_question_is_not_skipped():
    assert should_skip_rag("Привет, я Вадим, какие офисы в Минске?",
                           {"name": "Вадим"}, {"action": "clarify"}) is False


def test_name_plus_tool_intent_is_not_skipped():
    assert should_skip_rag("Я Вадим и хочу машину за 100 тысяч.",
                           {"name": "Вадим", "subject": "Легковой автомобиль", "cost": 100000},
                           {"action": "calculate"}) is False


def test_question_mark_blocks_skip():
    assert should_skip_rag("Я Вадим. Адрес в Минске?",
                           {"name": "Вадим"}, {}) is False


def test_no_name_patch_no_skip():
    assert should_skip_rag("Здравствуйте.", {}, {}) is False


def test_long_utterance_no_skip():
    assert should_skip_rag("Привет, я Вадим, очень рад познакомиться с вами",
                           {"name": "Вадим"}, {}) is False


def test_call_site_guard_skips_greeting_when_name_already_set():
    # Documentation-test for the call-site guard in backend/app.py:
    # once profile.name is set, the guard bails BEFORE calling should_skip_rag,
    # so a classifier-hallucinated {name: X} patch can no longer trigger the
    # greeting loop on a genuine question.
    class _Profile:
        name = "Женя"
    profile = _Profile()
    patches = {"name": "Минск"}  # classifier hallucination
    hints: dict = {}
    name_already = bool((profile.name or "").strip())
    # Guard expression mirrors app.py:
    skip = (not name_already) and should_skip_rag(
        "Ладно, подскажи адрес в Минске.", patches, hints
    )
    assert skip is False  # guard prevents the greeting loop


def test_call_site_guard_allows_skip_on_first_name_turn():
    # Counterpart: when profile.name is empty, the guard does NOT block, and
    # should_skip_rag still detects pure name capture -> skip=True.
    class _Profile:
        name = ""
    profile = _Profile()
    patches = {"name": "Женя"}
    hints: dict = {}
    name_already = bool((profile.name or "").strip())
    skip = (not name_already) and should_skip_rag(
        "Меня зовут Женя.", patches, hints
    )
    assert skip is True


# Bug 5 (live call 6dd5880b 2026-04-25) — funnel-aggression after name-only.
# Classifier emits a non-empty `action` (e.g. "clarify" or "conversation")
# alongside name=Никита on a bare "Привет, я Никита." Old guard blocked
# skip-RAG on ANY non-empty hints dict, so the bot dove into clarify funnel
# instead of emitting the open greeting.
def test_pure_name_capture_with_action_clarify_still_skips():
    """Classifier action='clarify' alone (no profile slots) shouldn't block."""
    assert should_skip_rag(
        "Привет, я Никита.",
        {"name": "Никита"},
        {"action": "clarify"},
    ) is True


def test_pure_name_capture_with_action_conversation_still_skips():
    assert should_skip_rag(
        "Здравствуйте, я Сергей.",
        {"name": "Сергей"},
        {"action": "conversation"},
    ) is True


def test_name_plus_subject_hint_blocks_skip():
    """If classifier extracted profile data (subject), user wants calc, not greeting."""
    assert should_skip_rag(
        "Я Вадим, хочу легковой.",
        {"name": "Вадим"},
        {"action": "calculate", "subject": "Легковой автомобиль"},
    ) is False


def test_name_plus_cost_hint_blocks_skip():
    assert should_skip_rag(
        "Я Вадим, сто тысяч.",
        {"name": "Вадим"},
        {"action": "calculate", "cost": 100000},
    ) is False


def test_action_only_hint_with_no_name_no_skip():
    """No name patch -> not a name-capture turn even if hints are non-profile."""
    assert should_skip_rag(
        "Здравствуйте.",
        {},
        {"action": "conversation"},
    ) is False
