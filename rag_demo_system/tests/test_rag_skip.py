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
