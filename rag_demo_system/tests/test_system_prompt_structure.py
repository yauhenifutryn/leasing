from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_prompt() -> str:
    return (ROOT / "config" / "system_prompt_ru.txt").read_text(encoding="utf-8")


def test_has_required_sections() -> None:
    prompt = _load_prompt()
    required = ["# Role", "# Personality", "# Goal", "# Guardrails", "# Instructions", "# Conversation Flow"]
    for section in required:
        assert section in prompt, f"Missing section: {section}"


def test_no_consent_section() -> None:
    prompt = _load_prompt()
    assert "согласие на обработку" not in prompt.lower()
    assert "consent" not in prompt.lower()


def test_name_frequency_rule() -> None:
    prompt = _load_prompt()
    assert "имени" in prompt.lower() or "имя" in prompt.lower() or "имен" in prompt.lower()
    assert any(x in prompt for x in ["1 раз", "не чаще", "редко", "не начинай каждый", "не начинайте каждый"])


def test_anti_specialist_rule() -> None:
    prompt = _load_prompt()
    assert "специалист" in prompt.lower()
    assert any(x in prompt.lower() for x in ["не предлагай", "не предлагайте", "только когда", "только если"])


def test_humor_allowed() -> None:
    prompt = _load_prompt()
    assert any(x in prompt.lower() for x in ["юмор", "шутк", "шутлив"])


def test_prompt_under_2000_tokens() -> None:
    prompt = _load_prompt()
    word_count = len(prompt.split())
    estimated_tokens = int(word_count * 1.5)
    assert estimated_tokens < 2000, f"Prompt too long: ~{estimated_tokens} tokens (target <2000)"


def test_example_utterances_present() -> None:
    prompt = _load_prompt()
    assert prompt.count('- "') >= 3 or prompt.count("- \"") >= 3, "Need at least 3 example utterances"
