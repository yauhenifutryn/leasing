from pathlib import Path


def test_system_prompt_forbids_reasoning_output():
    prompt_path = Path(__file__).resolve().parents[1] / "config" / "system_prompt_ru.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "Не выводите рассуждения" in prompt_text


def test_system_prompt_requires_final_marker() -> None:
    prompt_path = Path(__file__).resolve().parents[1] / "config" / "system_prompt_ru.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "FINAL:" in prompt_text
