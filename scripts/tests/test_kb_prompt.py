from pathlib import Path


def test_prompt_mentions_new_fields():
    text = Path("prompts/kb_entry_synthesis_ru.md").read_text(encoding="utf-8")
    for key in ["category", "subtopic", "keywords", "tags", "references"]:
        assert f"\"{key}\"" in text
