from scripts.kb_export import render_flat_markdown, render_structured_markdown


def sample_entries():
    return [
        {
            "category": "Условия",
            "subtopic": "НДС",
            "canonical_question": "Как учитывается НДС?",
            "best_answer": "Ответ",
            "keywords": ["НДС"],
            "tags": ["налоги"],
            "references": ["НК РБ"],
        }
    ]


def test_render_flat_contains_question():
    md = render_flat_markdown(sample_entries())
    assert "Вопрос" in md


def test_render_structured_groups():
    md = render_structured_markdown(sample_entries())
    assert "## Условия" in md
    assert "### НДС" in md
