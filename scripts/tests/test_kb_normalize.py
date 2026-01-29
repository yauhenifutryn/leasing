from scripts.kb_schema import normalize_entry


def test_normalize_entry_fills_defaults():
    entry = {
        "intent": "лизинг ИП",
        "canonical_question": "условия?",
        "best_answer": "ответ",
    }
    out = normalize_entry(entry)
    assert out["category"]
    assert out["subtopic"]
    assert isinstance(out["keywords"], list)
    assert isinstance(out["tags"], list)
    assert isinstance(out["references"], list)


def test_normalize_entry_trims_lists():
    entry = {
        "category": "  Финансы  ",
        "subtopic": "  НДС ",
        "keywords": ["  НДС ", "  ставка"],
        "tags": [" tax ", "  "],
        "references": ["  НК РБ  "],
    }
    out = normalize_entry(entry)
    assert out["category"] == "Финансы"
    assert out["subtopic"] == "НДС"
    assert out["keywords"] == ["НДС", "ставка"]
    assert out["tags"] == ["tax"]
    assert out["references"] == ["НК РБ"]
