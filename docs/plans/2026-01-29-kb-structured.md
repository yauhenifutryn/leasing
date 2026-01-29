# KB Structured Output and Metadata Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance `make kb` output for better RAG retrieval by adding structured metadata fields to KB JSON and producing a second, structured Markdown file while preserving the current flat Markdown output.

**Architecture:** Add optional metadata fields (`category`, `subtopic`, `keywords`, `tags`, `references`) in the synthesis prompt and normalize them in `50_build_kb.py`. Update `55_export_kb_markdown.py` to render both the existing flat Markdown and a new structured Markdown grouped by category and subtopic. Keep defaults backward compatible and avoid breaking existing consumers.

**Tech Stack:** Python 3.x, existing `scripts/*.py`, pytest for tests.

---

### Task 1: Add schema normalization for new metadata fields

**Files:**
- Create: `scripts/__init__.py`
- Modify: `scripts/50_build_kb.py`
- Create: `scripts/tests/test_kb_normalize.py`

**Step 1: Write failing tests**

```python
# scripts/tests/test_kb_normalize.py
from scripts.kb_schema import normalize_entry

def test_normalize_entry_fills_defaults():
    entry = {"intent": "лизинг ИП", "canonical_question": "условия?", "best_answer": "ответ"}
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest scripts/tests/test_kb_normalize.py -q`  
Expected: FAIL, missing `scripts.kb_schema.normalize_entry`.

**Step 3: Implement schema normalization**

Create `scripts/kb_schema.py` with `normalize_entry` and helper functions.  
Update `50_build_kb.py` to call `normalize_entry(kb_entry)` before appending.

**Step 4: Run tests to verify they pass**

Run: `pytest scripts/tests/test_kb_normalize.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/__init__.py scripts/kb_schema.py scripts/50_build_kb.py scripts/tests/test_kb_normalize.py
git commit -m "feat: normalize KB entries with structured metadata"
```

---

### Task 2: Update synthesis prompt for new fields

**Files:**
- Modify: `prompts/kb_entry_synthesis_ru.md`
- Create: `scripts/tests/test_kb_prompt.py`

**Step 1: Write failing test**

```python
# scripts/tests/test_kb_prompt.py
from pathlib import Path

def test_prompt_mentions_new_fields():
    text = Path("prompts/kb_entry_synthesis_ru.md").read_text(encoding="utf-8")
    for key in ["category", "subtopic", "keywords", "tags", "references"]:
        assert f"\\\"{key}\\\"" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest scripts/tests/test_kb_prompt.py -q`  
Expected: FAIL

**Step 3: Update prompt**

Add the new fields with short guidance and examples.

**Step 4: Run test to verify it passes**

Run: `pytest scripts/tests/test_kb_prompt.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add prompts/kb_entry_synthesis_ru.md scripts/tests/test_kb_prompt.py
git commit -m "docs: extend KB synthesis prompt with metadata fields"
```

---

### Task 3: Export structured Markdown alongside flat Markdown

**Files:**
- Modify: `scripts/55_export_kb_markdown.py`
- Create: `scripts/tests/test_kb_export.py`

**Step 1: Write failing tests**

```python
# scripts/tests/test_kb_export.py
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest scripts/tests/test_kb_export.py -q`  
Expected: FAIL, missing `scripts.kb_export`.

**Step 3: Implement exporter helpers**

Create `scripts/kb_export.py` with `render_flat_markdown` and `render_structured_markdown`.  
Update `55_export_kb_markdown.py` to use these helpers and add `--out-structured` with default `knowledge_base/kb_faq_ru_structured.md`.

**Step 4: Run tests to verify they pass**

Run: `pytest scripts/tests/test_kb_export.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/kb_export.py scripts/55_export_kb_markdown.py scripts/tests/test_kb_export.py
git commit -m "feat: add structured KB markdown export"
```

---

### Task 4: Update Makefile and run instructions

**Files:**
- Modify: `Makefile`
- Modify: `README.md`

**Step 1: Update Makefile**

Keep `kb-markdown` target unchanged but note that it now produces two Markdown files.

**Step 2: Update README**

Add a note: `make kb-markdown` generates `kb_faq_ru.md` and `kb_faq_ru_structured.md`.

**Step 3: Commit**

```bash
git add Makefile README.md
git commit -m "docs: document dual KB markdown outputs"
```

---

### Task 5: Verification

Run:
```bash
pytest scripts/tests -q
```

Manual check:
```bash
make kb
make kb-markdown
ls knowledge_base/kb_faq_ru.md knowledge_base/kb_faq_ru_structured.md
```

---

**Notes**
- Keep existing flat Markdown unchanged for compatibility.
- Structured Markdown is additive and not used by current RAG unless configured later.
