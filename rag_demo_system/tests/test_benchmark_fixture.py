"""
Validation tests for the Russian benchmark question fixture.

Verifies structural integrity of bench_questions_ru.jsonl:
- File exists and has 80+ questions
- All 5 required categories present
- question_id prefix matches category
- Required fields present on every question
- expected_keywords is a list with correct contents per category
- No duplicate question_ids
- text_ru contains Cyrillic characters
"""
import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "bench_questions_ru.jsonl"
VALID_CATEGORIES = {"short_factual", "long_factual", "kb_grounded", "ambiguous", "out_of_scope"}
CATEGORY_PREFIXES = {
    "short_factual": "sf",
    "long_factual": "lf",
    "kb_grounded": "kb",
    "ambiguous": "amb",
    "out_of_scope": "oos",
}


def _load_fixture():
    lines = FIXTURE_PATH.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_fixture_file_exists():
    assert FIXTURE_PATH.exists(), f"Fixture file not found at {FIXTURE_PATH}"


def test_fixture_has_80_plus_questions():
    questions = _load_fixture()
    assert len(questions) >= 80, f"Expected 80+ questions, got {len(questions)}"


def test_all_categories_present():
    questions = _load_fixture()
    categories = {q["category"] for q in questions}
    assert categories == VALID_CATEGORIES, (
        f"Missing categories: {VALID_CATEGORIES - categories}; "
        f"Extra categories: {categories - VALID_CATEGORIES}"
    )


def test_question_id_prefix_matches_category():
    questions = _load_fixture()
    for q in questions:
        prefix = q["question_id"].split("-")[0]
        expected_prefix = CATEGORY_PREFIXES[q["category"]]
        assert prefix == expected_prefix, (
            f"{q['question_id']} has wrong prefix for category '{q['category']}' "
            f"(expected prefix '{expected_prefix}')"
        )


def test_all_questions_have_required_fields():
    questions = _load_fixture()
    required = {"question_id", "category", "text_ru", "expected_keywords"}
    for q in questions:
        missing = required - q.keys()
        assert not missing, f"{q.get('question_id', '?')} missing fields: {missing}"


def test_expected_keywords_is_list():
    questions = _load_fixture()
    for q in questions:
        assert isinstance(q["expected_keywords"], list), (
            f"{q['question_id']} expected_keywords is not a list "
            f"(got {type(q['expected_keywords']).__name__})"
        )


def test_out_of_scope_has_empty_keywords():
    questions = _load_fixture()
    oos = [q for q in questions if q["category"] == "out_of_scope"]
    for q in oos:
        assert q["expected_keywords"] == [], (
            f"{q['question_id']} is out_of_scope but has non-empty expected_keywords: "
            f"{q['expected_keywords']}"
        )


def test_non_oos_has_keywords():
    questions = _load_fixture()
    non_oos = [q for q in questions if q["category"] != "out_of_scope"]
    for q in non_oos:
        assert len(q["expected_keywords"]) >= 2, (
            f"{q['question_id']} should have at least 2 expected_keywords "
            f"(got {len(q['expected_keywords'])})"
        )


def test_no_duplicate_question_ids():
    questions = _load_fixture()
    ids = [q["question_id"] for q in questions]
    duplicates = [qid for qid in set(ids) if ids.count(qid) > 1]
    assert len(ids) == len(set(ids)), f"Duplicate question_ids found: {duplicates}"


def test_text_ru_is_cyrillic():
    questions = _load_fixture()
    for q in questions:
        has_cyrillic = any("\u0400" <= c <= "\u04FF" for c in q["text_ru"])
        assert has_cyrillic, (
            f"{q['question_id']} text_ru has no Cyrillic characters: {q['text_ru']!r}"
        )


def test_category_values_are_valid():
    questions = _load_fixture()
    for q in questions:
        assert q["category"] in VALID_CATEGORIES, (
            f"{q['question_id']} has invalid category '{q['category']}'"
        )


def test_question_ids_have_numeric_suffix():
    questions = _load_fixture()
    for q in questions:
        parts = q["question_id"].split("-")
        assert len(parts) == 2, f"{q['question_id']} does not follow prefix-NN format"
        assert parts[1].isdigit(), (
            f"{q['question_id']} suffix '{parts[1]}' is not numeric"
        )
