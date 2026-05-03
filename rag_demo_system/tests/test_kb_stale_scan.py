"""Unit tests for kb_stale_scan's pattern matching and drift detection.

Section 7 Phase A.2 deliverable. Synthetic entries — no KB or model required.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent.parent
    script_path = repo_root / "rag_demo_system" / "scripts" / "kb_stale_scan.py"
    spec = importlib.util.spec_from_file_location("kb_stale_scan", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kb_stale_scan"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


kss = _load_module()


def _hit_values(hits, pattern_name):
    return sorted(h.value for h in hits if h.pattern_name == pattern_name)


def test_advance_percent_pattern_matches_both_phrasings():
    entry_a = {
        "intent": "ent-a",
        "category": "Условия",
        "best_answer": "Авансовый платёж 30% от стоимости.",
    }
    entry_b = {
        "intent": "ent-b",
        "category": "Условия",
        "best_answer": "Минимальный аванс — 10%, оформление быстрое.",
    }
    hits_a = kss.scan_entry(entry_a, 0, kss.PATTERNS)
    hits_b = kss.scan_entry(entry_b, 1, kss.PATTERNS)
    assert "30" in _hit_values(hits_a, "advance_percent")
    assert "10" in _hit_values(hits_b, "advance_percent")


def test_term_months_pattern():
    entry = {
        "intent": "ent",
        "category": "Условия",
        "best_answer": "Срок лизинга до 60 месяцев, возможен 36 месяцев.",
    }
    hits = kss.scan_entry(entry, 0, kss.PATTERNS)
    months = _hit_values(hits, "term_months")
    assert "60" in months
    assert "36" in months


def test_age_limit_patterns():
    entry = {
        "intent": "ent",
        "category": "Условия",
        "best_answer": "Возраст лизингополучателя от 21 до 75 лет.",
    }
    hits = kss.scan_entry(entry, 0, kss.PATTERNS)
    assert "21" in _hit_values(hits, "age_lower")
    assert "75" in _hit_values(hits, "age_upper")


def test_restrictive_only_pattern():
    entry = {
        "intent": "ent",
        "category": "Условия",
        "best_answer": "Финансируем только 5 марок техники, не более 3 единиц на договор.",
    }
    hits = kss.scan_entry(entry, 0, kss.PATTERNS)
    restrictive = _hit_values(hits, "restrictive_only")
    assert "5" in restrictive
    assert "3" in restrictive


def test_pdn_term_match():
    entry = {
        "intent": "ent",
        "category": "Платежи",
        "best_answer": "Учитывается ПДН и общая нагрузка на бюджет клиента.",
    }
    hits = kss.scan_entry(entry, 0, kss.PATTERNS)
    pdn_hits = [h for h in hits if h.pattern_name == "pdn"]
    assert len(pdn_hits) >= 2  # ПДН and нагрузка both should match


def test_scans_list_fields_too():
    entry = {
        "intent": "ent",
        "category": "Условия",
        "best_answer": "Лизинг физлицам.",
        "eligibility_rules": [
            "возраст от 21 до 70 лет",
            "минимальный аванс 0%",
        ],
    }
    hits = kss.scan_entry(entry, 0, kss.PATTERNS)
    # 70 in eligibility, 21 in eligibility, 0 in eligibility
    assert "70" in _hit_values(hits, "age_upper")
    assert "21" in _hit_values(hits, "age_lower")
    advance = _hit_values(hits, "advance_percent")
    assert "0" in advance
    # Each hit should record which field it came from
    elig_hits = [h for h in hits if h.field == "eligibility_rules"]
    assert len(elig_hits) >= 3


def test_drift_detected_when_same_category_multiple_values():
    entries = [
        {
            "intent": "physlico",
            "category": "Условия",
            "best_answer": "Минимальный аванс 0% для физлиц.",
        },
        {
            "intent": "ip",
            "category": "Условия",
            "best_answer": "Минимальный аванс 10% для ИП.",
        },
        {
            "intent": "old_doc",
            "category": "Условия",
            "best_answer": "Аванс — 20% (старый документ).",
        },
    ]
    all_hits = []
    for i, e in enumerate(entries):
        all_hits.extend(kss.scan_entry(e, i, kss.PATTERNS))
    # All three are in the same category — drift candidate
    advance_values = {h.value for h in all_hits if h.pattern_name == "advance_percent" and h.category == "Условия"}
    assert advance_values == {"0", "10", "20"}


def test_no_drift_when_same_value_across_entries():
    entries = [
        {"intent": "a", "category": "Условия", "best_answer": "Аванс 30%."},
        {"intent": "b", "category": "Условия", "best_answer": "Минимальный аванс 30% от стоимости."},
    ]
    all_hits = []
    for i, e in enumerate(entries):
        all_hits.extend(kss.scan_entry(e, i, kss.PATTERNS))
    advance_values = {h.value for h in all_hits if h.pattern_name == "advance_percent"}
    assert advance_values == {"30"}


def test_snippet_includes_context():
    entry = {
        "intent": "ent",
        "category": "Условия",
        "best_answer": "Подробное описание условий лизинга. Минимальный аванс 30% от стоимости. Иные правила.",
    }
    hits = kss.scan_entry(entry, 0, kss.PATTERNS)
    advance_hits = [h for h in hits if h.pattern_name == "advance_percent"]
    assert advance_hits
    assert "аванс" in advance_hits[0].snippet.lower()


def test_render_report_does_not_crash_on_empty_hits():
    entries = [{"intent": "x", "category": "Y", "best_answer": "no numbers here"}]
    all_hits = []
    for i, e in enumerate(entries):
        all_hits.extend(kss.scan_entry(e, i, kss.PATTERNS))
    out = kss.render_report(all_hits, kss.PATTERNS, entries, kss.REPO_ROOT / "knowledge_base" / "kb_faq_ru.yaml")
    assert "KB Stale-Number Scan" in out


def test_render_report_flags_drift_section():
    entries = [
        {"intent": "a", "category": "Условия", "best_answer": "Аванс 0%."},
        {"intent": "b", "category": "Условия", "best_answer": "Аванс 10%."},
    ]
    all_hits = []
    for i, e in enumerate(entries):
        all_hits.extend(kss.scan_entry(e, i, kss.PATTERNS))
    out = kss.render_report(all_hits, kss.PATTERNS, entries, kss.REPO_ROOT / "knowledge_base" / "kb_faq_ru.yaml")
    assert "Drift candidates" in out
    assert "advance_percent" in out
    assert "Условия" in out
