"""Unit tests for kb_cluster's clustering algorithm.

Section 7 Phase A.1 deliverable. Tests run without the embedding model
installed by injecting synthetic similarity matrices.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    """Load kb_cluster.py without importing rag_demo_system.scripts as a package
    (the scripts dir lacks an __init__.py)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    script_path = repo_root / "rag_demo_system" / "scripts" / "kb_cluster.py"
    spec = importlib.util.spec_from_file_location("kb_cluster", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kb_cluster"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


kb_cluster = _load_module()


def test_singletons_when_all_below_threshold():
    sims = [
        [0.0, 0.4, 0.3, 0.2],
        [0.4, 0.0, 0.5, 0.1],
        [0.3, 0.5, 0.0, 0.2],
        [0.2, 0.1, 0.2, 0.0],
    ]
    clusters = kb_cluster.cluster_by_threshold(sims, threshold=0.85)
    # All singletons -> 4 clusters of size 1
    assert len(clusters) == 4
    for c in clusters:
        assert len(c) == 1


def test_pair_clusters_above_threshold():
    sims = [
        [0.0, 0.92, 0.30, 0.20],
        [0.92, 0.0, 0.31, 0.22],
        [0.30, 0.31, 0.0, 0.95],
        [0.20, 0.22, 0.95, 0.0],
    ]
    clusters = kb_cluster.cluster_by_threshold(sims, threshold=0.85)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 2]
    members = {tuple(sorted(c)) for c in clusters}
    assert (0, 1) in members
    assert (2, 3) in members


def test_chain_via_single_linkage():
    # 0--1 at 0.90, 1--2 at 0.88, 0--2 at 0.70. Single-linkage clusters all three.
    sims = [
        [0.0, 0.90, 0.70, 0.10],
        [0.90, 0.0, 0.88, 0.10],
        [0.70, 0.88, 0.0, 0.10],
        [0.10, 0.10, 0.10, 0.0],
    ]
    clusters = kb_cluster.cluster_by_threshold(sims, threshold=0.85)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 3]
    triple = next(c for c in clusters if len(c) == 3)
    assert sorted(triple) == [0, 1, 2]


def test_clusters_sorted_largest_first():
    # 2 entries form a tight cluster; entries 2,3,4,5 form a larger chain.
    sims = [[0.0] * 6 for _ in range(6)]
    # tight pair
    sims[0][1] = sims[1][0] = 0.93
    # 4-chain
    sims[2][3] = sims[3][2] = 0.90
    sims[3][4] = sims[4][3] = 0.91
    sims[4][5] = sims[5][4] = 0.89
    clusters = kb_cluster.cluster_by_threshold(sims, threshold=0.85)
    # First cluster should be the 4-element one
    assert len(clusters[0]) == 4
    assert len(clusters[1]) == 2
    assert sorted(clusters[0]) == [2, 3, 4, 5]
    assert sorted(clusters[1]) == [0, 1]


def test_threshold_boundary_inclusive():
    # exactly equal to threshold should still cluster (>= comparison)
    sims = [
        [0.0, 0.85],
        [0.85, 0.0],
    ]
    clusters = kb_cluster.cluster_by_threshold(sims, threshold=0.85)
    assert len(clusters) == 1 and len(clusters[0]) == 2


def test_entry_text_concatenates_question_and_answer():
    entry = {
        "canonical_question": "Какие условия?",
        "best_answer": "Условия следующие.",
        "category": "Условия",  # should be ignored
    }
    text = kb_cluster.entry_text(entry)
    assert "Какие условия?" in text
    assert "Условия следующие." in text
    assert "category" not in text  # field name should not leak


def test_entry_text_handles_missing_fields():
    assert kb_cluster.entry_text({}) == ""
    assert kb_cluster.entry_text({"canonical_question": "X"}) == "X"
    assert kb_cluster.entry_text({"best_answer": "Y"}) == "Y"


def test_render_report_does_not_crash_on_empty_clusters():
    sims = [[0.0, 0.4], [0.4, 0.0]]
    clusters = kb_cluster.cluster_by_threshold(sims, threshold=0.85)
    entries = [
        {"intent": "intent-a", "canonical_question": "Q1?", "best_answer": "A1"},
        {"intent": "intent-b", "canonical_question": "Q2?", "best_answer": "A2"},
    ]
    out = kb_cluster.render_report(
        clusters,
        sims,
        entries,
        threshold=0.85,
        model_name="test-model",
        yaml_path=kb_cluster.REPO_ROOT / "knowledge_base" / "kb_faq_ru.yaml",
    )
    assert "KB Cluster Report" in out
    assert "Multi-member clusters (≥2): **0**" in out
    assert "(No clusters found at this threshold." in out


def test_render_report_marks_surgical_priority():
    sims = [[0.0] * 3 for _ in range(3)]
    sims[0][1] = sims[1][0] = 0.90
    sims[1][2] = sims[2][1] = 0.91
    sims[0][2] = sims[2][0] = 0.70
    clusters = kb_cluster.cluster_by_threshold(sims, threshold=0.85)
    entries = [
        {"intent": "office-a", "canonical_question": "Где офис?", "best_answer": "В Минске."},
        {"intent": "office-b", "canonical_question": "Адрес офиса?", "best_answer": "Минск."},
        {"intent": "office-c", "canonical_question": "Куда приехать?", "best_answer": "Минск."},
    ]
    out = kb_cluster.render_report(
        clusters,
        sims,
        entries,
        threshold=0.85,
        model_name="test-model",
        yaml_path=kb_cluster.REPO_ROOT / "knowledge_base" / "kb_faq_ru.yaml",
    )
    assert "SURGICAL-PASS PRIORITY" in out
    assert "office-a" in out and "office-b" in out and "office-c" in out
