"""Unit tests for the value_grounded() guard used by apply_turn's
deny-with-correction handling.

Codex adversarial confirmation pass (2026-04-20, E-Codex-2): the readback
deny-with-correction handling previously promoted any differing classifier
hint into pending_change without grounding — a plain "нет" turn carrying
stale numeric drift could stage an unspoken correction. apply_turn's
partition_patches now relies on value_grounded() for the same guarantee.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_value_grounded_rejects_stale_numeric_on_plain_deny():
    # Unit-level repro of the scenario Codex flagged: "нет" as utterance with
    # a stale cost carried from prior turn classifier drift. value_grounded
    # must return False so the readback block filters the delta out.
    from backend.classifier_schema import value_grounded

    assert value_grounded("cost", 120000, "нет") is False
    assert value_grounded("cost", 120000, "да давай 120 тысяч") is True
    # Enum case — "нет, грузовой" DOES ground the truck subject, so a
    # legitimate correction still passes through.
    assert value_grounded("subject", "Грузовой автомобиль", "нет, грузовой") is True
    # But "нет" alone cannot ground subject — even if classifier echoes one.
    assert value_grounded("subject", "Легковой автомобиль", "нет") is False
