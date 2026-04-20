"""Contract test for the READBACK_PENDING deny-with-correction grounding guard.

Codex adversarial confirmation pass (2026-04-20, E-Codex-2): the readback
deny-with-correction block previously promoted any differing classifier hint
into pending_change without grounding — a plain "нет" turn carrying stale
numeric drift could stage an unspoken correction. The guard reuses the same
value_grounded() helper the CHANGE_PENDING staging path already uses.

Full behavioural flow is exercised live via SIP; this is a source-level
regression fence so the guard can't be silently removed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_APP_PY = ROOT / "backend" / "app.py"


def test_readback_deny_block_uses_value_grounded():
    src = _APP_PY.read_text(encoding="utf-8")
    # Find the READBACK deny-with-correction block.
    idx = src.index("Deny-with-correction detection")
    block = src[idx : idx + 1200]
    # Grounding guard must be present in this block.
    assert "value_grounded" in block, (
        "READBACK deny-with-correction block must call value_grounded "
        "before staging a delta (Codex E-Codex-2 regression)"
    )
    # And the rejection path must log the dropped delta so prod regressions
    # surface in logs.
    assert "READBACK delta rejected" in block, (
        "ungrounded-delta rejection must log so we can see it in prod"
    )


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
