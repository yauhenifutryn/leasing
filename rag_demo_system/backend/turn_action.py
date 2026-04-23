"""TurnAction ADT — stub for Phase 3.A.

Only ProfileSnapshot is defined here for Phase 3.A's build_snapshot
consumer. The full 7-variant ADT lands in Phase 3.B (plan Task 6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProfileSnapshot:
    """Read-only projection of ClientProfile used by action payloads
    as anti-hallucination anchor in LLM prompts (E7) and as input to
    deterministic renderers (E8).
    """
    client_type: Optional[str]
    subject: Optional[str]
    cost: Optional[float]
    currency: Optional[str]
    original_cost: Optional[float]
    original_currency: Optional[str]
    condition_new: Optional[int]
    age_years: Optional[int]
    prepaid_pct: Optional[float]
    prepaid_amount: Optional[float]
    term_months: Optional[int]
    type_schedule: Optional[str]
