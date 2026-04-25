"""Rule for skipping RAG retrieval on pure name-capture turns.

Avoids the "Вадим -> director Vadim Dedkov from KB" hallucination pattern
by detecting turns where the client just introduces themselves.
"""

from __future__ import annotations


# Profile-data hint keys mirror the classifier slot fields. If any of these
# are present in `hints`, the user gave more than a name and we must NOT
# skip RAG / route to the open-greeting path. Non-profile hints (just
# `action` like "clarify" / "conversation") are tolerated — they're emitted
# by the SessionAgent on every turn and don't indicate calc-intent on
# their own. Bug 5 (live call 6dd5880b 2026-04-25): the previous rule
# `if hints: return False` was too strict and blocked the greeting path
# whenever the classifier returned an action label, dragging the bot into
# the clarify funnel after a bare "Привет, я Никита."
_PROFILE_HINT_KEYS: frozenset[str] = frozenset({
    "subject", "cost", "currency", "client_type", "condition_new",
    "age_years", "prepaid", "prepaid_pct", "prepaid_amount",
    "term", "term_months", "type_schedule",
})


def should_skip_rag(utterance: str, patches: dict, hints: dict) -> bool:
    """Return True if this turn is a pure name-capture and KB retrieval should be skipped."""
    if hints and any(k in _PROFILE_HINT_KEYS for k in hints):
        return False
    if not patches or set(patches.keys()) != {"name"}:
        return False
    if "?" in (utterance or ""):
        return False
    tokens = (utterance or "").strip().split()
    if len(tokens) > 5:
        return False
    return True
