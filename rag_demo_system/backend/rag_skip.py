"""Rule for skipping RAG retrieval on pure name-capture turns.

Avoids the "Вадим -> director Vadim Dedkov from KB" hallucination pattern
by detecting turns where the client just introduces themselves.
"""

from __future__ import annotations


def should_skip_rag(utterance: str, patches: dict, hints: dict) -> bool:
    """Return True if this turn is a pure name-capture and KB retrieval should be skipped."""
    if hints:
        return False
    if not patches or set(patches.keys()) != {"name"}:
        return False
    if "?" in (utterance or ""):
        return False
    tokens = (utterance or "").strip().split()
    if len(tokens) > 5:
        return False
    return True
