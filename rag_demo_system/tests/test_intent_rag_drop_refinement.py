"""When the classifier mis-labels a multi-slot setup as intent=RAG,
the schema must still keep slots that pass verbatim cue grounding —
without that, real calc setups fall to LLM fallback and the user
re-enters the whole chain.

Live regression (2026-05-09 chat regression run): "Юрлицо, недвижимость
в Минске за 250 тысяч долларов, подержанная." Classifier emitted
intent=RAG; pre-fix all slots got nulled; bot fell to FireLLMFallback
and asked generic "уточните срок и аванс" instead of routing to the
deterministic age clarify.

Original poison case (call d5174335 2026-04-27): "адреса офисов" →
classifier extracted subject=Недвижимость from word "офис" → schema
verbatim grounding accepted because subject cue regex once included
"офис". Profile got poisoned with subject=Недвижимость; user's later
BMW request was treated as a CHANGE.

Fix shape: still drop change_field/change_value unconditionally on
intent=RAG. For top-level slots, only drop when verbatim cue grounding
ALSO fails — the office-address case now fails because "офис" was
removed from the subject regex (the historical fix), so the poison is
double-protected by both the cue regex change AND the cue check here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.classifier_schema import parse_classifier_output  # noqa: E402


def test_realestate_setup_misclassified_rag_keeps_grounded_slots():
    """The F2 live regression: classifier emits intent=RAG for a
    clearly-calc setup utterance. Slots that ARE in the utterance must
    survive — the dispatcher needs them to route to age clarify.

    Plus intent must be rescued to TOOL so apply_turn's step 5b clarify
    gate (not _is_rag_turn) fires instead of LLM-fallback."""
    raw = json.dumps({
        "intent": "RAG",
        "subject": "Недвижимость",
        "client_type": "Юридическое лицо",
        "currency": "USD",
        "condition_new": 0,
        "cost": 250000,
    })
    utt = "Юрлицо, недвижимость в Минске за 250 тысяч долларов, подержанная."
    out = parse_classifier_output(raw, utt)
    assert out.subject == "Недвижимость", "subject leaked away"
    assert out.client_type == "Юридическое лицо", "client_type leaked away"
    assert out.currency == "USD", "currency leaked away"
    assert out.condition_new == 0, "condition_new leaked away"
    assert out.cost == 250000.0, "cost leaked away"
    assert out.intent == "TOOL", (
        "intent must be rescued to TOOL so dispatcher routes to clarify, "
        f"got: {out.intent!r}"
    )


def test_office_address_rag_poison_still_blocked():
    """Original poison case: classifier picks Недвижимость from the
    word 'офис' and emits intent=RAG. Subject cue regex no longer
    matches 'офис', so subject must still get nulled."""
    raw = json.dumps({
        "intent": "RAG",
        "subject": "Недвижимость",
    })
    utt = "Подскажите адреса офисов в Минске."
    out = parse_classifier_output(raw, utt)
    assert out.subject is None, "office-address case must still be poisoned"


def test_rag_change_field_value_always_dropped():
    """RAG turn must NEVER stage a parameter change. change_field /
    change_value are unconditionally nulled even on a RAG turn that
    happens to mention something — change-confirm is a deterministic
    flow, not LLM-driven."""
    raw = json.dumps({
        "intent": "RAG",
        "change_field": "currency",
        "change_value": "BYN",
    })
    utt = "А в каких валютах вы вообще считаете?"
    out = parse_classifier_output(raw, utt)
    assert out.change_field is None
    assert out.change_value is None


def test_rag_with_phantom_slots_drops_them():
    """Classifier on a clearly-RAG turn ('какие у вас офисы?') emits
    phantom slots not in the utterance — drop them as the original
    rule did."""
    raw = json.dumps({
        "intent": "RAG",
        "subject": "Легковой автомобиль",
        "currency": "EUR",
        "cost": 50000,
    })
    utt = "Какие у вас офисы в Минске?"
    out = parse_classifier_output(raw, utt)
    # 3 slots populated (multi-slot setup signal); per-field grounding
    # then drops them because none match the utterance.
    assert out.subject is None
    assert out.currency is None
    assert out.cost is None


def test_rag_solo_grounded_slot_still_dropped_meta_question():
    """Solo / pair slot mentions on a RAG turn still get blanket-
    dropped even when the cue regex would match — the meta-question
    case ('что такое аннуитет?' → type_schedule='0' grounds via cue).
    Pin the trade-off so future relaxations don't regress the pre-
    2026-05-09 anti-poison behaviour for legitimately-RAG turns.
    """
    raw = json.dumps({
        "intent": "RAG",
        "type_schedule": "0",
    })
    utt = "Что такое аннуитет?"
    out = parse_classifier_output(raw, utt)
    assert out.type_schedule is None, (
        "solo slot on RAG must blanket-drop even when cue would match"
    )


def test_rag_pair_slots_still_blanket_dropped():
    """Two-slot RAG emission below the 3-slot multi-slot-setup
    threshold — still blanket-drop."""
    raw = json.dumps({
        "intent": "RAG",
        "subject": "Недвижимость",
        "type_schedule": "0",
    })
    utt = "А какой график лучше для недвижимости?"
    out = parse_classifier_output(raw, utt)
    # User is asking ABOUT graph types for real estate, not selecting them.
    assert out.subject is None
    assert out.type_schedule is None
