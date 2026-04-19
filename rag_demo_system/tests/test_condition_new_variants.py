"""Fix 1.3 — condition_new robustness against phonetic/colloquial б/у variants.

Client complaint: assistant fails to set condition_new=0 when the user says
"бэу", "с пробегом", "не новый", "старый" or types "б-у" with a dash. The
existing regex only catches "б/у", "бу", "подержанный", "бывший".

Two layers are tested:
  1. `has_field_signal("condition_new", 0, utterance)` — used by the extras-
     staging path to decide whether a classifier-derived condition_new hint
     is grounded in the user's words.
  2. `filter_patches({"condition_new": 0}, utterance)` — must keep the patch
     for short single-word answers like "бэу" via the enum slot-fill bypass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_hygiene import filter_patches, has_field_signal


_USED_VARIANTS = [
    "б/у",            # baseline
    "бу",             # baseline
    "бэу",            # phonetic spelling — NEW
    "б-у",            # dash variant — NEW
    "подержанный",    # baseline (подержан\w+)
    "с пробегом",     # NEW — пробег\w*
    "не новый",       # NEW — не\s+нов\w+
    "старый",         # NEW
    "бывший в употреблении",  # baseline (бывш\w+)
]


@pytest.mark.parametrize("utterance", [
    "бэу мотоцикл",
    "возьму бу машину",
    "машина с пробегом",
    "не новый автомобиль",
    "старый грузовик",
    "авто б-у",
    "подержанная машина",
    "бывший в употреблении",
])
def test_has_field_signal_accepts_variant(utterance: str) -> None:
    assert has_field_signal("condition_new", 0, utterance), (
        f"has_field_signal must recognise condition_new cue in: {utterance!r}"
    )


def test_new_still_maps_to_one() -> None:
    # Positive-case sanity: "новый" must still signal condition_new=1.
    assert has_field_signal("condition_new", 1, "новый мотоцикл")


# Fix 1.10 — value-aware grounding. Contradictory or explicitly new-with-
# mileage-negation phrases must not ground condition_new=0.

def test_contradiction_rejects_used_grounding() -> None:
    # Codex adversarial review 2026-04-19: "новая машина без пробега"
    # previously grounded BOTH condition_new=0 and =1 because the "пробег"
    # token fired the used-cue regex while "новая" fired an implied
    # new-cue path. A bad classifier guess could then survive grounding
    # and push the session into the б/у flow.
    assert not has_field_signal("condition_new", 0, "новая машина без пробега")


def test_contradiction_accepts_new_grounding() -> None:
    # Mirror of the above: the same utterance must still ground =1 cleanly
    # because the client explicitly said "новая" and negated mileage.
    assert has_field_signal("condition_new", 1, "новая машина без пробега")


def test_bare_new_does_not_ground_used() -> None:
    assert not has_field_signal("condition_new", 0, "новый мотоцикл")


def test_bare_used_does_not_ground_new() -> None:
    assert not has_field_signal("condition_new", 1, "подержанная машина")
    assert not has_field_signal("condition_new", 1, "бэу мотоцикл")


def test_negated_new_is_used() -> None:
    # "не новый" is semantically used. Existing positive test already
    # covers the =0 direction; this locks in that =1 is rejected.
    assert not has_field_signal("condition_new", 1, "не новый автомобиль")


def test_zero_mileage_phrase_grounds_new() -> None:
    assert has_field_signal("condition_new", 1, "машина с нулевым пробегом")
    assert not has_field_signal("condition_new", 0, "машина с нулевым пробегом")


@pytest.mark.parametrize("utterance", [
    "бэу",
    "старый",
    "бу",
    "новый",
])
def test_slot_fill_single_word_variants_pass(utterance: str) -> None:
    # <2-token utterances must pass the noise filter via _ENUM_SLOT_FILL_WORDS
    # so the classifier's condition_new patch is not dropped.
    patches = {"condition_new": 0 if utterance != "новый" else 1}
    out = filter_patches(patches, utterance)
    assert "condition_new" in out, (
        f"single-word condition answer '{utterance}' must survive filter_patches"
    )
