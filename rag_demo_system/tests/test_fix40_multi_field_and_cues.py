"""Fix 40 — multi-field changes, prepaid slot, cues, and zero-ghost guard.

Four distinct regressions captured in sessions 242108b9 and 896671f8
(2026-04-18 logs):

40a. Sticky-patch guard dropped all-but-one field when classifier emitted
     change_field for one + values for several others. User says
     "грузовик за 50 тысяч на 7 лет юр.лицо" → only 1 of 4 patched.

40b. has_field_signal for term_months didn't understand "N лет" form —
     term_months=84 failed to match "на 7 лет" utterance.

40c. prepaid_pct shadowed prepaid_amount in direct-call params build, so
     switching from % to amount silently used the old %.

40d. _CLIENT_TYPE_CUE_RE had no "бизнес" entry. Classifier output for
     "Нет, я бизнес." dropped, profile stayed unset, bot re-asked.

40e. Classifier sometimes emits change_value=0 on non-numeric turns; the
     staging code accepted it (0 is not None/""), corrupting profile.term_months.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_hygiene import (  # noqa: E402
    filter_patches,
    has_field_signal,
    utterance_has_client_type_cue,
)


# ── 40b: has_field_signal understands "N лет" ─────────────────────────

def test_term_signal_from_n_years() -> None:
    assert has_field_signal("term_months", 84, "давай на 7 лет") is True


def test_term_signal_from_one_year() -> None:
    assert has_field_signal("term_months", 12, "один год") is True
    assert has_field_signal("term_months", 12, "на год") is True


def test_term_signal_half_year() -> None:
    assert has_field_signal("term_months", 18, "полтора года") is True
    assert has_field_signal("term_months", 6, "полгода") is True


def test_term_signal_from_months() -> None:
    assert has_field_signal("term_months", 36, "36 месяцев") is True


def test_term_signal_rejects_unrelated_utterance() -> None:
    assert has_field_signal("term_months", 84, "поменяй цвет") is False


def test_term_signal_rejects_wrong_year_count() -> None:
    # classifier emits 84 (7 years) but user said "на 3 года" — should not match
    assert has_field_signal("term_months", 84, "на 3 года") is False


# ── 40d: "бизнес" matches client_type cue ─────────────────────────────

def test_biznes_is_client_type_cue() -> None:
    assert utterance_has_client_type_cue("Нет, я бизнес.") is True
    assert utterance_has_client_type_cue("я бизнесмен") is True


def test_biznes_normalized_to_yur_litso() -> None:
    out = filter_patches({"client_type": "Юридическое лицо"}, "Нет, я бизнес.")
    assert out.get("client_type") == "Юридическое лицо"


def test_biznes_from_ip_classifier_still_accepted() -> None:
    # Classifier may emit ИП for "бизнес"; cue now matches, so hygiene accepts.
    out = filter_patches({"client_type": "ИП"}, "Нет, я бизнесмен.")
    # Accepted because cue matches — exact normalization is per classifier.
    assert "client_type" in out


def test_unrelated_utterance_still_drops_client_type() -> None:
    out = filter_patches({"client_type": "Юридическое лицо"}, "хочу машину")
    assert "client_type" not in out


def test_mikrobiznes_single_word_accepted() -> None:
    # Session a685ce41: "Микробизнес." was 1-word + no cue match, dropped.
    # After hotfix: enum-slot-fill whitelist + cue regex without \b for бизнес.
    assert utterance_has_client_type_cue("Микробизнес.") is True
    out = filter_patches({"client_type": "ИП"}, "Микробизнес.")
    assert out.get("client_type") == "ИП"


def test_malyy_biznes_accepted() -> None:
    assert utterance_has_client_type_cue("малый бизнес") is True


# ── 40e: has_field_signal rejects change_value=0 without literal 0 ────

def test_term_signal_rejects_implicit_zero() -> None:
    # Classifier emits change_value=0 on "нет, оставь как есть" — no literal 0.
    assert has_field_signal("term_months", 0, "нет, оставь как есть") is False


def test_term_signal_accepts_explicit_zero_digit() -> None:
    # User literally says "0" — accept (orchestrator OOR will catch later).
    assert has_field_signal("term_months", 0, "срок 0 месяцев") is True


# ── 40a: sticky-patch unlock with has_field_signal (integration) ──────
# (Sticky-patch is inline in app.py; the branch logic is tested via the
# has_field_signal helper above. Full integration happens in live testing.)
