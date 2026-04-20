from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.session import ClientProfile, ProfileState


def test_new_profile_is_collecting():
    p = ClientProfile()
    assert p.state == ProfileState.COLLECTING


def test_apply_pending_change_applies_and_clears():
    p = ClientProfile()
    p.term_months = 36
    p.pending_change = {"field": "term_months", "new_value": 60}
    p.apply_pending_change()
    assert p.term_months == 60
    assert p.pending_change is None


def test_client_type_ip_is_valid():
    p = ClientProfile(client_type="ИП")
    assert p.client_type == "ИП"


# --- Codex adversarial pass 4 (2026-04-20): fail-closed apply_pending_change ---

def test_apply_pending_change_returns_false_on_all_unknown_fields():
    # Payload with only unknown field → method logs warning, returns False,
    # leaves pending_change intact. Prevents state-loss (caller must NOT
    # advance to CONFIRMED when nothing was applied).
    p = ClientProfile(prepaid_pct=30.0)
    p.pending_change = {"changes": {"prepaid": {"old": 30, "new": 20}}}
    applied = p.apply_pending_change()
    assert applied is False
    assert p.prepaid_pct == 30.0  # unchanged
    assert p.pending_change is not None  # preserved for retry


def test_apply_pending_change_returns_true_on_mixed_known_unknown():
    # If at least one known field applied, other unknown fields are dropped
    # with a warning and the method still returns True (partial success).
    p = ClientProfile(term_months=36, prepaid_pct=30.0)
    p.pending_change = {
        "changes": {
            "term_months": {"old": 36, "new": 60},
            "prepaid": {"old": 30, "new": 20},  # unknown alias
        }
    }
    applied = p.apply_pending_change()
    assert applied is True
    assert p.term_months == 60
    assert p.prepaid_pct == 30.0  # unknown alias did NOT mutate it
    assert p.pending_change is None


def test_apply_pending_change_legacy_single_field_unknown_returns_false():
    p = ClientProfile()
    p.pending_change = {"field": "prepaid", "new_value": 20}  # unknown attr
    applied = p.apply_pending_change()
    assert applied is False
    assert p.pending_change is not None
