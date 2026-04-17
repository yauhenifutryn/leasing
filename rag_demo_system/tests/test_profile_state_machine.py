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
