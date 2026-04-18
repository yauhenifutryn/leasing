"""Contract + unit tests for stale-patch protection on identity fields.

Once client_type / subject / condition_new are captured, the classifier
must not silently overwrite them without an explicit change_field signal.
This prevents mid-session hallucinations from swapping Физическое↔Юридическое.
"""
from pathlib import Path


_APP_PY = Path(__file__).resolve().parents[1] / "backend" / "app.py"


def test_sticky_identity_fields_block_present():
    src = _APP_PY.read_text(encoding="utf-8")
    assert "_STICKY_IDENTITY_FIELDS" in src, "Fix 21 regression: sticky-fields set removed"
    assert '"client_type"' in src and '"subject"' in src and '"condition_new"' in src
    # The guard is gated on is_first_capture OR is_explicit_change
    assert "_is_first_capture or _is_explicit_change" in src, (
        "Fix 21 regression: stale-patch gate logic removed"
    )


def test_stale_client_type_log_message_present():
    src = _APP_PY.read_text(encoding="utf-8")
    assert "stale {_field} patch ignored" in src, (
        "Fix 21 regression: stale-patch warning log removed"
    )


def test_stale_guard_logic_unit():
    """Logic-level unit test of the stale-patch decision."""
    def _should_accept(current_val, new_val, explicit_change_field, field_name):
        if new_val in (None, ""):
            return False
        is_first_capture = current_val in (None, "")
        is_explicit_change = (explicit_change_field == field_name)
        if is_first_capture or is_explicit_change:
            return True
        if new_val != current_val:
            return False  # stale overwrite blocked
        return True  # same value, no-op

    # First capture — always accept
    assert _should_accept(None, "Физическое лицо", None, "client_type") is True
    assert _should_accept("", "Юридическое лицо", None, "client_type") is True

    # Classifier tries to flip without explicit change — block
    assert _should_accept("Физическое лицо", "Юридическое лицо", None, "client_type") is False

    # User explicitly changed it — accept
    assert _should_accept("Физическое лицо", "Юридическое лицо", "client_type", "client_type") is True

    # Change-field is for a different field — classifier drift blocked
    assert _should_accept("Физическое лицо", "Юридическое лицо", "term_months", "client_type") is False

    # Same value re-emitted — no-op accept
    assert _should_accept("Физическое лицо", "Физическое лицо", None, "client_type") is True

    # Empty new value — reject
    assert _should_accept("Физическое лицо", None, None, "client_type") is False
    assert _should_accept("Физическое лицо", "", None, "client_type") is False
