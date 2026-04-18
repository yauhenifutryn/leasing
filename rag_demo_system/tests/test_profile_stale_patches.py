"""Contract + unit tests for stale-patch protection on identity + numeric fields.

Once client_type / subject / condition_new (Fix 21) or cost / term_months /
prepaid_pct / prepaid_amount / type_schedule / currency / age_years (Fix 24)
are captured, the classifier must not silently overwrite them without an
explicit change_field signal. This prevents mid-session hallucinations:

- Fix 21 target: swapping Физическое↔Юридическое from "я физик" noise.
- Fix 24 target: a "А можно ли физическое лицо?" turn that should only
  request a client_type change must NOT also regress cost/term/prepaid to
  their earlier-calc values that the classifier still sees in its context.
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


def test_sticky_numeric_fields_block_present():
    """Fix 24: numeric fields must also live under sticky-patch protection."""
    src = _APP_PY.read_text(encoding="utf-8")
    assert "_STICKY_NUMERIC_FIELDS" in src, (
        "Fix 24 regression: numeric sticky-fields set removed"
    )
    for _f in ("cost", "term_months", "prepaid_pct", "prepaid_amount",
               "type_schedule", "currency", "age_years"):
        assert f'"{_f}"' in src, (
            f"Fix 24 regression: numeric field {_f!r} missing from sticky protection"
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


# --- Fix 24: numeric-field sticky-patch protection -------------------------
#
# We mirror the in-app loop in a tiny helper so we can exercise the Phase A
# -> Phase B -> attempted-regression scenario deterministically without
# spinning up the whole WebSocket pipeline.


_IDENTITY = ("client_type", "subject", "condition_new")
_NUMERIC = ("cost", "age_years", "prepaid_pct", "prepaid_amount",
            "term_months", "type_schedule", "currency")


def _apply_sticky(profile: dict, sa_parsed: dict) -> dict:
    """Simulates the post-Fix-24 loop in backend/app.py.

    Returns the dict of fields that would be patched onto the profile.
    """
    patches: dict = {}
    change_field_val = sa_parsed.get("change_field")
    for field in _IDENTITY + _NUMERIC:
        new_val = sa_parsed.get(field)
        if new_val is None or new_val == "":
            continue
        current_val = profile.get(field)
        is_first = current_val in (None, "")
        is_explicit = (change_field_val == field)
        if not is_explicit and field in ("prepaid_pct", "prepaid_amount"):
            if change_field_val in ("prepaid_pct", "prepaid_amount", "prepaid"):
                is_explicit = True
        if is_first or is_explicit:
            patches[field] = new_val
        # stale overwrite silently dropped
    return patches


def test_fix24_numeric_regression_blocked_on_client_type_change():
    """Bug e4eb325c: "А можно ли физическое лицо?" must not regress numerics.

    Profile at Phase B: cost=150000, subject=Грузовой, client_type=Юридическое,
    term=36, prepaid_pct=10. Classifier sees the short question, emits
    change_field=client_type with change_value=Физическое лицо AND re-emits
    stale Phase-A numerics (cost=80000, subject=Легковой, prepaid_pct=20).
    Only client_type should be updated; everything else must stay on Phase B.
    """
    profile = {
        "cost": 150000,
        "subject": "Грузовой автомобиль",
        "client_type": "Юридическое лицо",
        "condition_new": 1,
        "term_months": 36,
        "prepaid_pct": 10,
        "type_schedule": "0",
        "currency": "BYN",
    }
    sa_parsed = {
        "change_field": "client_type",
        "change_value": "Физическое лицо",
        # Classifier hallucinates Phase-A numerics from context window
        "client_type": "Физическое лицо",
        "subject": "Легковой автомобиль",
        "cost": 80000,
        "term_months": 36,  # same as current, coincidence
        "prepaid_pct": 20,
        "type_schedule": "0",
        "currency": "BYN",
    }
    patches = _apply_sticky(profile, sa_parsed)
    # Only the explicitly changed client_type patch must pass.
    assert patches == {"client_type": "Физическое лицо"}, (
        f"Fix 24 regression: expected only client_type patch, got {patches}"
    )


def test_fix24_first_capture_numerics_still_flow():
    """Fix 19 must still work: numeric first-capture patches always pass."""
    profile = {}  # fresh session
    sa_parsed = {
        "cost": 80000,
        "term_months": 36,
        "prepaid_pct": 20,
        "type_schedule": "0",
        "currency": "BYN",
        "subject": "Легковой автомобиль",
        "client_type": "Физическое лицо",
        "condition_new": 1,
    }
    patches = _apply_sticky(profile, sa_parsed)
    for k in ("cost", "term_months", "prepaid_pct", "type_schedule",
              "currency", "subject", "client_type", "condition_new"):
        assert patches.get(k) == sa_parsed[k], (
            f"Fix 19 regression: first-capture {k} dropped"
        )


def test_fix24_explicit_numeric_change_passes():
    """User says 'поменяй срок на 48' -> change_field=term_months passes."""
    profile = {"term_months": 36, "cost": 80000, "prepaid_pct": 20}
    sa_parsed = {
        "change_field": "term_months",
        "change_value": 48,
        "term_months": 48,
    }
    patches = _apply_sticky(profile, sa_parsed)
    assert patches == {"term_months": 48}


def test_fix24_same_numeric_reemit_is_noop():
    """Classifier re-emits exact current values — no patch, no error."""
    profile = {"cost": 150000, "term_months": 36, "prepaid_pct": 10}
    sa_parsed = {"cost": 150000, "term_months": 36, "prepaid_pct": 10}
    patches = _apply_sticky(profile, sa_parsed)
    # All equal current -> no explicit change, all stale drops. Empty patches.
    assert patches == {}


def test_fix24_prepaid_slot_unified():
    """change_field=prepaid_amount should also unlock prepaid_pct on same turn."""
    profile = {"prepaid_pct": 20, "prepaid_amount": 16000}
    sa_parsed = {
        "change_field": "prepaid_amount",
        "change_value": 14000,
        "prepaid_amount": 14000,
        "prepaid_pct": 17.5,  # classifier recomputed
    }
    patches = _apply_sticky(profile, sa_parsed)
    assert patches.get("prepaid_amount") == 14000
    assert patches.get("prepaid_pct") == 17.5


def test_fix24_identity_still_protected_under_numeric_change_field():
    """Fix 21 guarantee: change_field=term_months must NOT unlock subject."""
    profile = {
        "subject": "Грузовой автомобиль",
        "client_type": "Юридическое лицо",
        "term_months": 36,
    }
    sa_parsed = {
        "change_field": "term_months",
        "change_value": 48,
        "term_months": 48,
        "subject": "Легковой автомобиль",  # stale hallucination
        "client_type": "Физическое лицо",  # stale hallucination
    }
    patches = _apply_sticky(profile, sa_parsed)
    assert patches == {"term_months": 48}


def test_fix24_currency_also_sticky():
    """Currency: classifier must not silently flip USD->BYN mid-session."""
    profile = {"currency": "USD", "cost": 80000}
    sa_parsed = {
        # user question about something else; classifier hallucinates default
        "currency": "BYN",
    }
    patches = _apply_sticky(profile, sa_parsed)
    assert "currency" not in patches, (
        "Fix 24 regression: currency flipped without explicit change_field"
    )
