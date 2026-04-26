"""Contract tests for the fast-path confirmation skip.

On READBACK_PENDING / CHANGE_PENDING states, short confirmation utterances
must skip the classifier call (saves ~900ms per turn). This test verifies
the source-level contract — full integration flow is tested live.
"""
from pathlib import Path


_APP_PY = Path(__file__).resolve().parents[1] / "backend" / "app.py"


def test_fast_confirm_block_present():
    src = _APP_PY.read_text(encoding="utf-8")
    # Block identifiers that must exist (failing any of these = regression).
    assert "_fast_confirm = False" in src, "Fix 20 regression: fast-path init removed"
    assert "_CONFIRM_WORDS" in src, "Fix 20 regression: confirm-word set removed"
    assert "ProfileState.READBACK_PENDING" in src, "state gate missing"
    assert "ProfileState.CHANGE_PENDING" in src, "state gate missing"


def test_fast_confirm_synthesises_classifier_output():
    src = _APP_PY.read_text(encoding="utf-8")
    # In the _fast_confirm block apply_turn is the sole consumer, so the
    # block must synthesise a ClassifierOutput with is_confirmation=True
    # and `_skip` must include `_fast_confirm` so the classifier call is
    # bypassed entirely.
    idx = src.index("if _fast_confirm:")
    block = src[idx : idx + 600]
    assert "ClassifierOutput.model_validate" in block, (
        "fast-path must synthesise a ClassifierOutput for apply_turn"
    )
    assert '"is_confirmation": True' in block, (
        "fast-path synthesised output must carry is_confirmation=True"
    )

    # The skip flag must include _fast_confirm.
    skip_idx = src.index("_skip = (", idx)
    skip_block = src[skip_idx : skip_idx + 300]
    assert "_fast_confirm" in skip_block, "_skip must include _fast_confirm"


def test_fast_confirm_has_token_length_guard():
    # We must not match "да, я хочу пересчитать с новыми параметрами" as a pure
    # confirmation — word-count guard protects against this.
    src = _APP_PY.read_text(encoding="utf-8")
    idx = src.index("_fast_confirm = False")
    block = src[idx : idx + 1000]
    assert "message.split()" in block, "fast-path must guard on token count"
    assert "<= 3" in block, "fast-path word-count cap should be <=3"
