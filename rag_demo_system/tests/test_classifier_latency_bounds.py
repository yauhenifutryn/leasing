"""Contract tests for classifier latency-reduction bounds.

If someone refactors the classifier block and bumps these values back up,
latency regresses ~1-2 seconds. This test catches that.
"""
import inspect
from pathlib import Path


_APP_PY = Path(__file__).resolve().parents[1] / "backend" / "app.py"


def test_classifier_recent_turns_bound():
    """Fix-18: classifier dialog context must be capped at the most recent
    6 transcript entries (3 turn pairs). The literal [-6:] slice lives
    inside _build_classifier_user_prompt; scan the helper's source so the
    test survives future refactors that move the marker around app.py.
    """
    from backend.app import _build_classifier_user_prompt
    src = inspect.getsource(_build_classifier_user_prompt)
    # The slice must be [-6:] (3 pairs), not [-14:] or larger.
    assert "transcript[-6:]" in src, (
        "Fix 18 regression: classifier _recent_turns window grew beyond 3 pairs. "
        "This re-introduces ~1-2s of latency per turn."
    )
    # Ensure the old 14-window isn't present in the helper.
    assert "[-14:]" not in src, "old 7-pair window still present"


def test_classifier_max_tokens_bound():
    src = _APP_PY.read_text(encoding="utf-8")
    # Find the classifier's call_openai_compatible invocation and its max_tokens.
    # 2026-05-04: prompt was extracted to backend.classifier_prompt; locate the
    # call site via the function-call marker instead of the prompt text.
    marker = "system_prompt=build_classifier_system_prompt()"
    assert marker in src
    idx = src.index(marker)
    # Look for the max_tokens line within ~2000 chars of the call site (only
    # the bound is the contract; the window is just a locator).
    block = src[idx : idx + 2000]
    # max_tokens must stay <=180 to keep latency bounded. 160 is the current
    # safe value (full JSON ~140 tokens + 20 headroom).
    import re as _re
    m = _re.search(r"max_tokens\s*=\s*(\d+)", block)
    assert m, "max_tokens line missing in classifier block"
    assert int(m.group(1)) <= 180, (
        f"Fix 18 regression: classifier max_tokens={m.group(1)} exceeds 180 cap. "
        "Larger cap allows longer generations that hurt latency."
    )


def test_classifier_turn_text_truncation():
    """Fix-18: per-turn text inside the classifier dialog context must be
    truncated to <= 200 chars to bound classifier prompt length. Scan the
    extracted helper rather than app.py, so the test survives variable
    renames (e.g. _text -> text in commit 4cb30c5).
    """
    from backend.app import _build_classifier_user_prompt
    src = inspect.getsource(_build_classifier_user_prompt)
    # Accept either the length-check or a direct slice — semantically
    # equivalent for the 200-char-bound contract.
    assert "len(text) > 200" in src or "[:200]" in src, (
        "Fix 18 regression: classifier turn-text truncation removed."
    )
