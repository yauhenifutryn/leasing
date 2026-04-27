"""Contract tests for classifier latency-reduction bounds.

If someone refactors the classifier block and bumps these values back up,
latency regresses ~1-2 seconds. This test catches that.
"""
from pathlib import Path


_APP_PY = Path(__file__).resolve().parents[1] / "backend" / "app.py"


def test_classifier_recent_turns_bound():
    src = _APP_PY.read_text(encoding="utf-8")
    # The slice must be [-6:] (3 pairs), not [-14:] or larger.
    assert "chat_session.transcript[-6:]" in src, (
        "Fix 18 regression: classifier _recent_turns window grew beyond 3 pairs. "
        "This re-introduces ~1-2s of latency per turn."
    )
    # Ensure the old 14-window isn't present in the classifier block.
    # (Other slices may legitimately exist for other purposes; we only check
    # the specific classifier context line.)
    classifier_marker = "# Build conversation context"
    assert classifier_marker in src
    idx = src.index(classifier_marker)
    block = src[idx : idx + 400]
    assert "[-14:]" not in block, "old 7-pair window still present"


def test_classifier_max_tokens_bound():
    src = _APP_PY.read_text(encoding="utf-8")
    # Find the classifier's call_openai_compatible invocation and its max_tokens.
    marker = "Ты SessionAgent голосового бота"
    assert marker in src
    idx = src.index(marker)
    # Look for the max_tokens line within ~10000 chars of the system prompt
    # (the SessionAgent prompt is ~5500 chars after the 2026-04-27 mid-collection
    # RAG-drift priority block was added; add headroom for the
    # call_openai_compatible args block that follows).
    block = src[idx : idx + 10000]
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
    src = _APP_PY.read_text(encoding="utf-8")
    # Truncation must be present so long bot responses don't bloat the prompt.
    assert "len(_text) > 200" in src, (
        "Fix 18 regression: classifier turn-text truncation removed."
    )
