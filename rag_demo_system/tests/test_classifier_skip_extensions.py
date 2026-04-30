"""Bug 12 + a487323-replacement: tests for _SKIP_CLASSIFIER and _skip
predicate extensions.

Bug 12 (live calls 2026-04-29): bare backchannels ("Угу.", "Ага.",
"Ксения.") were sent through the classifier and routed to clarify
prompts. Add them to _SKIP_CLASSIFIER so the classifier never sees
them and apply_turn falls naturally to FireLLMFallback.

a487323 replacement (Codex finding): the FAST-PATH _fast_deny flag
exists at backend/app.py:950-999 but the _skip predicate at
1009-1015 doesn't include it. Adding `or _fast_deny` to _skip is the
one-line fix that the original cherry-pick was trying to ship.

These tests exercise the module-level _SKIP_CLASSIFIER constant
directly. The full request-handler (chat / process_user_text) wiring
is covered by the live call evidence and the existing classifier
fast-path test suite (test_classifier_fast_path_confirm.py).
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_skip_classifier_includes_backchannel_words():
    """Bug 12: bare backchannels must be in the skip set so they bypass
    the classifier round-trip and route to FireLLMFallback by default."""
    from backend import app as backend_app

    src = Path(backend_app.__file__).read_text(encoding="utf-8")

    # Anchor on the literal _SKIP_CLASSIFIER assignment site.
    assert "_SKIP_CLASSIFIER" in src, (
        "_SKIP_CLASSIFIER constant assignment should exist in backend.app"
    )
    # Each new backchannel word must appear inside the constant block.
    # The block is short (a single set literal); grep is sufficient.
    for word in ("угу", "ага", "мгм", "ксения", "ксюша"):
        assert f'"{word}"' in src, (
            f"_SKIP_CLASSIFIER must include {word!r}; live calls 2026-04-29 "
            f"showed the bot routing this bare backchannel to clarify"
        )


def test_skip_predicate_honors_fast_deny():
    """a487323 replacement: the `_skip` predicate must include the
    `_fast_deny` short-circuit. Without this, a one-word "нет" /
    "отмена" in CHANGE_PENDING bypasses the classifier (good) but
    then re-enters the classifier path through `tool_schemas and
    not _skip` (bad), wasting the fast-path saving."""
    from backend import app as backend_app

    src = Path(backend_app.__file__).read_text(encoding="utf-8")
    # The literal predicate. Order doesn't matter (`_fast_confirm` may
    # come first), but `_fast_deny` MUST appear in the same predicate
    # next to `_fast_confirm` and before the `_msg_stripped in
    # _SKIP_CLASSIFIER` clause.
    assert "_fast_deny" in src, "_fast_deny flag must exist (was added by FAST-PATH deny)"

    # Find the `_skip = (` block and verify _fast_deny appears inside.
    idx = src.find("_skip = (")
    assert idx >= 0, "_skip = ( ... ) predicate not found in backend.app"
    # Slice forward up to the closing ')' of that assignment. This is
    # bounded because the predicate is short.
    end = src.find(")", idx)
    block = src[idx:end + 1]
    assert "_fast_deny" in block, (
        "a487323 replacement: `_skip` predicate must include _fast_deny "
        "so a fast-path deny in pending states also bypasses the "
        "classifier round-trip. Predicate block was:\n" + block
    )
