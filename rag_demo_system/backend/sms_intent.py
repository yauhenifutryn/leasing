"""SMS-intent detection.

Used by the legacy orchestrator (app.py:1014-1015) and any other path that
needs to decide whether the current user utterance is a request to send the
last calculator result by SMS.

Two paths fire `True`:

  1. Explicit keyword: "отправ", "смс", "sms", "пришли" anywhere in the
     utterance. Existing pre-Section-3 behaviour.

  2. Affirmation-after-calc: the bot's post-calc prompt always offers SMS
     ("Хотите изменить параметры или отправить график по СМС?"), so a
     short bare affirmation in the immediately following turn is the user
     accepting the offer. Without this, Qwen3.5 sometimes hallucinates
     "график отправлен" without invoking the send_sms tool. Live regression
     pinned in test_sms_intent.py.

The affirmation path is gated on:
  - last entry in tool_calls_history is a successful calculator call
  - utterance is short (≤ 6 words) and starts/equals an affirmation token
  - utterance does NOT contain a change-keyword ("измени", "поменя",
    "пересчит", "другой", "переделай") that would indicate the user
    wants to alter params instead of confirming SMS
  - profile state is NOT READBACK_PENDING or CHANGE_PENDING (Bug 15,
    live call 1cae210d 2026-04-25). In those states the user's "Да" is
    confirming a profile transition, not the SMS offer — sending now
    would deliver stale calc params from before the change.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


_EXPLICIT_TRIGGERS: tuple[str, ...] = ("отправ", "смс", "sms", "пришли")

_AFFIRMATIONS: frozenset[str] = frozenset({
    "да", "ага", "конечно", "пожалуйста", "хорошо", "ок", "окей",
    "давай", "открой", "пиши", "отправь",
})

_CHANGE_KEYWORDS: tuple[str, ...] = (
    "измен", "поменя", "пересчит", "другой", "переделай",
    "не нужно", "не надо",
    # Graph-type stems: any utterance naming a graph type after a calc is
    # a change request, not an SMS confirm. Live regression on call
    # 6a9d359b 2026-04-26: "Да, давай сделаем линейный график" was caught
    # by the leading "да" affirmation path before this. Belt-and-suspenders
    # — the primary gate now lives in app.py at the SMS direct-fire site
    # and consults the classifier's structured change signals.
    "линейн", "аннуит",
    # Change directive forms not covered by "поменя"/"измен":
    "сдела",  # сделай / сделаем / сделать
    "поставь", "ставь",
)

_NEGATIONS: tuple[str, ...] = ("нет", "не", "не-а")


def detect_sms_intent(
    message: str,
    tool_calls_history: Iterable[dict[str, Any]],
    profile_state: Optional[Any] = None,
) -> bool:
    """True if the user's utterance should trigger SMS-direct-send.

    Args:
        message: latest user utterance (raw, before any normalisation).
        tool_calls_history: cumulative session tool-call history. Only the
            most recent entry matters for the affirmation path.
        profile_state: ProfileState (or its string value). When provided
            and equal to READBACK_PENDING or CHANGE_PENDING, both paths
            fail closed — the user's confirmation belongs to that pending
            profile transition, not to the SMS offer. Bug 15.
    """
    if not message:
        return False

    # Bug 15 — never SMS while a readback or change-confirm is in flight.
    # The user's affirmation (or even an explicit "отправь смс") is
    # ambiguous in those states; the safe default is to let the profile
    # transition complete and the calculator re-run before SMS becomes
    # eligible again.
    if profile_state is not None:
        state_value = getattr(profile_state, "value", profile_state)
        if state_value in ("READBACK_PENDING", "CHANGE_PENDING"):
            return False

    msg_lower = message.lower()

    # Path 1 — explicit keyword.
    if any(t in msg_lower for t in _EXPLICIT_TRIGGERS):
        return True

    # Path 2 — affirmation after successful calc.
    history_list = list(tool_calls_history)
    if not history_list:
        return False
    last = history_list[-1]
    last_was_calc_ok = (
        isinstance(last, dict)
        and last.get("tool") == "calculator"
        and isinstance(last.get("result"), dict)
        and last["result"].get("ok") is True
    )
    if not last_was_calc_ok:
        return False

    # Normalise: strip terminal punctuation, replace internal commas with
    # spaces so "Да, открой" splits cleanly into ["да", "открой"].
    msg_norm = msg_lower.strip().rstrip(".!?,…").replace(",", " ")
    # Collapse repeated whitespace produced by comma replacement.
    msg_norm = " ".join(msg_norm.split())

    # Reject change-of-mind utterances even if they start with affirmation.
    if any(c in msg_norm for c in _CHANGE_KEYWORDS):
        return False

    # Reject pure negations.
    if msg_norm in _NEGATIONS or any(
        msg_norm.startswith(neg + " ") for neg in _NEGATIONS
    ):
        return False

    # Word-count guard: a long utterance is unlikely to be a bare SMS confirm.
    words = msg_norm.split()
    if len(words) > 6:
        return False

    # Reject question-like utterances even if short and affirmative.
    if "?" in message or any(
        msg_norm.startswith(q) for q in ("а ", "что ", "почему ", "как ", "когда ")
    ):
        return False

    # Match if the utterance equals or starts with an affirmation token.
    if msg_norm in _AFFIRMATIONS:
        return True
    first_word = words[0] if words else ""
    if first_word in _AFFIRMATIONS:
        return True

    return False
