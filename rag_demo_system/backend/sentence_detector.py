"""Sentence detector for streaming LLM token output.

Accumulates tokens and emits complete sentences, splitting on sentence-ending
punctuation followed by a space.  Preserves common Russian (and currency)
abbreviations so that an internal period does not trigger a false split.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Abbreviation list
# ---------------------------------------------------------------------------
# Abbreviations whose trailing period must NOT be treated as a sentence
# boundary when they appear *mid-sentence* (i.e. followed by continuation
# text).  Order: longer patterns first so regex alternation picks them up
# before shorter prefixes.
_ABBREVS: tuple[str, ...] = (
    "и т.д.",
    "и т.п.",
    "т.е.",
    "т.д.",
    "т.п.",
    "т.к.",
    "корп.",
    "стр.",
    "руб.",
    "USD.",
    "EUR.",
    "BYN.",
    "ул.",
    "кв.",
    "пр.",
    "др.",
    "г.",
    "д.",
)

# Regex: does the string *end* with one of the abbreviations?
_ABBREV_TAIL_RE = re.compile(
    r"(?:"
    + "|".join(re.escape(a) for a in _ABBREVS)
    + r")$"
)

# Sentence-ending patterns: a terminator followed by at least one space.
# No end-of-string anchor so we can find boundaries mid-buffer.
_SENTENCE_END_RE = re.compile(r"(\.\.\.|\.|!|\?)\s")


class SentenceDetector:
    """Accumulates streamed tokens and yields full sentences."""

    def __init__(self) -> None:
        self._buf: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, token: str) -> list[str]:
        """Add *token* to the buffer and return any complete sentences.

        A sentence is emitted when the buffer contains sentence-ending
        punctuation followed by a space, **unless** the period belongs to
        a known abbreviation that is followed by continuation text.
        """
        self._buf += token
        return self._try_emit()

    def flush(self) -> str | None:
        """Return whatever remains in the buffer, or ``None`` if empty.

        Call this when the LLM stream has ended to retrieve the last
        (possibly incomplete) sentence.
        """
        rest = self._buf.strip()
        self._buf = ""
        return rest or None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _try_emit(self) -> list[str]:
        """Scan the buffer for valid sentence boundaries and split."""
        sentences: list[str] = []

        while True:
            split = self._find_split()
            if split is None:
                break
            sentence = self._buf[:split].rstrip()
            sentences.append(sentence)
            self._buf = self._buf[split:]

        return sentences

    def _find_split(self) -> int | None:
        """Return the buffer index just past the first valid sentence end.

        Returns ``None`` when no valid split point exists yet.
        """
        start = 0
        while True:
            m = _SENTENCE_END_RE.search(self._buf, pos=start)
            if m is None:
                return None

            end = m.end()                        # index right after the space
            prefix = self._buf[:end].rstrip()    # text up to (incl.) punctuation
            rest = self._buf[end:]               # text after the space

            # If the punctuation belongs to an abbreviation AND there is
            # continuation text after it, this is not a real sentence
            # boundary.  Skip it and keep scanning.
            if _ABBREV_TAIL_RE.search(prefix) and rest.strip():
                start = end
                continue

            # Valid sentence boundary (or abbreviation at very end of
            # buffer with no continuation).
            return end

        return None  # unreachable, but keeps mypy happy
