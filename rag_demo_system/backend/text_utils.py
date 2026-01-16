from __future__ import annotations

import re
from collections.abc import Iterable, Iterator


def clean_answer(text: str) -> str:
    cleaned = text.strip()
    parts = re.split(r"(?i)\bfinal\s*:\s*", cleaned)
    if len(parts) > 1:
        cleaned = parts[-1].strip()
    cleaned = re.sub(r"^\s*\*{0,2}ответ\*{0,2}\s*[:\-]\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^\s*ответ\s*[:\-]\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^\s*\*+\s*", "", cleaned)
    return cleaned.strip()


def iter_final_text(chunks: Iterable[str]) -> Iterator[str]:
    buffer = ""
    found = False
    for chunk in chunks:
        if not chunk:
            continue
        if found:
            yield chunk
            continue
        buffer += chunk
        marker_index = buffer.upper().find("FINAL:")
        if marker_index == -1:
            if len(buffer) > 32:
                buffer = buffer[-16:]
            continue
        found = True
        remainder = buffer[marker_index + len("FINAL:") :]
        buffer = ""
        if remainder:
            yield remainder
