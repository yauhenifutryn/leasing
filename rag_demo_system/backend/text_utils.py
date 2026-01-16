from __future__ import annotations

import re


def clean_answer(text: str) -> str:
    cleaned = text.strip()
    parts = re.split(r"(?i)\bfinal\s*:\s*", cleaned)
    if len(parts) > 1:
        cleaned = parts[-1].strip()
    cleaned = re.sub(r"^\s*\*{0,2}ответ\*{0,2}\s*[:\-]\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^\s*ответ\s*[:\-]\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^\s*\*+\s*", "", cleaned)
    return cleaned.strip()
