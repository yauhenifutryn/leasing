from __future__ import annotations

import re
from collections.abc import Iterable, Iterator


_BANNED_PHRASES = [
    "к сожалению",
    "понимаю ваше беспокойство",
    "понимаю вашу ситуацию",
    "в базе знаний нет информации",
    "в предоставленных фрагментах базы знаний",
]


def clean_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned)
    cleaned = re.sub(r"(?i)</?think>", "", cleaned)
    parts = re.split(r"(?i)\bfinal\s*:\s*", cleaned)
    if len(parts) > 1:
        cleaned = parts[-1].strip()
    cleaned = re.sub(r"^\s*\*{0,2}ответ\*{0,2}\s*[:\-]\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^\s*ответ\s*[:\-]\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^\s*\*+\s*", "", cleaned)
    # Strip banned phrases the LLM ignores prompt rules about
    for phrase in _BANNED_PHRASES:
        cleaned = re.sub(re.escape(phrase) + r"[,.]?\s*", "", cleaned, flags=re.I)
    return cleaned.strip()


def strip_leading_name(text: str, name: str, turn_number: int) -> str:
    """Remove client name from the start of response unless it's a 'name turn'.

    Name is allowed on turns 1, 5, 10, ... (every 5th) and when the text
    naturally uses the name mid-sentence. This prevents the LLM habit of
    starting every response with the client's name.
    """
    if not name or not text:
        return text
    if turn_number > 0 and turn_number % 5 == 1:
        return text  # allow name on this turn
    # Strip "Name, " or "Name! " from the beginning
    pattern = re.compile(r"^" + re.escape(name) + r"[,!]?\s*", re.I)
    return pattern.sub("", text)


def sanitize_rewrite(text: str) -> str:
    cleaned = clean_answer(text)
    if not cleaned:
        return ""
    first_line = cleaned.splitlines()[0].strip()
    first_line = first_line.strip("\"'“”")
    if not first_line:
        return ""
    if re.search(r"[.!?]", first_line):
        return ""
    if len(first_line) > 80:
        return ""
    if len(first_line.split()) > 10:
        return ""
    return first_line.strip()


def _emit_visible(text: str, carry: str) -> tuple[list[str], str]:
    data = carry + text
    out_parts: list[str] = []
    while True:
        idx = data.upper().find("FINAL:")
        if idx == -1:
            if len(data) <= 6:
                return out_parts, data
            out_parts.append(data[:-6])
            return out_parts, data[-6:]
        if idx:
            out_parts.append(data[:idx])
        data = data[idx + len("FINAL:") :]


def iter_final_text(chunks: Iterable[str]) -> Iterator[str]:
    buffer = ""
    in_think = False
    for chunk in chunks:
        if not chunk:
            continue
        buffer += chunk
        while buffer:
            if in_think:
                end = buffer.lower().find("</think>")
                if end == -1:
                    if len(buffer) > 16:
                        buffer = buffer[-16:]
                    break
                buffer = buffer[end + len("</think>") :]
                in_think = False
                continue
            start = buffer.lower().find("<think>")
            if start != -1:
                visible = buffer[:start]
                buffer = buffer[start + len("<think>") :]
                in_think = True
            else:
                visible = buffer
                buffer = ""
            if visible:
                cleaned = re.sub(r"(?i)\bfinal\s*:\s*", "", visible)
                if cleaned:
                    yield cleaned
