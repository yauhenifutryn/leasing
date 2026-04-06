from __future__ import annotations

import re
from collections.abc import Iterable, Iterator


_BANNED_PHRASES = [
    "к сожалению",
    "понимаю ваше беспокойство",
    "понимаю вашу ситуацию",
    "шутки не работают",
    "нет шутки в запасе",
    "не нарушать деловой этикет",
]
# Regex patterns for phrases the LLM varies creatively
_BANNED_PATTERNS = [
    # Drop entire sentence that mentions "база знаний" in any form
    re.compile(r"[^.!?]*(?:в\s+(?:моей\s+|нашей\s+)?базе\s+(?:знаний|данных)|в\s+предоставленных\s+фрагментах)[^.!?]*[.!?]?\s*", re.I),
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
    # Strip "Ксения:" role label the LLM sometimes outputs
    cleaned = re.sub(r"^Ксения\s*:\s*", "", cleaned, flags=re.I)
    # Strip banned phrases the LLM ignores prompt rules about
    for phrase in _BANNED_PHRASES:
        cleaned = re.sub(re.escape(phrase) + r"[,.]?\s*", "", cleaned, flags=re.I)
    for pattern in _BANNED_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Replace raw URLs/emails with spoken forms (LLM sometimes ignores prompt rule)
    cleaned = re.sub(r"(?:(?:на|по адресу)\s+(?:нашем\s+)?(?:сайте?[:\s]*)?)?https?://mikro-leasing\.by/akcii\S*", "на сайте микро-лизинг точка бай в разделе акции", cleaned, flags=re.I)
    cleaned = re.sub(r"(?:(?:на|по адресу)\s+(?:нашем\s+)?(?:сайте?[:\s]*)?)?https?://mikro-leasing\.by\S*", "на сайте микро-лизинг точка бай", cleaned, flags=re.I)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"\bwww\.\S+", "", cleaned)
    cleaned = re.sub(r"[\w.-]+@[\w.-]+\.\w+", "инфо собака микро-лизинг точка бай", cleaned)
    # Fix male gender forms the LLM uses despite female persona prompt
    cleaned = re.sub(r"\bя рад\b", "я рада", cleaned)
    cleaned = re.sub(r"\bЯ рад\b", "Я рада", cleaned)
    cleaned = re.sub(r"\bРад,", "Рада,", cleaned)
    cleaned = re.sub(r"\bРад ", "Рада ", cleaned)
    cleaned = re.sub(r"\bбуду рад\b", "буду рада", cleaned)
    cleaned = re.sub(r"\bя смог\b", "я смогла", cleaned)
    cleaned = re.sub(r"\bсмог помочь", "смогла помочь", cleaned)
    cleaned = re.sub(r"\bя мог\b", "я могла", cleaned)
    cleaned = re.sub(r"\bчтобы я мог\b", "чтобы я могла", cleaned)
    cleaned = re.sub(r"\bя готов\b", "я готова", cleaned)
    cleaned = re.sub(r"\bголосовой помощник\b", "голосовая помощница", cleaned, flags=re.I)
    cleaned = re.sub(r"\bголосовой консультант\b", "голосовая помощница", cleaned, flags=re.I)
    return cleaned.strip()


def _name_forms(name: str) -> list[str]:
    """Get all grammatical case forms of a Russian name using pymorphy3."""
    forms = {name.lower()}
    try:
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()
        parsed = morph.parse(name)[0]
        # Add nominative (base form)
        forms.add(parsed.normal_form)
        # Add all case forms
        for case in ("nomn", "gent", "datv", "accs", "ablt", "loct"):
            inflected = parsed.inflect({case})
            if inflected:
                forms.add(inflected.word)
    except Exception:  # noqa: BLE001
        pass
    return sorted(forms, key=len, reverse=True)  # longest first to avoid partial matches


def strip_name_from_response(text: str, name: str, turn_number: int) -> str:
    """Control client name frequency in responses.

    Name is allowed on turns 1, 6, 11, ... (every 5th). On other turns,
    only VOCATIVE uses are removed (addressing the person):
      - "Никита, ..." at the start
      - "..., Никита," or "..., Никита!" in the middle/end
    The name is KEPT when it's part of the semantic content
    (e.g., "Вас зовут Никита" in response to "как меня зовут?").
    """
    if not name or not text:
        return text
    if turn_number > 0 and turn_number % 10 == 1:
        return text  # allow name on this turn
    # Only strip vocative patterns (name used for addressing, not content)
    for form in _name_forms(name):
        esc = re.escape(form)
        # "Никита, ..." at sentence start
        text = re.sub(r"(?:^|\.\s+)" + esc + r",\s*", lambda m: m.group(0)[:m.group(0).find(form[0])] if "." in m.group(0) else "", text, flags=re.I)
        # ", Никита," or ", Никита!" in middle/end
        text = re.sub(r",\s*" + esc + r"\s*[,!.]", lambda m: m.group(0)[-1] if m.group(0)[-1] in ".!" else "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# Regex to detect street addresses in Russian text (all grammatical cases)
_ADDRESS_RE = re.compile(
    r"(?:улиц[аеыуой]|ул\.|проспект[аеуом]?|пр-т[аеуом]?|пр\."
    r"|переулк[аеуом]?|пер\.|бульвар[аеуом]?|б-р|набережн[аяойуюые]|наб\.)"
    r"\s+[А-ЯЁа-яё][А-ЯЁа-яё\s\-]+,?\s*(?:дом\s+)?\d+[А-Яа-я]?",
    re.I,
)

def validate_addresses(text: str, context_chunks: list[str]) -> str:
    """Remove sentences with hallucinated addresses.

    If an address in the LLM response doesn't appear in the retrieved
    context, the ENTIRE sentence containing it is removed (not just the
    address) to avoid garbled output like "офис находится на Точный адрес...".
    """
    if not context_chunks:
        return text
    context_joined = " ".join(context_chunks).lower()

    # Split into sentences, check each for hallucinated addresses
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean_sentences = []
    for sentence in sentences:
        addresses = _ADDRESS_RE.findall(sentence)
        if not addresses:
            clean_sentences.append(sentence)
            continue
        # Check if ALL addresses in this sentence are in context
        all_valid = True
        for addr in addresses:
            street_words = re.findall(r"[А-ЯЁа-яё]{3,}", addr)
            found = any(w.lower() in context_joined for w in street_words)
            if not found:
                all_valid = False
                break
        if all_valid:
            clean_sentences.append(sentence)
        # else: drop the entire sentence silently

    return " ".join(clean_sentences)


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
