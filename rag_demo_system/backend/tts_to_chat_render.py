"""Convert TTS-phonetic strings to chat display form.

The KB is shared by voice and chat. KB authors write phonetic forms so
Silero TTS pronounces things naturally — "сайт микро-лизинг точка бай" sounds
right when spoken; "site mikro-lizing.by" does not. In chat we want the
opposite: the displayable digit/symbol form. This module is the chat-only
post-processor; the KB stays untouched.

Apply at the end of `_text_process_utterance` to the assembled `full_reply`,
right before persistence + the broadcast event.
"""
from __future__ import annotations

import re

# Order matters: more-specific patterns run first so a longer phrase like
# "инфо собака микро-лизинг точка бай" is matched as a single email construct
# rather than three independent replacements competing.

# Email: "<local> собака <host> точка <tld>" → "<local>@<host>.<tld>"
_EMAIL_FULL_RE = re.compile(
    r"(\w[\w-]*)\s+собака\s+([\w-]+(?:\s*-\s*[\w-]+)*)\s+точка\s+(бай|ру|ком|орг|нет)",
    re.IGNORECASE | re.UNICODE,
)

# Domain: "<host> точка <tld>" → "<host>.<tld>"
_DOMAIN_RE = re.compile(
    r"([\w-]+(?:\s*-\s*[\w-]+)*)\s+точка\s+(бай|ру|ком|орг|нет)\b",
    re.IGNORECASE | re.UNICODE,
)

# "X собака Y" (unattached email-ish, no domain dot) → "X@Y"
_SOBAKA_SHORT_RE = re.compile(
    r"(\w[\w-]*)\s+собака\s+(\w[\w-]*)",
    re.IGNORECASE | re.UNICODE,
)

# Address: "..., 6, а" → "..., 6А" (Cyrillic 'а' lowercase to uppercase 'А' attached)
_HOUSE_LETTER_RE = re.compile(
    r"(\d+)\s*,\s*а\b",
    re.UNICODE,
)

# Currency words → ISO codes
_CURRENCY_PATTERNS = [
    (re.compile(r"\bбелорусск(?:их|ий)\s+рубл(?:ей|ь|я)\b", re.IGNORECASE | re.UNICODE), "BYN"),
    (re.compile(r"\bдоллар(?:ов|а|у)?(?:\s+США)?\b", re.IGNORECASE | re.UNICODE), "USD"),
    (re.compile(r"\bевро\b", re.IGNORECASE | re.UNICODE), "EUR"),
]

# Phonetic abbreviation strings the LLM may emit when inheriting TTS rules.
# Replace WITH the readable abbreviation form (which is also what TTS will
# pronounce correctly because the system prompt tells it to read each letter).
_ABBR_PHONETIC = [
    (re.compile(r"\bпэ-дэ-эн\b", re.IGNORECASE | re.UNICODE), "ПДН"),
    (re.compile(r"\bэр-эф\b", re.IGNORECASE | re.UNICODE), "РФ"),
    (re.compile(r"\bвэ-эн-жэ\b", re.IGNORECASE | re.UNICODE), "ВНЖ"),
    (re.compile(r"\bбэ-уай-эн\b", re.IGNORECASE | re.UNICODE), "BYN"),
]

_TLD_MAP = {
    "бай": "by",
    "ру": "ru",
    "ком": "com",
    "орг": "org",
    "нет": "net",
}


def _collapse_dashes(host_phrase: str) -> str:
    """Collapse 'микро - лизинг' / 'микро-лизинг' into 'микро-лизинг'."""
    return re.sub(r"\s*-\s*", "-", host_phrase).strip()


def tts_to_chat_render(text: str) -> str:
    """Apply chat-display transformations to a TTS-friendly string.

    Idempotent — running twice yields the same output as running once.
    Safe on empty strings. Conservative on ambiguity: a pattern only fires
    when the surrounding tokens look like the intended construct.
    """
    if not text:
        return text

    # 1. Email construct first (longest pattern wins).
    def _email_sub(m: re.Match) -> str:
        local = m.group(1)
        host = _collapse_dashes(m.group(2))
        tld = _TLD_MAP.get(m.group(3).lower(), m.group(3))
        return f"{local}@{host}.{tld}"

    text = _EMAIL_FULL_RE.sub(_email_sub, text)

    # 2. Domain construct.
    def _domain_sub(m: re.Match) -> str:
        host = _collapse_dashes(m.group(1))
        tld = _TLD_MAP.get(m.group(2).lower(), m.group(2))
        return f"{host}.{tld}"

    text = _DOMAIN_RE.sub(_domain_sub, text)

    # 3. Lone "X собака Y" (after the email pattern took the long forms).
    text = _SOBAKA_SHORT_RE.sub(r"\1@\2", text)

    # 4. Address letter suffixes "6, а" → "6А".
    text = _HOUSE_LETTER_RE.sub(lambda m: f"{m.group(1)}А", text)

    # 5. Currency words → ISO.
    for rx, code in _CURRENCY_PATTERNS:
        text = rx.sub(code, text)

    # 6. Phonetic abbreviation echoes.
    for rx, abbr in _ABBR_PHONETIC:
        text = rx.sub(abbr, text)

    return text
