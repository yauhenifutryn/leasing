from __future__ import annotations

import re
from typing import Any


SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")


def split_sentences(text: str) -> list[str]:
    raw = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    return raw if raw else [text.strip()]


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in WORD_RE.findall(text)}


def attach_citations(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sentences = split_sentences(answer)
    chunk_tokens = [(c["chunk_id"], _tokenize(c.get("text", ""))) for c in chunks]

    citations = []
    for idx, sent in enumerate(sentences):
        sent_tokens = _tokenize(sent)
        best_id = None
        best_score = 0.0
        for chunk_id, tokens in chunk_tokens:
            if not tokens or not sent_tokens:
                continue
            overlap = sent_tokens.intersection(tokens)
            score = len(overlap) / max(1, len(sent_tokens))
            if score > best_score:
                best_score = score
                best_id = chunk_id
        citations.append({"sentence_index": idx, "sentence": sent, "chunk_id": best_id, "score": best_score})
    return citations
