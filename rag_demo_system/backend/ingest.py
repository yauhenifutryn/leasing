from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    text: str
    heading_path: List[str]
    source: str
    doc_name: str
    start_char: int
    end_char: int


@dataclass
class Block:
    text: str
    heading_path: List[str]
    start_char: int
    end_char: int
    kind: str


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
LIST_RE = re.compile(r"^\s*([-*+]\s+|\d+\.)\s+.+")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*-{3,}")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")


def _token_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def parse_blocks(md_text: str) -> list[Block]:
    lines = md_text.splitlines()
    heading_stack: list[str] = []
    blocks: list[Block] = []

    buffer: list[str] = []
    buffer_start = 0
    buffer_end = 0
    buffer_kind = "paragraph"

    def flush() -> None:
        nonlocal buffer, buffer_start, buffer_end, buffer_kind
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(
                Block(
                    text=text,
                    heading_path=heading_stack.copy(),
                    start_char=buffer_start,
                    end_char=buffer_end,
                    kind=buffer_kind,
                )
            )
        buffer = []
        buffer_kind = "paragraph"

    offset = 0
    idx = 0
    in_code = False
    code_start = 0
    code_buffer: list[str] = []

    while idx < len(lines):
        line = lines[idx]
        line_start = offset
        offset += len(line) + 1

        match = HEADING_RE.match(line.strip())
        if match and not in_code:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            if level <= len(heading_stack):
                heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(title)
            idx += 1
            continue

        if line.strip().startswith("```"):
            if not in_code:
                flush()
                in_code = True
                code_start = line_start
                code_buffer = [line]
                idx += 1
                continue
            in_code = False
            code_buffer.append(line)
            blocks.append(
                Block(
                    text="\n".join(code_buffer).strip(),
                    heading_path=heading_stack.copy(),
                    start_char=code_start,
                    end_char=offset,
                    kind="code",
                )
            )
            code_buffer = []
            idx += 1
            continue

        if in_code:
            code_buffer.append(line)
            idx += 1
            continue

        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        is_table = "|" in line and TABLE_SEP_RE.search(next_line)
        if is_table:
            flush()
            table_start = line_start
            table_lines = [line]
            idx += 1
            while idx < len(lines):
                table_lines.append(lines[idx])
                if idx + 1 >= len(lines):
                    idx += 1
                    break
                if not lines[idx + 1].strip():
                    idx += 1
                    break
                if "|" not in lines[idx + 1]:
                    idx += 1
                    break
                idx += 1
            end_char = table_start + sum(len(l) + 1 for l in table_lines)
            blocks.append(
                Block(
                    text="\n".join(table_lines).strip(),
                    heading_path=heading_stack.copy(),
                    start_char=table_start,
                    end_char=end_char,
                    kind="table",
                )
            )
            continue

        if LIST_RE.match(line):
            if buffer and buffer_kind != "list":
                flush()
            if not buffer:
                buffer_start = line_start
                buffer_kind = "list"
            buffer.append(line)
            buffer_end = offset
            idx += 1
            continue

        if not line.strip():
            flush()
            idx += 1
            continue

        if not buffer:
            buffer_start = line_start
        buffer.append(line)
        buffer_end = offset
        idx += 1

    flush()
    return blocks


def _split_paragraph(text: str, chunk_size_tokens: int) -> list[str]:
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text]
    chunks: list[str] = []
    buffer: list[str] = []
    tokens = 0
    for sent in sentences:
        sent_tokens = _token_count(sent)
        if tokens + sent_tokens > chunk_size_tokens and buffer:
            chunks.append(" ".join(buffer).strip())
            buffer = [sent]
            tokens = sent_tokens
        else:
            buffer.append(sent)
            tokens += sent_tokens
    if buffer:
        chunks.append(" ".join(buffer).strip())
    return chunks


def chunk_blocks(blocks: list[Block], chunk_size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    i = 0
    while i < len(blocks):
        start_block = blocks[i]
        heading_path = start_block.heading_path
        start_char = start_block.start_char
        buffer: list[Block] = []
        tokens = 0
        j = i
        while j < len(blocks):
            block = blocks[j]
            if block.heading_path != heading_path and tokens > 0:
                break
            block_tokens = _token_count(block.text)
            if tokens + block_tokens > chunk_size_tokens and tokens > 0:
                break
            buffer.append(block)
            tokens += block_tokens
            j += 1
            if tokens >= chunk_size_tokens:
                break

        if not buffer:
            buffer = [blocks[i]]
            j = i + 1

        # If a single paragraph is too large, split by sentences
        assembled_parts: list[str] = []
        for block in buffer:
            if block.kind == "paragraph" and _token_count(block.text) > chunk_size_tokens:
                assembled_parts.extend(_split_paragraph(block.text, chunk_size_tokens))
            else:
                assembled_parts.append(block.text)

        end_char = buffer[-1].end_char
        text = "\n\n".join(assembled_parts).strip()
        chunk_id = f"chunk:{start_char}:{end_char}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                heading_path=heading_path,
                source="",
                doc_name="",
                start_char=start_char,
                end_char=end_char,
            )
        )

        if overlap_tokens > 0:
            overlap_count = 0
            overlap_blocks = 0
            for block in reversed(buffer):
                overlap_count += _token_count(block.text)
                overlap_blocks += 1
                if overlap_count >= overlap_tokens:
                    break
            i = max(i + 1, j - overlap_blocks)
        else:
            i = j if j > i else i + 1

    return chunks


def build_chunks(md_path: Path, chunk_size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    raw = md_path.read_text(encoding="utf-8")
    blocks = parse_blocks(raw)
    chunks = chunk_blocks(blocks, chunk_size_tokens, overlap_tokens)

    doc_name = md_path.name
    for chunk in chunks:
        heading_prefix = " / ".join(chunk.heading_path) if chunk.heading_path else "Без раздела"
        chunk.text = f"{heading_prefix}\n\n{chunk.text}".strip()
        chunk.source = str(md_path)
        chunk.doc_name = doc_name

    return chunks
