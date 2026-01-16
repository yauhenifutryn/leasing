from pathlib import Path

from backend.ingest import build_chunks


def test_chunking(tmp_path: Path):
    md = "# Заголовок\n\nТекст раздела.\n\n## Подраздел\n\nДоп текст."
    path = tmp_path / "kb_test.md"
    path.write_text(md, encoding="utf-8")
    chunks = build_chunks(path, chunk_size_tokens=120, overlap_tokens=20)
    assert len(chunks) >= 1
