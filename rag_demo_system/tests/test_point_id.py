import uuid
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ingest import Chunk
from backend import rag


def test_make_point_struct_uses_uuid_and_payload_chunk_id() -> None:
    chunk = Chunk(
        chunk_id="chunk:45:769",
        text="example text",
        heading_path=["H1"],
        source="source.md",
        doc_name="kb.md",
        start_char=45,
        end_char=769,
    )
    vector = [0.0, 0.1, 0.2]
    point = rag.make_point_struct(chunk, vector)

    assert isinstance(point.id, str)
    uuid.UUID(point.id)
    assert point.payload.get("chunk_id") == "chunk:45:769"


def test_normalize_point_id_accepts_uuid_string() -> None:
    value = str(uuid.uuid4())
    out = rag.normalize_point_id(value)
    assert isinstance(out, str)
    uuid.UUID(out)
