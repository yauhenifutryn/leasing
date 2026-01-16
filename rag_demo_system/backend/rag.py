from __future__ import annotations

from typing import Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .ingest import Chunk

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "micro-leasing-kb")


def normalize_point_id(value: str | int | uuid.UUID) -> str | int:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            uuid.UUID(value)
            return value
        except ValueError:
            pass
        if value.isdigit():
            return int(value)
        return str(uuid.uuid5(_NAMESPACE, value))
    raise TypeError("Unsupported point id type")


def make_point_struct(chunk: Chunk, vector: list[float]) -> qmodels.PointStruct:
    normalized_id = normalize_point_id(chunk.chunk_id)
    return qmodels.PointStruct(
        id=normalized_id,
        vector=vector,
        payload={
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "heading_path": chunk.heading_path,
            "source": chunk.source,
            "doc_name": chunk.doc_name,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        },
    )


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    existing = client.get_collections().collections
    if any(c.name == name for c in existing):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )


def upsert_chunks(client: QdrantClient, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(make_point_struct(chunk, vector))
    client.upsert(collection_name=collection, points=points)


def search(client: QdrantClient, collection: str, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    # Qdrant client compatibility: search() (older) vs query_points() (newer)
    if hasattr(client, 'search'):
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
    elif hasattr(client, 'query_points'):
        res = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        results = getattr(res, 'points', res)
    else:
        raise RuntimeError('Unsupported qdrant-client: neither search() nor query_points() exists')
    out: list[dict[str, Any]] = []
    for hit in results:
        payload = hit.payload or {}
        out.append(
            {
                "chunk_id": str(payload.get("chunk_id") or hit.id),
                "text": str(payload.get("text", "")),
                "score": float(hit.score or 0.0),
                "heading_path": list(payload.get("heading_path") or []),
                "source": str(payload.get("source", "")),
                "doc_name": str(payload.get("doc_name", "")),
                "start_char": int(payload.get("start_char") or 0),
                "end_char": int(payload.get("end_char") or 0),
            }
        )
    return out
