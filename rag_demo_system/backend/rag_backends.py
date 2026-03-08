from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


@dataclass
class ProviderResponse:
    answer: str
    backend: str
    conversation_ref: dict[str, Any] = field(default_factory=dict)
    used_knowledge: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, Any] = field(default_factory=dict)
    incomplete: bool = False
    can_barge_in: bool = True


class RagProvider(Protocol):
    backend_name: str

    def status(self) -> dict[str, Any]:
        ...


def resolve_backend(
    requested: str | None,
    providers: dict[str, RagProvider | Any],
) -> RagProvider | Any:
    name = (requested or "our_rag").strip() or "our_rag"
    if name in providers:
        return providers[name]
    return providers["our_rag"]


def map_dify_retriever_resources(resources: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    used: list[dict[str, Any]] = []
    for item in resources:
        doc_name = item.get("document_name") or "dify_document"
        segment_position = item.get("segment_position")
        chunk_suffix = segment_position if segment_position is not None else "unknown"
        metadata = item.get("metadata") or {}
        used.append(
            {
                "chunk_id": f"{doc_name}:{chunk_suffix}",
                "doc_name": doc_name,
                "heading_path": list(metadata.get("heading_path") or []),
                "snippet": item.get("content") or "",
                "score": item.get("score"),
            }
        )
    return used


def build_backend_status(
    *,
    launch_mode: str,
    rag_statuses: dict[str, dict[str, Any]],
    voice_statuses: dict[str, dict[str, Any]],
    llm_status: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    backends: dict[str, dict[str, Any]] = {}
    backends.update(rag_statuses)
    backends.update(voice_statuses)
    backends.update(llm_status)
    return {
        "ok": True,
        "launch_mode": launch_mode,
        "backends": backends,
    }
