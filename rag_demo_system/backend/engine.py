from __future__ import annotations

import json
from pathlib import Path
import logging
from typing import Any

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .cache import TTLCache
from .ingest import Chunk, build_chunks
from .query import load_abbreviations, normalize_query
from .rag import ensure_collection, search, upsert_chunks
from .llm import call_openai_compatible
from .rerank import Reranker
from .settings import Settings

logger = logging.getLogger("rag_demo")


def iter_batches(items: list[Chunk], batch_size: int) -> list[list[Chunk]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    out: list[list[Chunk]] = []
    for i in range(0, len(items), batch_size):
        out.append(items[i : i + batch_size])
    return out


class RAGEngine:
    def __init__(self, settings: Settings, state_dir: Path) -> None:
        self.settings = settings
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.embedder: SentenceTransformer | None = None
        self.reranker: Reranker | None = None
        self.bm25: BM25Okapi | None = None
        self.bm25_chunks: list[Chunk] = []
        self.abbrev = load_abbreviations(self.settings.query_rewrite.abbreviations_path)
        self.query_cache = TTLCache(ttl_sec=60)
        self.rerank_cache = TTLCache(ttl_sec=60)

    def _get_embedder(self) -> SentenceTransformer:
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.settings.embedding.model_name, device=self.settings.embedding.device)
        return self.embedder

    def _get_reranker(self) -> Reranker | None:
        if not self.settings.reranker.enabled:
            return None
        if self.reranker is None:
            self.reranker = Reranker(
                model_name=self.settings.reranker.model_name,
                device=self.settings.reranker.device,
                batch_size=self.settings.reranker.batch_size,
            )
        return self.reranker

    def _tokenize(self, text: str) -> list[str]:
        return [t for t in text.lower().split() if t]

    def _save_chunks(self, chunks: list[Chunk]) -> None:
        out = [chunk.__dict__ for chunk in chunks]
        (self.state_dir / "chunks.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_chunks(self) -> list[Chunk]:
        path = self.state_dir / "chunks.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Chunk(**item) for item in data]

    def index(self) -> dict[str, Any]:
        kb_path = self.settings.app.kb_markdown_path
        if not kb_path.exists():
            return {"ok": False, "error": f"KB not found: {kb_path}"}

        chunks = build_chunks(
            kb_path,
            self.settings.embedding.chunk_size_tokens,
            self.settings.embedding.chunk_overlap_tokens,
        )

        embedder = self._get_embedder()
        client = QdrantClient(url=self.settings.qdrant.url)
        batch_size = self.settings.embedding.batch_size
        total = len(chunks)
        inserted = 0
        logger.info("Indexing KB: %s", kb_path)
        logger.info("Chunks total: %s", total)

        for idx, batch in enumerate(iter_batches(chunks, batch_size)):
            texts = [f"passage: {c.text}" for c in batch]
            vectors = embedder.encode(texts, normalize_embeddings=True)
            vectors = vectors.tolist() if hasattr(vectors, "tolist") else vectors
            if idx == 0:
                ensure_collection(client, self.settings.qdrant.collection, vector_size=len(vectors[0]))
            upsert_chunks(client, self.settings.qdrant.collection, batch, vectors)
            inserted += len(batch)
            logger.info("Upserted %s/%s", inserted, total)

        self._save_chunks(chunks)
        self._build_bm25(chunks)
        return {"ok": True, "chunks": len(chunks), "inserted": inserted}

    def _build_bm25(self, chunks: list[Chunk]) -> None:
        self.bm25_chunks = chunks
        corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def _ensure_bm25(self) -> None:
        if self.bm25 is not None:
            return
        chunks = self._load_chunks()
        if chunks:
            self._build_bm25(chunks)

    def retrieve(self, query: str) -> dict[str, Any]:
        original_query = query
        normalized = normalize_query(query, self.abbrev)
        rewritten = normalized
        if self.settings.llm.base_url and self.settings.llm.model:
            try:
                llm_resp = call_openai_compatible(
                    base_url=self.settings.llm.base_url,
                    model=self.settings.llm.model,
                    system_prompt=(
                        "Переформулируй запрос клиента в короткий поисковый запрос для базы знаний. "
                        "Без домыслов, только ключевые слова. Верни одну строку."
                    ),
                    user_prompt=f"Запрос клиента: {normalized}",
                    temperature=0.0,
                    max_tokens=48,
                    timeout_sec=8,
                )
                candidate = llm_resp.text.strip().splitlines()[0].strip()
                if candidate:
                    rewritten = candidate
            except Exception:
                rewritten = normalized

        embedder = self._get_embedder()
        cached = self.query_cache.get(rewritten)
        if cached is None:
            q_vec = embedder.encode([f"query: {rewritten}"], normalize_embeddings=True)
            q_vec = q_vec.tolist() if hasattr(q_vec, "tolist") else q_vec
            self.query_cache.set(rewritten, q_vec[0])
        else:
            q_vec = [cached]

        client = QdrantClient(url=self.settings.qdrant.url)
        vector_hits = search(client, self.settings.qdrant.collection, q_vec[0], self.settings.retrieval.vector_top_k)

        self._ensure_bm25()
        bm25_hits: list[Chunk] = []
        if self.bm25 and self.bm25_chunks:
            scores = self.bm25.get_scores(self._tokenize(rewritten))
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.settings.retrieval.bm25_top_k]
            bm25_hits = [self.bm25_chunks[i] for i in top_idx]

        merged: dict[str, dict[str, Any]] = {}
        for hit in vector_hits:
            merged[hit["chunk_id"]] = hit
        for chunk in bm25_hits:
            merged.setdefault(
                chunk.chunk_id,
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "score": 0.0,
                    "heading_path": chunk.heading_path,
                    "source": chunk.source,
                    "doc_name": chunk.doc_name,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                },
            )

        candidates = list(merged.values())

        reranker = self._get_reranker()
        if reranker:
            cache_key = normalized + ":" + ",".join(sorted([c["chunk_id"] for c in candidates]))
            cached_rerank = self.rerank_cache.get(cache_key)
            if cached_rerank is None:
                rerank_scores = reranker.rerank(rewritten, candidates)
                score_map = {r.chunk_id: r.score for r in rerank_scores}
                self.rerank_cache.set(cache_key, score_map)
            else:
                score_map = cached_rerank
            for c in candidates:
                c["rerank_score"] = float(score_map.get(c["chunk_id"], 0.0))
        else:
            if not self.settings.reranker.allow_no_rerank:
                return {"ok": False, "error": "Reranker disabled and allow_no_rerank=false"}
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)

        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        top_rerank_score = candidates[0].get("rerank_score", 0.0) if candidates else 0.0

        filtered = [
            c
            for c in candidates
            if c.get("rerank_score", 0.0) >= self.settings.retrieval.min_rerank_score
        ]

        weak = False
        final: list[dict[str, Any]] = []
        total_tokens = 0
        if not filtered and candidates:
            weak = True
            filtered = candidates[: self.settings.retrieval.final_top_n]
        for c in filtered:
            tokens = len(c.get("text", "").split())
            if total_tokens + tokens > self.settings.retrieval.context_max_tokens:
                break
            final.append(c)
            total_tokens += tokens
            if len(final) >= self.settings.retrieval.final_top_n:
                break

        return {
            "ok": True,
            "query": original_query,
            "normalized_query": normalized,
            "rewritten_query": rewritten,
            "candidates": candidates,
            "final": final,
            "weak": weak,
            "top_rerank_score": float(top_rerank_score),
        }
