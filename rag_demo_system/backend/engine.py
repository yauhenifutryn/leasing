from __future__ import annotations

import json
import time
from pathlib import Path
import logging
from typing import Any

from rank_bm25 import BM25Okapi

from .cache import TTLCache, LRUCache
from .ingest import Chunk, build_chunks
from .query import load_abbreviations, normalize_query, expand_synonyms
from .rag import ensure_collection, search, upsert_chunks
from .rerank import Reranker
from .retrieval_utils import filter_vector_hits
from .settings import Settings, RetrievalConfig, RerankerConfig

logger = logging.getLogger("rag_demo")


def iter_batches(items: list[Chunk], batch_size: int) -> list[list[Chunk]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    out: list[list[Chunk]] = []
    for i in range(0, len(items), batch_size):
        out.append(items[i : i + batch_size])
    return out


def effective_retrieval(config: RetrievalConfig, fast: bool, voice_fast: bool = False) -> dict[str, int]:
    if voice_fast:
        return {
            "vector_top_k": config.voice_vector_top_k,
            "bm25_top_k": config.voice_bm25_top_k,
            "final_top_n": config.voice_final_top_n,
            "context_max_tokens": config.voice_context_max_tokens,
        }
    if not fast:
        return {
            "vector_top_k": config.vector_top_k,
            "bm25_top_k": config.bm25_top_k,
            "final_top_n": config.final_top_n,
            "context_max_tokens": config.context_max_tokens,
        }
    return {
        "vector_top_k": config.fast_vector_top_k,
        "bm25_top_k": config.fast_bm25_top_k,
        "final_top_n": config.fast_final_top_n,
        "context_max_tokens": config.fast_context_max_tokens,
    }


def should_rerank(config: RerankerConfig, voice_fast: bool) -> bool:
    # Reranking is fast (~20ms) and critical for quality. Always use it.
    return bool(config.enabled)


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
        self.embed_cache = LRUCache(max_size=512)
        self.query_cache = TTLCache(ttl_sec=45)
        self.rerank_cache = TTLCache(ttl_sec=45)

    def _get_embedder(self) -> Any:
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.settings.embedding.model_name, device=self.settings.embedding.device)
        return self.embedder

    def _get_reranker(self, allow_rerank: bool) -> Reranker | None:
        if not allow_rerank:
            return None
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
        from qdrant_client import QdrantClient

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

    def retrieve(self, query: str, fast: bool = False, voice_fast: bool = False, session_id: str | None = None) -> dict[str, Any]:
        original_query = query
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        normalized = normalize_query(query, self.abbrev)
        normalized = expand_synonyms(normalized)
        timings["normalize_ms"] = (time.perf_counter() - t0) * 1000
        rewritten = normalized
        session_key = session_id or "anon"
        cache_key = f"{session_key}:{rewritten}:fast={fast}:voice={voice_fast}"

        cached = self.query_cache.get(cache_key)
        if cached:
            cached["cache_hit"] = True
            cached["timings"] = {"cache_hit": 1.0, "total_ms": 0.0}
            return cached

        embedder = self._get_embedder()
        embed_key = f"{session_key}:{rewritten}"
        t_embed = time.perf_counter()
        cached_embed = self.embed_cache.get(embed_key)
        if cached_embed is None:
            q_vec = embedder.encode([f"query: {rewritten}"], normalize_embeddings=True)
            q_vec = q_vec.tolist() if hasattr(q_vec, "tolist") else q_vec
            self.embed_cache.set(embed_key, q_vec[0])
        else:
            q_vec = [cached_embed]
        timings["embed_ms"] = (time.perf_counter() - t_embed) * 1000

        from qdrant_client import QdrantClient

        client = QdrantClient(url=self.settings.qdrant.url)
        retrieval_cfg = effective_retrieval(self.settings.retrieval, fast, voice_fast=voice_fast)
        t_qdrant = time.perf_counter()
        vector_hits = search(client, self.settings.qdrant.collection, q_vec[0], retrieval_cfg["vector_top_k"])
        vector_hits = filter_vector_hits(vector_hits, self.settings.retrieval.score_threshold)
        timings["qdrant_ms"] = (time.perf_counter() - t_qdrant) * 1000

        self._ensure_bm25()
        bm25_hits: list[Chunk] = []
        if self.bm25 and self.bm25_chunks:
            t_bm25 = time.perf_counter()
            scores = self.bm25.get_scores(self._tokenize(rewritten))
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                : retrieval_cfg["bm25_top_k"]
            ]
            bm25_hits = [self.bm25_chunks[i] for i in top_idx]
            timings["bm25_ms"] = (time.perf_counter() - t_bm25) * 1000
        else:
            timings["bm25_ms"] = 0.0

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

        allow_rerank = should_rerank(self.settings.reranker, voice_fast=voice_fast)
        reranker = self._get_reranker(allow_rerank=allow_rerank)
        if reranker:
            t_rerank = time.perf_counter()
            rerank_key = normalized + ":" + ",".join(sorted([c["chunk_id"] for c in candidates]))
            cached_rerank = self.rerank_cache.get(rerank_key)
            if cached_rerank is None:
                rerank_scores = reranker.rerank(rewritten, candidates)
                score_map = {r.chunk_id: r.score for r in rerank_scores}
                self.rerank_cache.set(rerank_key, score_map)
            else:
                score_map = cached_rerank
            for c in candidates:
                c["rerank_score"] = float(score_map.get(c["chunk_id"], 0.0))
            timings["rerank_ms"] = (time.perf_counter() - t_rerank) * 1000
        else:
            if allow_rerank:
                if not self.settings.reranker.allow_no_rerank:
                    return {"ok": False, "error": "Reranker disabled and allow_no_rerank=false"}
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)
            timings["rerank_ms"] = 0.0

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
            filtered = candidates[: retrieval_cfg["final_top_n"]]
        for c in filtered:
            tokens = len(c.get("text", "").split())
            if total_tokens + tokens > retrieval_cfg["context_max_tokens"]:
                break
            final.append(c)
            total_tokens += tokens
            if len(final) >= retrieval_cfg["final_top_n"]:
                break

        timings["total_ms"] = (
            timings.get("normalize_ms", 0.0)
            + timings.get("embed_ms", 0.0)
            + timings.get("qdrant_ms", 0.0)
            + timings.get("bm25_ms", 0.0)
            + timings.get("rerank_ms", 0.0)
        )

        result = {
            "ok": True,
            "query": original_query,
            "normalized_query": normalized,
            "rewritten_query": rewritten,
            "candidates": candidates,
            "final": final,
            "weak": weak,
            "top_rerank_score": float(top_rerank_score),
            "timings": timings,
            "cache_hit": False,
        }
        self.query_cache.set(cache_key, result)
        return result
