"""FastAPI service that powers the optional KB-viz live overlay.

Three jobs:
  1. Serve the static kb_viz_{2d,3d}.html files produced by render_viz.py, so
     the client opens them over HTTPS from the same origin as the overlay
     endpoint (avoiding CORS complications from file:// origins).
  2. Expose POST /overlay_query which embeds a user's question, projects the
     embedding through the fitted UMAP reducer for the requested plot kind,
     and returns the top-K nearest KB chunks from Qdrant.
  3. Expose POST /feedback so the client can confirm whether the top-K
     chunks answered their question. Feedback is appended as a JSONL record
     (one line per submission) to .state/kb_viz_feedback.jsonl, matching the
     shape used by the existing self-improvement pipeline
     (.state/analysis/session_reports.jsonl). A future aggregation script
     can union the two sources.

Auth is optional. Set KB_VIZ_OVERLAY_TOKEN in the environment to require a
bearer token on /overlay_query and /feedback. Unset means open. No rate
limit; the user rebuilds the server daily so the attack surface is minimal.

Env vars:
    KB_VIZ_RESULTS_DIR    default: rag_demo_system/results
    KB_VIZ_STATE_DIR      default: rag_demo_system/.state
    KB_VIZ_EMBED_MODEL    default: intfloat/multilingual-e5-large
    KB_VIZ_EMBED_DEVICE   default: cpu
    KB_VIZ_QDRANT_URL     default: http://localhost:6333
    KB_VIZ_QDRANT_COLL    default: micro_leasing_kb
    KB_VIZ_OVERLAY_TOKEN  optional bearer token

Run:
    uvicorn rag_demo_system.services.kb_viz_service:app --host 0.0.0.0 --port 8500
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("kb_viz_service")

DEFAULT_RESULTS_DIR = Path("rag_demo_system/results")
DEFAULT_STATE_DIR = Path("rag_demo_system/.state")
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_EMBED_DEVICE = "cpu"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLL = "micro_leasing_kb"
DEFAULT_TOP_K = 5
TEXT_PREVIEW_CHARS = 180
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
PROFILES_FILE_NAME = "kb_viz_profiles.json"
PROFILE_NAME_MAX_CHARS = 128
FEEDBACK_LOG_NAME = "kb_viz_feedback.jsonl"
MAX_COMMENT_CHARS = 2000


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


class _State:
    """Lazily loaded heavy dependencies (model, reducers, qdrant client)."""

    def __init__(self) -> None:
        self.results_dir = Path(_env("KB_VIZ_RESULTS_DIR", str(DEFAULT_RESULTS_DIR)))
        self.state_dir = Path(_env("KB_VIZ_STATE_DIR", str(DEFAULT_STATE_DIR)))
        self.embed_model_name = _env("KB_VIZ_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.embed_device = _env("KB_VIZ_EMBED_DEVICE", DEFAULT_EMBED_DEVICE)
        self.qdrant_url = _env("KB_VIZ_QDRANT_URL", DEFAULT_QDRANT_URL)
        self.qdrant_coll = _env("KB_VIZ_QDRANT_COLL", DEFAULT_QDRANT_COLL)
        self.token = os.getenv("KB_VIZ_OVERLAY_TOKEN") or None
        self._lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._profiles_lock = threading.Lock()
        self._embedder: Any = None
        self._reducers: dict[str, Any] = {}
        self._qdrant: Any = None

    def feedback_log_path(self) -> Path:
        return self.state_dir / FEEDBACK_LOG_NAME

    def profiles_path(self) -> Path:
        return self.state_dir / PROFILES_FILE_NAME

    def append_feedback(self, record: dict[str, Any]) -> Path:
        """Append a feedback record to the JSONL log (thread-safe)."""
        path = self.feedback_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._feedback_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        return path

    def list_profiles(self) -> list[dict[str, Any]]:
        """Return stored profiles sorted by last_seen_ts desc."""
        path = self.profiles_path()
        if not path.exists():
            return []
        with self._profiles_lock:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
        profiles = data.get("profiles") or []
        return sorted(profiles, key=lambda p: p.get("last_seen_ts") or "", reverse=True)

    def upsert_profile(self, name: str) -> dict[str, Any]:
        """Create profile if new, else touch last_seen_ts. Thread-safe."""
        name = (name or "").strip()[:PROFILE_NAME_MAX_CHARS]
        if not name:
            raise ValueError("profile name required")
        path = self.profiles_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        with self._profiles_lock:
            data: dict[str, Any] = {"profiles": []}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    data = {"profiles": []}
            profiles = data.get("profiles") or []
            match = next((p for p in profiles if p.get("name") == name), None)
            if match:
                match["last_seen_ts"] = now
                result = match
            else:
                result = {"name": name, "created_ts": now, "last_seen_ts": now}
                profiles.append(result)
            data["profiles"] = profiles
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def get_embedder(self) -> Any:
        with self._lock:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    "Loading embedding model %s on %s", self.embed_model_name, self.embed_device
                )
                self._embedder = SentenceTransformer(
                    self.embed_model_name, device=self.embed_device
                )
        return self._embedder

    def get_reducer(self, kind: str) -> Any:
        with self._lock:
            if kind not in self._reducers:
                import joblib

                path = self.results_dir / f"umap_{kind}.joblib"
                if not path.exists():
                    raise FileNotFoundError(
                        f"UMAP reducer for '{kind}' not found at {path}. "
                        "Run render_viz.py first."
                    )
                self._reducers[kind] = joblib.load(path)
        return self._reducers[kind]

    def get_qdrant(self) -> Any:
        with self._lock:
            if self._qdrant is None:
                from qdrant_client import QdrantClient

                self._qdrant = QdrantClient(url=self.qdrant_url)
        return self._qdrant


STATE = _State()


class OverlayRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    kind: str = Field("3d", pattern="^(2d|3d)$")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    client_id: str | None = Field(default=None, max_length=128)


class OverlayMatch(BaseModel):
    chunk_id: str
    score: float
    section: str
    text_preview: str


class OverlayResponse(BaseModel):
    query_id: str
    kind: str
    position: list[float]
    top_k: list[OverlayMatch]


class FeedbackMatchRef(BaseModel):
    """Lightweight reference back to a match shown in the overlay.

    The client echoes what it saw (chunk_id + section + score) so the
    feedback record is self-contained without a server-side session cache.
    """

    chunk_id: str = Field(..., max_length=256)
    section: str = Field("", max_length=256)
    score: float | None = None


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., min_length=1, max_length=64)
    query_text: str = Field(..., min_length=1, max_length=2000)
    kind: str = Field(..., pattern="^(2d|3d)$")
    verdict: str = Field(..., pattern="^(correct|wrong)$")
    comment: str | None = Field(default=None, max_length=MAX_COMMENT_CHARS)
    top_k: list[FeedbackMatchRef] = Field(default_factory=list, max_length=20)
    client_id: str | None = Field(default=None, max_length=128)


class FeedbackResponse(BaseModel):
    ok: bool
    log_path: str


class ChunkCoverage(BaseModel):
    correct: int
    wrong: int
    last_ts: str | None = None
    last_verdict: str | None = None
    section: str = ""


class UserCoverage(BaseModel):
    """Per-user aggregate so the UI can render distinct-colored traces."""
    correct_chunks: list[str] = Field(default_factory=list)
    wrong_chunks: list[str] = Field(default_factory=list)
    total_events: int = 0


class CoverageResponse(BaseModel):
    total_feedback: int
    unique_chunks_validated: int
    per_chunk: dict[str, ChunkCoverage]
    per_section: dict[str, dict[str, int]]
    per_user: dict[str, UserCoverage] = Field(default_factory=dict)


class ProfileRecord(BaseModel):
    name: str
    created_ts: str
    last_seen_ts: str


class ProfilesResponse(BaseModel):
    profiles: list[ProfileRecord]


class ProfileUpsertRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=PROFILE_NAME_MAX_CHARS)


app = FastAPI(title="KB Viz Overlay Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


def _require_token(authorization: str | None) -> None:
    if not STATE.token:
        return
    expected = f"Bearer {STATE.token}"
    provided = authorization or ""
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def _embed_query(text: str) -> list[float]:
    embedder = STATE.get_embedder()
    vecs = embedder.encode([QUERY_PREFIX + text], normalize_embeddings=True)
    if hasattr(vecs, "tolist"):
        vecs = vecs.tolist()
    return list(vecs[0])


def _project(vector: list[float], kind: str) -> list[float]:
    import numpy as np

    reducer = STATE.get_reducer(kind)
    arr = np.asarray([vector], dtype=np.float32)
    coords = reducer.transform(arr)
    return [float(x) for x in coords[0]]


def _top_k(vector: list[float], k: int) -> list[OverlayMatch]:
    client = STATE.get_qdrant()
    try:
        results = client.search(
            collection_name=STATE.qdrant_coll,
            query_vector=vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
    except AttributeError:
        res = client.query_points(
            collection_name=STATE.qdrant_coll,
            query=vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        results = getattr(res, "points", res)

    out: list[OverlayMatch] = []
    for hit in results:
        payload = dict(hit.payload or {})
        heading = payload.get("heading_path") or []
        section = heading[0] if heading else ""
        text = str(payload.get("text", ""))
        preview = text if len(text) <= TEXT_PREVIEW_CHARS else text[:TEXT_PREVIEW_CHARS].rstrip() + "…"
        out.append(
            OverlayMatch(
                chunk_id=str(payload.get("chunk_id", hit.id)),
                score=float(hit.score or 0.0),
                section=str(section),
                text_preview=preview,
            )
        )
    return out


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "kb_viz",
        "results_dir": str(STATE.results_dir),
        "embed_model": STATE.embed_model_name,
        "embed_device": STATE.embed_device,
        "qdrant_url": STATE.qdrant_url,
        "token_required": STATE.token is not None,
    }


@app.get("/")
def serve_2d() -> Any:
    path = STATE.results_dir / "kb_viz_2d.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Not generated: {path}")
    return FileResponse(path, media_type="text/html")


@app.get("/3d")
def serve_3d() -> Any:
    path = STATE.results_dir / "kb_viz_3d.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Not generated: {path}")
    return FileResponse(path, media_type="text/html")


@app.get("/profiles", response_model=ProfilesResponse)
def list_profiles(authorization: str | None = Header(default=None)) -> ProfilesResponse:
    """List known profile names so returning users can pick instead of retyping.

    Gated behind the same bearer as write endpoints when a token is set, so
    a cross-origin page cannot silently harvest participant names.
    """
    _require_token(authorization)
    return ProfilesResponse(profiles=[ProfileRecord(**p) for p in STATE.list_profiles()])


@app.post("/profiles", response_model=ProfileRecord)
def upsert_profile(
    payload: ProfileUpsertRequest,
    authorization: str | None = Header(default=None),
) -> ProfileRecord:
    """Create a new profile or touch an existing one."""
    _require_token(authorization)
    try:
        rec = STATE.upsert_profile(payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return ProfileRecord(**rec)


def _maybe_touch_profile(name: str | None) -> None:
    """Best-effort profile touch. Never fails the caller."""
    if not name or not name.strip():
        return
    try:
        STATE.upsert_profile(name)
    except Exception:
        logger.exception("profile touch failed for %s", name)


@app.post("/overlay_query", response_model=OverlayResponse)
def overlay_query(
    payload: OverlayRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> OverlayResponse:
    _require_token(authorization)

    try:
        vec = _embed_query(payload.text)
    except Exception as exc:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=503, detail=f"Embedding failed: {exc}")

    try:
        position = _project(vec, payload.kind)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Projection failed")
        raise HTTPException(status_code=500, detail=f"Projection failed: {exc}")

    try:
        matches = _top_k(vec, payload.top_k)
    except Exception as exc:
        logger.exception("Qdrant search failed")
        raise HTTPException(status_code=503, detail=f"Qdrant search failed: {exc}")

    _maybe_touch_profile(payload.client_id)

    return OverlayResponse(
        query_id=uuid.uuid4().hex,
        kind=payload.kind,
        position=position,
        top_k=matches,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(
    payload: FeedbackRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> FeedbackResponse:
    _require_token(authorization)

    if payload.verdict == "wrong" and not (payload.comment and payload.comment.strip()):
        raise HTTPException(
            status_code=422,
            detail="comment is required when verdict is 'wrong'",
        )

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query_id": payload.query_id,
        "query_text": payload.query_text,
        "kind": payload.kind,
        "verdict": payload.verdict,
        "comment": (payload.comment or "").strip() or None,
        "top_k": [m.model_dump() for m in payload.top_k],
        "client_id": payload.client_id,
        "remote_addr": request.client.host if request.client else None,
    }
    path = STATE.append_feedback(record)
    _maybe_touch_profile(payload.client_id)
    return FeedbackResponse(ok=True, log_path=str(path))


def _compute_coverage() -> CoverageResponse:
    """Walk the feedback JSONL and tally per-chunk / per-section / per-user.

    Cheap enough to recompute on every request while the log stays small
    (one KB demo session is well under 500 entries). If the file grows
    large a cache + mtime check would be the obvious optimization.

    per_user lets the UI render one distinct-colored trace per validator so
    concurrent clients can see "user A validated these, user B those" at a
    glance without clicking into individual chunks.
    """
    log = STATE.feedback_log_path()
    per_chunk: dict[str, ChunkCoverage] = {}
    per_section: dict[str, dict[str, int]] = {}
    per_user: dict[str, UserCoverage] = {}
    per_user_correct_sets: dict[str, set[str]] = {}
    per_user_wrong_sets: dict[str, set[str]] = {}
    total_feedback = 0

    if log.exists():
        for raw in log.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_feedback += 1
            verdict = rec.get("verdict", "")
            ts = rec.get("ts")
            user_key = _user_key(rec)
            if user_key not in per_user:
                per_user[user_key] = UserCoverage()
                per_user_correct_sets[user_key] = set()
                per_user_wrong_sets[user_key] = set()
            per_user[user_key].total_events += 1
            sections_this_record: set[str] = set()
            for m in rec.get("top_k") or []:
                cid = str(m.get("chunk_id", "")).strip()
                if not cid:
                    continue
                section = str(m.get("section", "")).strip()
                cov = per_chunk.get(cid)
                if cov is None:
                    cov = ChunkCoverage(correct=0, wrong=0, section=section)
                    per_chunk[cid] = cov
                if verdict == "correct":
                    cov.correct += 1
                    per_user_correct_sets[user_key].add(cid)
                elif verdict == "wrong":
                    cov.wrong += 1
                    per_user_wrong_sets[user_key].add(cid)
                cov.last_ts = ts
                cov.last_verdict = verdict or None
                if section and not cov.section:
                    cov.section = section
                sections_this_record.add(section or "Без раздела")

            for sec_key in sections_this_record:
                sec = per_section.setdefault(
                    sec_key, {"correct": 0, "wrong": 0, "unique_chunks": 0}
                )
                if verdict in ("correct", "wrong"):
                    sec[verdict] += 1

        seen_per_section: dict[str, set[str]] = {}
        for cid, cov in per_chunk.items():
            sec_key = cov.section or "Без раздела"
            seen_per_section.setdefault(sec_key, set()).add(cid)
        for sec_key, chunk_ids in seen_per_section.items():
            if sec_key not in per_section:
                per_section[sec_key] = {"correct": 0, "wrong": 0, "unique_chunks": 0}
            per_section[sec_key]["unique_chunks"] = len(chunk_ids)

        for user_key in per_user:
            per_user[user_key].correct_chunks = sorted(per_user_correct_sets[user_key])
            per_user[user_key].wrong_chunks = sorted(per_user_wrong_sets[user_key])

    return CoverageResponse(
        total_feedback=total_feedback,
        unique_chunks_validated=len(per_chunk),
        per_chunk=per_chunk,
        per_section=per_section,
        per_user=per_user,
    )


def _user_key(rec: dict[str, Any]) -> str:
    """Best-effort label for who authored a feedback record."""
    cid = rec.get("client_id")
    if isinstance(cid, str) and cid.strip():
        return cid.strip()
    addr = rec.get("remote_addr")
    if isinstance(addr, str) and addr.strip():
        return f"ip:{addr.strip()}"
    return "anon"


@app.get("/coverage", response_model=CoverageResponse)
def coverage(authorization: str | None = Header(default=None)) -> CoverageResponse:
    """Aggregate validation tallies. Gated behind the same bearer as writes
    so cross-origin pages cannot harvest participant names or IPs when a
    token is configured.
    """
    _require_token(authorization)
    return _compute_coverage()


def _main() -> int:
    import uvicorn

    port = int(os.getenv("KB_VIZ_PORT", "8500"))
    host = os.getenv("KB_VIZ_HOST", "0.0.0.0")
    uvicorn.run(
        "rag_demo_system.services.kb_viz_service:app",
        host=host,
        port=port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
