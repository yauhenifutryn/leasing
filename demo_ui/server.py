#!/usr/bin/env python3
import argparse
import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATE_DIR = Path(__file__).resolve().parent / ".state"


def load_dotenv_if_present(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key in os.environ:
            continue
        os.environ[key] = value


# Load `.env` (gitignored) so demo UI works without manual `export ...`
load_dotenv_if_present(REPO_ROOT / ".env")

STATE_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def safe_join(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise ValueError("Path escapes root")
    return candidate


ALLOWED_READ_DIRS = {
    "insights_per_call": REPO_ROOT / "insights_per_call",
    "insights_global": REPO_ROOT / "insights_global",
    "knowledge_base": REPO_ROOT / "knowledge_base",
    "nlu_output": REPO_ROOT / "nlu_output",
    "transcripts_clean": REPO_ROOT / "transcripts_clean",
}

ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac"}


WHITELIST_TASKS: dict[str, list[str]] = {
    "check": ["make", "check"],
    "analyze_calls": ["make", "analyze-calls"],
    "nlu_export": ["make", "nlu-export"],
    "rollup": ["make", "rollup"],
    "aggregate": ["make", "aggregate"],
    "dedup": ["make", "dedup"],
    "kb": ["make", "kb"],
    "kb_markdown": ["make", "kb-markdown"],
    # Optional: can be slow locally; keep behind explicit click in UI
    "transcribe": ["make", "transcribe"],
    # Optional: start streamlit review UI
    "review_ui": [sys.executable, "-m", "streamlit", "run", "scripts/review_app.py", "--server.port", "8501"],
}


@dataclass
class Session:
    id: str
    created_at: str
    name: str
    notes: str = ""
    audio_files: list[str] = field(default_factory=list)


@dataclass
class Run:
    id: str
    session_id: str
    task: str
    command: list[str]
    status: str  # queued|running|success|failed|stopped
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    log_path: str | None = None


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sessions_path = root / "sessions.json"
        self.runs_path = root / "runs.jsonl"
        self.feedback_path = root / "feedback.jsonl"
        self._lock = threading.Lock()
        self.sessions: dict[str, Session] = {}
        self.runs: dict[str, Run] = {}
        self._load()

    def _load(self) -> None:
        if self.sessions_path.exists():
            payload = json.loads(self.sessions_path.read_text(encoding="utf-8"))
            for item in payload:
                sess = Session(
                    id=item.get("id", ""),
                    created_at=item.get("created_at", now_iso()),
                    name=item.get("name", "Session"),
                    notes=item.get("notes", ""),
                    audio_files=item.get("audio_files") or [],
                )
                self.sessions[sess.id] = sess
        if self.runs_path.exists():
            for line in self.runs_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                run = Run(**item)
                self.runs[run.id] = run

    def _persist_sessions(self) -> None:
        self.sessions_path.write_text(json.dumps([asdict(s) for s in self.sessions.values()], ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_run(self, run: Run) -> None:
        with self.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(run), ensure_ascii=False))
            handle.write("\n")

    def create_session(self, name: str) -> Session:
        with self._lock:
            sess = Session(id=str(uuid.uuid4()), created_at=now_iso(), name=name)
            self.sessions[sess.id] = sess
            self._persist_sessions()
            return sess

    def set_session_audio(self, session_id: str, files: list[str]) -> None:
        with self._lock:
            sess = self.sessions.get(session_id)
            if sess is None:
                return
            sess.audio_files = files
            self._persist_sessions()

    def list_sessions(self) -> list[Session]:
        with self._lock:
            return sorted(self.sessions.values(), key=lambda s: s.created_at, reverse=True)

    def create_run(self, session_id: str, task: str, command: list[str]) -> Run:
        log_dir = self.root / "logs" / session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(uuid.uuid4())
        log_path = log_dir / f"{run_id}.log"
        run = Run(
            id=run_id,
            session_id=session_id,
            task=task,
            command=command,
            status="queued",
            created_at=now_iso(),
            log_path=str(log_path),
        )
        with self._lock:
            self.runs[run.id] = run
            self._append_run(run)
        return run

    def update_run(self, run: Run) -> None:
        with self._lock:
            self.runs[run.id] = run
            self._append_run(run)

    def get_run(self, run_id: str) -> Run | None:
        with self._lock:
            return self.runs.get(run_id)

    def list_runs(self, session_id: str | None = None, limit: int = 50) -> list[Run]:
        with self._lock:
            runs = list(self.runs.values())
        if session_id:
            runs = [r for r in runs if r.session_id == session_id]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[:limit]

    def add_feedback(self, session_id: str | None, payload: dict[str, Any]) -> None:
        record = {
            "ts": now_iso(),
            "session_id": session_id,
            **payload,
        }
        with self.feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


class Runner:
    def __init__(self, store: StateStore) -> None:
        self.store = store
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def enqueue(self, run_id: str) -> None:
        self.queue.put(run_id)

    def stop(self, run_id: str) -> bool:
        with self._lock:
            proc = self.processes.get(run_id)
        if proc is None:
            return False
        try:
            proc.terminate()
            return True
        except Exception:
            return False

    def _worker(self) -> None:
        while True:
            run_id = self.queue.get()
            run = self.store.get_run(run_id)
            if run is None:
                continue
            run.status = "running"
            run.started_at = now_iso()
            self.store.update_run(run)

            log_path = Path(run.log_path) if run.log_path else (STATE_DIR / "logs" / f"{run.id}.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"[{now_iso()}] START {run.task}: {' '.join(run.command)}\n")
                log.flush()
                proc = subprocess.Popen(
                    run.command,
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={**os.environ},
                )
                with self._lock:
                    self.processes[run.id] = proc
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                finally:
                    return_code = proc.wait()
                    with self._lock:
                        self.processes.pop(run.id, None)

            run.return_code = return_code
            run.finished_at = now_iso()
            if return_code == 0:
                run.status = "success"
            elif return_code == -15:
                run.status = "stopped"
            else:
                run.status = "failed"
            self.store.update_run(run)


STORE = StateStore(STATE_DIR)
RUNNER = Runner(STORE)


class Handler(BaseHTTPRequestHandler):
    server_version = "LeasingDemoUI/0.1"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, "application/json; charset=utf-8", json_bytes(payload))

    def _bad(self, message: str, status: int = 400) -> None:
        self._json(status, {"error": message})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = (STATIC_DIR / "index.html").read_bytes()
            return self._send(200, "text/html; charset=utf-8", body)
        if parsed.path.startswith("/static/"):
            rel = parsed.path.removeprefix("/static/")
            fp = safe_join(STATIC_DIR, rel)
            if not fp.exists() or fp.is_dir():
                return self._send(404, "text/plain; charset=utf-8", b"Not found")
            ctype = "application/octet-stream"
            if fp.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            elif fp.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            elif fp.suffix == ".svg":
                ctype = "image/svg+xml; charset=utf-8"
            return self._send(200, ctype, fp.read_bytes())

        if parsed.path == "/api/health":
            return self._json(200, {"ok": True, "ts": now_iso()})
        if parsed.path == "/api/sessions":
            sessions = [asdict(s) for s in STORE.list_sessions()]
            return self._json(200, {"sessions": sessions})
        if parsed.path == "/api/runs":
            qs = parse_qs(parsed.query)
            session_id = (qs.get("session_id") or [None])[0]
            runs = [asdict(r) for r in STORE.list_runs(session_id=session_id)]
            return self._json(200, {"runs": runs})
        if parsed.path == "/api/tasks":
            return self._json(200, {"tasks": sorted(WHITELIST_TASKS.keys())})
        if parsed.path == "/api/env":
            env = {
                "OPENAI_MODEL": os.getenv("OPENAI_MODEL", ""),
                "REVIEW_OPENAI_MODEL": os.getenv("REVIEW_OPENAI_MODEL", ""),
                "OPENAI_API_KEY": "SET" if os.getenv("OPENAI_API_KEY") else "MISSING",
                "HUGGINGFACE_TOKEN": "SET" if os.getenv("HUGGINGFACE_TOKEN") else "MISSING",
            }
            return self._json(200, {"env": env})
        if parsed.path == "/api/audio":
            audio_dir = REPO_ROOT / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            files: list[dict[str, Any]] = []
            for p in sorted(audio_dir.iterdir()):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in ALLOWED_AUDIO_EXTS:
                    continue
                st = p.stat()
                files.append({"name": p.name, "size": st.st_size, "mtime": int(st.st_mtime)})
            return self._json(200, {"files": files})
        if parsed.path == "/api/metrics":
            return self._json(200, {"metrics": compute_metrics()})
        if parsed.path == "/api/log":
            qs = parse_qs(parsed.query)
            run_id = (qs.get("run_id") or [None])[0]
            if not run_id:
                return self._bad("run_id required")
            run = STORE.get_run(run_id)
            if run is None or not run.log_path:
                return self._bad("unknown run_id", 404)
            fp = Path(run.log_path)
            if not fp.exists():
                return self._json(200, {"lines": []})
            limit = int((qs.get("limit") or ["300"])[0])
            lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
            return self._json(200, {"lines": lines})
        if parsed.path == "/api/files":
            qs = parse_qs(parsed.query)
            kind = (qs.get("kind") or [None])[0]
            if kind not in ALLOWED_READ_DIRS:
                return self._bad("invalid kind", 400)
            root = ALLOWED_READ_DIRS[kind]
            if not root.exists():
                return self._json(200, {"files": []})
            files = sorted([p.name for p in root.glob("*.json")])[:1000]
            return self._json(200, {"files": files})
        if parsed.path == "/api/file":
            qs = parse_qs(parsed.query)
            kind = (qs.get("kind") or [None])[0]
            name = (qs.get("name") or [None])[0]
            if kind not in ALLOWED_READ_DIRS or not name:
                return self._bad("kind and name required")
            root = ALLOWED_READ_DIRS[kind]
            fp = safe_join(root, name)
            if not fp.exists():
                return self._bad("not found", 404)
            if fp.suffix != ".json":
                return self._bad("only .json allowed", 400)
            text = fp.read_text(encoding="utf-8", errors="ignore")
            return self._json(200, {"name": name, "text": text})

        return self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except Exception:
            return self._bad("invalid json")

        if parsed.path == "/api/session":
            name = (payload.get("name") or "").strip() or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            sess = STORE.create_session(name=name)
            return self._json(200, {"session": asdict(sess)})

        if parsed.path == "/api/run":
            session_id = payload.get("session_id")
            task = payload.get("task")
            if not session_id or session_id not in STORE.sessions:
                return self._bad("invalid session_id")
            if task not in WHITELIST_TASKS:
                return self._bad("invalid task")
            command = list(WHITELIST_TASKS[task])
            if command and command[0] == "make":
                command = ["make", f"PY={sys.executable}", *command[1:]]
            run = STORE.create_run(session_id=session_id, task=task, command=command)
            RUNNER.enqueue(run.id)
            return self._json(200, {"run": asdict(run)})

        if parsed.path == "/api/session/audio":
            session_id = payload.get("session_id")
            files = payload.get("files")
            if not session_id or session_id not in STORE.sessions:
                return self._bad("invalid session_id")
            if not isinstance(files, list):
                return self._bad("files must be a list")
            audio_dir = REPO_ROOT / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            cleaned: list[str] = []
            for name in files:
                if not isinstance(name, str):
                    continue
                if "/" in name or "\\" in name:
                    continue
                p = audio_dir / name
                if not p.exists():
                    continue
                if p.suffix.lower() not in ALLOWED_AUDIO_EXTS:
                    continue
                cleaned.append(name)
            STORE.set_session_audio(session_id=session_id, files=cleaned)
            return self._json(200, {"ok": True, "files": cleaned})

        if parsed.path == "/api/audio/upload":
            files = payload.get("files")
            if not isinstance(files, list) or not files:
                return self._bad("files must be a non-empty list")
            audio_dir = REPO_ROOT / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            saved: list[str] = []
            for item in files:
                if not isinstance(item, dict):
                    continue
                name = (item.get("name") or "").strip()
                data_b64 = (item.get("data_base64") or "").strip()
                if not name or "/" in name or "\\" in name:
                    continue
                if Path(name).suffix.lower() not in ALLOWED_AUDIO_EXTS:
                    continue
                if not data_b64:
                    continue
                try:
                    raw = base64.b64decode(data_b64, validate=True)
                except Exception:
                    continue
                if len(raw) > 200 * 1024 * 1024:
                    continue
                (audio_dir / name).write_bytes(raw)
                saved.append(name)
            return self._json(200, {"ok": True, "saved": saved})

        if parsed.path == "/api/stop":
            run_id = payload.get("run_id")
            if not run_id:
                return self._bad("run_id required")
            ok = RUNNER.stop(run_id)
            return self._json(200, {"ok": ok})

        if parsed.path == "/api/feedback":
            session_id = payload.get("session_id")
            message = (payload.get("message") or "").strip()
            if not message:
                return self._bad("message required")
            STORE.add_feedback(session_id=session_id, payload={"message": message})
            return self._json(200, {"ok": True})

        return self._send(404, "text/plain; charset=utf-8", b"Not found")


def compute_metrics(max_files: int = 5000) -> dict[str, Any]:
    per_call_dir = REPO_ROOT / "insights_per_call"
    files = sorted(per_call_dir.glob("*.json")) if per_call_dir.exists() else []
    files = files[:max_files]

    def norm_status(s: Any) -> str:
        val = str(s or "").strip().lower()
        if val in {"resolved", "fully_resolved", "solved"}:
            return "resolved"
        if val in {"partially_resolved", "partial", "partially"}:
            return "partial"
        if val in {"unresolved", "not_resolved", "not resolved", "failed"}:
            return "unresolved"
        return "unknown"

    resolution = {"resolved": 0, "partial": 0, "unresolved": 0, "unknown": 0}
    emotions: dict[str, int] = {}
    quality_flags: dict[str, int] = {}
    unresolved_reasons: dict[str, int] = {}

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        st = norm_status(data.get("resolution_status"))
        resolution[st] = resolution.get(st, 0) + 1

        emo = data.get("emotions") or {}
        if isinstance(emo, dict):
            client_emo = (emo.get("client") or "").strip().lower()
            if client_emo:
                emotions[client_emo] = emotions.get(client_emo, 0) + 1

        qf = data.get("quality_flags") or []
        if isinstance(qf, list):
            for x in qf:
                if isinstance(x, str) and x.strip():
                    key = x.strip()
                    quality_flags[key] = quality_flags.get(key, 0) + 1

        if st in {"unresolved", "partial"}:
            reason = ""
            handoff = data.get("handoff")
            if isinstance(handoff, str):
                reason = handoff.strip()
            elif isinstance(handoff, dict):
                for k in ("reason", "type", "needed"):
                    v = handoff.get(k)
                    if isinstance(v, str) and v.strip():
                        reason = v.strip()
                        break
            if not reason and isinstance(qf, list) and qf:
                first = qf[0]
                if isinstance(first, str) and first.strip():
                    reason = first.strip()
            if not reason:
                mi = data.get("main_issue")
                if isinstance(mi, str) and mi.strip():
                    reason = mi.strip()
            if reason:
                unresolved_reasons[reason] = unresolved_reasons.get(reason, 0) + 1

    top_reasons: list[dict[str, Any]] = []
    top_path = REPO_ROOT / "insights_global" / "global_top_intents.json"
    if top_path.exists():
        try:
            payload = json.loads(top_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for row in payload[:20]:
                    if not isinstance(row, dict):
                        continue
                    label = row.get("intent") or row.get("label") or row.get("name")
                    value = row.get("count") or row.get("value")
                    if isinstance(label, str) and isinstance(value, int):
                        top_reasons.append({"label": label, "value": value})
        except Exception:
            pass

    def top_n(d: dict[str, int], n: int = 10) -> list[dict[str, Any]]:
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [{"label": k, "value": v} for k, v in items]

    return {
        "total_calls": sum(resolution.values()),
        "resolution": resolution,
        "top_reasons": top_reasons,
        "quality_flags_top": top_n(quality_flags, 10),
        "emotions_top": top_n(emotions, 8),
        "unresolved_reasons_top": top_n(unresolved_reasons, 10),
        "note": f"Computed from {min(len(files), max_files)} per-call JSON files.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("DEMO_UI_PORT", "8787")))
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[demo_ui] Serving on http://{args.host}:{args.port} (repo: {REPO_ROOT})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
