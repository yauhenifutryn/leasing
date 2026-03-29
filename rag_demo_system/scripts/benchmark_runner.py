"""
Benchmark runner CLI.

Executes the benchmark question fixture via the voice WebSocket pipeline
and writes JSONL results to an output file.

Usage:
    python benchmark_runner.py \\
        --fixture fixtures/bench_questions_ru.jsonl \\
        --profile baseline \\
        --output results/bench_baseline_20260326.jsonl \\
        --ws-url ws://localhost:8787/ws/voice \\
        --backend-url http://localhost:8787 \\
        [--timeout 30] \\
        [--warmup 3]

The runner loads the base .env file then overlays the selected benchmark
profile (.env.bench.<name>) before reading BENCH_* variables.

Per D-05 (research): first N questions are flagged warmup=true and excluded
from aggregate statistics. Default N=3.

Per D-06: on timeout or disconnect, error is recorded and the runner
continues to the next question.

Per Pitfall 2: a fresh WebSocket connection is opened per question to
avoid session-state contamination between questions.

Per Pitfall 3: question text is converted to audio via POST /api/tts on
the backend before sending over WebSocket. The runner never calls a TTS
sidecar directly.

Per Pitfall 4: JSONL output uses ensure_ascii=False so Cyrillic text is
preserved as-is in the output file.

Per Pitfall 5: session.update is sent first and session.updated is awaited
before any audio is sent. This guarantees the correct stack_id.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
import websockets


# ---------------------------------------------------------------------------
# Pure helper functions (importable for unit testing without side effects)
# ---------------------------------------------------------------------------

def is_warmup(index: int, warmup_count: int) -> bool:
    """Return True if question at `index` (0-based) is a warmup question.

    The first `warmup_count` questions are warmup and excluded from stats.
    """
    return index < warmup_count


def compute_keyword_hits(
    answer: str,
    expected_keywords: list[str],
) -> tuple[list[str], float | None]:
    """Compute which expected keywords appear in the answer (case-insensitive).

    Returns (hits, rate) where:
    - hits: list of keywords found in the answer
    - rate: len(hits) / len(expected_keywords), or None if expected_keywords is empty
    """
    if not expected_keywords:
        return [], None
    answer_lower = answer.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    rate = len(hits) / len(expected_keywords)
    return hits, rate


def build_result_dict(
    *,
    question_id: str,
    stack_id: str,
    warmup: bool,
    transcript: str | None,
    answer: str | None,
    retrieved_chunks: list[Any],
    speech_stopped: float | None,
    stt_done: float | None,
    retrieval_done: float | None,
    llm_first_token: float | None,
    tts_first_chunk: float | None,
    playback_started: float | None,
    primary_kpi_ms: float | None,
    llm_ttfb_ms: float | None,
    keyword_hits: list[str],
    keyword_hit_rate: float | None,
    error: str | None,
) -> dict[str, Any]:
    """Build the canonical 17-field JSONL result record for one benchmark question."""
    return {
        "question_id": question_id,
        "stack_id": stack_id,
        "warmup": warmup,
        "transcript": transcript,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "speech_stopped": speech_stopped,
        "stt_done": stt_done,
        "retrieval_done": retrieval_done,
        "llm_first_token": llm_first_token,
        "tts_first_chunk": tts_first_chunk,
        "playback_started": playback_started,
        "primary_kpi_ms": primary_kpi_ms,
        "llm_ttfb_ms": llm_ttfb_ms,
        "keyword_hits": keyword_hits,
        "keyword_hit_rate": keyword_hit_rate,
        "error": error,
    }


def build_error_result(
    *,
    question_id: str,
    stack_id: str,
    warmup: bool,
    speech_stopped: float | None,
    error_message: str,
) -> dict[str, Any]:
    """Build a result record for a question that failed with an error.

    All timing fields are set to None; keyword fields are empty.
    """
    return build_result_dict(
        question_id=question_id,
        stack_id=stack_id,
        warmup=warmup,
        transcript=None,
        answer=None,
        retrieved_chunks=[],
        speech_stopped=speech_stopped,
        stt_done=None,
        retrieval_done=None,
        llm_first_token=None,
        tts_first_chunk=None,
        playback_started=None,
        primary_kpi_ms=None,
        llm_ttfb_ms=None,
        keyword_hits=[],
        keyword_hit_rate=None,
        error=error_message,
    )


def _synthesize_question_audio(backend_url: str, text: str, tts_provider: str) -> str:
    """Call POST /api/tts to convert question text to base64 audio.

    Returns the audio_b64 string. Raises on HTTP error.
    """
    url = backend_url.rstrip("/") + "/api/tts"
    resp = requests.post(
        url,
        json={"text": text, "tts_provider": tts_provider},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["audio_b64"]


async def _run_one_question(
    *,
    question: dict[str, Any],
    question_index: int,
    warmup_count: int,
    ws_url: str,
    backend_url: str,
    backend: str,
    brain_model: str,
    stt_provider: str,
    tts_provider: str,
    timeout_sec: float,
) -> dict[str, Any]:
    """Run a single benchmark question over a fresh WebSocket connection.

    Opens a new connection, sends session.update, synthesizes audio via
    POST /api/tts, sends the audio buffer, and collects events until
    response.done.
    """
    qid = question["question_id"]
    text = question["text_ru"]
    expected_keywords = question.get("expected_keywords", [])
    warmup = is_warmup(question_index, warmup_count)

    # Synthesize question text to audio before opening WebSocket
    audio_b64 = _synthesize_question_audio(backend_url, text, tts_provider)

    t_speech_stopped: float | None = None
    transcript: str | None = None
    answer_parts: list[str] = []
    retrieved_chunks: list[Any] = []
    timings: dict[str, Any] = {}
    stack_id = "unknown"

    async with websockets.connect(ws_url) as ws:
        # Step 1: Send session.update and wait for session.updated
        await ws.send(json.dumps({
            "type": "session.update",
            "backend": backend,
            "brain_model": brain_model,
            "stt_provider": stt_provider,
            "tts_provider": tts_provider,
        }))

        session_updated = await asyncio.wait_for(
            _recv_until(ws, "session.updated"),
            timeout=timeout_sec,
        )
        stack_id = session_updated.get("stack_id", "unknown")

        # Step 2: Record speech_stopped timestamp and send audio
        t_speech_stopped = time.time()
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Step 3: Collect events until response.done
        async for raw in _stream_until_done(ws, timeout=timeout_sec):
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "conversation.item.input_audio_transcription.completed":
                transcript = msg.get("transcription")

            elif msg_type == "response.output_text.delta":
                delta = msg.get("delta", "")
                if delta:
                    answer_parts.append(delta)

            elif msg_type == "response.done":
                retrieved_chunks = msg.get("used_knowledge", [])
                timings = msg.get("timings", {})
                # Extract answer from response.done if not already assembled
                if not answer_parts:
                    answer_parts.append(msg.get("answer", "") or "")
                break

    answer = "".join(answer_parts)

    # Extract timing fields from the backend's timings dict (from response.done event)
    stt_done = timings.get("t_stt_done")
    retrieval_done = timings.get("t_retrieval_done")
    llm_first_token = timings.get("t_llm_first_token")
    tts_first_chunk = timings.get("t_tts_first_chunk")
    playback_started = timings.get("t_playback_started")

    # Compute latency KPIs in milliseconds
    primary_kpi_ms: float | None = None
    if playback_started is not None and t_speech_stopped is not None:
        primary_kpi_ms = (playback_started - t_speech_stopped) * 1000.0

    llm_ttfb_ms: float | None = None
    if llm_first_token is not None and t_speech_stopped is not None:
        llm_ttfb_ms = (llm_first_token - t_speech_stopped) * 1000.0

    keyword_hits, keyword_hit_rate = compute_keyword_hits(answer, expected_keywords)

    return build_result_dict(
        question_id=qid,
        stack_id=stack_id,
        warmup=warmup,
        transcript=transcript,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        speech_stopped=t_speech_stopped,
        stt_done=stt_done,
        retrieval_done=retrieval_done,
        llm_first_token=llm_first_token,
        tts_first_chunk=tts_first_chunk,
        playback_started=playback_started,
        primary_kpi_ms=primary_kpi_ms,
        llm_ttfb_ms=llm_ttfb_ms,
        keyword_hits=keyword_hits,
        keyword_hit_rate=keyword_hit_rate,
        error=None,
    )


async def _recv_until(ws, msg_type: str) -> dict[str, Any]:
    """Receive WebSocket messages until one with the given type is found."""
    async for raw in ws:
        msg = json.loads(raw)
        if msg.get("type") == msg_type:
            return msg
    raise RuntimeError(f"WebSocket closed before receiving {msg_type!r}")


async def _stream_until_done(ws, timeout: float):
    """Async generator yielding raw WebSocket messages until response.done or timeout."""
    deadline = time.monotonic() + timeout
    async for raw in ws:
        yield raw
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError("Benchmark timeout waiting for response.done")
        msg = json.loads(raw)
        if msg.get("type") == "response.done":
            return


async def _run_benchmark(
    *,
    fixture_path: Path,
    ws_url: str,
    backend_url: str,
    output_path: Path,
    backend: str,
    brain_model: str,
    stt_provider: str,
    tts_provider: str,
    timeout_sec: float,
    warmup_count: int,
) -> None:
    """Core benchmark loop: execute each question and write JSONL results."""
    questions = []
    with fixture_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(questions)
    error_count = 0
    non_warmup_kpis: list[float] = []

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, question in enumerate(questions):
            qid = question["question_id"]
            warmup = is_warmup(idx, warmup_count)
            print(
                f"[{idx + 1}/{total}] {qid}"
                f"{' (warmup)' if warmup else ''}",
                flush=True,
            )

            t_speech_stopped = time.time()
            try:
                result = await asyncio.wait_for(
                    _run_one_question(
                        question=question,
                        question_index=idx,
                        warmup_count=warmup_count,
                        ws_url=ws_url,
                        backend_url=backend_url,
                        backend=backend,
                        brain_model=brain_model,
                        stt_provider=stt_provider,
                        tts_provider=tts_provider,
                        timeout_sec=timeout_sec,
                    ),
                    timeout=timeout_sec + 10,
                )
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR: {err_msg}", flush=True)
                error_count += 1
                result = build_error_result(
                    question_id=qid,
                    stack_id="unknown",
                    warmup=warmup,
                    speech_stopped=t_speech_stopped,
                    error_message=err_msg,
                )

            if not warmup and result["primary_kpi_ms"] is not None:
                non_warmup_kpis.append(result["primary_kpi_ms"])

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    # Print summary
    print("\n--- Benchmark Summary ---")
    print(f"Total questions: {total}")
    print(f"Warmup questions: {warmup_count}")
    print(f"Errors: {error_count}")
    if non_warmup_kpis:
        avg_kpi = sum(non_warmup_kpis) / len(non_warmup_kpis)
        print(f"Primary KPI mean (non-warmup): {avg_kpi:.1f} ms")
    else:
        print("Primary KPI: N/A (all results had errors or missing playback_started)")
    print(f"Results written to: {output_path}")


def _load_env_profile(base_env: Path, profile_name: str, root_dir: Path) -> None:
    """Load base .env then overlay the benchmark profile .env.bench.<name>."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Warning: python-dotenv not installed; env vars not loaded from .env files", file=sys.stderr)
        return

    if base_env.exists():
        load_dotenv(base_env)

    profile_env = root_dir / f".env.bench.{profile_name}"
    if profile_env.exists():
        load_dotenv(profile_env, override=True)
    else:
        print(f"Warning: profile file {profile_env} not found; using existing env vars", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the benchmark question fixture against a voice WebSocket backend and write JSONL results.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("fixtures/bench_questions_ru.jsonl"),
        help="Path to the JSONL benchmark fixture file (default: fixtures/bench_questions_ru.jsonl)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="baseline",
        help="Benchmark env profile name; loads .env.bench.<name> overlay (default: baseline)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file path (default: results/bench_<profile>_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default="ws://localhost:8787/ws/voice",
        help="WebSocket URL for the voice endpoint (default: ws://localhost:8787/ws/voice)",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8787",
        help="HTTP base URL for the backend REST API (default: http://localhost:8787)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-question timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup questions (default: 3)",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location if not absolute
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    fixture_path = args.fixture if args.fixture.is_absolute() else (root_dir / args.fixture)

    # Load env profile
    base_env = root_dir / ".env"
    _load_env_profile(base_env, args.profile, root_dir)

    # Read BENCH_* vars (with fallbacks from CLI defaults or env)
    backend = os.getenv("BENCH_BACKEND", "our_rag")
    brain_model = os.getenv("BENCH_BRAIN_MODEL", "Qwen/Qwen3-30B-A3B")
    stt_provider = os.getenv("BENCH_STT_PROVIDER", "whisper")
    tts_provider = os.getenv("BENCH_TTS_PROVIDER", "qwen3_tts")

    # Determine output path
    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else (root_dir / args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = root_dir / "results" / f"bench_{args.profile}_{timestamp}.jsonl"

    print("=== Benchmark Runner ===")
    print(f"  Fixture:     {fixture_path}")
    print(f"  Profile:     {args.profile}")
    print(f"  Backend:     {backend}")
    print(f"  Brain model: {brain_model}")
    print(f"  STT:         {stt_provider}")
    print(f"  TTS:         {tts_provider}")
    print(f"  WS URL:      {args.ws_url}")
    print(f"  Backend URL: {args.backend_url}")
    print(f"  Output:      {output_path}")
    print(f"  Warmup:      {args.warmup}")
    print(f"  Timeout:     {args.timeout}s")
    print()

    asyncio.run(
        _run_benchmark(
            fixture_path=fixture_path,
            ws_url=args.ws_url,
            backend_url=args.backend_url,
            output_path=output_path,
            backend=backend,
            brain_model=brain_model,
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            timeout_sec=args.timeout,
            warmup_count=args.warmup,
        )
    )
