"""Text-only benchmark runner for RAG and brain comparison.

Uses /api/chat instead of the voice WebSocket. No STT/TTS needed.
Much faster: ~2-5 seconds per question instead of 10-50.

Usage:
    python scripts/benchmark_text.py --profile baseline --output results/text_baseline.jsonl
    python scripts/benchmark_text.py --profile brain_upgrade --output results/text_brain.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def run_question(
    backend_url: str,
    session_id: str,
    text: str,
    backend: str,
    brain_model: str,
) -> dict:
    resp = requests.post(
        backend_url.rstrip("/") + "/api/chat",
        json={
            "message": text,
            "backend": backend,
            "session_id": session_id,
            "brain_model": brain_model,
            "fast": True,
            "mode": "voice_fast",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Text-only benchmark runner")
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    fixture_path = root_dir / "fixtures" / "bench_questions_ru.jsonl"

    # Load env
    try:
        from dotenv import load_dotenv
        base_env = root_dir / ".env"
        if base_env.exists():
            load_dotenv(base_env, override=True)
        profile_env = root_dir / f".env.bench.{args.profile}"
        if profile_env.exists():
            load_dotenv(profile_env, override=True)
    except ImportError:
        pass

    backend = os.getenv("BENCH_BACKEND", "our_rag")
    brain_model = os.getenv("BENCH_BRAIN_MODEL", "Qwen/Qwen3-30B-A3B")

    output_path = args.output or (root_dir / "results" / f"text_{args.profile}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = []
    with fixture_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    total = len(questions)
    print(f"=== Text Benchmark: {args.profile} ===")
    print(f"  Backend:     {backend}")
    print(f"  Brain:       {brain_model}")
    print(f"  Questions:   {total}")
    print(f"  Output:      {output_path}")
    print()

    # Give consent once
    consent_sid = f"bench_{args.profile}_{int(time.time())}"
    requests.post(
        args.backend_url.rstrip("/") + "/api/chat",
        json={"message": "да, согласен", "session_id": consent_sid},
        timeout=15,
    )

    errors = 0
    latencies = []

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, q in enumerate(questions):
            qid = q["question_id"]
            text = q["text_ru"]
            expected = q.get("expected_keywords", [])
            warmup = idx < args.warmup

            label = f"[{idx+1}/{total}] {qid}"
            if warmup:
                label += " (warmup)"

            t0 = time.time()
            try:
                result = run_question(args.backend_url, consent_sid, text, backend, brain_model)
                t1 = time.time()
                elapsed_ms = (t1 - t0) * 1000

                answer = result.get("answer", "")
                chunks = len(result.get("used_knowledge", []))
                timings = result.get("timings", {})

                # Keyword match
                answer_lower = answer.lower()
                matched = [kw for kw in expected if kw.lower() in answer_lower]
                keyword_score = len(matched) / max(len(expected), 1)

                record = {
                    "question_id": qid,
                    "profile": args.profile,
                    "backend": backend,
                    "brain_model": brain_model,
                    "question": text,
                    "answer": answer,
                    "chunks_used": chunks,
                    "keyword_score": round(keyword_score, 3),
                    "keywords_matched": matched,
                    "keywords_expected": expected,
                    "total_ms": round(elapsed_ms, 1),
                    "embed_ms": round(timings.get("embed_ms", 0), 1),
                    "rerank_ms": round(timings.get("rerank_ms", 0), 1),
                    "rag_ms": round(timings.get("total_ms", 0), 1),
                    "llm_ms": round(timings.get("llm_total_ms", 0), 1),
                    "tokens_per_sec": round(timings.get("llm_tokens_per_sec", 0), 1),
                    "warmup": warmup,
                    "error": None,
                }

                if not warmup:
                    latencies.append(elapsed_ms)

                status = f"{elapsed_ms:.0f}ms | {chunks} chunks | kw:{keyword_score:.0%}"
                print(f"{label} -- {status}", flush=True)

            except Exception as exc:
                t1 = time.time()
                errors += 1
                record = {
                    "question_id": qid,
                    "profile": args.profile,
                    "error": str(exc),
                    "warmup": warmup,
                }
                print(f"{label} -- ERROR: {exc}", flush=True)

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    # Summary
    print()
    print("--- Summary ---")
    print(f"Total: {total} | Errors: {errors} | Warmup: {args.warmup}")
    if latencies:
        avg = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        print(f"Avg latency: {avg:.0f}ms | P50: {p50:.0f}ms")
        kw_scores = [json.loads(line).get("keyword_score", 0) for line in output_path.read_text().splitlines() if line.strip() and not json.loads(line).get("warmup")]
        if kw_scores:
            print(f"Avg keyword score: {sum(kw_scores)/len(kw_scores):.1%}")
    print(f"Results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
