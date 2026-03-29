"""Text-quality benchmark for Qwen3-Omni hybrid mode.

Sends questions through: RAG retrieval (backend) -> Omni /chat (with context chunks).
Omni produces text + audio but we only evaluate the text answer quality.
No STT needed (we send text, not audio).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def retrieve_chunks(backend_url: str, question: str) -> list[str]:
    """Use the backend's RAG retrieval endpoint (no LLM needed)."""
    resp = requests.post(
        backend_url.rstrip("/") + "/api/retrieve",
        json={"query": question},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    final = data.get("final") or []
    return [c.get("text", "") for c in final if isinstance(c, dict)]


def ask_omni(omni_url: str, context_chunks: list[str], question: str) -> dict:
    """Send question + context to Omni and get text response."""
    # Omni expects audio_b64 but for text-only benchmark we send empty audio
    # and put the question in a context chunk so Omni has the text
    chunks_with_question = context_chunks + [f"Вопрос клиента: {question}"]
    resp = requests.post(
        omni_url.rstrip("/") + "/chat",
        json={
            "audio_b64": "",
            "context_chunks": chunks_with_question,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Omni text-quality benchmark")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--omni-url", default="http://localhost:8002")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    fixture_path = root_dir / "fixtures" / "bench_questions_ru.jsonl"
    output_path = args.output or (root_dir / "results" / f"text_omni_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = []
    with fixture_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    total = len(questions)
    print(f"=== Omni Text Benchmark ===")
    print(f"  Backend:   {args.backend_url} (RAG retrieval)")
    print(f"  Omni:      {args.omni_url}")
    print(f"  Questions: {total}")
    print(f"  Output:    {output_path}")
    print()

    # No consent needed: /api/retrieve doesn't go through consent flow

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
                # Step 1: RAG retrieval via backend
                chunks = retrieve_chunks(args.backend_url, text)

                # Step 2: Send to Omni with context
                omni_result = ask_omni(args.omni_url, chunks, text)
                t1 = time.time()
                elapsed_ms = (t1 - t0) * 1000

                answer = omni_result.get("text", "")
                has_audio = bool(omni_result.get("audio_b64", ""))

                # Keyword match
                answer_lower = answer.lower()
                matched = [kw for kw in expected if kw.lower() in answer_lower]
                keyword_score = len(matched) / max(len(expected), 1)

                record = {
                    "question_id": qid,
                    "profile": "omni_hybrid",
                    "backend": "our_rag",
                    "brain_model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    "question": text,
                    "answer": answer,
                    "chunks_used": len(chunks),
                    "has_audio": has_audio,
                    "keyword_score": round(keyword_score, 3),
                    "keywords_matched": matched,
                    "keywords_expected": expected,
                    "total_ms": round(elapsed_ms, 1),
                    "warmup": warmup,
                    "error": None,
                }

                if not warmup:
                    latencies.append(elapsed_ms)

                status = f"{elapsed_ms:.0f}ms | {len(chunks)} chunks | kw:{keyword_score:.0%} | audio:{'yes' if has_audio else 'no'}"
                print(f"{label} -- {status}", flush=True)

            except Exception as exc:
                t1 = time.time()
                errors += 1
                record = {
                    "question_id": qid,
                    "profile": "omni_hybrid",
                    "error": str(exc),
                    "warmup": warmup,
                }
                print(f"{label} -- ERROR: {exc}", flush=True)

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print()
    print("--- Summary ---")
    print(f"Total: {total} | Errors: {errors} | Warmup: {args.warmup}")
    if latencies:
        avg = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        print(f"Avg latency: {avg:.0f}ms | P50: {p50:.0f}ms")
    print(f"Results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
