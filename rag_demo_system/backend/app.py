from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .citations import attach_citations
from .text_utils import clean_answer, iter_final_text
from .consent import (
    consent_denied_response,
    consent_granted_response,
    consent_request,
    detect_consent,
)
from .engine import RAGEngine
from .settings import load_settings
from .state import StateStore
from .router import route_non_rag

settings = load_settings()
state = StateStore(Path(__file__).resolve().parents[1] / ".state")
engine = RAGEngine(settings, Path(__file__).resolve().parents[1] / ".state")

app = FastAPI(title="Micro Leasing RAG Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    rebuild: bool = False


class RetrieveRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    stream: bool = False


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "kb": str(settings.app.kb_markdown_path)}


@app.post("/api/index")
async def index_kb(payload: IndexRequest) -> dict[str, Any]:
    result = engine.index()
    if not result.get("ok"):
        return result
    state.log({"event": "index", "chunks": result.get("chunks")})
    return result


@app.post("/api/retrieve")
async def retrieve(payload: RetrieveRequest) -> dict[str, Any]:
    query = payload.query.strip()
    if not query:
        return {"ok": False, "error": "empty query"}
    result = engine.retrieve(query)
    return result


@app.post("/api/chat")
async def chat(payload: ChatRequest, stream: bool = False) -> Any:
    message = payload.message.strip()
    if not message:
        return {"ok": False, "error": "empty message"}

    session_id = payload.session_id or str(uuid.uuid4())
    session = state.get(session_id) or state.create(session_id)

    decision = detect_consent(message)
    if session.consent_denied:
        response = {
            "ok": True,
            "session_id": session_id,
            "answer": consent_denied_response(),
            "consent": "denied",
            "chunks": [],
            "citations": [],
        }
        return _stream_or_json(response, stream)

    if not session.consent_given:
        if decision == "denied":
            session.consent_denied = True
            state.update(session)
            response = {
                "ok": True,
                "session_id": session_id,
                "answer": consent_denied_response(),
                "consent": "denied",
                "chunks": [],
                "citations": [],
            }
            return _stream_or_json(response, stream)
        if decision == "granted":
            session.consent_given = True
            state.update(session)
            response = {
                "ok": True,
                "session_id": session_id,
                "answer": consent_granted_response(),
                "consent": "granted",
                "chunks": [],
                "citations": [],
            }
            return _stream_or_json(response, stream)
        response = {
            "ok": True,
            "session_id": session_id,
            "answer": consent_request(),
            "consent": "needed",
            "chunks": [],
            "citations": [],
        }
        return _stream_or_json(response, stream)

    routed = route_non_rag(message, settings.llm.base_url, settings.llm.model)
    if routed:
        state.log({"event": "router", "kind": routed.kind, "session_id": session_id})
        response = {
            "ok": True,
            "session_id": session_id,
            "answer": routed.response,
            "consent": "granted",
            "chunks": [],
            "citations": [],
        }
        return _stream_or_json(response, stream)

    retrieval = engine.retrieve(message)
    if not retrieval.get("ok"):
        response = {
            "ok": True,
            "session_id": session_id,
            "answer": settings.app.strict_refusal_text,
            "consent": "granted",
            "chunks": [],
            "citations": [],
        }
        return _stream_or_json(response, stream)

    final_chunks = retrieval.get("final") or []
    weak_context = bool(retrieval.get("weak"))
    if not final_chunks:
        state.log({"event": "no_context", "query": message, "session_id": session_id})
        response = {
            "ok": True,
            "session_id": session_id,
            "answer": settings.app.strict_refusal_text,
            "consent": "granted",
            "chunks": [],
            "citations": [],
        }
        return _stream_or_json(response, stream)

    system_prompt = settings.app.system_prompt_path.read_text(encoding="utf-8")
    context_block = "\n\n".join(
        [f"[Fragment {i+1}]\n{c['text']}" for i, c in enumerate(final_chunks)]
    )

    expanded = any(trigger in message.lower() for trigger in settings.llm.expand_triggers)
    length_hint = (
        f"Ответ должен быть {settings.llm.concise_sentences_min}–{settings.llm.concise_sentences_max} коротких предложений."
        if not expanded
        else "Можно ответить подробнее, но только на основе контекста."
    )

    weak_hint = ""
    if weak_context:
        weak_hint = (
            "Контекст может быть неполным. Дай ближайшую релевантную информацию из фрагментов, "
            "скажи, что точных данных может не хватать, и задай уточняющий вопрос.\n\n"
        )
    user_prompt = (
        "Ответь строго на основе следующих фрагментов базы знаний. "
        "Если ответа нет — верни точный отказ.\n\n"
        f"{length_hint}\n\n"
        f"{weak_hint}{context_block}\n\nВопрос клиента: {message}"
    )

    from .llm import call_openai_compatible, iter_openai_compatible_stream

    if stream:
        def gen() -> Any:
            streamed_parts: list[str] = []
            had_final = False
            try:
                stream_iter = iter_openai_compatible_stream(
                    base_url=settings.llm.base_url,
                    model=settings.llm.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=settings.llm.temperature,
                    max_tokens=settings.llm.max_tokens,
                    timeout_sec=settings.llm.timeout_sec,
                )
                for chunk in iter_final_text(stream_iter):
                    had_final = True
                    streamed_parts.append(chunk)
                    yield f"data: {json.dumps({'type': 'delta', 'text': chunk}, ensure_ascii=False)}\\n\\n"
            except Exception as exc:
                state.log({"event": "llm_error", "error": str(exc), "session_id": session_id})
                error_payload = {
                    "type": "final",
                    "ok": True,
                    "session_id": session_id,
                    "answer": (
                        "LLM не настроен или недоступен. "
                        "Проверьте RAG_LLM_BASE_URL и RAG_LLM_MODEL, затем перезапустите backend."
                    ),
                    "consent": "granted",
                    "chunks": [],
                    "used_knowledge": [],
                    "citations": [],
                }
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\\n\\n"
                return

            if had_final:
                answer_text = clean_answer("".join(streamed_parts))
            else:
                answer_text = settings.app.strict_refusal_text

            citations = attach_citations(answer_text, final_chunks) if had_final else []
            used_knowledge = [
                {
                    "chunk_id": c["chunk_id"],
                    "heading_path": c.get("heading_path", []),
                    "snippet": c.get("text", "")[:280],
                    "source": c.get("source", ""),
                    "doc_name": c.get("doc_name", ""),
                }
                for c in final_chunks
            ] if had_final else []

            state.log({
                "event": "chat",
                "session_id": session_id,
                "question": message,
                "normalized_query": retrieval.get("normalized_query"),
                "rewritten_query": retrieval.get("rewritten_query"),
                "top_rerank_score": retrieval.get("top_rerank_score"),
                "chunks": [c.get("chunk_id") for c in final_chunks],
            })

            final_payload = {
                "type": "final",
                "ok": True,
                "session_id": session_id,
                "answer": answer_text,
                "consent": "granted",
                "chunks": final_chunks if had_final else [],
                "used_knowledge": used_knowledge,
                "citations": citations,
            }
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\\n\\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        llm_resp = call_openai_compatible(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout_sec=settings.llm.timeout_sec,
        )
        answer = clean_answer(llm_resp.text) or settings.app.strict_refusal_text
    except Exception as exc:
        state.log({"event": "llm_error", "error": str(exc), "session_id": session_id})
        response = {
            "ok": True,
            "session_id": session_id,
            "answer": (
                "LLM не настроен или недоступен. "
                "Проверьте RAG_LLM_BASE_URL и RAG_LLM_MODEL, затем перезапустите backend."
            ),
            "consent": "granted",
            "chunks": [],
            "citations": [],
        }
        return _stream_or_json(response, stream)
    citations = attach_citations(answer, final_chunks)
    used_knowledge = [
        {
            "chunk_id": c["chunk_id"],
            "heading_path": c.get("heading_path", []),
            "snippet": c.get("text", "")[:280],
            "source": c.get("source", ""),
            "doc_name": c.get("doc_name", ""),
        }
        for c in final_chunks
    ]

    state.log({
        "event": "chat",
        "session_id": session_id,
        "question": message,
        "normalized_query": retrieval.get("normalized_query"),
        "rewritten_query": retrieval.get("rewritten_query"),
        "top_rerank_score": retrieval.get("top_rerank_score"),
        "chunks": [c.get("chunk_id") for c in final_chunks],
    })

    response = {
        "ok": True,
        "session_id": session_id,
        "answer": answer,
        "consent": "granted",
        "chunks": final_chunks,
        "used_knowledge": used_knowledge,
        "citations": citations,
        "retrieval": {
            "normalized_query": retrieval.get("normalized_query"),
            "candidates": retrieval.get("candidates"),
        },
    }
    return _stream_or_json(response, stream)


@app.get("/api/logs")
async def logs(limit: int = 200) -> dict[str, Any]:
    return {"ok": True, "items": state.tail_logs(limit=limit)}


@app.post("/api/voice/start")
async def voice_start() -> JSONResponse:
    return JSONResponse(status_code=501, content={"ok": False, "error": "Voice not implemented"})


@app.post("/api/voice/stop")
async def voice_stop() -> JSONResponse:
    return JSONResponse(status_code=501, content={"ok": False, "error": "Voice not implemented"})


@app.get("/api/voice/status")
async def voice_status() -> JSONResponse:
    return JSONResponse(status_code=501, content={"ok": False, "error": "Voice not implemented"})


def _stream_or_json(payload: dict[str, Any], stream: bool) -> Any:
    if not stream:
        return payload

    def gen() -> Any:
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\\n\\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
