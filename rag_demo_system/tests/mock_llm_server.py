"""Mock OpenAI-compatible LLM server for local SIP testing."""
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

CANNED_RESPONSE = (
    "По предварительному расчёту, ежемесячный платёж по лизингу Volkswagen Tiguan "
    "составит примерно 1 250 белорусских рублей при сроке 5 лет. "
    "Хотите, я отправлю подробный расчёт по СМС?"
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)

    if not stream:
        return {
            "id": "mock-1",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": CANNED_RESPONSE},
                "finish_reason": "stop",
            }],
        }

    def generate():
        for word in CANNED_RESPONSE.split():
            chunk = {
                "id": "mock-1",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8787)
