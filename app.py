"""FastAPI LLM Chat — streaming multimodal chat powered by Claude."""

from __future__ import annotations

import os
from typing import Any

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="LLM Chat", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

_CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
_MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
_SYSTEM = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise, and thoughtful AI assistant. "
    "When shown images, describe and analyze them carefully.",
)


class ChatRequest(BaseModel):
    message: str = ""
    image_b64: str | None = None
    media_type: str | None = None
    history: list[dict[str, Any]] = []


def _build_content(text: str, image_b64: str | None, media_type: str | None) -> Any:
    if image_b64 and media_type:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_b64,
                },
            },
            {"type": "text", "text": text or "What do you see in this image?"},
        ]
    return text


async def _sse_generator(messages: list[dict[str, Any]]):
    """Async generator yielding SSE-formatted chunks from Claude."""
    import json

    with _CLIENT.messages.stream(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield f"data: {json.dumps({'token': text})}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    from pathlib import Path

    html = Path("templates/index.html").read_text()
    return HTMLResponse(content=html)


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip() and not req.image_b64:
        raise HTTPException(status_code=400, detail="No input provided")

    messages = list(req.history)
    messages.append(
        {"role": "user", "content": _build_content(req.message, req.image_b64, req.media_type)}
    )

    return StreamingResponse(
        _sse_generator(messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
