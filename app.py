"""Flask LLM Chat — multimodal streaming chat powered by Claude."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Generator
from typing import Any

import anthropic
from dotenv import load_dotenv
from flask import Flask, Response, render_template, request, stream_with_context
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

_CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
_MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
_SYSTEM = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise, and thoughtful AI assistant. "
    "When shown images, describe and analyze them carefully.",
)


def _build_content(text: str, image_b64: str | None, media_type: str | None) -> Any:
    """Build Anthropic content block supporting text and optional image."""
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


def _stream_reply(messages: list[dict[str, Any]]) -> Generator[str, None, None]:
    """Yield SSE-formatted text chunks from Claude."""
    with _CLIENT.messages.stream(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield f"data: {json.dumps({'token': text})}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/chat")
def chat() -> Response:
    payload = request.get_json(force=True)
    history: list[dict[str, Any]] = payload.get("history", [])
    user_text: str = payload.get("message", "").strip()
    image_b64: str | None = payload.get("image_b64")
    media_type: str | None = payload.get("media_type")

    if not user_text and not image_b64:
        return Response("No input", status=400)

    messages: list[dict[str, Any]] = list(history)
    messages.append({"role": "user", "content": _build_content(user_text, image_b64, media_type)})

    return Response(
        stream_with_context(_stream_reply(messages)),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
