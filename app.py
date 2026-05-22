"""
FastAPI LLM Chat — inference runs on our from-scratch GPT transformer.

No external LLM API. Model weights: GPT-2 (open, free) loaded into
our custom transformer implementation in model/transformer.py.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any

import tiktoken
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model.load_gpt2 import load_gpt2
from model.transformer import GPT

app = FastAPI(title="LLM Chat — From Scratch", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

_MODEL_SIZE = os.getenv("GPT2_MODEL", "gpt2")         # gpt2 | gpt2-medium | gpt2-large
_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
_TOP_K = int(os.getenv("TOP_K", "40"))
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model + tokenizer — loaded once on first request, cached forever
# ---------------------------------------------------------------------------

_model: GPT | None = None
_enc: tiktoken.Encoding | None = None
_lock = threading.Lock()


def _get_model_and_enc() -> tuple[GPT, tiktoken.Encoding]:
    global _model, _enc
    if _model is None:
        with _lock:
            if _model is None:
                enc = tiktoken.get_encoding("gpt2")
                model = load_gpt2(_MODEL_SIZE)
                model.to(_DEVICE)
                model.eval()
                _model, _enc = model, enc
    return _model, _enc  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

_EOT = 50256  # <|endoftext|> in GPT-2 tokenizer


def _build_prompt(history: list[dict[str, str]], message: str) -> str:
    """Flatten chat history into a plain text prompt for GPT-2."""
    lines: list[str] = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if isinstance(content, str):
            lines.append(f"{role.capitalize()}: {content}")
    lines.append(f"Human: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _run_inference(prompt: str) -> str:
    """Blocking inference — call in a thread to avoid blocking the event loop."""
    model, enc = _get_model_and_enc()
    ids = enc.encode(prompt)
    # crop to leave room for generation
    max_ctx = model.cfg.context_len - _MAX_NEW_TOKENS
    ids = ids[-max_ctx:]
    idx = torch.tensor([ids], device=_DEVICE)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=_MAX_NEW_TOKENS,
            temperature=_TEMPERATURE,
            top_k=_TOP_K,
        )

    new_ids = out[0, len(ids):].tolist()
    # stop at <|endoftext|> if generated
    if _EOT in new_ids:
        new_ids = new_ids[: new_ids.index(_EOT)]

    return enc.decode(new_ids)


async def _stream_tokens(text: str) -> AsyncGenerator[str, None]:
    """Fake per-word streaming so the UI gets progressive output."""
    import json

    words = text.split(" ")
    for i, word in enumerate(words):
        token = word if i == len(words) - 1 else word + " "
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0.02)
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = ""
    history: list[dict[str, Any]] = []


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    from pathlib import Path
    return HTMLResponse(content=Path("templates/index.html").read_text())


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="No input provided")

    prompt = _build_prompt(req.history, req.message)

    # run blocking inference in a thread pool so FastAPI stays async
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, _run_inference, prompt)

    return StreamingResponse(
        _stream_tokens(reply),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/model-info")
async def model_info() -> dict[str, Any]:
    model, _ = _get_model_and_enc()
    return {
        "model": _MODEL_SIZE,
        "parameters_M": round(model.num_parameters() / 1e6, 1),
        "context_len": model.cfg.context_len,
        "device": _DEVICE,
        "implementation": "from-scratch (model/transformer.py)",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
