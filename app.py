"""FastAPI LLM Chat — pure from-scratch GPT-2 inference, zero external APIs."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import tiktoken
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from model.load_gpt2 import load_gpt2
from model.transformer import GPT

load_dotenv()

app = FastAPI(title="LLM Chat — From Scratch", version="1.0.0")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_SIZE = os.getenv("GPT2_MODEL", "gpt2")
_MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "256"))
_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.85"))
_TOP_K = int(os.getenv("TOP_K", "50"))
_EOT = 50256

_model: GPT | None = None
_enc: tiktoken.Encoding | None = None
_lock = threading.Lock()


def _get_model() -> tuple[GPT, tiktoken.Encoding]:
    global _model, _enc
    if _model is None:
        with _lock:
            if _model is None:
                enc = tiktoken.get_encoding("gpt2")
                mdl = load_gpt2(_MODEL_SIZE)
                mdl.to(_DEVICE)
                mdl.eval()
                _model, _enc = mdl, enc
    return _model, _enc  # type: ignore[return-value]


def _build_prompt(history: list[dict[str, Any]], message: str) -> str:
    lines: list[str] = []
    for t in history:
        role = t.get("role", "")
        content = t.get("content", "")
        if isinstance(content, str) and content.strip():
            lines.append(f"{role.capitalize()}: {content}")
    lines.append(f"Human: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _run_inference(prompt: str) -> str:
    model, enc = _get_model()
    ids = enc.encode(prompt)
    max_ctx = model.cfg.context_len - _MAX_NEW
    ids = ids[-max_ctx:]
    idx = torch.tensor([ids], device=_DEVICE)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=_MAX_NEW,
                              temperature=_TEMPERATURE, top_k=_TOP_K)
    new_ids = out[0, len(ids):].tolist()
    if _EOT in new_ids:
        new_ids = new_ids[: new_ids.index(_EOT)]
    return enc.decode(new_ids).strip()


async def _stream(message: str, history: list[dict[str, Any]]) -> AsyncGenerator[str, None]:
    prompt = _build_prompt(history, message)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _run_inference, prompt)
    words = text.split(" ")
    for i, word in enumerate(words):
        token = word if i == len(words) - 1 else word + " "
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0.02)
    yield "data: [DONE]\n\n"


class ChatRequest(BaseModel):
    message: str = ""
    history: list[dict[str, Any]] = []


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=Path("templates/index.html").read_text())


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="No input")
    return StreamingResponse(
        _stream(req.message, req.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/model-info")
async def model_info() -> dict[str, Any]:
    try:
        mdl, _ = _get_model()
        return {
            "model": _MODEL_SIZE,
            "params_M": round(mdl.num_parameters() / 1e6, 1),
            "device": _DEVICE,
            "backend": "from-scratch (model/transformer.py)",
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "backend": "gpt2-scratch"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
