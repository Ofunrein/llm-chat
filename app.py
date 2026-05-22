"""
FastAPI LLM Chat — from-scratch GPT inference.

Backend priority:
  1. Fine-tuned checkpoint (CHECKPOINT_PATH env var) — trained on arXiv abstracts
  2. GPT-2 pretrained weights (GPT2_MODEL env var, default "gpt2")

No external APIs. All inference runs locally via model/transformer.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from model.transformer import GPT, TransformerConfig

load_dotenv()

app = FastAPI(title="LLM Chat — From Scratch", version="1.0.0")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "256"))
_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.85"))
_TOP_K = int(os.getenv("TOP_K", "50"))
_CHECKPOINT = os.getenv("CHECKPOINT_PATH", "")
_GPT2_MODEL = os.getenv("GPT2_MODEL", "gpt2")

_model: GPT | None = None
_encode: Any = None          # callable: str → list[int]
_decode: Any = None          # callable: list[int] → str
_backend_name = "loading…"
_lock = threading.Lock()
_EOT = 50256


def _load_model() -> None:
    global _model, _encode, _decode, _backend_name

    if Path(_CHECKPOINT).is_file():
        # --- fine-tuned checkpoint ---
        from model.tokenizer import BPETokenizer

        ckpt = torch.load(_CHECKPOINT, map_location=_DEVICE)
        cfg: TransformerConfig = ckpt["config"]
        mdl = GPT(cfg)
        mdl.load_state_dict(ckpt["model"])
        tok_path = Path(_CHECKPOINT).parent / "tokenizer.json"
        if not tok_path.exists():
            tok_path = Path("tokenizer.json")
        tok = BPETokenizer.load(tok_path)
        _encode = tok.encode
        _decode = tok.decode
        _backend_name = f"fine-tuned checkpoint ({cfg.n_layers}L/{cfg.d_model}d)"
    else:
        # --- GPT-2 pretrained ---
        import tiktoken
        from model.load_gpt2 import load_gpt2

        enc = tiktoken.get_encoding("gpt2")
        mdl = load_gpt2(_GPT2_MODEL)
        _encode = enc.encode
        _decode = enc.decode
        _backend_name = f"GPT-2 {_GPT2_MODEL} pretrained"

    mdl.to(_DEVICE)
    mdl.eval()
    _model = mdl


def _get_model() -> GPT:
    if _model is None:
        with _lock:
            if _model is None:
                _load_model()
    return _model  # type: ignore[return-value]


def _build_prompt(history: list[dict[str, Any]], message: str) -> str:
    """
    For abstract-completion use case: treat message as a research topic or
    partial abstract opening. Model continues it.
    """
    lines: list[str] = []
    for t in history:
        role = t.get("role", "")
        content = t.get("content", "")
        if isinstance(content, str) and content.strip():
            label = "Abstract" if role == "assistant" else "Topic"
            lines.append(f"{label}: {content.strip()}")
    lines.append(f"Topic: {message.strip()}")
    lines.append("Abstract:")
    return "\n\n".join(lines)


def _run_inference(prompt: str) -> str:
    model = _get_model()
    ids = _encode(prompt)
    max_ctx = model.cfg.context_len - _MAX_NEW
    ids = ids[-max_ctx:]
    idx = torch.tensor([ids], device=_DEVICE)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=_MAX_NEW,
                              temperature=_TEMPERATURE, top_k=_TOP_K)
    new_ids = out[0, len(ids):].tolist()
    # stop at <|endoftext|> or first double-newline
    if _EOT in new_ids:
        new_ids = new_ids[: new_ids.index(_EOT)]
    text = _decode(new_ids)
    # trim at first paragraph break if too long
    if "\n\n" in text:
        text = text[: text.index("\n\n")]
    return text.strip()


async def _stream(message: str, history: list[dict[str, Any]]) -> AsyncGenerator[str, None]:
    prompt = _build_prompt(history, message)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _run_inference, prompt)
    words = text.split(" ")
    for i, word in enumerate(words):
        token = word if i == len(words) - 1 else word + " "
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0.018)
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
        mdl = _get_model()
        return {
            "backend": _backend_name,
            "params_M": round(mdl.num_parameters() / 1e6, 1),
            "device": _DEVICE,
            "context_len": mdl.cfg.context_len,
            "implementation": "model/transformer.py",
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
