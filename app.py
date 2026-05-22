"""
FastAPI LLM Chat — from-scratch GPT inference.

Backend priority:
  1. Fine-tuned checkpoint (CHECKPOINT_PATH env var)
  2. GPT-2 pretrained weights (GPT2_MODEL env var, default "gpt2")

No external APIs. All inference via model/transformer.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from model.transformer import GPT, TransformerConfig

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "200"))
_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.85"))
_TOP_K = int(os.getenv("TOP_K", "40"))
_CHECKPOINT = os.getenv("CHECKPOINT_PATH", "")
_GPT2_MODEL = os.getenv("GPT2_MODEL", "gpt2")
_EOT = 50256

# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

_model: GPT | None = None
_encode: Any = None
_decode: Any = None
_backend_name = "loading…"
_load_error: str | None = None
_lock = threading.Lock()


def _load_model() -> None:
    global _model, _encode, _decode, _backend_name, _load_error
    try:
        if Path(_CHECKPOINT).is_file():
            from model.tokenizer import BPETokenizer
            ckpt = torch.load(_CHECKPOINT, map_location=_DEVICE, weights_only=False)
            cfg: TransformerConfig = ckpt["config"]
            mdl = GPT(cfg)
            mdl.load_state_dict(ckpt["model"])
            tok_path = Path(_CHECKPOINT).parent / "tokenizer.json"
            if not tok_path.exists():
                tok_path = Path("tokenizer.json")
            tok = BPETokenizer.load(tok_path)
            _encode = tok.encode
            _decode = tok.decode
            _backend_name = f"fine-tuned ({cfg.n_layers}L/{cfg.d_model}d/arXiv)"
            log.info("Loaded fine-tuned checkpoint: %s", _CHECKPOINT)
        else:
            import tiktoken
            from model.load_gpt2 import load_gpt2
            log.info("Loading GPT-2 (%s)…", _GPT2_MODEL)
            enc = tiktoken.get_encoding("gpt2")
            mdl = load_gpt2(_GPT2_MODEL)
            _encode = enc.encode
            _decode = enc.decode
            _backend_name = f"GPT-2 {_GPT2_MODEL} · from-scratch weights"
            log.info("GPT-2 loaded (%dM params)", mdl.num_parameters() // 1_000_000)

        mdl.to(_DEVICE)
        mdl.eval()
        _model = mdl
    except Exception as e:
        _load_error = str(e)
        log.exception("Model load failed: %s", e)


def _get_model() -> GPT:
    if _model is None:
        with _lock:
            if _model is None:
                _load_model()
    if _model is None:
        raise RuntimeError(_load_error or "Model failed to load")
    return _model


# ---------------------------------------------------------------------------
# Lifespan — pre-warm model on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(application: FastAPI):  # type: ignore[type-arg]
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _get_model)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="LLM Chat — From Scratch", version="1.0.0", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _build_prompt(history: list[dict[str, Any]], message: str) -> str:
    """
    Build a plain-text continuation prompt.
    GPT-2 (WebText-trained) works best with natural prose continuation.
    """
    parts: list[str] = []
    for t in history:
        content = t.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    parts.append(message.strip())
    return " ".join(parts)


def _run_inference(prompt: str) -> str:
    model = _get_model()
    ids = _encode(prompt)
    max_ctx = model.cfg.context_len - _MAX_NEW
    ids = ids[-max_ctx:]
    idx = torch.tensor([ids], device=_DEVICE)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=_MAX_NEW,
            temperature=_TEMPERATURE,
            top_k=_TOP_K,
        )

    new_ids = out[0, len(ids):].tolist()
    # stop at <|endoftext|>
    if _EOT in new_ids:
        new_ids = new_ids[: new_ids.index(_EOT)]
    if not new_ids:
        return "[model returned empty output — try a different prompt]"
    text = _decode(new_ids)
    # trim at double newline (paragraph boundary)
    if "\n\n" in text:
        text = text[: text.index("\n\n")]
    return text.strip()


async def _sse_stream(message: str, history: list[dict[str, Any]]) -> AsyncGenerator[str, None]:
    prompt = _build_prompt(history, message)
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(None, _run_inference, prompt)
    except Exception as e:
        yield f"data: {json.dumps({'token': f'Error: {e}'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    words = text.split(" ")
    for i, word in enumerate(words):
        token = word if i == len(words) - 1 else word + " "
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0.018)
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
        _sse_stream(req.message, req.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/model-info")
async def model_info() -> JSONResponse:
    if _load_error:
        return JSONResponse({"error": _load_error}, status_code=500)
    if _model is None:
        return JSONResponse({"backend": _backend_name, "status": "loading", "device": _DEVICE})
    mdl = _model
    return JSONResponse({
        "backend": _backend_name,
        "params_M": round(mdl.num_parameters() / 1e6, 1),
        "device": _DEVICE,
        "context_len": mdl.cfg.context_len,
        "implementation": "model/transformer.py",
    })


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": "ready" if _model else "loading"}


@app.get("/debug")
async def debug() -> JSONResponse:
    """Show installed packages and import status — remove before prod."""
    import importlib.util
    import subprocess
    import sys

    torch_spec = importlib.util.find_spec("torch")
    info: dict[str, Any] = {
        "python": sys.version,
        "torch_spec": str(torch_spec),
        "model_error": _load_error,
        "model_loaded": _model is not None,
    }
    try:
        result = subprocess.run(
            ["pip3", "list", "--format=columns"],
            capture_output=True, text=True, timeout=10
        )
        lines = [l for l in result.stdout.splitlines() if any(k in l.lower() for k in ["torch", "transform", "tiktoken", "fastapi"])]
        info["installed"] = lines
    except Exception as e:
        info["pip_error"] = str(e)
    return JSONResponse(info)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
