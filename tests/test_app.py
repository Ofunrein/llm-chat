"""Tests for from-scratch GPT chat app."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]


# stub heavy deps before any app import
for _n in ["torch", "transformers", "tiktoken", "regex"]:
    _make_stub(_n)

# give torch the attributes app.py reads at module level
import torch as _torch_real  # noqa: E402 -- might already be installed

_torch_mod = sys.modules["torch"]
if not hasattr(_torch_mod, "cuda"):
    _torch_mod.cuda = MagicMock()  # type: ignore[attr-defined]
    _torch_mod.cuda.is_available = MagicMock(return_value=False)

# stub model sub-packages
for _n in ["model", "model.transformer", "model.load_gpt2"]:
    _make_stub(_n)

_lgpt2 = sys.modules["model.load_gpt2"]
_lgpt2.load_gpt2 = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

_tiktoken = sys.modules["tiktoken"]
_tiktoken.get_encoding = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

# stub GPT class in model.transformer
_mtransformer = sys.modules["model.transformer"]
_mtransformer.GPT = MagicMock()  # type: ignore[attr-defined]

# now safe to import app
from app import _build_prompt, app  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


def test_build_prompt_basic() -> None:
    prompt = _build_prompt([], "Hello")
    assert "Human: Hello" in prompt
    assert "Assistant:" in prompt


def test_build_prompt_with_history() -> None:
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    prompt = _build_prompt(history, "How are you?")
    assert "User: Hi" in prompt
    assert "Assistant: Hello!" in prompt
    assert "Human: How are you?" in prompt


def test_index_returns_200() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "LLM Chat" in resp.text


def test_chat_no_input_returns_400() -> None:
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 400
