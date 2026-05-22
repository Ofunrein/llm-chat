"""Tests for from-scratch GPT chat app — mocks all heavy deps."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules.setdefault(name, m)
    return sys.modules[name]


# --- stub before any app import ---
for n in ["torch", "tiktoken", "transformers", "regex"]:
    stub(n)

_torch = sys.modules["torch"]
_torch.cuda = MagicMock()  # type: ignore[attr-defined]
_torch.cuda.is_available = MagicMock(return_value=False)
_torch.tensor = MagicMock()  # type: ignore[attr-defined]
_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))  # type: ignore[attr-defined]

_tiktoken = sys.modules["tiktoken"]
_tiktoken.get_encoding = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

for n in ["model", "model.transformer", "model.load_gpt2"]:
    stub(n)

_lgpt2 = sys.modules["model.load_gpt2"]
_lgpt2.load_gpt2 = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

_mt = sys.modules["model.transformer"]
_mt.GPT = MagicMock()  # type: ignore[attr-defined]

# safe to import now
from app import _build_prompt, app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


def test_build_prompt_empty_history() -> None:
    p = _build_prompt([], "Hello")
    assert "Human: Hello" in p
    assert "Assistant:" in p


def test_build_prompt_with_history() -> None:
    h = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey!"}]
    p = _build_prompt(h, "How are you?")
    assert "User: Hi" in p
    assert "Assistant: Hey!" in p
    assert "Human: How are you?" in p


def test_index_ok() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "LLM Chat" in r.text


def test_chat_empty_returns_400() -> None:
    r = client.post("/chat", json={"message": ""})
    assert r.status_code == 400


def test_health_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
