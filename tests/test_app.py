"""Tests for FastAPI LLM Chat app — stubs all heavy deps per-test."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _stub_heavy_deps(monkeypatch: pytest.MonkeyPatch):
    """Stub torch/tiktoken/transformers/model so app.py imports without GPU."""

    def stub(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    for n in ["torch", "tiktoken", "transformers", "regex"]:
        stub(n)

    _torch = sys.modules["torch"]
    _torch.cuda = MagicMock()  # type: ignore[attr-defined]
    _torch.cuda.is_available = MagicMock(return_value=False)
    _torch.tensor = MagicMock()  # type: ignore[attr-defined]
    _torch.no_grad = MagicMock(  # type: ignore[attr-defined]
        return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )
    )
    sys.modules["tiktoken"].get_encoding = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

    for n in ["model.transformer", "model.load_gpt2"]:
        stub(n)

    _mt = sys.modules["model.transformer"]
    _mt.GPT = MagicMock()  # type: ignore[attr-defined]
    _mt.TransformerConfig = MagicMock()  # type: ignore[attr-defined]

    sys.modules["model.load_gpt2"].load_gpt2 = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

    # remove cached app so it re-imports with stubs active
    monkeypatch.delitem(sys.modules, "app", raising=False)

    yield

    # monkeypatch auto-restores sys.modules on teardown


@pytest.fixture()
def _app():
    import app as _app_mod
    return _app_mod


@pytest.fixture()
def client(_app):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient
    return TestClient(_app.app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_prompt_empty_history(_app) -> None:  # type: ignore[no-untyped-def]
    p = _app._build_prompt([], "We propose a novel method")
    assert "We propose a novel method" in p


def test_build_prompt_with_history(_app) -> None:  # type: ignore[no-untyped-def]
    h = [
        {"role": "user", "content": "attention mechanisms"},
        {"role": "assistant", "content": "We propose"},
    ]
    p = _app._build_prompt(h, "transformers scale")
    assert "attention mechanisms" in p
    assert "We propose" in p
    assert "transformers scale" in p


def test_index_ok(client) -> None:  # type: ignore[no-untyped-def]
    r = client.get("/")
    assert r.status_code == 200
    assert "LLM Chat" in r.text


def test_chat_empty_returns_400(client) -> None:  # type: ignore[no-untyped-def]
    r = client.post("/chat", json={"message": ""})
    assert r.status_code == 400


def test_health_ok(client) -> None:  # type: ignore[no-untyped-def]
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
