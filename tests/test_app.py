"""Tests for FastAPI LLM Chat."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from app import _build_content, app  # noqa: E402

client = TestClient(app)


def test_build_content_text_only() -> None:
    assert _build_content("hello", None, None) == "hello"


def test_build_content_with_image() -> None:
    result = _build_content("describe this", "abc123", "image/jpeg")
    assert isinstance(result, list)
    assert result[0]["type"] == "image"
    assert result[0]["source"]["data"] == "abc123"
    assert result[1]["text"] == "describe this"


def test_build_content_image_no_text() -> None:
    result = _build_content("", "abc123", "image/png")
    assert isinstance(result, list)
    assert result[1]["text"] == "What do you see in this image?"


def test_index_returns_200() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "LLM Chat" in resp.text


def test_chat_no_input_returns_400() -> None:
    resp = client.post("/chat", json={})
    assert resp.status_code == 400
