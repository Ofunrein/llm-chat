"""Tests for message conversion and chat app logic."""

from __future__ import annotations

import importlib
import os
import sys

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

import app as chat_app


def test_build_content_text_only() -> None:
    result = chat_app._build_content("hello", None, None)
    assert result == "hello"


def test_build_content_with_image() -> None:
    result = chat_app._build_content("describe this", "abc123", "image/jpeg")
    assert isinstance(result, list)
    assert result[0]["type"] == "image"
    assert result[0]["source"]["data"] == "abc123"
    assert result[1]["type"] == "text"
    assert result[1]["text"] == "describe this"


def test_build_content_image_no_text() -> None:
    result = chat_app._build_content("", "abc123", "image/png")
    assert isinstance(result, list)
    assert result[1]["text"] == "What do you see in this image?"


def test_index_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    client = chat_app.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200


def test_chat_no_input_returns_400() -> None:
    client = chat_app.app.test_client()
    resp = client.post("/chat", json={})
    assert resp.status_code == 400
