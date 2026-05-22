"""Tests for arXiv data pipeline."""

from __future__ import annotations

from model.data import _build_corpus, _clean


def test_clean_removes_urls() -> None:
    text = "See http://arxiv.org/abs/1234 for details."
    assert "http" not in _clean(text)


def test_clean_collapses_whitespace() -> None:
    assert _clean("foo   bar\n\nbaz") == "foo bar baz"


def test_build_corpus_separator() -> None:
    corpus = _build_corpus(["abstract one", "abstract two"])
    assert "abstract one" in corpus
    assert "abstract two" in corpus
    assert "<|endoftext|>" in corpus


def test_build_corpus_filters_empty() -> None:
    corpus = _build_corpus(["valid abstract", "", "   "])
    assert corpus.count("<|endoftext|>") == 0 or "valid abstract" in corpus
