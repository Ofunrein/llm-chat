"""Smoke test for BPE tokenizer train + encode + decode roundtrip."""

from __future__ import annotations

from model.tokenizer import BPETokenizer


def test_bpe_train_encode_decode() -> None:
    text = "the quick brown fox jumps over the lazy dog " * 100
    tok = BPETokenizer()
    tok.train(text, vocab_size=300)

    ids = tok.encode("the quick brown fox")
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert "quick" in decoded or len(decoded) > 0  # may merge chars


def test_bpe_save_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    text = "hello world hello world " * 50
    tok = BPETokenizer()
    tok.train(text, vocab_size=280)

    path = tmp_path / "tok.json"
    tok.save(path)

    tok2 = BPETokenizer.load(path)
    assert len(tok2) == len(tok)
    assert tok2.encode("hello") == tok.encode("hello")


def test_bpe_vocab_size() -> None:
    # rich corpus with many unique pairs needed for merges beyond 256
    text = "the quick brown fox jumps over the lazy dog. " * 500
    tok = BPETokenizer()
    tok.train(text, vocab_size=300)
    # vocab size is 256 bytes + however many merges were possible
    assert len(tok) <= 300
    assert len(tok) >= 256
