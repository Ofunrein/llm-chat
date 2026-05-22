"""Unit tests for the from-scratch transformer."""

from __future__ import annotations

import pytest


def test_transformer_forward() -> None:
    """GPT forward pass produces correct output shapes."""
    torch = pytest.importorskip("torch")
    from model.transformer import GPT, TransformerConfig

    cfg = TransformerConfig(
        vocab_size=256, context_len=32, d_model=64,
        n_heads=4, n_layers=2, d_ff=128, dropout=0.0,
    )
    model = GPT(cfg)
    model.eval()

    idx = torch.randint(0, 256, (2, 16))  # batch=2, seq=16
    logits, loss = model(idx, targets=idx)

    assert logits.shape == (2, 16, 256), f"unexpected logits shape {logits.shape}"
    assert loss is not None
    assert loss.item() > 0


def test_transformer_generate() -> None:
    """GPT.generate returns extended sequence."""
    torch = pytest.importorskip("torch")
    from model.transformer import GPT, TransformerConfig

    cfg = TransformerConfig(
        vocab_size=256, context_len=32, d_model=64,
        n_heads=4, n_layers=2, d_ff=128, dropout=0.0,
    )
    model = GPT(cfg)
    model.eval()

    idx = torch.randint(0, 256, (1, 4))
    out = model.generate(idx, max_new_tokens=8, temperature=1.0, top_k=10)

    assert out.shape == (1, 12), f"expected (1,12) got {out.shape}"


def test_num_parameters() -> None:
    pytest.importorskip("torch")
    from model.transformer import GPT, TransformerConfig

    cfg = TransformerConfig(
        vocab_size=1000, context_len=128, d_model=64,
        n_heads=4, n_layers=2, d_ff=128,
    )
    model = GPT(cfg)
    n = model.num_parameters()
    assert n > 0
    assert n < 10_000_000  # small test model
