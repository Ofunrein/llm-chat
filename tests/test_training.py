"""Integration test: full train → checkpoint → inference pipeline."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest


def test_train_produces_checkpoint(tmp_path: pathlib.Path) -> None:
    """Run 5 training steps and verify a checkpoint is saved."""
    torch = pytest.importorskip("torch")
    from model.data import _build_corpus
    from model.synthetic_data import generate
    from model.tokenizer import BPETokenizer
    from model.train import _parser, train

    # tiny dataset
    corpus = _build_corpus(generate(50))
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=300)
    ids = tok.encode(corpus[:20000])
    data_path = tmp_path / "train.bin"
    np.array(ids, dtype=np.uint16).tofile(data_path)
    tok.save(tmp_path / "tokenizer.json")

    import sys
    sys.argv = [
        "train",
        "--data", str(data_path),
        "--vocab-size", "300",
        "--context-len", "32",
        "--n-layers", "2",
        "--d-model", "64",
        "--n-heads", "4",
        "--d-ff", "128",
        "--batch-size", "2",
        "--max-steps", "5",
        "--warmup-steps", "1",
        "--log-every", "5",
        "--save-every", "5",
        "--out", str(tmp_path / "ckpt"),
        "--device", "cpu",
    ]
    train(_parser().parse_args())

    ckpts = list((tmp_path / "ckpt").glob("ckpt_*.pt"))
    assert ckpts, "No checkpoint saved"
    ckpt = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    assert "model" in ckpt
    assert "config" in ckpt


def test_checkpoint_inference(tmp_path: pathlib.Path) -> None:
    """Load a trained checkpoint and run generation."""
    torch = pytest.importorskip("torch")
    from model.data import _build_corpus
    from model.synthetic_data import generate
    from model.tokenizer import BPETokenizer
    from model.train import _parser, train
    from model.transformer import GPT

    corpus = _build_corpus(generate(50))
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=300)
    ids = tok.encode(corpus[:20000])
    data_path = tmp_path / "train.bin"
    np.array(ids, dtype=np.uint16).tofile(data_path)
    tok.save(tmp_path / "tokenizer.json")

    import sys
    sys.argv = [
        "train",
        "--data", str(data_path),
        "--vocab-size", "300",
        "--context-len", "32",
        "--n-layers", "2",
        "--d-model", "64",
        "--n-heads", "4",
        "--d-ff", "128",
        "--batch-size", "2",
        "--max-steps", "5",
        "--warmup-steps", "1",
        "--log-every", "5",
        "--save-every", "5",
        "--out", str(tmp_path / "ckpt"),
        "--device", "cpu",
    ]
    train(_parser().parse_args())

    ckpt_path = sorted((tmp_path / "ckpt").glob("ckpt_*.pt"))[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt_ids = tok.encode("We propose")
    idx = torch.tensor([prompt_ids])
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=10)

    assert out.shape[1] == len(prompt_ids) + 10
    decoded = tok.decode(out[0].tolist())
    assert len(decoded) > 0
