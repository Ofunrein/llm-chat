"""Evaluate a trained GPT checkpoint — computes perplexity on val.bin."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from model.transformer import GPT, TransformerConfig


def evaluate(args: argparse.Namespace) -> float:
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg: TransformerConfig = ckpt["config"]
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    T = cfg.context_len
    B = args.batch_size
    n_batches = args.n_batches

    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i in range(n_batches):
            start = (i * B * T) % max(1, len(data) - B * T - 1)
            chunk = data[start : start + B * T + 1].astype(np.int64)
            x = torch.tensor(chunk[:-1], dtype=torch.long).view(B, T).to(device)
            y = torch.tensor(chunk[1:], dtype=torch.long).view(B, T).to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            n += 1

    avg_loss = total_loss / n
    ppl = math.exp(avg_loss)
    print(f"val loss: {avg_loss:.4f} | perplexity: {ppl:.2f}")
    return ppl


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate GPT checkpoint on val.bin")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-bin", dest="val_bin", default="data/val.bin")
    p.add_argument("--batch-size", type=int, dest="batch_size", default=4)
    p.add_argument("--n-batches", type=int, dest="n_batches", default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    evaluate(_parser().parse_args())
