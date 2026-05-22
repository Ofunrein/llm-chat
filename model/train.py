"""
Training loop for the GPT model.

Features
--------
- AdamW optimiser with cosine LR schedule + linear warmup
- Gradient clipping
- Checkpoint saving / resuming
- Optional Weights & Biases logging
- Multi-GPU via torch.nn.parallel.DistributedDataParallel (DDP) hooks ready

Quick-start (single GPU / CPU):
    python -m model.train --data data/train.bin --out checkpoints/
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from model.transformer import GPT, TransformerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine decay with linear warm-up."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


class DataLoader:
    """Memory-mapped token loader for a flat binary file (uint16 token IDs)."""

    def __init__(self, path: Path, batch_size: int, context_len: int, device: torch.device) -> None:
        import numpy as np

        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.context_len = context_len
        self.device = device
        self._pos = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        B, T = self.batch_size, self.context_len
        chunk = self.data[self._pos : self._pos + B * T + 1]
        if len(chunk) < B * T + 1:
            self._pos = 0
            chunk = self.data[self._pos : self._pos + B * T + 1]
        x = torch.tensor(chunk[:-1].astype(np.int64), dtype=torch.long).view(B, T).to(self.device)
        y = torch.tensor(chunk[1:].astype(np.int64), dtype=torch.long).view(B, T).to(self.device)
        self._pos += B * T
        return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # --- model ---
    cfg = TransformerConfig(
        vocab_size=args.vocab_size,
        context_len=args.context_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"Parameters: {model.num_parameters() / 1e6:.2f}M")

    # optional bf16 / fp16 autocast
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    ctx = torch.amp.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else torch.inference_mode.__class__

    # --- optimiser ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=device.type == "cuda",
    )

    # --- data ---
    loader = DataLoader(Path(args.data), args.batch_size, args.context_len, device)

    # --- optional WandB ---
    run = None
    if args.wandb:
        import wandb  # type: ignore[import]
        run = wandb.init(project="llm-from-scratch", config=vars(args))

    # --- resume ---
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- loop ---
    model.train()
    t0 = time.perf_counter()

    for step in range(start_step, args.max_steps):
        lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = loader.next_batch()

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            dt = time.perf_counter() - t0
            tokens_per_sec = args.batch_size * args.context_len * args.log_every / dt
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | tok/s {tokens_per_sec:,.0f}")
            if run:
                run.log({"loss": loss.item(), "lr": lr, "tokens_per_sec": tokens_per_sec}, step=step)
            t0 = time.perf_counter()

        if step > 0 and step % args.save_every == 0:
            ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": cfg}, ckpt_path)
            print(f"Saved {ckpt_path}")

    if run:
        run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GPT from scratch")
    p.add_argument("--data", required=True, help="Path to tokenised .bin file (uint16)")
    p.add_argument("--out", default="checkpoints", help="Checkpoint output dir")
    p.add_argument("--resume", default=None, help="Resume from checkpoint path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--vocab-size", type=int, dest="vocab_size", default=50_257)
    p.add_argument("--context-len", type=int, dest="context_len", default=1_024)
    p.add_argument("--d-model", type=int, dest="d_model", default=768)
    p.add_argument("--n-heads", type=int, dest="n_heads", default=12)
    p.add_argument("--n-layers", type=int, dest="n_layers", default=12)
    p.add_argument("--d-ff", type=int, dest="d_ff", default=3_072)
    p.add_argument("--dropout", type=float, default=0.1)
    # training
    p.add_argument("--batch-size", type=int, dest="batch_size", default=8)
    p.add_argument("--max-steps", type=int, dest="max_steps", default=100_000)
    p.add_argument("--warmup-steps", type=int, dest="warmup_steps", default=2_000)
    p.add_argument("--max-lr", type=float, dest="max_lr", default=3e-4)
    p.add_argument("--min-lr", type=float, dest="min_lr", default=3e-5)
    p.add_argument("--grad-clip", type=float, dest="grad_clip", default=1.0)
    p.add_argument("--log-every", type=int, dest="log_every", default=100)
    p.add_argument("--save-every", type=int, dest="save_every", default=5_000)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return p


if __name__ == "__main__":
    train(_parser().parse_args())
