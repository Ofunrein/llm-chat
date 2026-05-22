"""
Transformer LLM — implemented from scratch with PyTorch.

Architecture follows the GPT-2 / nanoGPT design:
  - Token + positional embeddings
  - N × (LayerNorm → Multi-Head Self-Attention → LayerNorm → MLP) blocks
  - Final LayerNorm + language model head (weight-tied to embedding)

Reference: Vaswani et al. 2017, Brown et al. 2020, Karpathy nanoGPT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TransformerConfig:
    vocab_size: int = 50_257        # GPT-2 default (tiktoken "gpt2")
    context_len: int = 1_024        # max sequence length
    d_model: int = 768              # embedding / hidden dim
    n_heads: int = 12               # attention heads
    n_layers: int = 12              # transformer blocks
    d_ff: int = 3_072               # feedforward hidden dim (≈ 4 × d_model)
    dropout: float = 0.1
    bias: bool = True               # bias in linear projections


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = math.sqrt(self.d_head)

        # fused QKV projection
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # causal mask: lower-triangular
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_len, cfg.context_len)).view(
                1, 1, cfg.context_len, cfg.context_len
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape  # batch, time-steps, channels

        # project to Q, K, V
        qkv = self.qkv(x)                                    # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)                       # each (B, T, C)

        # reshape to (B, n_heads, T, d_head)
        def reshape(t: Tensor) -> Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / self.scale        # (B, h, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                        # (B, h, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    """A single transformer block: pre-LayerNorm residual architecture."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    GPT-style autoregressive language model.

    Usage
    -----
    cfg = TransformerConfig(vocab_size=50257, n_layers=12, d_model=768)
    model = GPT(cfg)
    logits, loss = model(idx, targets)          # training
    tokens = model.generate(idx, max_new=200)   # inference
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying: token embedding ↔ LM head (reduces params ~10–20%)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2-style weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # scale residual projections by 1/√(2·n_layers) per GPT-2 paper
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.context_len, f"Sequence length {T} > context_len {self.cfg.context_len}"

        device = idx.device
        pos = torch.arange(T, device=device).unsqueeze(0)           # (1, T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))       # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                                    # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Auto-regressively sample *max_new_tokens* tokens."""
        for _ in range(max_new_tokens):
            # crop to context window
            idx_cond = idx[:, -self.cfg.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature              # (B, vocab)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    @classmethod
    def gpt2_small(cls) -> "GPT":
        return cls(TransformerConfig(n_layers=12, d_model=768, n_heads=12, d_ff=3072))

    @classmethod
    def gpt2_medium(cls) -> "GPT":
        return cls(TransformerConfig(n_layers=24, d_model=1024, n_heads=16, d_ff=4096))

    @classmethod
    def gpt2_large(cls) -> "GPT":
        return cls(TransformerConfig(n_layers=36, d_model=1280, n_heads=20, d_ff=5120))

    @classmethod
    def gpt2_xl(cls) -> "GPT":
        return cls(TransformerConfig(n_layers=48, d_model=1600, n_heads=25, d_ff=6400))
