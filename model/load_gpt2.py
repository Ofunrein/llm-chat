"""
Load pretrained GPT-2 weights into our from-scratch GPT model.

This lets you run real inference without an API or training from scratch.
Reference: Karpathy, nanoGPT (https://github.com/karpathy/nanoGPT)
"""

from __future__ import annotations

import torch

from model.transformer import GPT, TransformerConfig


def load_gpt2(size: str = "gpt2") -> GPT:
    """
    Download and load GPT-2 weights into our custom transformer.

    Parameters
    ----------
    size : str
        One of "gpt2" (117M), "gpt2-medium" (345M),
        "gpt2-large" (762M), "gpt2-xl" (1558M).
    """
    from transformers import GPT2LMHeadModel  # type: ignore[import]

    size_to_cfg = {
        "gpt2":        TransformerConfig(n_layers=12, d_model=768,  n_heads=12, d_ff=3072),
        "gpt2-medium": TransformerConfig(n_layers=24, d_model=1024, n_heads=16, d_ff=4096),
        "gpt2-large":  TransformerConfig(n_layers=36, d_model=1280, n_heads=20, d_ff=5120),
        "gpt2-xl":     TransformerConfig(n_layers=48, d_model=1600, n_heads=25, d_ff=6400),
    }
    cfg = size_to_cfg[size]
    cfg.vocab_size = 50_257
    cfg.dropout = 0.0  # eval mode

    model = GPT(cfg)
    hf = GPT2LMHeadModel.from_pretrained(size)
    hf_sd = hf.state_dict()
    our_sd = model.state_dict()

    # transpose Conv1D weights (HF stores them transposed)
    transposed = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    mapping = _build_weight_mapping(cfg.n_layers)
    for our_key, hf_key in mapping.items():
        if our_key not in our_sd:
            continue
        w = hf_sd[hf_key]
        if any(hf_key.endswith(t) for t in transposed):
            w = w.T
        assert our_sd[our_key].shape == w.shape, f"Shape mismatch: {our_key} {our_sd[our_key].shape} vs {hf_key} {w.shape}"
        our_sd[our_key].copy_(w)

    model.load_state_dict(our_sd)
    model.eval()
    print(f"Loaded {size} ({model.num_parameters() / 1e6:.0f}M params)")
    return model


def _build_weight_mapping(n_layers: int) -> dict[str, str]:
    """Map our parameter names to HF GPT-2 parameter names."""
    m: dict[str, str] = {
        "tok_emb.weight": "transformer.wte.weight",
        "pos_emb.weight": "transformer.wpe.weight",
        "ln_f.weight":    "transformer.ln_f.weight",
        "ln_f.bias":      "transformer.ln_f.bias",
    }
    for i in range(n_layers):
        p = f"blocks.{i}"
        h = f"transformer.h.{i}"
        m.update({
            f"{p}.ln1.weight":        f"{h}.ln_1.weight",
            f"{p}.ln1.bias":          f"{h}.ln_1.bias",
            f"{p}.ln2.weight":        f"{h}.ln_2.weight",
            f"{p}.ln2.bias":          f"{h}.ln_2.bias",
            f"{p}.attn.qkv.weight":   f"{h}.attn.c_attn.weight",
            f"{p}.attn.qkv.bias":     f"{h}.attn.c_attn.bias",
            f"{p}.attn.out_proj.weight": f"{h}.attn.c_proj.weight",
            f"{p}.attn.out_proj.bias":   f"{h}.attn.c_proj.bias",
            f"{p}.ff.net.0.weight":   f"{h}.mlp.c_fc.weight",
            f"{p}.ff.net.0.bias":     f"{h}.mlp.c_fc.bias",
            f"{p}.ff.net.2.weight":   f"{h}.mlp.c_proj.weight",
            f"{p}.ff.net.2.bias":     f"{h}.mlp.c_proj.bias",
        })
    return m


if __name__ == "__main__":
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    model = load_gpt2("gpt2")

    prompt = "The transformer architecture is"
    idx = torch.tensor([enc.encode(prompt)])
    out = model.generate(idx, max_new_tokens=50, temperature=0.8, top_k=40)
    print(enc.decode(out[0].tolist()))
