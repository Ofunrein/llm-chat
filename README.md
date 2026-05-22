---
title: LLM Chat
emoji: 💬
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# llm-chat

Production-quality LLM project in two parts:

1. **Chat app** — streaming chat UI powered by GPT-2 running on our from-scratch transformer (FastAPI + Docker, HF Spaces)
2. **LLM from scratch** — GPT-style transformer + BPE tokenizer + training loop in PyTorch, zero Hugging Face dependency for inference

No external LLM APIs. The transformer in `model/transformer.py` IS the model.

---

## Live Demo

🤗 **[Open on Hugging Face Spaces](https://huggingface.co/spaces/Ofunrein/llm-chat)**

---

## Architecture

```
llm-chat/
├── app.py               # FastAPI — async SSE streaming, /chat, /model-info, /health
├── templates/
│   └── index.html       # Dark glassmorphism UI, animated canvas orbs, inline CSS+JS
├── model/
│   ├── transformer.py   # GPT architecture from scratch: CausalSelfAttention, FeedForward, Block, GPT
│   ├── tokenizer.py     # BPE tokenizer from scratch: train, encode, decode, save, load
│   ├── train.py         # Training loop: AdamW, cosine LR, warmup, grad clip, WandB
│   └── load_gpt2.py     # Load pretrained GPT-2 weights into our transformer (weight mapping)
├── tests/
│   └── test_app.py      # pytest — 5 tests, no GPU required
├── Dockerfile           # HF Spaces deployment (torch CPU, uvicorn)
└── pyproject.toml       # uv-managed deps
```

---

## From-Scratch Stack

| Component | File | Details |
|---|---|---|
| Multi-head causal self-attention | `model/transformer.py` | Fused QKV, causal mask, scaled dot-product |
| Feed-forward (GELU) | `model/transformer.py` | Pre-LayerNorm residual architecture |
| Learned positional + token embeddings | `model/transformer.py` | Weight-tied LM head |
| GPT-2 weight initialisation | `model/transformer.py` | `std=0.02`, residual scaling `1/√(2N)` |
| Top-k sampling + temperature | `model/transformer.py` | `model.generate()` |
| BPE tokenizer | `model/tokenizer.py` | Full train/encode/decode/save/load |
| Training loop | `model/train.py` | AdamW, cosine decay, linear warmup, grad clip |
| GPT-2 weight loading | `model/load_gpt2.py` | Maps HF → our param names, transposes Conv1D |

### Model sizes

| Variant | Layers | d_model | Heads | Params |
|---|---|---|---|---|
| `GPT.gpt2_small()` | 12 | 768 | 12 | ~117M |
| `GPT.gpt2_medium()` | 24 | 1024 | 16 | ~345M |
| `GPT.gpt2_large()` | 36 | 1280 | 20 | ~762M |
| `GPT.gpt2_xl()` | 48 | 1600 | 25 | ~1.5B |

---

## Quick Start (local)

```bash
git clone https://github.com/Ofunrein/llm-chat && cd llm-chat
uv sync
uv run python app.py
# → http://localhost:7860
```

### Docker

```bash
docker build -t llm-chat .
docker run -p 7860:7860 llm-chat
```

---

## Train Your Own

```bash
# 1. tokenise a corpus
python -c "
import numpy as np
from model.tokenizer import BPETokenizer
text = open('data/input.txt').read()
tok = BPETokenizer()
tok.train(text, vocab_size=1000)
tok.save('tokenizer.json')
np.array(tok.encode(text), dtype=np.uint16).tofile('data/train.bin')
"

# 2. train (CPU demo)
python -m model.train \
  --data data/train.bin --vocab-size 1000 \
  --n-layers 4 --d-model 128 --n-heads 4 \
  --context-len 256 --batch-size 4 --max-steps 2000
```

---

## Tests

```bash
uv run pytest tests/ -v   # 5 tests, no GPU needed
```

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Type hints throughout |
| Framework | FastAPI | Async-native, `StreamingResponse`, OpenAPI docs at `/docs` |
| Inference | PyTorch (CPU/CUDA) | The from-scratch transformer |
| Package manager | uv | 10–100× faster than pip |
| Deploy | HF Spaces (Docker) | Free, supports PyTorch, persistent container |

> **Kubernetes**: overkill for a single-model demo. Add a `k8s/` dir with a Deployment + Service + HPA if you need multi-replica auto-scaling — the Dockerfile is already production-ready.

---

## License

MIT
