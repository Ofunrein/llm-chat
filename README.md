# llm-chat

A production-quality LLM project in two parts:

1. **Chat app** — streaming multimodal chat interface (Flask + Claude API, deployable to Vercel)
2. **LLM from scratch** — GPT-style transformer + BPE tokenizer implemented with PyTorch, no Hugging Face

---

## Demo

> **Live demo:** deployed on Vercel — see below after setup

---

## Architecture

```
llm-chat/
├── app.py              # Flask app — SSE streaming, multimodal endpoints
├── static/
│   ├── css/main.css    # Dark glassmorphism UI
│   └── js/main.js      # Streaming SSE client, drag-and-drop images
├── templates/
│   └── index.html      # Single-page chat interface
├── model/
│   ├── transformer.py  # GPT architecture from scratch (attention, MLP, blocks)
│   ├── tokenizer.py    # BPE tokenizer from scratch (no tiktoken dependency)
│   └── train.py        # Training loop — cosine LR, gradient clipping, WandB
├── tests/
│   └── test_app.py     # pytest unit + integration tests
├── vercel.json         # Vercel Python deployment config
└── pyproject.toml      # uv-managed dependencies
```

---

## Chat App

### Features

- **Streaming responses** via Server-Sent Events (SSE) — token-by-token output
- **Multimodal input** — attach or drag-and-drop images; Claude analyzes them alongside text
- **Multi-turn history** — full conversation context sent on each request
- **Dark glassmorphism UI** — animated gradient orbs, custom-styled chat bubbles
- **Zero JS frameworks** — vanilla ES2022, no React/Vue, fast first paint

### Quick start

```bash
# 1. clone
git clone https://github.com/Ofunrein/llm-chat && cd llm-chat

# 2. install dependencies (requires uv)
uv sync

# 3. set your API key
cp .env.example .env
# edit .env → ANTHROPIC_API_KEY=sk-ant-...

# 4. run
uv run python app.py
# → http://localhost:5000
```

### Deploy to Vercel

```bash
npm i -g vercel
vercel --prod
# set env var: ANTHROPIC_API_KEY in Vercel dashboard
```

---

## LLM From Scratch

A clean, well-commented implementation of the GPT architecture — no `transformers` library, no abstraction layers.

### What's implemented

| Component | File | Notes |
|---|---|---|
| Multi-head causal self-attention | `model/transformer.py` | Fused QKV, causal mask, scaled dot-product |
| Feed-forward network | `model/transformer.py` | GELU, pre-LayerNorm residual |
| Positional + token embeddings | `model/transformer.py` | Learned, weight-tied LM head |
| Weight initialisation | `model/transformer.py` | GPT-2 style, residual scaling |
| Top-k sampling + temperature | `model/transformer.py` | `model.generate()` |
| BPE tokenizer | `model/tokenizer.py` | Full train/encode/decode/save/load |
| Training loop | `model/train.py` | AdamW, cosine LR, warmup, grad clip, checkpoints |
| WandB logging | `model/train.py` | Optional `--wandb` flag |

### Model sizes

| Variant | Layers | d_model | Heads | Params |
|---|---|---|---|---|
| `GPT.gpt2_small()` | 12 | 768 | 12 | ~117M |
| `GPT.gpt2_medium()` | 24 | 1024 | 16 | ~345M |
| `GPT.gpt2_large()` | 36 | 1280 | 20 | ~762M |
| `GPT.gpt2_xl()` | 48 | 1600 | 25 | ~1.5B |

### Train a model

```bash
# tokenise your dataset first (e.g. TinyShakespeare)
python -c "
import numpy as np
text = open('data/input.txt').read()
from model.tokenizer import BPETokenizer
tok = BPETokenizer()
tok.train(text, vocab_size=1000)
ids = tok.encode(text)
np.array(ids, dtype=np.uint16).tofile('data/train.bin')
tok.save('tokenizer.json')
"

# train (CPU demo — use --device cuda for real training)
python -m model.train \
  --data data/train.bin \
  --vocab-size 1000 \
  --n-layers 4 \
  --d-model 128 \
  --n-heads 4 \
  --context-len 256 \
  --batch-size 4 \
  --max-steps 2000 \
  --device cpu
```

### Run inference

```python
import torch
from model.transformer import GPT, TransformerConfig
from model.tokenizer import BPETokenizer

tok = BPETokenizer.load("tokenizer.json")
cfg = TransformerConfig(vocab_size=len(tok), n_layers=4, d_model=128, n_heads=4, context_len=256)
model = GPT(cfg)
model.load_state_dict(torch.load("checkpoints/ckpt_0002000.pt")["model"])
model.eval()

prompt = tok.encode("To be or not to be")
idx = torch.tensor([prompt])
out = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=40)
print(tok.decode(out[0].tolist()))
```

---

## Tests

```bash
uv run pytest tests/ -v
```

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Type hints, modern syntax |
| Web framework | Flask | Lightweight, SSE-friendly |
| LLM API | Anthropic Claude | Best-in-class reasoning + vision |
| Deep learning | PyTorch | Imperative, debuggable, industry standard |
| Package manager | uv | 10–100× faster than pip |
| Linting | ruff + mypy | Fast, strict |
| Deploy | Vercel | Zero-config, free tier |

---

## License

MIT
