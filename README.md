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

A production-quality LLM project in two parts:

1. **Chat app** — streaming multimodal chat interface (FastAPI + Claude API, deployable to Vercel & HF Spaces)
2. **LLM from scratch** — GPT-style transformer + BPE tokenizer implemented with PyTorch, no Hugging Face

---

## Architecture

```
llm-chat/
├── app.py              # FastAPI — async SSE streaming, multimodal endpoints
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
├── Dockerfile          # HF Spaces + self-host
├── vercel.json         # Vercel Python deployment config
└── pyproject.toml      # uv-managed dependencies
```

---

## Chat App

### Features

- **Streaming responses** via Server-Sent Events — token-by-token output with `asyncio`
- **Multimodal input** — attach or drag-and-drop images; Claude analyzes them alongside text
- **Multi-turn history** — full conversation context sent on each request
- **Dark glassmorphism UI** — animated gradient orbs, custom-styled chat bubbles
- **Zero JS frameworks** — vanilla ES2022, fast first paint
- **Auto-generated API docs** — FastAPI `/docs` endpoint (OpenAPI 3.1)

### Quick start

```bash
git clone https://github.com/Ofunrein/llm-chat && cd llm-chat
uv sync
cp .env.example .env  # add ANTHROPIC_API_KEY
uv run python app.py
# → http://localhost:8000  |  API docs: http://localhost:8000/docs
```

### Deploy — Vercel

```bash
vercel --prod
# set ANTHROPIC_API_KEY in Vercel dashboard → Environment Variables
```

### Deploy — Hugging Face Spaces

```bash
huggingface-cli login
huggingface-cli repo create llm-chat --type space --space-sdk docker
git remote add hf https://huggingface.co/spaces/Ofunrein/llm-chat
git push hf main
# set ANTHROPIC_API_KEY in Space Settings → Repository secrets
```

---

## LLM From Scratch

Clean, well-commented GPT implementation — no `transformers`, no abstraction layers.

### What's implemented

| Component | File | Notes |
|---|---|---|
| Multi-head causal self-attention | `model/transformer.py` | Fused QKV, causal mask, scaled dot-product |
| Feed-forward network | `model/transformer.py` | GELU, pre-LayerNorm residual |
| Token + positional embeddings | `model/transformer.py` | Learned, weight-tied LM head |
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

### Train (tiny demo)

```bash
python -m model.train \
  --data data/train.bin \
  --vocab-size 1000 \
  --n-layers 4 --d-model 128 --n-heads 4 \
  --context-len 256 --batch-size 4 --max-steps 2000
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
| Web framework | **FastAPI** | Async-native, `StreamingResponse`, OpenAPI docs |
| LLM API | Anthropic Claude | Best-in-class reasoning + vision |
| Deep learning | PyTorch | Imperative, debuggable, industry standard |
| Package manager | uv | 10–100× faster than pip |
| Linting | ruff + mypy | Fast, strict |
| Deploy | Vercel + HF Spaces | Zero-config, free tier |

---

## License

MIT
