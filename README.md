# llm-chat

Production-quality LLM project in two parts:

1. **Chat app** — streaming chat UI powered by GPT-2 medium (345M) running on our from-scratch transformer (FastAPI + Docker, HF Spaces)
2. **LLM from scratch** — GPT-style transformer + BPE tokenizer + training loop in PyTorch, zero Hugging Face dependency for inference

No external LLM APIs. The transformer in `model/transformer.py` IS the model. HF `transformers` is used only at startup to download pretrained GPT-2 tensors, which are then copied into our own `GPT` module — every forward pass runs through `model/transformer.py`.

---

## Live Demo

🤗 **[Open on Hugging Face Spaces](https://huggingface.co/spaces/Ofunrein/llm-chat)**

Try: *"What is the capital of Texas?"* → `Austin.`

---

## Architecture

```
llm-chat/
├── app.py               # FastAPI — async SSE streaming, /chat, /model-info, /health
│                          + few-shot Q:/A: prompt builder, stop-token trimming
├── templates/
│   └── index.html       # Dark glassmorphism UI, animated canvas orbs, inline CSS+JS
├── model/
│   ├── transformer.py   # GPT architecture from scratch: CausalSelfAttention, FeedForward, Block, GPT
│   ├── tokenizer.py     # BPE tokenizer from scratch: train, encode, decode, save, load
│   ├── train.py         # Training loop: AdamW, cosine LR, warmup, grad clip, WandB
│   ├── load_gpt2.py     # Load pretrained GPT-2 weights into our transformer (weight mapping)
│   ├── data.py          # arXiv dataset downloader + BPE tokenization pipeline
│   ├── synthetic_data.py # Synthetic ML abstract generator for local dev
│   └── eval.py          # Perplexity evaluation on val set
├── scripts/
│   └── train_e2e.sh     # End-to-end: download → tokenize → train → eval
├── configs/
│   └── train_arxiv_small.sh
├── tests/
│   ├── test_app.py      # pytest — 17 tests, no GPU required
│   ├── test_training.py # train→checkpoint→inference integration
│   ├── test_transformer.py
│   ├── test_tokenizer.py
│   └── test_data.py
├── Dockerfile           # HF Spaces deployment (torch CPU, uvicorn, gpt2-medium pre-cached)
└── pyproject.toml       # uv-managed deps
```

---

## How Chat Works

Base GPT-2 is a continuation model, not a chat model. To make it answer questions instead of free-associating, `app.py` wraps every user turn in a few-shot Q:/A: harness:

```
The following is a question-answering assistant. It answers each
question concisely and factually.

Q: What is the capital of France?
A: Paris.

Q: Who wrote Romeo and Juliet?
A: William Shakespeare.

Q: What is the largest planet in our solar system?
A: Jupiter.

Q: What is 2 plus 2?
A: 4.

Q: <user message>
A:
```

Generation stops on the next `\nQ:` (or `\n\n`) so the model emits exactly one answer turn, never a hallucinated next question.

Defaults: `temperature=0.7`, `top_k=40`, `max_new_tokens=80`. Override via env vars `TEMPERATURE`, `TOP_K`, `MAX_NEW_TOKENS`.

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
| Few-shot QA prompt + stop tokens | `app.py` | `_build_prompt`, `_run_inference` |

### Model sizes

| Variant | Layers | d_model | Heads | Params | Default? |
|---|---|---|---|---|---|
| `GPT.gpt2_small()` | 12 | 768 | 12 | ~124M | — |
| `GPT.gpt2_medium()` | 24 | 1024 | 16 | ~354M | ✓ deployed |
| `GPT.gpt2_large()` | 36 | 1280 | 20 | ~762M | — |
| `GPT.gpt2_xl()` | 48 | 1600 | 25 | ~1.5B | — |

`gpt2-medium` is the deployed default — `gpt2-small` is too weak for factual QA. Switch via `GPT2_MODEL` env var.

---

## Endpoints

| Route | Method | Body | Returns |
|---|---|---|---|
| `/` | GET | — | Chat UI (HTML) |
| `/chat` | POST | `{"message": str, "history": [{"role","content"}, …]}` | SSE stream of `{"token": "..."}` events, terminated by `[DONE]` |
| `/model-info` | GET | — | `{backend, params_M, device, context_len, implementation}` |
| `/health` | GET | — | `{status, model}` |

---

## Quick Start (local)

```bash
git clone https://github.com/Ofunrein/llm-chat && cd llm-chat
uv sync
uv run python app.py
# → http://localhost:7860
```

Optional env:

```bash
export GPT2_MODEL=gpt2-medium    # gpt2 | gpt2-medium | gpt2-large | gpt2-xl
export TEMPERATURE=0.7
export TOP_K=40
export MAX_NEW_TOKENS=80
export CHECKPOINT_PATH=/path/to/ckpt.pt   # use a fine-tuned checkpoint instead
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

Then point the app at the checkpoint:

```bash
CHECKPOINT_PATH=runs/<run>/ckpt.pt uv run python app.py
```

---

## Tests

```bash
uv run pytest tests/ -v   # 17 tests, no GPU needed
```

---

## Deploy (HF Spaces)

```bash
git remote add hf https://Ofunrein:<HF_TOKEN>@huggingface.co/spaces/Ofunrein/llm-chat
git push hf main
```

Build pulls `gpt2-medium` weights at image-build time so cold-start is fast (~30s instead of ~2min).

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Type hints throughout |
| Framework | FastAPI | Async-native, `StreamingResponse`, OpenAPI docs at `/docs` |
| Inference | PyTorch (CPU/CUDA) | The from-scratch transformer |
| Package manager | uv | 10–100× faster than pip |
| Deploy | HF Spaces (Docker) | Free, supports PyTorch, persistent container |

> **Kubernetes**: overkill for a single-model demo. Add a `k8s/` dir with a Deployment + Service + HPA if you need multi-replica auto-scaling — the Dockerfile is already production-ready.

---

## License

MIT
