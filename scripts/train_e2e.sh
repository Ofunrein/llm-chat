#!/usr/bin/env bash
# Run end-to-end arXiv training on HF Spaces (or any machine with internet + CPU/GPU).
# Usage: bash scripts/train_e2e.sh [--gpu] [--max 50000]
#
# Steps:
#   1. Download arXiv ML abstracts (or synthetic fallback)
#   2. Train BPE tokenizer + encode corpus
#   3. Train GPT from scratch
#   4. Evaluate perplexity on val set
#   5. Print inference sample

set -euo pipefail

MAX=${MAX:-20000}
VOCAB=${VOCAB:-8000}
STEPS=${STEPS:-10000}
LR=${LR:-3e-4}
DEVICE=${DEVICE:-cpu}
OUT=checkpoints/arxiv-e2e

# parse flags
for arg in "$@"; do
  case $arg in
    --gpu)  DEVICE=cuda ;;
    --max=*) MAX="${arg#*=}" ;;
  esac
done

echo "=== Step 1: Download arXiv abstracts (max=$MAX) ==="
python3 -m model.data \
  --max "$MAX" \
  --vocab-size "$VOCAB" \
  --out data \
  --source s2 || python3 -m model.data \
  --max "$MAX" \
  --vocab-size "$VOCAB" \
  --out data \
  --source arxiv

echo ""
echo "=== Step 2: Train GPT ==="
python3 -m model.train \
  --data data/train.bin \
  --vocab-size "$VOCAB" \
  --context-len 128 \
  --n-layers 6 \
  --d-model 256 \
  --n-heads 8 \
  --d-ff 1024 \
  --dropout 0.1 \
  --batch-size 8 \
  --max-steps "$STEPS" \
  --warmup-steps 300 \
  --max-lr "$LR" \
  --min-lr 3e-5 \
  --log-every 100 \
  --save-every 2000 \
  --device "$DEVICE" \
  --out "$OUT"

echo ""
echo "=== Step 3: Evaluate ==="
CKPT=$(ls "$OUT"/ckpt_*.pt 2>/dev/null | sort | tail -1)
if [ -n "$CKPT" ]; then
  python3 -m model.eval \
    --checkpoint "$CKPT" \
    --val-bin data/val.bin \
    --device "$DEVICE"

  echo ""
  echo "=== Step 4: Sample inference ==="
  python3 - <<'PYEOF'
import torch
from model.transformer import GPT
from model.tokenizer import BPETokenizer
import os, glob

ckpt_files = sorted(glob.glob("checkpoints/arxiv-e2e/ckpt_*.pt"))
if not ckpt_files:
    print("No checkpoint found")
    exit()

ckpt = torch.load(ckpt_files[-1], map_location="cpu", weights_only=False)
cfg = ckpt["config"]
model = GPT(cfg)
model.load_state_dict(ckpt["model"])
model.eval()

tok = BPETokenizer.load("tokenizer.json")
prompt = "We propose a novel approach to"
ids = tok.encode(prompt)
idx = torch.tensor([ids])
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=80, temperature=0.8, top_k=40)
new_ids = out[0, len(ids):].tolist()
print(f"Prompt: {prompt}")
print(f"Output: {tok.decode(new_ids)}")
PYEOF
else
  echo "No checkpoint produced — check training logs above"
fi

echo ""
echo "=== Done. To serve this checkpoint: ==="
echo "  CHECKPOINT_PATH=$CKPT uvicorn app:app --port 7860"
