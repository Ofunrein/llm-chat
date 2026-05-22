#!/usr/bin/env bash
# Train a small GPT on arXiv ML abstracts (CPU-friendly, ~2h on HF Spaces)
set -e

python -m model.data \
  --max 50000 \
  --vocab-size 8000 \
  --out data

python -m model.train \
  --data data/train.bin \
  --vocab-size 8000 \
  --context-len 256 \
  --n-layers 6 \
  --d-model 256 \
  --n-heads 8 \
  --d-ff 1024 \
  --dropout 0.1 \
  --batch-size 8 \
  --max-steps 20000 \
  --warmup-steps 500 \
  --max-lr 3e-4 \
  --min-lr 3e-5 \
  --grad-clip 1.0 \
  --log-every 100 \
  --save-every 2000 \
  --out checkpoints/arxiv-small \
  "$@"
