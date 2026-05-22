"""
Generate a synthetic arXiv-style ML abstracts dataset for local testing.

For production training, use: python -m model.data --source arxiv
which downloads real abstracts (works on HF Spaces / no proxy).
"""

from __future__ import annotations

import random
from pathlib import Path

_TEMPLATES = [
    "We propose {method}, a novel approach to {task} that achieves state-of-the-art performance on {benchmark}. Our method leverages {technique} to {benefit}. Extensive experiments demonstrate that {method} outperforms existing baselines by {margin}% on standard benchmarks.",
    "Large language models have demonstrated remarkable capabilities in {task}. However, {challenge} remains an open problem. In this work, we introduce {method}, which addresses this limitation through {technique}. We show that {method} achieves {margin}% improvement while reducing computational cost.",
    "We present {method}, a scalable framework for {task}. Unlike prior approaches that rely on {old_approach}, {method} directly {benefit} using {technique}. We validate our approach on {benchmark} and show consistent improvements across diverse settings.",
    "Self-supervised learning has emerged as a powerful paradigm for {task}. We introduce {method}, which {benefit} without requiring labeled data. Our key insight is that {technique} provides a strong learning signal. {method} achieves competitive performance with supervised methods on {benchmark}.",
    "Transformers have revolutionized {task}, yet {challenge} limits their applicability. We propose {method}, which {benefit} through {technique}. Theoretical analysis shows that {method} reduces complexity from O(n²) to O(n log n) while maintaining expressivity.",
    "We study the problem of {task} in the low-data regime. Existing methods suffer from {challenge}. Our approach, {method}, exploits {technique} to {benefit}. Experiments on {benchmark} confirm that {method} outperforms prior work, especially when labeled examples are scarce.",
    "The attention mechanism is central to modern {task} systems. We identify a fundamental limitation: {challenge}. To address this, we propose {method}, a {technique}-based approach that {benefit}. Ablation studies validate each component of {method}.",
    "Generalization remains a central challenge in {task}. We introduce {method}, which {benefit} by incorporating {technique} into the training procedure. We provide theoretical guarantees showing that {method} achieves {margin}% lower generalization error under distribution shift.",
]

_METHODS = ["TransForge", "AttnNet", "ScaleViT", "DiffuBERT", "SparseGPT", "HyperAttn", "MixLayer", "LoFTR-LM", "CausalFlow", "PolyFormer", "GradAlign", "ContrastBench"]
_TASKS = ["natural language understanding", "text generation", "machine translation", "question answering", "few-shot learning", "instruction following", "code generation", "summarization", "reasoning", "multimodal alignment"]
_BENCHMARKS = ["GLUE", "SuperGLUE", "MMLU", "BIG-Bench", "HumanEval", "MATH", "HellaSwag", "WMT-14", "SQuAD 2.0", "ARC-Challenge"]
_TECHNIQUES = ["contrastive learning", "sparse attention", "mixture-of-experts", "retrieval augmentation", "knowledge distillation", "gradient checkpointing", "prefix tuning", "quantization-aware training", "in-context learning", "chain-of-thought prompting"]
_CHALLENGES = ["quadratic attention complexity", "catastrophic forgetting", "data inefficiency", "poor out-of-distribution generalization", "hallucination", "prompt sensitivity", "high memory footprint", "lack of interpretability"]
_BENEFITS = ["achieves linear scaling", "reduces hallucination", "improves sample efficiency", "enables zero-shot transfer", "maintains accuracy under compression", "handles longer contexts", "improves calibration"]
_OLD = ["fine-tuning all parameters", "dense attention", "data augmentation", "task-specific heads", "autoregressive decoding"]
_TITLES = [
    "{method}: {technique} for {task}",
    "Scaling {task} with {technique}",
    "{method}: A {technique} Approach to {task}",
    "Towards Efficient {task} via {technique}",
    "{method}: Improving {task} through {technique}",
]


def _abstract() -> str:
    m = random.choice(_METHODS) + str(random.randint(1, 99))
    t = random.choice(_TASKS)
    b = random.choice(_BENCHMARKS)
    te = random.choice(_TECHNIQUES)
    ch = random.choice(_CHALLENGES)
    be = random.choice(_BENEFITS)
    oa = random.choice(_OLD)
    mg = random.randint(2, 18)
    title_tmpl = random.choice(_TITLES)
    title = title_tmpl.format(method=m, task=t.title(), technique=te.title())
    body = random.choice(_TEMPLATES).format(
        method=m, task=t, benchmark=b, technique=te,
        challenge=ch, benefit=be, old_approach=oa, margin=mg,
    )
    return f"{title}\n{body}"


def generate(n: int = 10_000, seed: int = 42) -> list[str]:
    random.seed(seed)
    return [_abstract() for _ in range(n)]


if __name__ == "__main__":
    import argparse

    import numpy as np

    from model.data import _build_corpus
    from model.tokenizer import BPETokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--vocab-size", type=int, dest="vocab_size", default=4_000)
    p.add_argument("--out", default="data")
    args = p.parse_args()

    print(f"Generating {args.n:,} synthetic abstracts…")
    abstracts = generate(args.n)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    raw = out / "synthetic_abstracts.txt"
    raw.write_text("\n\n---\n\n".join(abstracts))
    print(f"Raw text → {raw} ({raw.stat().st_size / 1e3:.0f} KB)")

    corpus = _build_corpus(abstracts)

    print(f"Training BPE tokenizer (vocab={args.vocab_size})…")
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=args.vocab_size, verbose=True)
    tok.save("tokenizer.json")

    ids = tok.encode(corpus)
    split = int(len(ids) * 0.9)
    np.array(ids[:split], dtype=np.uint16).tofile(out / "train.bin")
    np.array(ids[split:], dtype=np.uint16).tofile(out / "val.bin")
    print(f"train.bin: {split:,} tokens | val.bin: {len(ids)-split:,} tokens")
