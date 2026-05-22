"""
Download and preprocess arXiv ML/AI/NLP abstracts for LLM training.

Sources (all open-access via Semantic Scholar / arXiv bulk API):
  - cs.AI  — Artificial Intelligence
  - cs.LG  — Machine Learning
  - cs.CL  — Computation and Language (NLP)
  - stat.ML — Statistics / ML

Output
------
  data/train.bin  — uint16 token IDs (90% split)
  data/val.bin    — uint16 token IDs (10% split)
  data/arxiv_abstracts.txt — raw text (optional, for inspection)
  tokenizer.json  — BPE tokenizer trained on corpus

Usage
-----
  uv run python -m model.data              # default: ~50k abstracts
  uv run python -m model.data --max 200000 # larger run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL", "stat.ML"}
_S2_BULK = (
    "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    "?fields=title,abstract,year"
    "&fieldsOfStudy=Computer+Science"
    "&openAccessPdf"
)


def _fetch_s2(max_abstracts: int) -> list[str]:
    """Pull abstracts from Semantic Scholar bulk search (no auth needed, free)."""
    import urllib.request

    abstracts: list[str] = []
    token: str | None = None
    print(f"Fetching up to {max_abstracts:,} abstracts from Semantic Scholar…")

    while len(abstracts) < max_abstracts:
        url = _S2_BULK + "&limit=500"
        if token:
            url += f"&token={token}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llm-from-scratch/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"  fetch error: {e}", file=sys.stderr)
            break

        for paper in data.get("data", []):
            abstract = (paper.get("abstract") or "").strip()
            title = (paper.get("title") or "").strip()
            if abstract and len(abstract) > 80:
                abstracts.append(f"{title}\n{abstract}")

        token = data.get("token")
        print(f"  fetched {len(abstracts):,}", end="\r", flush=True)
        if not token or len(abstracts) >= max_abstracts:
            break

    print(f"\nDone — {len(abstracts):,} abstracts")
    return abstracts[:max_abstracts]


def _fetch_arxiv_sample(max_abstracts: int) -> list[str]:
    """
    Fallback: pull from arXiv API (slower but no rate limit for small sets).
    Returns list of 'Title\\nAbstract' strings.
    """
    import time
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    ns = "http://www.w3.org/2005/Atom"
    base = "https://export.arxiv.org/api/query"
    query = "cat:cs.LG+OR+cat:cs.AI+OR+cat:cs.CL+OR+cat:stat.ML"
    abstracts: list[str] = []
    start = 0
    batch = 200

    print(f"Fetching up to {max_abstracts:,} abstracts from arXiv API…")
    while len(abstracts) < max_abstracts:
        params = urllib.parse.urlencode({
            "search_query": query,
            "start": start,
            "max_results": batch,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        try:
            req = urllib.request.Request(
                f"{base}?{params}",
                headers={"User-Agent": "llm-from-scratch/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                tree = ET.parse(r)
        except Exception as e:
            print(f"  arXiv error: {e}", file=sys.stderr)
            break

        entries = tree.findall(f"{{{ns}}}entry")
        if not entries:
            break

        for entry in entries:
            title_el = entry.find(f"{{{ns}}}title")
            summary_el = entry.find(f"{{{ns}}}summary")
            title = (title_el.text or "").strip() if title_el is not None else ""
            abstract = (summary_el.text or "").strip() if summary_el is not None else ""
            abstract = re.sub(r"\s+", " ", abstract)
            if abstract and len(abstract) > 80:
                abstracts.append(f"{title}\n{abstract}")

        start += batch
        print(f"  fetched {len(abstracts):,}", end="\r", flush=True)
        time.sleep(3)  # arXiv rate limit: 1 req/3s

    print(f"\nDone — {len(abstracts):,} abstracts")
    return abstracts[:max_abstracts]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_SEP = "\n\n<|endoftext|>\n\n"


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()


def _build_corpus(abstracts: list[str]) -> str:
    cleaned = [_clean(a) for a in abstracts if _clean(a)]
    return _SEP.join(cleaned)


# ---------------------------------------------------------------------------
# Tokenise + save
# ---------------------------------------------------------------------------

def _tokenize_and_split(
    corpus: str,
    vocab_size: int,
    train_ratio: float,
    out_dir: Path,
) -> None:
    from model.tokenizer import BPETokenizer

    print(f"Training BPE tokenizer (vocab_size={vocab_size:,})…")
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab_size, verbose=True)
    tok_path = out_dir.parent / "tokenizer.json"
    tok.save(tok_path)
    print(f"Tokenizer saved → {tok_path}")

    print("Encoding corpus…")
    ids = tok.encode(corpus)
    print(f"Total tokens: {len(ids):,}")

    split = int(len(ids) * train_ratio)
    train_ids = np.array(ids[:split], dtype=np.uint16)
    val_ids = np.array(ids[split:], dtype=np.uint16)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")
    print(f"Saved train.bin ({len(train_ids):,} tokens) + val.bin ({len(val_ids):,} tokens) → {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)

    # fetch
    if args.source == "arxiv":
        abstracts = _fetch_arxiv_sample(args.max)
    else:
        try:
            abstracts = _fetch_s2(args.max)
            if not abstracts:
                raise RuntimeError("empty")
        except Exception:
            print("S2 failed, falling back to arXiv API…")
            abstracts = _fetch_arxiv_sample(args.max)

    if not abstracts:
        print("ERROR: no abstracts fetched", file=sys.stderr)
        sys.exit(1)

    # save raw text
    raw_path = out_dir / "arxiv_abstracts.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("\n\n---\n\n".join(abstracts), encoding="utf-8")
    print(f"Raw text saved → {raw_path} ({raw_path.stat().st_size / 1e6:.1f} MB)")

    corpus = _build_corpus(abstracts)
    _tokenize_and_split(corpus, args.vocab_size, args.train_ratio, out_dir)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download + preprocess arXiv ML abstracts")
    p.add_argument("--max", type=int, default=50_000, help="Max abstracts to fetch")
    p.add_argument("--vocab-size", type=int, dest="vocab_size", default=8_000)
    p.add_argument("--train-ratio", type=float, dest="train_ratio", default=0.9)
    p.add_argument("--out", default="data", help="Output directory")
    p.add_argument("--source", choices=["s2", "arxiv"], default="s2",
                   help="Data source: Semantic Scholar (s2) or arXiv API (arxiv)")
    return p


if __name__ == "__main__":
    main(_parser().parse_args())
