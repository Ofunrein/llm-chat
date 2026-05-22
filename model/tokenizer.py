"""
Byte-Pair Encoding (BPE) tokenizer — built from scratch.

Implements the original Sennrich et al. (2016) algorithm:
  1. Initialise vocabulary as individual UTF-8 bytes.
  2. Count pair frequencies in the corpus.
  3. Merge the most frequent pair into a new token.
  4. Repeat until vocab_size is reached.

Usage
-----
tok = BPETokenizer()
tok.train(text, vocab_size=1000)
ids = tok.encode("hello world")
text = tok.decode(ids)
tok.save("tokenizer.json")
tok2 = BPETokenizer.load("tokenizer.json")
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


class BPETokenizer:
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []          # ordered merge rules
        self.vocab: dict[int, bytes] = {}                # id → bytes
        self._encoder: dict[bytes, int] = {}             # bytes → id

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train BPE on *text* until the vocabulary reaches *vocab_size*."""
        assert vocab_size >= 256, "vocab_size must be >= 256 (byte vocabulary)"

        # initialise with all 256 possible byte values
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._encoder = {v: k for k, v in self.vocab.items()}

        # tokenise into bytes
        ids = list(text.encode("utf-8"))

        num_merges = vocab_size - 256
        for i in range(num_merges):
            counts = _pair_counts(ids)
            if not counts:
                break
            best = max(counts, key=counts.__getitem__)
            new_id = 256 + i
            ids = _merge(ids, best, new_id)
            self.merges.append(best)
            new_bytes = self.vocab[best[0]] + self.vocab[best[1]]
            self.vocab[new_id] = new_bytes
            self._encoder[new_bytes] = new_id
            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i+1}/{num_merges}: {best} → {new_id} ({new_bytes!r})")

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        ids = list(text.encode("utf-8"))
        for pair in self.merges:
            ids = _merge(ids, pair, self._encoder[self.vocab[pair[0]] + self.vocab[pair[1]]])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string (lossy for invalid UTF-8)."""
        raw = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {
            "merges": self.merges,
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tok = cls()
        data = json.loads(Path(path).read_text())
        tok.merges = [tuple(m) for m in data["merges"]]  # type: ignore[assignment]
        tok.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        tok._encoder = {v: k for k, v in tok.vocab.items()}
        return tok

    def __len__(self) -> int:
        return len(self.vocab)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of *pair* in *ids* with *new_id*."""
    result: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result
