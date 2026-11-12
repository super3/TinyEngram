"""Registered n-gram allowlist for a domain table.

A ``PhraseList`` stores, per n-gram order, the sorted int64 keys
(see ``ngram_keys``) of the phrases a table is allowed to fire on. Membership is
exact (binary search). ``to_bloom`` builds the compact serving-time filter.

File format (``.npz``): ``keys_<n>`` arrays plus a JSON ``meta`` blob with the
compressed vocab size, the tokenizer it was built with and selection stats.
"""

import json
from typing import Dict, Optional

import numpy as np

from .bloom import BloomFilter
from .ngram_keys import check_packable, tag_keys


class PhraseList:
    def __init__(self, keys_by_order: Dict[int, np.ndarray], vocab_size: int, meta: Optional[dict] = None):
        self.vocab_size = int(vocab_size)
        self.keys = {}
        for n, keys in keys_by_order.items():
            n = int(n)
            check_packable(self.vocab_size, n)
            self.keys[n] = np.unique(np.asarray(keys, dtype=np.int64))
        self.meta = dict(meta or {})

    @property
    def orders(self):
        return sorted(self.keys)

    def __len__(self):
        return int(sum(k.size for k in self.keys.values()))

    def contains(self, keys: np.ndarray, n: int) -> np.ndarray:
        """Exact membership for order-``n`` keys of any shape. ``-1`` is never a member."""
        keys = np.asarray(keys, dtype=np.int64)
        table = self.keys.get(int(n))
        if table is None or table.size == 0:
            return np.zeros(keys.shape, dtype=bool)
        idx = np.searchsorted(table, keys)
        idx = np.minimum(idx, table.size - 1)
        return (table[idx] == keys) & (keys >= 0)

    def to_bloom(self, fp_rate: float = 0.01, seed: int = 0) -> BloomFilter:
        bf = BloomFilter.for_capacity(len(self), fp_rate=fp_rate, seed=seed)
        for n, keys in self.keys.items():
            bf.add(tag_keys(keys, n))
        return bf

    def save(self, path: str) -> None:
        meta = dict(self.meta)
        meta["vocab_size"] = self.vocab_size
        arrays = {f"keys_{n}": k for n, k in self.keys.items()}
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "PhraseList":
        with np.load(path) as z:
            meta = json.loads(z["meta"].tobytes().decode())
            keys = {int(name.split("_", 1)[1]): z[name] for name in z.files if name.startswith("keys_")}
        return cls(keys, vocab_size=meta.pop("vocab_size"), meta=meta)
