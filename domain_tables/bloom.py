"""A small vectorized Bloom filter over int64 keys (numpy only).

Sized from the expected item count and target false-positive rate:
``m = -n ln(p) / ln(2)^2`` bits and ``k = (m / n) ln 2`` probes, i.e. about
9.6 bits per key for p = 1%. Probes use double hashing over two splitmix64 mixes.
There are no false negatives.
"""

import json
import math

import numpy as np

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    # uint64 arithmetic wraps mod 2**64, which is what splitmix64 wants.
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


class BloomFilter:
    def __init__(self, num_bits: int, num_hashes: int, seed: int = 0, bits: np.ndarray = None):
        self.num_bits = int(max(8, num_bits))
        self.num_hashes = int(max(1, num_hashes))
        self.seed = int(seed)
        n_bytes = (self.num_bits + 7) // 8
        if bits is None:
            bits = np.zeros(n_bytes, dtype=np.uint8)
        if bits.shape != (n_bytes,):
            raise ValueError("bit array has the wrong size")
        self.bits = bits
        self.num_items = 0

    @classmethod
    def for_capacity(cls, n_items: int, fp_rate: float = 0.01, seed: int = 0) -> "BloomFilter":
        n_items = max(1, int(n_items))
        m = math.ceil(-n_items * math.log(fp_rate) / (math.log(2) ** 2))
        k = max(1, round(m / n_items * math.log(2)))
        return cls(m, k, seed=seed)

    def _positions(self, keys: np.ndarray) -> np.ndarray:
        k = np.asarray(keys, dtype=np.int64).reshape(-1).view(np.uint64)
        seed = np.uint64(self.seed)
        with np.errstate(over="ignore"):
            h1 = _splitmix64(k ^ seed)
            h2 = _splitmix64(k ^ (seed + np.uint64(0x632BE59BD9B4E019))) | np.uint64(1)
            i = np.arange(self.num_hashes, dtype=np.uint64)
            pos = (h1[:, None] + i[None, :] * h2[:, None]) % np.uint64(self.num_bits)
        return pos.astype(np.int64)  # [N, k]

    def add(self, keys: np.ndarray) -> None:
        keys = np.asarray(keys, dtype=np.int64).reshape(-1)
        keys = keys[keys >= 0]
        if keys.size == 0:
            return
        pos = self._positions(keys).reshape(-1)
        np.bitwise_or.at(self.bits, pos >> 3, (np.uint8(1) << (pos & 7).astype(np.uint8)))
        self.num_items += int(keys.size)

    def contains(self, keys: np.ndarray) -> np.ndarray:
        """Vectorized membership test. Negative keys are never members."""
        keys = np.asarray(keys, dtype=np.int64)
        shape = keys.shape
        flat = keys.reshape(-1)
        out = np.zeros(flat.shape, dtype=bool)
        valid = flat >= 0
        if valid.any():
            pos = self._positions(flat[valid])
            hit = (self.bits[pos >> 3] >> (pos & 7).astype(np.uint8)) & np.uint8(1)
            out[valid] = hit.all(axis=1)
        return out.reshape(shape)

    @property
    def size_bytes(self) -> int:
        return int(self.bits.nbytes)

    def save(self, path: str) -> None:
        meta = {"num_bits": self.num_bits, "num_hashes": self.num_hashes,
                "seed": self.seed, "num_items": self.num_items}
        np.savez(path, bits=self.bits, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))

    @classmethod
    def load(cls, path: str) -> "BloomFilter":
        with np.load(path) as z:
            meta = json.loads(z["meta"].tobytes().decode())
            bf = cls(meta["num_bits"], meta["num_hashes"], seed=meta["seed"], bits=z["bits"].copy())
        bf.num_items = meta.get("num_items", 0)
        return bf
