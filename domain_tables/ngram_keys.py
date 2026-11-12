"""Exact integer keys for n-grams over compressed token ids.

The Engram hash (multiplicative XOR, mod a prime per head) is lossy and depends on
the layer seed. Phrase selection and Bloom filters need a key that identifies an
n-gram exactly and is the same for every layer, so we pack the ``n`` compressed
token ids into one int64 in base ``V`` (``V`` = compressed vocab size).

Keys for different orders live in the same integer range, so ``tag_keys`` adds the
order in the top bits when several orders share one structure (e.g. a Bloom filter).
"""

import numpy as np

ORDER_SHIFT = 58  # order tag lives in bits 58..62; packed keys must stay below 2**58


def check_packable(vocab_size: int, max_order: int) -> None:
    if int(vocab_size) ** int(max_order) >= (1 << ORDER_SHIFT):
        raise ValueError(
            f"compressed vocab {vocab_size} ** order {max_order} does not fit in "
            f"{ORDER_SHIFT} bits; use a smaller max n-gram order"
        )


def ngram_keys(compressed_ids: np.ndarray, n: int, vocab_size: int, n_prefix: int = 0) -> np.ndarray:
    """Pack every length-``n`` window ending at each position into an int64 key.

    Args:
        compressed_ids: ``[B, T]`` int array of compressed token ids. Negative ids are
            treated as invalid (any window containing one gets key ``-1``).
        n: n-gram order (>= 1).
        vocab_size: compressed vocabulary size ``V``.
        n_prefix: number of leading columns that are history only (e.g. tokens kept
            from a previous decoding step). They provide context but no keys are
            returned for them.

    Returns:
        ``[B, T - n_prefix]`` int64 array. Positions without ``n`` real tokens of
        history get ``-1``.
    """
    x = np.asarray(compressed_ids, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("compressed_ids must be [B, T]")
    check_packable(vocab_size, n)
    B, T = x.shape
    keys = np.full((B, T), -1, dtype=np.int64)
    if T >= n:
        acc = np.zeros((B, T - n + 1), dtype=np.int64)
        valid = np.ones((B, T - n + 1), dtype=bool)
        for i in range(n):
            window = x[:, i : T - n + 1 + i]
            valid &= window >= 0
            acc = acc * vocab_size + np.where(window >= 0, window, 0)
        keys[:, n - 1 :] = np.where(valid, acc, -1)
    return keys[:, n_prefix:]


def tag_keys(keys: np.ndarray, n: int) -> np.ndarray:
    """Put the order ``n`` into the top bits so keys of different orders never collide."""
    keys = np.asarray(keys, dtype=np.int64)
    return np.where(keys >= 0, keys | (np.int64(n) << np.int64(ORDER_SHIFT)), -1)
