"""Phrase selection: register only n-grams that are much more frequent in the domain.

Counts bigrams/trigrams (over compressed token ids, the same ids the Engram hash
uses) in a domain corpus and a general corpus, then scores each domain n-gram with
the log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn,
2008, "Fightin' Words"). N-grams with enough domain support and a high z-score are
kept; common phrases are left to the base model.
"""

import glob
import json
import os
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

from .ngram_keys import ngram_keys


# --------------------------------------------------------------------- corpora


def _parse_filter(expr: Optional[str]):
    """``"language==en,domain==biomedical,educational_score>4.0"`` -> list of predicates."""
    if not expr:
        return []
    preds = []
    for part in expr.split(","):
        part = part.strip()
        for op in ("==", ">=", "<=", "!=", ">", "<"):
            if op in part:
                col, val = (s.strip() for s in part.split(op, 1))
                preds.append((col, op, val))
                break
        else:
            raise ValueError(f"cannot parse filter clause {part!r}")
    return preds


def _apply_filter(table, preds):
    import pyarrow.compute as pc

    if not preds:
        return table
    mask = None
    for col, op, val in preds:
        column = table.column(col)
        if op in ("==", "!=") and not _is_number(val):
            cond = pc.equal(column, val) if op == "==" else pc.not_equal(column, val)
        else:
            num = float(val)
            cond = {
                "==": pc.equal, "!=": pc.not_equal, ">": pc.greater,
                "<": pc.less, ">=": pc.greater_equal, "<=": pc.less_equal,
            }[op](column, num)
        mask = cond if mask is None else pc.and_(mask, cond)
    return table.filter(mask)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def iter_texts(
    files: Optional[str] = None,
    text_column: str = "text",
    filter_expr: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_config: Optional[str] = None,
    hf_split: str = "train",
    max_docs: Optional[int] = None,
    batch_size: int = 1024,
) -> Iterator[List[str]]:
    """Yield batches of texts from local parquet/jsonl/txt files or a streamed HF dataset."""
    seen = 0

    def take(batch):
        nonlocal seen
        if max_docs is not None:
            batch = batch[: max(0, max_docs - seen)]
        seen += len(batch)
        return batch

    if hf_dataset:
        from datasets import load_dataset

        ds = load_dataset(hf_dataset, hf_config, split=hf_split, streaming=True)
        buf = []
        for row in ds:
            buf.append(row[text_column])
            if len(buf) == batch_size:
                out = take(buf)
                if out:
                    yield out
                buf = []
                if max_docs is not None and seen >= max_docs:
                    return
        if buf:
            out = take(buf)
            if out:
                yield out
        return

    paths = sorted(glob.glob(files)) if files else []
    if not paths:
        raise FileNotFoundError(f"no files match {files!r}")
    preds = _parse_filter(filter_expr)
    for path in paths:
        if path.endswith(".parquet"):
            import pyarrow as pa
            import pyarrow.parquet as pq

            cols = sorted({text_column} | {c for c, _, _ in preds})
            for rb in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=cols):
                table = _apply_filter(pa.Table.from_batches([rb]), preds)
                out = take(table.column(text_column).to_pylist())
                if out:
                    yield out
                if max_docs is not None and seen >= max_docs:
                    return
        else:
            buf = []
            with open(path) as f:
                for line in f:
                    if path.endswith(".jsonl"):
                        row = json.loads(line)
                        text = row[text_column]
                    else:
                        text = line.rstrip("\n")
                    if text:
                        buf.append(text)
                    if len(buf) == batch_size:
                        out = take(buf)
                        if out:
                            yield out
                        buf = []
                        if max_docs is not None and seen >= max_docs:
                            return
            if buf:
                out = take(buf)
                if out:
                    yield out
        if max_docs is not None and seen >= max_docs:
            return


# -------------------------------------------------------------------- counting


def _merge_counts(keys_a, cnt_a, keys_b, cnt_b):
    keys = np.concatenate([keys_a, keys_b])
    cnts = np.concatenate([cnt_a, cnt_b])
    order = np.argsort(keys, kind="stable")
    keys, cnts = keys[order], cnts[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(keys)) + 1])
    return keys[starts], np.add.reduceat(cnts, starts)


class NgramCounter:
    """Exact n-gram counts over compressed ids, merged in numpy chunks."""

    def __init__(self, orders: Iterable[int], vocab_size: int, flush_every: int = 20_000_000):
        self.orders = list(orders)
        self.vocab_size = int(vocab_size)
        self.flush_every = int(flush_every)
        self.keys = {n: np.zeros(0, dtype=np.int64) for n in self.orders}
        self.counts = {n: np.zeros(0, dtype=np.int64) for n in self.orders}
        self.totals = {n: 0 for n in self.orders}
        self.tokens = 0
        self.docs = 0
        self._buf = {n: [] for n in self.orders}
        self._buffered = 0

    def add_docs(self, docs: List[np.ndarray]) -> None:
        """``docs``: list of 1-D compressed id arrays. N-grams never cross document boundaries."""
        if not docs:
            return
        sep = np.full(1, -1, dtype=np.int64)
        flat = np.concatenate([np.concatenate([np.asarray(d, dtype=np.int64), sep]) for d in docs])[None, :]
        self.tokens += int(sum(len(d) for d in docs))
        self.docs += len(docs)
        for n in self.orders:
            k = ngram_keys(flat, n, self.vocab_size)[0]
            k = k[k >= 0]
            self._buf[n].append(k)
            self.totals[n] += int(k.size)
            self._buffered += int(k.size)
        if self._buffered >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        for n in self.orders:
            if not self._buf[n]:
                continue
            arr = np.concatenate(self._buf[n])
            uniq, cnt = np.unique(arr, return_counts=True)
            self.keys[n], self.counts[n] = _merge_counts(self.keys[n], self.counts[n], uniq, cnt.astype(np.int64))
            self._buf[n] = []
        self._buffered = 0

    def save(self, path: str) -> None:
        self.flush()
        arrays = {}
        for n in self.orders:
            arrays[f"keys_{n}"] = self.keys[n]
            arrays[f"counts_{n}"] = self.counts[n]
            arrays[f"total_{n}"] = np.array(self.totals[n])
        arrays["meta"] = np.array([self.vocab_size, self.tokens, self.docs])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "NgramCounter":
        with np.load(path) as z:
            orders = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("keys_"))
            vocab_size, tokens, docs = (int(v) for v in z["meta"])
            c = cls(orders, vocab_size)
            for n in orders:
                c.keys[n], c.counts[n], c.totals[n] = z[f"keys_{n}"], z[f"counts_{n}"], int(z[f"total_{n}"])
        c.tokens, c.docs = tokens, docs
        return c

    def lookup(self, n: int, keys: np.ndarray) -> np.ndarray:
        table, cnt = self.keys[n], self.counts[n]
        if table.size == 0:
            return np.zeros(keys.shape, dtype=np.int64)
        idx = np.minimum(np.searchsorted(table, keys), table.size - 1)
        return np.where(table[idx] == keys, cnt[idx], 0)


def count_corpus(batches: Iterable[List[str]], tokenizer, compressor, orders, vocab_size, log_every=0) -> NgramCounter:
    counter = NgramCounter(orders, vocab_size)
    for i, texts in enumerate(batches):
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        counter.add_docs([compressor(np.asarray(x, dtype=np.int64)) for x in ids if len(x)])
        if log_every and (i + 1) % log_every == 0:
            print(f"  {counter.docs:,} docs, {counter.tokens:,} tokens")
    counter.flush()
    return counter


# --------------------------------------------------------------------- scoring


def log_odds_z(y_d, n_d, y_g, n_g, alpha0: float):
    """Log-odds ratio z-score with an informative Dirichlet prior from the pooled counts."""
    y_d = y_d.astype(np.float64)
    y_g = y_g.astype(np.float64)
    alpha = alpha0 * (y_d + y_g) / float(n_d + n_g)
    delta = (np.log(y_d + alpha) - np.log(n_d + alpha0 - y_d - alpha)) - (
        np.log(y_g + alpha) - np.log(n_g + alpha0 - y_g - alpha)
    )
    var = 1.0 / (y_d + alpha) + 1.0 / (y_g + alpha)
    return delta / np.sqrt(var)


def split_keys(keys: np.ndarray, n: int, vocab_size: int) -> np.ndarray:
    """Unpack order-``n`` keys into their compressed ids: ``[len(keys), n]``."""
    keys = np.asarray(keys, dtype=np.int64)
    out = np.empty((keys.size, n), dtype=np.int64)
    rest = keys.copy()
    for i in range(n - 1, -1, -1):
        out[:, i] = rest % vocab_size
        rest //= vocab_size
    return out


def token_classes(tokenizer, lookup_table: np.ndarray, vocab_size: int):
    """Per compressed id: does it contain a letter, and is it layout (tab/newline) only."""
    rep = representative_tokens(lookup_table, vocab_size)
    texts = tokenizer.batch_decode([[int(t)] for t in rep])
    has_alpha = np.array([any(ch.isalpha() for ch in t) for t in texts], dtype=bool)
    is_layout = np.array([("\t" in t or "\n" in t) and not t.strip() for t in texts], dtype=bool)
    return has_alpha, is_layout


def lexical_filter(has_alpha: np.ndarray, is_layout: np.ndarray, vocab_size: int):
    """Keep n-grams with at least one token containing a letter and no layout-only token.

    Drops numbers, punctuation runs and table formatting (e.g. "\\t0.", "0.05"), which are
    frequent in scientific text but carry no domain knowledge worth a table row.
    """

    def keep(keys, n):
        ids = split_keys(keys, n, vocab_size)
        return has_alpha[ids].any(axis=1) & ~is_layout[ids].any(axis=1)

    return keep


def select_phrases(
    domain: NgramCounter,
    general: NgramCounter,
    min_count: int = 5,
    min_z: float = 3.0,
    min_ratio: float = 2.0,
    max_per_order: Optional[Dict[int, int]] = None,
    alpha0: float = 1000.0,
    keep_fn=None,
):
    """Return ``{n: (keys, z, domain_count, general_count)}`` sorted by z, descending.

    ``keep_fn(keys, n) -> bool mask`` optionally filters candidates (e.g. ``lexical_filter``).
    """
    out = {}
    for n in domain.orders:
        keys, y_d = domain.keys[n], domain.counts[n]
        keep = y_d >= min_count
        keys, y_d = keys[keep], y_d[keep]
        if keep_fn is not None and keys.size:
            keep = keep_fn(keys, n)
            keys, y_d = keys[keep], y_d[keep]
        y_g = general.lookup(n, keys)
        n_d, n_g = max(domain.totals[n], 1), max(general.totals[n], 1)
        z = log_odds_z(y_d, n_d, y_g, n_g, alpha0)
        # rate ratio with add-one smoothing on the general side
        ratio = (y_d / n_d) / ((y_g + 1.0) / n_g)
        keep = (z >= min_z) & (ratio >= min_ratio)
        keys, z, y_d, y_g = keys[keep], z[keep], y_d[keep], y_g[keep]
        order = np.argsort(-z, kind="stable")
        cap = (max_per_order or {}).get(n)
        if cap:
            order = order[:cap]
        out[n] = (keys[order], z[order], y_d[order], y_g[order])
    return out


def decode_key(key: int, n: int, vocab_size: int, representative: np.ndarray, tokenizer) -> str:
    ids = []
    for _ in range(n):
        ids.append(int(key % vocab_size))
        key //= vocab_size
    ids = [int(representative[c]) for c in reversed(ids)]
    return tokenizer.decode(ids)


def representative_tokens(lookup_table: np.ndarray, vocab_size: int) -> np.ndarray:
    """For each compressed id, the first original token id that maps to it (for display)."""
    rep = np.zeros(vocab_size, dtype=np.int64)
    uniq, first = np.unique(lookup_table, return_index=True)
    rep[uniq] = first
    return rep
