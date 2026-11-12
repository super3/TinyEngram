"""Build a domain phrase list (+ Bloom filter) for a DomainEngram table.

Example (Biomed-Enriched vs FineWeb-Edu):

    python tools/select_phrases.py \
        --tokenizer Qwen/Qwen3-0.6B \
        --domain_files "dataset/biomed/*.parquet" \
        --domain_filter "language==en,domain==biomedical,educational_score>4.0" \
        --domain_max_docs 400000 \
        --general_dataset HuggingFaceFW/fineweb-edu --general_config sample-10BT \
        --general_max_docs 400000 \
        --out phrase_lists/biomed_v1

Writes ``<out>.npz`` (exact keys), ``<out>.bloom.npz`` and ``<out>.report.json``
(selection stats and the top phrases decoded, for inspection).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engram import get_compressed_tokenizer  # noqa: E402
from domain_tables.phrase_list import PhraseList  # noqa: E402
from domain_tables.phrase_select import (  # noqa: E402
    NgramCounter,
    count_corpus,
    decode_key,
    iter_texts,
    lexical_filter,
    representative_tokens,
    select_phrases,
    token_classes,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True, help="tokenizer of the base model the table targets")
    p.add_argument("--orders", type=int, nargs="+", default=[2, 3])
    for side in ("domain", "general"):
        p.add_argument(f"--{side}_files", default=None, help="glob of .parquet/.jsonl/.txt files")
        p.add_argument(f"--{side}_dataset", default=None, help="HF dataset id (streamed)")
        p.add_argument(f"--{side}_config", default=None)
        p.add_argument(f"--{side}_split", default="train")
        p.add_argument(f"--{side}_text_column", default="text")
        p.add_argument(f"--{side}_filter", default=None, help='e.g. "language==en,educational_score>4.0"')
        p.add_argument(f"--{side}_max_docs", type=int, default=None)
    p.add_argument("--min_count", type=int, default=5, help="min occurrences in the domain corpus")
    p.add_argument("--min_z", type=float, default=3.0, help="min log-odds z-score")
    p.add_argument("--min_ratio", type=float, default=2.0, help="min domain/general rate ratio")
    p.add_argument("--max_per_order", type=int, nargs="+", default=[200000, 200000])
    p.add_argument("--alpha0", type=float, default=1000.0, help="Dirichlet prior mass")
    p.add_argument("--no_lexical_filter", action="store_true",
                   help="keep numeric/punctuation/layout n-grams (dropped by default)")
    p.add_argument("--bloom_fp_rate", type=float, default=0.01)
    p.add_argument("--report_top", type=int, default=100)
    p.add_argument("--counts_dir", default=None,
                   help="cache n-gram counts here; reused on later runs (for threshold sweeps)")
    p.add_argument("--out", required=True, help="output prefix")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ct = get_compressed_tokenizer(args.tokenizer)
    V = len(ct)
    print(f"tokenizer {args.tokenizer}: {len(tok)} ids -> {V} compressed ids")

    counters = {}
    for side in ("domain", "general"):
        t0 = time.time()
        a = vars(args)
        cache = os.path.join(args.counts_dir, f"{side}_counts.npz") if args.counts_dir else None
        if cache and os.path.exists(cache):
            print(f"loading cached {side} counts from {cache}")
            counters[side] = NgramCounter.load(cache)
        else:
            print(f"counting {side} corpus ...")
            batches = iter_texts(
                files=a[f"{side}_files"], text_column=a[f"{side}_text_column"], filter_expr=a[f"{side}_filter"],
                hf_dataset=a[f"{side}_dataset"], hf_config=a[f"{side}_config"], hf_split=a[f"{side}_split"],
                max_docs=a[f"{side}_max_docs"],
            )
            counters[side] = count_corpus(batches, tok, ct, args.orders, V, log_every=50)
            if cache:
                os.makedirs(args.counts_dir, exist_ok=True)
                counters[side].save(cache)
        c = counters[side]
        print(f"  {side}: {c.docs:,} docs, {c.tokens:,} tokens, "
              + ", ".join(f"{n}-grams {c.totals[n]:,} ({c.keys[n].size:,} distinct)" for n in args.orders)
              + f" [{time.time() - t0:.0f}s]")

    caps = dict(zip(args.orders, args.max_per_order))
    keep_fn = None
    if not args.no_lexical_filter:
        keep_fn = lexical_filter(*token_classes(tok, ct.lookup_table, V), vocab_size=V)
    sel = select_phrases(counters["domain"], counters["general"], min_count=args.min_count, min_z=args.min_z,
                         min_ratio=args.min_ratio, max_per_order=caps, alpha0=args.alpha0, keep_fn=keep_fn)

    rep = representative_tokens(ct.lookup_table, V)
    report = {"args": vars(args), "compressed_vocab": V, "orders": {}}
    dom, gen = counters["domain"], counters["general"]
    for n, (keys, z, y_d, y_g) in sel.items():
        token_cov = float(y_d.sum() / max(dom.totals[n], 1))
        gen_cov = float(y_g.sum() / max(gen.totals[n], 1))
        report["orders"][str(n)] = {
            "selected": int(keys.size),
            "candidates": int((dom.counts[n] >= args.min_count).sum()),
            "domain_occurrence_coverage": token_cov,
            "general_occurrence_coverage": gen_cov,
            "top": [
                {"phrase": decode_key(int(k), n, V, rep, tok), "z": round(float(zz), 2),
                 "domain": int(d), "general": int(g)}
                for k, zz, d, g in zip(keys[: args.report_top], z, y_d, y_g)
            ],
        }
        print(f"order {n}: selected {keys.size:,}; they cover {token_cov:.1%} of domain and "
              f"{gen_cov:.1%} of general {n}-gram occurrences")
        for row in report["orders"][str(n)]["top"][:15]:
            print(f"    {row['z']:8.1f}  d={row['domain']:<7} g={row['general']:<6} {row['phrase']!r}")

    meta = {"tokenizer": args.tokenizer, "selection": {k: v for k, v in vars(args).items() if k != "out"},
            "domain_docs": dom.docs, "domain_tokens": dom.tokens,
            "general_docs": counters["general"].docs, "general_tokens": counters["general"].tokens}
    pl = PhraseList({n: s[0] for n, s in sel.items()}, vocab_size=V, meta=meta)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pl.save(args.out + ".npz")
    bloom = pl.to_bloom(fp_rate=args.bloom_fp_rate)
    bloom.save(args.out + ".bloom.npz")
    report["bloom_bytes"] = bloom.size_bytes
    with open(args.out + ".report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}.npz ({len(pl):,} phrases), {args.out}.bloom.npz ({bloom.size_bytes / 1e6:.2f} MB), "
          f"{args.out}.report.json")


if __name__ == "__main__":
    main()
    # Streaming `datasets` leaves background threads that can crash interpreter shutdown
    # after all outputs are written; exit directly.
    sys.stdout.flush()
    os._exit(0)
