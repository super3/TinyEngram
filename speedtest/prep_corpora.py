"""Write the text corpora for the static n-gram caches and the evaluation prompts (test #2).

med.txt     biomedical training text (Biomed-Enriched, en, biomedical, score > 4.0)
gen.txt     general web text (FineWeb-Edu sample-10BT)
prompts.json  held-out prompts: biomedical (score < 4.0 docs) and later FineWeb-Edu docs
"""

import argparse
import json


def biomed(path, keep):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    for rb in f.iter_batches(batch_size=4096, columns=["text", "language", "domain", "educational_score"]):
        t = rb.to_pydict()
        for text, la, do, ed in zip(t["text"], t["language"], t["domain"], t["educational_score"]):
            if la == "en" and do == "biomedical" and keep(ed or 0):
                yield text


def fineweb():
    from datasets import load_dataset
    for row in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True):
        yield row["text"]


def write_corpus(texts, path, max_bytes):
    n = 0
    used = 0
    with open(path, "w") as f:
        for t in texts:
            f.write(t.strip() + "\n\n")
            n += len(t.encode()) + 2
            used += 1
            if n >= max_bytes:
                break
    return used


def prompt_of(text, chars):
    text = " ".join(text.split())
    if len(text) < chars + 200:
        return None
    cut = text[:chars]
    return cut[: cut.rfind(" ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biomed", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--corpus_mb", type=float, default=20)
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--prompt_chars", type=int, default=400)
    args = ap.parse_args()
    mb = int(args.corpus_mb * 1e6)
    med_docs = write_corpus(biomed(args.biomed, lambda e: e > 4.0), f"{args.out_dir}/med.txt", mb)
    gen_iter = fineweb()
    gen_docs = write_corpus(gen_iter, f"{args.out_dir}/gen.txt", mb)
    prompts = []
    for text in biomed(args.biomed, lambda e: e < 4.0):
        p = prompt_of(text, args.prompt_chars)
        if p:
            prompts.append({"domain": "biomed", "id": len(prompts), "text": p})
        if len(prompts) >= args.n_prompts:
            break
    n_gen = 0
    for i, text in enumerate(gen_iter):  # continues after the docs used for gen.txt
        if i < 1000:
            continue
        p = prompt_of(text, args.prompt_chars)
        if p:
            prompts.append({"domain": "general", "id": len(prompts), "text": p})
            n_gen += 1
        if n_gen >= args.n_prompts:
            break
    json.dump(prompts, open(f"{args.out_dir}/prompts.json", "w"), indent=1)
    print(f"med.txt {med_docs} docs, gen.txt {gen_docs} docs, prompts {len(prompts)}")


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
