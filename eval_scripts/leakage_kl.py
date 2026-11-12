"""Measure how much a table changes the base model's predictions, per corpus.

For each corpus, runs every batch twice (table off, table on) and reports, over real
(non-padding) next-token positions:

  * mean KL(base || base+table) in nats, overall and on positions where the table fired
  * top-1 flip rate: how often the argmax prediction changes
  * fire rate from table telemetry
  * changed fraction: positions whose logits moved at all. This is larger than the fire
    rate even with strict masking, because attention carries what the table injected at a
    matched position into every later position
  * mean NLL with the table off and on (loss delta)

A selective table should show near-zero KL and flips on general text and a clear
NLL drop on held-out domain text. This is cheaper and more sensitive than benchmark
accuracy for catching leakage.

    python eval_scripts/leakage_kl.py --base_model Qwen/Qwen3-0.6B-Base --table_dir runs/.../final \
        --corpus domain_eval:files=dataset/biomed/*.parquet,filter=language==en;domain==biomedical;educational_score<4.0 \
        --corpus general:hf=HuggingFaceFW/fineweb-edu,config=sample-10BT \
        --max_docs 500 --out results/leakage_table.json

Corpus spec: ``name:key=value,...`` with keys files | hf | config | split | column | filter
(use ``;`` between filter clauses).
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_corpus(spec):
    name, _, rest = spec.partition(":")
    kv = dict(item.split("=", 1) for item in rest.split(",") if item)
    return name, kv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--table_dir", required=True)
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--max_docs", type=int, default=500)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selection_backend", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from domain_tables import DomainEngram
    from domain_tables.io import load_model_with_tables
    from domain_tables.phrase_select import iter_texts
    from engram_qwen import set_skip_engram

    overrides = {"telemetry": True}
    if args.selection_backend:
        overrides["selection_backend"] = args.selection_backend
    model = load_model_with_tables(args.base_model, args.table_dir, dtype=getattr(torch, args.dtype), **overrides)
    model.to(args.device).eval()
    tables = [m for m in model.modules() if isinstance(m, DomainEngram)]
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    report = {"base_model": args.base_model, "table_dir": args.table_dir, "corpora": {}}
    for spec in args.corpus:
        name, kv = parse_corpus(spec)
        batches = iter_texts(
            files=kv.get("files"), hf_dataset=kv.get("hf"), hf_config=kv.get("config"),
            hf_split=kv.get("split", "train"), text_column=kv.get("column", "text"),
            filter_expr=kv.get("filter", "").replace(";", ",") or None,
            max_docs=args.max_docs, batch_size=args.batch_size,
        )
        for m in tables:
            m.reset_telemetry()
        acc = {"positions": 0, "kl_sum": 0.0, "flips": 0, "nll_off": 0.0, "nll_on": 0.0,
               "fired_positions": 0, "kl_sum_fired": 0.0}
        for texts in batches:
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
            ids = enc.input_ids.to(args.device)
            mask = enc.attention_mask.to(args.device)
            with torch.no_grad():
                set_skip_engram(True)
                logits_off = model(ids, attention_mask=mask, use_cache=False).logits[:, :-1].float()
                set_skip_engram(False)
                logits_on = model(ids, attention_mask=mask, use_cache=False).logits[:, :-1].float()
            valid = mask[:, 1:].bool() & mask[:, :-1].bool()
            target = ids[:, 1:]
            lp_off = F.log_softmax(logits_off, -1)
            lp_on = F.log_softmax(logits_on, -1)
            kl = (lp_off.exp() * (lp_off - lp_on)).sum(-1)  # KL(base || table) per position
            changed = (logits_on - logits_off).abs().amax(-1) > 0  # where the table altered anything
            acc["positions"] += int(valid.sum())
            acc["kl_sum"] += float(kl[valid].sum())
            acc["fired_positions"] += int((valid & changed).sum())
            acc["kl_sum_fired"] += float(kl[valid & changed].sum())
            acc["flips"] += int((lp_off.argmax(-1) != lp_on.argmax(-1))[valid].sum())
            acc["nll_off"] += float(-lp_off.gather(-1, target[..., None])[..., 0][valid].sum())
            acc["nll_on"] += float(-lp_on.gather(-1, target[..., None])[..., 0][valid].sum())
        n = max(acc["positions"], 1)
        tele = [m.get_telemetry() for m in tables]
        res = {
            "positions": acc["positions"],
            "mean_kl": acc["kl_sum"] / n,
            "mean_kl_where_changed": acc["kl_sum_fired"] / max(acc["fired_positions"], 1),
            "changed_frac": acc["fired_positions"] / n,
            "top1_flip_rate": acc["flips"] / n,
            "nll_off": acc["nll_off"] / n,
            "nll_on": acc["nll_on"] / n,
            "nll_delta": (acc["nll_on"] - acc["nll_off"]) / n,
            # the table is not called on the "off" pass, so telemetry covers the "on" pass only
            "fire_rate": sum(t["fire_rate"] for t in tele) / max(len(tele), 1),
            "telemetry": tele,
        }
        report["corpora"][name] = res
        print(f"{name}: KL={res['mean_kl']:.5f} flips={res['top1_flip_rate']:.4f} "
              f"changed={res['changed_frac']:.3f} fire={res['fire_rate']:.3f} "
              f"NLL {res['nll_off']:.4f} -> {res['nll_on']:.4f} ({res['nll_delta']:+.4f})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)  # streaming datasets threads can crash interpreter shutdown


if __name__ == "__main__":
    main()
