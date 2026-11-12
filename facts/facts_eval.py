"""Evaluate one arm on the made-up-facts test (test #4).

Arms: base | table (a saved domain table) | lora (a saved LoRA adapter) | rag (base model with
the best-matching training document put in the prompt, chosen by word overlap).

Metrics, separately for companies that were in the training documents ("trained") and
companies that never were ("heldout"):
  exact  greedy answer (up to 16 tokens, first line) contains the true value
  mc     the true value has the highest log-likelihood among it and three distractors
Plus canary extraction: does greedy continuation of "The internal access code for X is"
reproduce the planted code?
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOP = set("q a the of in is was which what who by and for its it to year city company founded product "
           "flagship headquartered".split())


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


class Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.sets = [words(d) for d in docs]

    def top(self, text):
        q = words(text)
        best = max(range(len(self.docs)), key=lambda i: len(q & self.sets[i]))
        return self.docs[best]


def load(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.arm in ("base", "rag"):
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    elif args.arm == "lora":
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
        model = PeftModel.from_pretrained(model, args.lora_dir).merge_and_unload()
    else:
        from domain_tables.io import load_model_with_tables
        from engram_qwen import set_skip_engram
        model = load_model_with_tables(args.base_model, args.table_dir, dtype=dtype)
        set_skip_engram(False)
    return model.to(args.device).eval(), tok


@torch.no_grad()
def generate_all(model, tok, prompts, device, batch, max_new=16):
    """Greedy continuations; prompts of equal token length are batched, so no padding is needed."""
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    by_len = defaultdict(list)
    for i, ids in enumerate(enc):
        by_len[len(ids)].append(i)
    out = [None] * len(prompts)
    for L, idx in by_len.items():
        for s in range(0, len(idx), batch):
            chunk = idx[s:s + batch]
            ids = torch.tensor([enc[i] for i in chunk], device=device)
            g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            for i, row in zip(chunk, g[:, L:].tolist()):
                out[i] = tok.decode(row, skip_special_tokens=True)
    return out


@torch.no_grad()
def answer_logliks(model, tok, prompts, answers, device, batch):
    """Sum log-probability of ' answer' after each prompt (right-padded batches)."""
    pairs = []
    for p, a in zip(prompts, answers):
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ai = tok(" " + a, add_special_tokens=False)["input_ids"]
        pairs.append((pi, ai))
    scores = []
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        L = max(len(pi) + len(ai) for pi, ai in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, (pi, ai) in enumerate(chunk):
            seq = pi + ai
            ids[r, :len(seq)] = torch.tensor(seq)
            mask[r, :len(seq)] = 1
        logits = model(ids.to(device), attention_mask=mask.to(device), use_cache=False).logits.float()
        logp = F.log_softmax(logits[:, :-1], -1)
        for r, (pi, ai) in enumerate(chunk):
            tgt = torch.tensor(ai, device=device)
            pos = torch.arange(len(pi) - 1, len(pi) - 1 + len(ai), device=device)
            scores.append(float(logp[r, pos, tgt].sum()))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "table", "lora", "rag"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--table_dir", default=None)
    ap.add_argument("--lora_dir", default=None)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qs = json.load(open(f"{args.data_dir}/questions.json"))
    canaries = json.load(open(f"{args.data_dir}/canaries.json"))
    docs = json.load(open(f"{args.data_dir}/docs.json"))
    model, tok = load(args)

    def with_context(p):
        return p if args.arm != "rag" else "Context: " + retr.top(p) + "\n" + p

    retr = Retriever(docs) if args.arm == "rag" else None
    prompts = [with_context(q["question"]) for q in qs]

    gens = generate_all(model, tok, prompts, args.device, args.batch)
    exact = [q["answer"].lower() in g.split("\n")[0].lower() for q, g in zip(qs, gens)]

    cand_prompts, cand_answers = [], []
    for p, q in zip(prompts, qs):
        for a in [q["answer"]] + q["distractors"]:
            cand_prompts.append(p)
            cand_answers.append(a)
    ll = answer_logliks(model, tok, cand_prompts, cand_answers, args.device, args.batch)
    mc = [max(range(4), key=lambda k: ll[4 * i + k]) == 0 for i in range(len(qs))]

    res = {}
    for split in ("trained", "heldout"):
        idx = [i for i, q in enumerate(qs) if q["split"] == split]
        by_attr = {}
        for attr in sorted({qs[i]["attr"] for i in idx}):
            j = [i for i in idx if qs[i]["attr"] == attr]
            by_attr[attr] = {"exact": sum(exact[i] for i in j) / len(j), "mc": sum(mc[i] for i in j) / len(j)}
        res[split] = {"n": len(idx), "exact": sum(exact[i] for i in idx) / len(idx),
                      "mc": sum(mc[i] for i in idx) / len(idx), "by_attr": by_attr}

    cp = [with_context(c["prompt"]) for c in canaries]
    cg = generate_all(model, tok, cp, args.device, args.batch, max_new=10)
    extracted = [c["code"] in g for c, g in zip(canaries, cg)]

    out = {"name": args.name or args.arm, "arm": args.arm, "table_dir": args.table_dir, "lora_dir": args.lora_dir,
           "results": res, "canaries": {"n": len(canaries), "extracted": int(sum(extracted))},
           "examples": [{"q": q["question"], "answer": q["answer"], "gen": g}
                        for q, g in list(zip(qs, gens))[:12]]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    t, h = res["trained"], res["heldout"]
    print(f"{out['name']}: trained exact {t['exact']:.3f} mc {t['mc']:.3f} | heldout exact {h['exact']:.3f} "
          f"mc {h['mc']:.3f} | canaries {out['canaries']['extracted']}/{len(canaries)}", flush=True)


if __name__ == "__main__":
    main()
