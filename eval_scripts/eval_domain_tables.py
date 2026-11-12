"""Evaluate one arm (base / table / lora) on domain and out-of-domain suites.

Each suite is run as its own lm-eval call. For table arms, table telemetry (fire rate,
gate, injection norm) is reset before each call and recorded after it, so the output
shows how often a table fires on domain tasks versus unrelated ones.

    python eval_scripts/eval_domain_tables.py --arm base  --base_model Qwen/Qwen3-0.6B-Base --out results/base.json
    python eval_scripts/eval_domain_tables.py --arm table --base_model Qwen/Qwen3-0.6B-Base \
        --table_dir runs/p1/table_Nov05/final --out results/table.json
    python eval_scripts/eval_domain_tables.py --arm lora  --base_model Qwen/Qwen3-0.6B-Base \
        --lora_dir runs/p1/lora_Nov05/final --out results/lora.json
    python eval_scripts/compare_arms.py results/*.json

Suites (``--suites``, comma separated): domain, ood, ood_full, ppl. ``--limit`` caps the
examples per task for quick checks.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# MMLU subjects counted as biomedical (excluded from the OOD MMLU average, as in
# doc/experiments/catastrophic_forgetting.md).
MMLU_BIOMED = [
    "anatomy", "clinical_knowledge", "college_biology", "college_medicine", "high_school_biology",
    "human_aging", "medical_genetics", "nutrition", "professional_medicine", "virology",
]
MMLU_ALL = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts",
    "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory",
    "professional_accounting", "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
]
MMLU_NON_BIOMED = [s for s in MMLU_ALL if s not in MMLU_BIOMED]

# name -> list of lm-eval tasks run in one call. Calls are kept small so telemetry is per call.
SUITES = {
    "domain": {
        "mmlu_clinical_knowledge": ["mmlu_clinical_knowledge"],
        "mmlu_medical_genetics": ["mmlu_medical_genetics"],
        "mmlu_professional_medicine": ["mmlu_professional_medicine"],
        "mmlu_anatomy": ["mmlu_anatomy"],
        "mmlu_college_medicine": ["mmlu_college_medicine"],
        "mmlu_college_biology": ["mmlu_college_biology"],
        "pubmedqa": ["pubmedqa"],
        "medmcqa": ["medmcqa"],
        "medqa_4options": ["medqa_4options"],
    },
    "ood": {
        "mmlu_non_biomed": [f"mmlu_{s}" for s in MMLU_NON_BIOMED],
        "truthfulqa_mc1": ["truthfulqa_mc1"],
        "truthfulqa_mc2": ["truthfulqa_mc2"],
        "arc_challenge": ["arc_challenge"],
        "hellaswag": ["hellaswag"],
        "winogrande": ["winogrande"],
    },
    "ood_full": {
        "piqa": ["piqa"],
        "arc_easy": ["arc_easy"],
        "lambada_openai": ["lambada_openai"],
    },
    "ppl": {
        "wikitext": ["wikitext"],
    },
}
PRIMARY_METRIC = ["acc_norm,none", "acc,none", "mc2,none", "word_perplexity,none", "exact_match,strict-match"]
# acc_norm is preferred where lm-eval reports it (arc, hellaswag); MMLU/PubMedQA report acc only.
PREFER_ACC = {"pubmedqa", "medmcqa", "medqa_4options", "truthfulqa_mc1", "winogrande"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=["base", "table", "lora"])
    p.add_argument("--name", default=None, help="label in the results (default: arm or table dir)")
    p.add_argument("--base_model", required=True)
    p.add_argument("--table_dir", default=None)
    p.add_argument("--lora_dir", default=None)
    p.add_argument("--suites", default="domain,ood")
    p.add_argument("--only", default=None, help="comma separated suite entries to run (e.g. pubmedqa,arc_challenge)")
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--batch_size", default="auto")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--selection_backend", default=None, help="override the table's exact|bloom selection")
    p.add_argument("--disable_table", action="store_true", help="load the table but switch it off (sanity check)")
    p.add_argument("--num_fewshot", type=int, default=None)
    p.add_argument("--samples_dir", default=None,
                   help="also save per-question predictions (doc id, target, predicted choice, correct) "
                        "as <samples_dir>/<task>.jsonl, for paired tests and answer-letter checks")
    p.add_argument("--out", required=True)
    return p.parse_args()


def load(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.arm == "base":
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    elif args.arm == "lora":
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
        model = PeftModel.from_pretrained(model, args.lora_dir).merge_and_unload()
    else:
        from domain_tables.io import load_model_with_tables
        from engram_qwen import set_skip_engram

        overrides = {"telemetry": True}
        if args.selection_backend:
            overrides["selection_backend"] = args.selection_backend
        model = load_model_with_tables(args.base_model, args.table_dir, dtype=dtype, **overrides)
        set_skip_engram(bool(args.disable_table))
    return model.to(args.device).eval(), tok


def table_modules(model):
    from domain_tables import DomainEngram

    return [m for m in model.modules() if isinstance(m, DomainEngram)]


def primary(task, metrics):
    keys = (["acc,none"] + PRIMARY_METRIC) if task in PREFER_ACC else PRIMARY_METRIC
    for k in keys:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return k.split(",")[0], float(metrics[k])
    return None, None


def save_samples(samples, out_dir):
    """Write one line per question: predicted choice index, target and correctness."""
    os.makedirs(out_dir, exist_ok=True)
    for task, rows in samples.items():
        with open(os.path.join(out_dir, f"{task}.jsonl"), "w") as f:
            for r in rows:
                lls = [x[0] if isinstance(x, (list, tuple)) else x for x in r.get("filtered_resps", [])]
                lls = [float(v[0]) if isinstance(v, (list, tuple)) else float(v) for v in lls]
                pred = max(range(len(lls)), key=lls.__getitem__) if lls else None
                f.write(json.dumps({"doc_id": r.get("doc_id"), "target": r.get("target"), "pred": pred,
                                    "acc": r.get("acc"), "loglik": lls}) + "\n")


def main():
    args = parse_args()
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    model, tok = load(args)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size, max_length=args.max_length,
              device=args.device)
    tables = table_modules(model)

    only = set(args.only.split(",")) if args.only else None
    out = {
        "name": args.name or (os.path.basename(os.path.dirname(args.table_dir.rstrip("/"))) if args.table_dir else args.arm),
        "arm": args.arm, "base_model": args.base_model, "table_dir": args.table_dir, "lora_dir": args.lora_dir,
        "limit": args.limit, "disable_table": args.disable_table, "results": {}, "telemetry": {},
    }
    if args.table_dir:
        with open(os.path.join(args.table_dir, "table_config.json")) as f:
            out["table_config"] = json.load(f)

    for suite in args.suites.split(","):
        for entry, tasks in SUITES[suite].items():
            if not tasks or (only and entry not in only):
                continue
            for m in tables:
                m.reset_telemetry()
            t0 = time.time()
            try:
                res = simple_evaluate(model=lm, tasks=tasks, limit=args.limit, num_fewshot=args.num_fewshot,
                                      batch_size=args.batch_size, log_samples=bool(args.samples_dir))
            except Exception as exc:  # keep the other suites running; record why this one failed
                print(f"[{suite}] {entry}: FAILED: {exc!r}", flush=True)
                out["results"][entry] = {"suite": suite, "metric": None, "score": None, "error": repr(exc)}
                continue
            per_task = {t: res["results"][t] for t in tasks if t in res["results"]}
            if args.samples_dir:
                save_samples(res.get("samples", {}), args.samples_dir)
            if len(tasks) > 1:
                # Macro average over subtasks (MMLU groups); lm-eval's own group score is size-weighted.
                vals = [primary(t, r)[1] for t, r in per_task.items() if primary(t, r)[1] is not None]
                metric = primary(tasks[0], per_task.get(tasks[0], {}))[0]
                score = sum(vals) / len(vals) if vals else None
            else:
                metric, score = primary(tasks[0], per_task.get(tasks[0], {}))
            stderr = None
            if len(tasks) == 1 and metric:
                stderr = per_task.get(tasks[0], {}).get(f"{metric}_stderr,none")
                stderr = stderr if isinstance(stderr, (int, float)) else None
            out["results"][entry] = {"suite": suite, "metric": metric, "score": score, "stderr": stderr,
                                     "seconds": round(time.time() - t0, 1),
                                     "tasks": {t: {k: v for k, v in r.items() if isinstance(v, (int, float))}
                                               for t, r in per_task.items()}}
            if tables:
                out["telemetry"][entry] = [m.get_telemetry() for m in tables]
            fire = ""
            if tables:
                rates = [t["fire_rate"] for t in out["telemetry"][entry]]
                fire = f"  fire_rate={sum(rates) / len(rates):.3f}"
            print(f"[{suite}] {entry}: {metric}={score}{fire}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
