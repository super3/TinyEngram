# Phase 1: one medical table, done right

Proposal for brief task 5. Everything here runs on the RTX PRO 6000 with the scripts in
this repo; nothing in it has been run on a GPU yet. Read
[phase0_findings.md](./phase0_findings.md) first for the code-level findings this plan
relies on.

## Questions

1. **Content or pathway?** Does a trained table beat the same architecture with frozen
   random rows (`random_table`) and with zero rows (`zero_table`, the gate, bias and
   conv alone)? Upstream's +4.7 points on *non*-biomedical MMLU suggests a large
   pathway share.
2. **Does selectivity cost accuracy?** Does a phrase-list table with strict masking keep
   the domain gain while cutting leakage on general text?
3. **How does it compare with LoRA?** Both the brief's r16 and a parameter-matched rank.

## Arms (matched data, same document order)

| Arm | `--arm` | Trains | Qwen3-0.6B | Qwen3-8B |
|---|---|---|---|---|
| Frozen base | (eval only) | nothing | 0 | 0 |
| Plain table | `table` | rows, projections, gate, conv | 31.5M | ~56M |
| Selective table | `table` + `PHRASES=` | same, only registered n-grams | 31.5M | ~56M |
| Random-frozen control | `random_table` | projections, gate, conv | 8.4M | ~34M |
| Pathway-only control | `zero_table` | projections, gate, conv | 8.4M | ~34M |
| LoRA r16 | `lora` | adapters | 10.1M | 43.6M |
| LoRA matched | `lora`, `LORA_R=48` (0.6B) or `21` (8B) | adapters | 30.3M | ~57M |

Run the random and zero controls for the selective table too (`PHRASES=... TAG=selective
bash train/run_phase1.sh table random_table zero_table`).

## Step 0: phrase lists

```bash
python tools/select_phrases.py --tokenizer Qwen/Qwen3-0.6B-Base --domain_files "dataset/biomed/data/*.parquet" --domain_filter "language==en,domain==biomedical,educational_score>4.0" --domain_max_docs 400000 --general_files "dataset/biomed/data/*.parquet" --general_filter "language==en,domain==other" --general_max_docs 400000 --min_ratio 5 --counts_dir phrase_lists/counts_sci --out phrase_lists/biomed_sci_r5
```

Swap the general side for `--general_dataset HuggingFaceFW/fineweb-edu --general_config
sample-10BT` to get the broad list. Qwen3-0.6B and Qwen3-8B share a tokenizer, so a list
built for one works for the other.

What the CPU runs showed (100k domain docs = 27M tokens):

**FineWeb-Edu as background** (30k docs, 32M tokens):

| min ratio | bigrams | domain bigram coverage | general bigram coverage | trigrams | domain | general |
|---|---|---|---|---|---|---|
| 2 | 73,171 | 38.6% | 6.1% | 62,970 | 19.4% | 2.1% |
| 5 | 47,475 | 25.4% | 1.9% | 47,352 | 15.2% | 0.8% |
| 10 | 25,364 | 16.2% | 0.6% | 26,589 | 11.1% | 0.3% |
| 25 | 6,405 | 8.0% | 0.1% | 6,272 | 6.3% | 0.1% |

At every threshold the top-ranked phrases are **academic register**, not medicine:
`compared to`, `this study`, `. Furthermore,`, `et al.`, `, respectively`. The
background corpus decides what counts as "domain".

**Scientific background** (Biomed-Enriched `domain == other`, 100k docs, 14M tokens), ratio ≥ 5:
16,644 bigrams and 8,390 trigrams covering 12.2% and 4.0% of domain occurrences (0.8%
and 0.3% of background ones). Register phrases mostly cancel. What remains is methods,
units and biology: `patients with`, `the expression of`, `the activation of`, `μM`,
`IL-`, `CD4`, `autophagy`, `macrophages`, `95% CI`.

Plan: build both lists (broad = FineWeb-Edu, ratio ≥ 5; specific = scientific background,
ratio ≥ 5) and run each as a selective arm. If the broad list wins, the table is mostly
learning domain register; if the specific list holds up, it is learning domain terms.
Read `<out>.report.json` before training: the top phrases are the fastest sanity check.
Phrases there are decoded from compressed (lowercased, normalized) ids, so case and
spacing are approximate (`et al.` shows as `etal.`).

Table size with a phrase list: only registered n-grams write rows, so size each order's
per-head table near the number of phrases (for example `VOCAB="20000 10000"` for the
specific list) and check `occupied_frac_l*` in the training logs.

## Step 1: learning-rate pilot

TinyEngram trained at 1e-5 (biomedical) and 2e-5 (function calls). With `value_proj`
zero-initialized, only it gets gradient on the first step, and with Adam each element
moves about `lr` per step, so 1e-5 over ~1,000 steps barely moves rows from their
N(0, 0.02) init. That would make `table` and `random_table` look alike for the wrong reason.

Run 300-step pilots (`TRAIN_DOCS=57600`) of `table` and `random_table` at 1e-5, 1e-4
and 1e-3, and LoRA at 1e-4 and 2e-4, on Qwen3-0.6B-Base. Give each arm its best LR by
eval loss. `run_phase1.sh` defaults to 1e-4 for tables and 2e-4 for LoRA until the
pilot says otherwise.

## Step 2: main runs

Data: 190k training docs (~50M tokens), 1 epoch, 192 docs per optimizer step
(~990 steps), `--group_by_length`, max length 1,500, cosine schedule, seed 42. On 0.6B,
repeat `table` and `random_table` with seeds 1 and 2 to measure run-to-run spread.

```bash
MODEL=Qwen/Qwen3-0.6B-Base bash train/run_phase1.sh
```

```bash
MODEL=Qwen/Qwen3-0.6B-Base PHRASES=phrase_lists/biomed_sci_r5.npz VOCAB="20000 10000" TAG=selective_sci bash train/run_phase1.sh table random_table zero_table
```

```bash
MODEL=Qwen/Qwen3-0.6B-Base LORA_R=48 TAG=lora48 bash train/run_phase1.sh lora
```

For continuity with upstream, also run `table` and `random_table` once on the
post-trained `Qwen/Qwen3-0.6B`.

**Qwen3-8B-Base** (36 layers, hidden 4096): the same relative depths are layers
`6 9 17 22`. Keep 10k/1k, 8 heads and 512 dims per n-gram (the projections grow to
1024 → 4096).

```bash
MODEL=Qwen/Qwen3-8B-Base LAYERS="6 9 17 22" BS=8 ACC=24 bash train/run_phase1.sh
```

Rough cost on the PRO 6000, to be replaced by the measured `throughput.json`:

| Model | GFLOP per real token | Assumed throughput | 50M tokens | VRAM |
|---|---|---|---|---|
| Qwen3-0.6B | ~3.2 (forward, recompute and backward from the lowest table up) | 20-40k tok/s | 20-40 min per arm | < 20 GB |
| Qwen3-8B | ~43 | ~3.7k tok/s (40% of 404 TFLOPS) | ~4 h per arm | ~40-60 GB at BS 8 x 1,500 |

The 8B logits alone are 8 × 1,500 × 151,936 values, about 7 GB in fp32 for the loss;
lower `BS` and raise `ACC` if memory is tight. Non-reentrant checkpointing (set
automatically for table arms) skips backprop through the layers below the lowest table.

## Step 3: evaluation

```bash
TRUST_REMOTE_CODE=1 MODEL=Qwen/Qwen3-0.6B-Base RUNS=runs/phase1/Qwen3-0.6B-Base/plain bash eval_scripts/run_phase1_eval.sh
```

This evaluates the base model and every `final/` under `RUNS`, runs the leakage check
for table arms, and writes `summary.md` through `eval_scripts/compare_arms.py`.

| Suite | Tasks (lm-eval) | Size |
|---|---|---|
| domain | `mmlu_clinical_knowledge`, `mmlu_medical_genetics`, `mmlu_professional_medicine`, `mmlu_anatomy`, `mmlu_college_medicine`, `mmlu_college_biology`, `pubmedqa`, `medmcqa`, `medqa_4options` | 100-4,183 questions each |
| ood | MMLU without its 10 biomedical subjects (47 subjects, macro average), `truthfulqa_mc1`, `truthfulqa_mc2`, `arc_challenge`, `hellaswag`, `winogrande` | |
| leakage | `leakage_kl.py`: KL(base ‖ base+table), top-1 flips and NLL change on FineWeb-Edu and on held-out Biomed (score < 4) | 500 docs each |

`--limit 50` gives a fast smoke pass. `--suites ood_full` adds PIQA, ARC-Easy and
LAMBADA; `ppl` adds WikiText perplexity. `TRUST_REMOTE_CODE=1` is needed only for
PubMedQA (its dataset uses a loading script).

## Decision rules

- **Content effect** = `table` − `random_table` on the pooled domain score. Count it as
  real only if it exceeds about 2 pooled standard errors and has the same sign across
  seeds. MMLU medical genetics alone has SE ≈ 5 points; MedMCQA (4,183 questions) and
  MedQA (1,273) are the low-noise tasks.
- **Pathway effect** = `random_table` − base and `zero_table` − base.
- **Retention**: OOD average within about 1 SE of base, and for selective tables a
  general-text KL that is a small fraction of the domain-text KL.
- **Selectivity**: fire rate on domain tasks versus OOD tasks (telemetry in each result
  file and in `summary.md`). Fire rate is not effect: attention spreads an injection to
  later tokens, so judge leakage by KL and flips.
- **Go to Phase 2** if a selective table keeps most (say ≥ 70%) of the plain table's
  content effect with clearly lower general-text KL.

## Known gaps

- Telemetry in lm-eval runs counts padded positions (HFLM calls the model without an
  attention mask), so fire rates there are slight underestimates; `leakage_kl.py`
  masks padding.
- The OOD MMLU score is a macro average over subjects; upstream's table used lm-eval's
  group score. Compare like with like.
- Multi-table loading, the router and per-request switching are Phase 2 work (hooks
  listed in phase0_findings.md, section 5).
