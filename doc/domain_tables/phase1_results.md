# Phase 1 results: trained table vs random-table control

Runs on 2026-11-05, one RTX PRO 6000 Blackwell (96 GB) rented on Clore.ai. Raw outputs
are in [`results/`](../../results/): per-task lm-eval scores, training logs, telemetry,
leakage measurements and the exact job script for each run.

## Summary

- **TinyEngram's numbers reproduce.** Our scores for the post-trained Qwen3-0.6B ("chat")
  match the published baseline exactly, and a trained table lifts them to within about 2
  points of the published "Engram SFT" numbers.
- **A random, frozen table gives the same gain.** With the table rows fixed at their
  random init and only the gate, projections and conv trained, the chat model improves as
  much as with a trained table. The trained rows did move (20–50% of their initial norm),
  so the table was trained; what it learned made no measurable difference.
- **On Qwen3-0.6B-Base neither table helps.** Base alone scores 56.0% on the three headline
  MMLU subsets, far above chat + table (42.8%), and both tables cost about 2 points.
- **Non-medical scores rise on the chat model too.** That points to the added pathway
  restoring raw-text likelihood scoring that chat tuning degraded, not to stored
  biomedical knowledge.

## Results

Headline subsets (MMLU clinical knowledge, medical genetics, professional medicine),
zero-shot, accuracy:

| Setup | Clinical | Genetics | Prof. medicine | Average |
|---|---|---|---|---|
| Chat alone | 33.6 | 37.0 | 32.0 | 34.2 |
| Chat + trained table | 43.8 | 41.0 | 43.8 | 42.8 |
| Chat + random table | 44.9 | 43.0 | 41.9 | 43.3 |
| *TinyEngram published, chat + table* | *44.2* | *44.0* | *45.6* | *44.6* |
| Base alone | 60.0 | 58.0 | 50.0 | 56.0 |
| Base + trained table | 57.7 | 55.0 | 48.2 | 53.6 |
| Base + random table | 58.5 | 55.0 | 48.2 | 53.9 |
| Base + trained phrase-list table (LR 1e-3) | 56.2 | 57.0 | 47.8 | 53.7 |
| Base + random phrase-list table (LR 1e-3) | 57.4 | 54.0 | 48.2 | 53.2 |

Suite averages: medical = 6 MMLU medical subsets + MedQA (4 options); general = MMLU
without its 10 biomedical subjects (macro average), TruthfulQA MC1/MC2, ARC-Challenge,
HellaSwag, WinoGrande.

| Setup | Medical (7) | General (6) |
|---|---|---|
| Chat alone | 35.4 | 41.6 |
| Chat + trained table | 40.8 | 43.3 |
| Chat + random table | 42.0 | 43.6 |
| Base alone | 51.9 | 46.5 |
| Base + trained table | 49.9 | 46.3 |
| Base + random table | 49.9 | 46.1 |
| Base + trained phrase-list table | 50.6 | 46.5 |
| Base + random phrase-list table | 50.3 | 46.3 |

Standard errors per task are 1.3–5 points (MMLU medical genetics: 100 questions, ±5);
see each run's `summary.md`.

Next-token loss on held-out text (`leakage_kl.py`, 300 docs each), table off → on:

| Setup | Held-out biomedical | FineWeb-Edu (general) | KL on general text |
|---|---|---|---|
| Chat + trained table | 3.319 → 3.128 | 3.050 → 2.960 | 0.166 |
| Chat + random table | 3.319 → 3.147 | 3.050 → 2.947 | 0.136 |
| Base + trained table | 2.941 → 2.874 | 2.640 → 2.670 | 0.033 |
| Base + random table | 2.941 → 2.878 | 2.640 → 2.669 | 0.032 |
| Base + trained phrase-list table | 2.941 → 2.915 | 2.640 → 2.647 | 0.008 |
| Base + random phrase-list table | 2.941 → 2.917 | 2.640 → 2.646 | 0.006 |

On the chat model both tables also lower loss on general web text, consistent with a
repair of the chat model rather than domain memory. The phrase-list table fires on about
10% of biomedical tokens and 3% of general tokens, and its effect on general text is
about 4x smaller than the always-on table's.

Table movement (run 3, chat): trained rows moved by a mean L2 of 0.035–0.083 from their
shared initialization (initial row norm 0.159), across all rows and all four layers.

## Setup

| Run | Base model | Table | LR | Folder |
|---|---|---|---|---|
| 1 | Qwen3-0.6B-Base | plain hashed, 10k/1k | 1e-4 | `results/clore_run1` |
| 2 | Qwen3-0.6B-Base | phrase list (biomedical vs Biomed-Enriched `domain==other`, ratio ≥ 5; 52k bigrams, 47k trigrams), 40k/20k, strict masking | 1e-3 | `results/clore_run2` |
| 3 | Qwen3-0.6B (post-trained) | plain hashed, 10k/1k | 1e-4 | `results/clore_run3` |

Shared settings: layers 5/7/13/17, `n_embed_per_ngram` 512, 8 heads per order;
190,000 Biomed-Enriched training docs (`en`, `biomedical`, score > 4.0; 49.6M non-padding
tokens), 1 epoch, 192 docs per step (990 steps), cosine schedule, length-grouped batches,
BF16 base, value projection zero-initialized, table seed 1234. Trained and random arms
share the seeded table init, data order and hyperparameters; they differ only in whether
the table rows receive gradients. Evaluation: lm-eval 0.4.9.1, zero-shot, BF16.
Training ran at ~28k non-padding tokens/s with 47 GB peak memory, about 29 minutes per
arm. Total rental cost: $5.23 for three runs plus two failed starts.

CPU cross-check (`results/cpu_bench`, FP32): chat 32.1 / 38.0 / 31.6 and Base
59.6 / 57.0 / 51.5 on the headline subsets, within noise of the GPU numbers.

## Not covered

- One seed per arm.
- PubMedQA and MedMCQA did not run: lm-eval loads both through dataset loading scripts
  that need `trust_remote_code`, which was not enabled.
- TinyEngram's own learning rate (1e-5) and step count were not replicated; our chat +
  table result lands near theirs regardless.
- The table weights were not kept (only results), so further evaluation of these exact
  tables needs a rerun.
