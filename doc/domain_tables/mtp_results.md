# Test 3: does an n-gram table help a draft (MTP) head?

Run on 2026-11-12, one RTX PRO 6000 on Clore.ai ($0.98). Script: `mtp/mtp_table_test.py`;
raw output: `results/clore_run5_mtp/mtp_results.json`.

Setup: frozen Qwen3-0.6B-Base. Medusa-style draft heads (EAGLE-style input: base hidden
state at t plus the embedding of the base's next token) propose 4 tokens ahead. Heads were
trained for one pass over 10M tokens: FineWeb-Edu (generic) or Biomed-Enriched (all
others). Evaluation: the base generated 128 greedy tokens from 300 held-out prompts per
domain; the metric is tokens produced per verification step (1 + accepted draft tokens)
under greedy chain drafting. Output quality is identical by construction; only speed
differs.

| Drafter | Trainable params | Biomed tokens/step | Biomed 1st-draft accept | General tokens/step | General 1st-draft accept |
|---|---|---|---|---|---|
| generic head | 9.4M | 1.605 | 0.454 | 1.587 | 0.468 |
| domain head | 9.4M | 1.681 | 0.499 | 1.413 | 0.351 |
| domain head, 4x wider | 34.6M | **1.727** | 0.519 | 1.443 | 0.368 |
| domain head + learned n-gram table | 31.1M | 1.716 | 0.520 | 1.428 | 0.359 |
| domain head + random frozen table | 10.5M | 1.682 | 0.502 | 1.414 | 0.349 |
| n-gram lookup only (no training) | 0 | 1.459 | 0.292 | 1.210 | 0.165 |
| domain head + n-gram lookup (2 branches) | 9.4M | **1.857** | 0.573 | 1.501 | 0.407 |

Findings:
- Training the head on domain text helps on that domain (+4.7% tokens/step) and hurts on
  general text (−11%).
- The learned table helps (+2.1% over the plain domain head) and the random table does
  not, so here the table's contents matter. But a wider head with similar parameters does
  as well or slightly better (+2.7%).
- The biggest gain comes from a training-free n-gram lookup used as a second draft branch:
  +10.5% over the domain head on biomedical text (+18% over the generic head), at the cost
  of verifying two branches per step.

Caveats: one seed; small heads trained briefly, so absolute acceptance is low (1.6–1.9
tokens/step, where mature EAGLE-style drafters reach 3+); acceptance counted, not
wall-clock speed; no position-level significance test (differences of ~0.03 tokens/step
over ~36k draft positions per domain).
