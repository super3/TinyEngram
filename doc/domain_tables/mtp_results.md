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

## Run 6: general head + domain lookup, general lookup, chained drafting

Same setup and seeds (`results/clore_run6_mtp`, $0.68); the neural heads reproduce run 5
exactly. New drafters: a lookup counted on the general (FineWeb-Edu) training text, the
generic head paired with each lookup as a second draft branch, and a single chained draft
in which the lookup supplies each next token it knows and the head fills in the rest.

| Drafter | Biomed tokens/step | General tokens/step |
|---|---|---|
| generic head | 1.605 | 1.587 |
| domain head | 1.681 | 1.413 |
| lookup, biomed counts | 1.459 | 1.210 |
| lookup, general counts | 1.333 | 1.333 |
| **generic head + biomed lookup** (2 branches) | 1.799 | 1.661 |
| generic head + general lookup (2 branches) | 1.704 | 1.717 |
| **generic head + both lookups** (3 branches) | 1.831 | **1.747** |
| domain head + biomed lookup (2 branches) | **1.857** | 1.501 |
| generic > biomed lookup (chained, lookup first) | 1.492 | 1.333 |
| domain > biomed lookup (chained, lookup first) | 1.495 | 1.282 |
| generic > general lookup (chained, lookup first) | 1.384 | 1.371 |

- A shared generic head with a swappable domain lookup gets most of the domain head's gain
  on biomedical text (1.80 vs 1.86) without its loss on general text (1.66 vs 1.50); adding
  a general lookup as a third branch is the best all-rounder (1.83 / 1.75).
- Lookups help as extra branches, not as replacements: letting the lookup override the
  head whenever it has an entry is worse than the head alone. (llama.cpp's stacking orders
  the two the other way, head first and n-gram extension after, which was not tested.)
- Extra branches cost extra verification per step (2–3 drafts of 4 tokens instead of one);
  wall-clock speed was not measured.

Trained head weights (`results/clore_run6_mtp/heads/`, 460 MB) are kept out of git.

## Base model answer letters (MMLU)

Also from run 6: Qwen3-0.6B-Base on all MMLU subjects with every answer saved
(`results/clore_run6_mtp/samples/base`), compared with the chat model from run 4 on the
same 12,509 questions:

| Model | A | B | C | D | Accuracy |
|---|---|---|---|---|---|
| Base | 28.6% | 31.5% | 18.6% | 21.3% | 49.6% |
| chat | 56.9% | 11.4% | 12.2% | 19.5% | 39.7% |
| correct answers | 22.9% | 24.3% | 25.6% | 27.2% | |

The Base model has only a mild letter skew; the chat model's strong preference for "A" is
what every table (including the zero table) removed in run 4, which is consistent with
tables helping the chat model and not the Base model.
