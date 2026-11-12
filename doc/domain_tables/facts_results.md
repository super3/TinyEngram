# Test 4: can a table learn facts the model cannot know?

Run on 2026-11-12, one RTX PRO 6000 on Clore.ai ($0.51). Data: `facts/make_facts.py`
(400 fictional companies × 4 attributes, 4 paraphrased training documents each, 20
planted "access codes"); evaluation: `facts/facts_eval.py`; raw output:
`results/clore_run8_facts`.

Qwen3-0.6B-Base, frozen. Each arm trained 30 epochs on the 1,600 documents (≈ 3,000
steps, batch 16). Questions are phrased differently from the training text.

| Arm | Exact recall | 4-way choice | Canaries extracted | Held-out companies (exact / choice) | MMLU non-biomedical |
|---|---|---|---|---|---|
| base model | 0.6% | 26.1% | 0/20 | 0.0% / 24.7% | **51.8%** |
| zero table (pathway only) | 0.1% | 26.8% | 0/20 | 0.3% / 25.3% | 24.3% |
| random frozen table | 6.6% | 33.7% | 0/20 | 0.3% / 21.0% | 28.0% |
| trained table, LR 1e-3 | 18.2% | 44.1% | 0/20 | 0.3% / 22.0% | 32.1% |
| trained table, LR 5e-3 | 22.3% | 68.3% | 13/20 | 0.3% / 22.8% | 27.3% |
| LoRA r16, LR 5e-4 | 57.6% | 96.4% | 19/20 | 1.0% / 22.3% | 28.3% |
| retrieval (best-matching document in the prompt) | 92.5% | 100% | 20/20 | 0.3% / 25.5% | (base model) |

Findings:
- Tables can store new facts: the trained table beats the random and zero tables by a wide
  margin (22% vs 7% vs 0% exact recall). This is the first test in this project where the
  table's contents clearly matter.
- LoRA learns the same facts much better (58% exact, 96% choice), and retrieval is better
  still (93%, 100%) with no training.
- Facts that are learned can be extracted: 13 of 20 planted codes from the faster-trained
  table, 19 from LoRA.
- **Every trained arm destroyed general ability:** non-biomedical MMLU fell from 51.8% to
  24–32% (chance is 25%), including the zero table, which has no n-gram information at
  all. Thirty epochs on a tiny, repetitive corpus with no general-text replay is far too
  aggressive; the frozen base does not by itself protect against this, because the added
  pathway can still steer every token. Untested: fewer epochs, early stopping, mixing in
  general text, phrase-list (strict) masking so the table only fires on the company names.
