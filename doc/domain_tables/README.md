# Domain n-gram tables

Tools for testing whether small, independently trained n-gram tables can add domain
knowledge to a frozen LLM without hurting general performance. Built on TinyEngram's
Engram module.

- [phase0_findings.md](./phase0_findings.md): environment check, how the Engram module
  works, verified issues, data statistics, and the single-GPU reproduction recipe.
- [phase1_plan.md](./phase1_plan.md): arms, configs for Qwen3-0.6B and Qwen3-8B,
  phrase-selection results, evaluation and decision rules.
- [phase1_results.md](./phase1_results.md): GPU results. A random, frozen table matches
  the trained table, and Qwen3-0.6B-Base needs no table; raw outputs in `results/`.

## What is here

| Path | Purpose |
|---|---|
| `domain_tables/table.py` | `DomainEngram` / `DomainTableConfig`: Engram plus phrase-list masking, occupancy, frozen-table controls, telemetry, correct cached decoding |
| `domain_tables/model.py` | `DomainEngramQwenForCausalLM` (EngramQwen with `DomainEngram` layers) |
| `domain_tables/phrase_list.py`, `bloom.py`, `ngram_keys.py` | exact n-gram keys, allowlists and Bloom filters |
| `domain_tables/phrase_select.py`, `tools/select_phrases.py` | build a phrase list from a domain and a background corpus |
| `domain_tables/io.py` | table-only checkpoints: `save_tables`, `load_tables`, `load_model_with_tables` |
| `train/train_domain_table.py` | train one arm: `table`, `random_table`, `zero_table` or `lora` |
| `train/run_phase1.sh` | all Phase 1 arms for one base model |
| `train/train_biomedical_1gpu.sh` | Phase 0: upstream biomedical recipe on one GPU |
| `eval_scripts/eval_domain_tables.py` | lm-eval domain and OOD suites, with per-task table telemetry |
| `eval_scripts/leakage_kl.py` | KL, top-1 flips and NLL change from a table, on any corpus |
| `eval_scripts/compare_arms.py`, `run_phase1_eval.sh` | evaluate all arms and write a comparison table |
| `tests/test_domain_tables.py` | CPU tests (tiny random Qwen3 + the real Qwen3 tokenizer) |

## Quick start

```bash
python -m pytest tests -q
```

```python
from domain_tables import DomainTableConfig
from domain_tables.model import DomainEngramQwenForCausalLM

cfg = DomainTableConfig(
    tokenizer_name_or_path="Qwen/Qwen3-0.6B-Base",
    engram_vocab_size=[10000, 1000],
    layer_ids=[5, 7, 13, 17],
    phrase_list_path="phrase_lists/biomed_sci_r5.npz",  # optional: fire only on these n-grams
    table_mode="trained",                             # or random_frozen / zero_frozen
    telemetry=True,
)
model = DomainEngramQwenForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", engram_cfg=cfg)
```

Load a trained table onto the frozen base:

```python
from domain_tables.io import load_model_with_tables

model = load_model_with_tables("Qwen/Qwen3-0.6B-Base", "runs/phase1/.../final", selection_backend="bloom")
```

Use `engram_qwen.set_skip_engram(True)` to switch every table off, or set
`module.enabled = False` on a single `DomainEngram` layer.
