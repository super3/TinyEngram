# Phase 0: environment, code review and reproduction plan

Status as of 2026-11-04. Covers tasks 1, 2 and 4 of the experiment brief. Task 3
(GPU reproduction) is prepared but not run: this session had no GPU. Phase 1 configs
and the eval harness (task 5) are in [phase1_plan.md](./phase1_plan.md).

## 1. Environment check

This work ran in a cloud container, not on the RTX PRO 6000 workstation:

| Item | Value |
|---|---|
| GPU | none (`nvidia-smi` not present) |
| CPU / RAM | 4 vCPU, 15 GB |
| Free disk | 30 GB |
| Python | 3.11 (3.10 not installed; repo pins are tested on 3.10) |
| Stack | torch 2.8.0+cpu, transformers 4.57.3, lm-eval 0.4.9.1 (repo pins) |

What CPU allowed: unit tests, short training runs of all four arms on real
Qwen3-0.6B-Base, eval-harness and leakage runs on a handful of examples, dataset
statistics and phrase selection over 27M domain tokens. No accuracy numbers here are
meaningful; they only show that the pipeline runs end to end.

On the workstation, check first:

```bash
nvidia-smi
```

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

```bash
df -h .
```

Blackwell (sm_120) needs a CUDA 12.8+ PyTorch build. The repo pins `torch==2.8.0` from
the cu126 index; if `torch.cuda` warns that sm_120 is not supported, swap the index line
in `requirements.txt` for `https://download.pytorch.org/whl/cu128`.

## 2. How the module works

Per Engram layer (`engram.py`, wired in by `engram_qwen.py`):

1. **Token compression.** `CompressedTokenizer` normalizes every vocab entry (NFKC,
   NFD, strip accents, lowercase, collapse whitespace) and merges ids that become
   equal: Qwen3's 151,669 ids become 107,457. So `Heart`, ` heart` and `HEART` share
   n-gram ids. N-grams are over these compressed ids, not raw token ids.
2. **Hashing.** For each layer, `n` odd random multipliers are drawn from
   `seed + 10007 * layer_id`. An n-gram's mix is `XOR_k(token_k * m_k)`. Head `j` of
   order `n` uses `mix mod p_j`, with distinct primes just above the configured size:
   10007...10079 for the 8 bigram heads and 1009...1049 for the 8 trigram heads (10k/1k).
   All 8 heads of one order reduce the same mix with different moduli, so they are
   less independent than 8 separate hash functions. Positions at the start of a
   sequence pad with `pad_id=2` (DeepSeek's default; for Qwen that is the token `#`).
3. **Lookup.** One `nn.Embedding` holds all heads (offsets per head), 64 dims per head
   (`n_embed_per_ngram / n_head_per_ngram = 512 / 8`). 16 rows are concatenated into a
   1,024-wide vector.
4. **Gate.** `key = key_proj(e)`, `query = hidden state entering the layer`;
   `gate = sigmoid(signed_sqrt(<RMSNorm(key), RMSNorm(query)> / sqrt(D)))`. The gate is
   always in (0, 1): there is no off state.
5. **Value and conv.** `v = gate * value_proj(e)`, then `out = v + SiLU(conv(v))` where
   the conv is depthwise and causal with kernel 4 and dilation 3, so each position also
   sees the previous 9 positions' values.
6. **Injection.** `out` is added to the residual stream before decoder layers
   `5, 7, 13, 17` of Qwen3-0.6B (28 layers).
7. **Training.** Base frozen. The script zero-initializes `value_proj`.
   `from_pretrained` re-initializes every missing Engram weight with N(0, 0.02),
   including the ShortConv weights that `engram.py` zero-initializes, so the conv does
   not start at zero in practice. The table rows start at N(0, 0.02).

Size for the 10k/1k setup on Qwen3-0.6B: **31.5M trainable parameters** (22.7M table
rows, 8.4M projections, rest norms and conv). Saved as a table-only file it is 127 MB
in fp32.

## 3. Findings (verified on CPU unless marked)

| # | Finding | Impact | Status |
|---|---|---|---|
| 1 | **Cached decoding is wrong.** With a KV cache, `EngramQwenModel` passes only the new token to Engram, so n-grams hash the token against padding and the conv has no history. Measured: max logit difference 0.98 between full forward and incremental decoding (random `value_proj`). | Any `generate()`-based eval or serving. Log-likelihood multiple-choice evals (MMLU, PubMedQA, TruthfulQA MC) do a full forward pass and are unaffected. | Fixed in `DomainEngram` (keeps n-gram and conv history on the cache); tests check that incremental logits and `generate()` match. Base `Engram` now logs a warning. |
| 2 | **Plain hashed tables always fire.** Every n-gram maps to some row and the gate is never 0. Telemetry: fire rate 1.000 on clinical knowledge, TruthfulQA and ARC alike. | Selectivity rests entirely on the soft gate. | Phrase lists (exact or Bloom) and strict masking added. |
| 3 | **Occupancy masking alone cannot fix #2 for small tables.** A 3-step CPU run (4.6k tokens) already touched 34% of the 10k/1k slots; the training split has ~495M tokens and ~2.2M distinct bigrams per 27M tokens. Every slot is occupied within the first few hundred steps. | Occupancy is only informative for large tables or together with a phrase list. | Implemented (`track_occupancy`, `occupancy_mask`) with that caveat. |
| 4 | **Masking the injection does not mask its effect.** With strict masking, a phrase-list table fired on 3.8% of general-text tokens, yet 64% of positions had different logits (86% on domain text). Attention carries what was injected at a matched position into every later position. | "Only contributes when its phrases appear" holds for the injection, not its downstream effect. Composition of many tables can still interfere through attention. Measure with KL, not fire rate. | `eval_scripts/leakage_kl.py` reports KL, top-1 flips and the changed fraction. |
| 5 | **Upstream used the post-trained Qwen3-0.6B**, not -Base (`/nasdata/model/Qwen/Qwen3-0___6B` is ModelScope's name for `Qwen/Qwen3-0.6B`; the README links it too). Its MMLU baseline is 0.4034. The biomedical table also raised **non-biomedical** MMLU by +4.7 points, with humanities, social science and STEM all up. | A biomedical memory cannot explain gains on law or econometrics. Part of the gain is probably the new pathway re-tuning a chat model for raw-text likelihood scoring. | Phase 1 runs on -Base and keeps the random- and zero-table controls. (Interpretation, not verified.) |
| 6 | **LoRA r16 is not parameter-matched.** On Qwen3-0.6B, LoRA r16 on all linear layers is 10.1M parameters, not the ~30M the README states for the comparison. The matched rank for the 31.5M table is about r48. On Qwen3-8B, r16 is 43.6M against ~56M for the table (matched r ≈ 21). | The README's LoRA-vs-Engram forgetting comparison had LoRA at about 1/3 to 1/2 the capacity. | Phase 1 runs r16 (as the brief says) and a matched rank. |
| 7 | **Reproduction details.** `train_biomedical.sh` ships 20k/2k ("Medium") while the README headline numbers are the 10k/1k ("Small") column. The data filter also requires `domain == "biomedical"` (the brief lists only `en` and the score). The step count behind the reported numbers is not stated: the eval scripts point at `checkpoint-500` (biomedical tasks, 20k run) and `checkpoint-3500` (MMLU). The scripts also set `HF_ENDPOINT` to a gitee mirror. | Exact reproduction needs a checkpoint sweep. | `train/train_biomedical_1gpu.sh` defaults to 10k/1k with no mirror and saves every 500 steps. |
| 8 | **Padding waste.** Per-batch padding at micro-batch 24 leaves only **36%** real tokens (51% at 8). | Roughly 2.5x wasted compute. | Phase 1 uses `--group_by_length`. |
| 9 | **Small eval sets.** MMLU medical genetics has 100 questions (SE ≈ 5 points); clinical knowledge 265 and professional medicine 272 (SE ≈ 3 points); PubMedQA 500 (SE ≈ 2.2 points). | The README's differences between vocab sizes (1-4 points) are within noise; base-to-Engram gains are 2-4 SE. | `compare_arms.py` prints the baseline's SE next to each task. |
| 10 | Each Engram layer rebuilt the compressed tokenizer and hashed n-grams for all layers, then used one. | Startup time and 4x hashing work per step. | Tokenizer cached in `engram.py`; `DomainEngram` hashes only its own layer. |
| 11 | The `*_engram.py` eval scripts add `eval_scripts/tinyengram` to `sys.path`, which does not exist. | Running them from `eval_scripts/` fails to import `engram_qwen`. | Repo root added to the path. |
| 12 | lm-eval's `pubmedqa` loads `bigbio/pubmed_qa` through a dataset script. | Needs `trust_remote_code`; not run in this session. | `--trust_remote_code` flag (off by default) in the new harness. |

Checked and **not** a problem: the training scripts do not write the Engram config
explicitly, but `transformers` passes the `engram_cfg` keyword from `from_pretrained`
into `GenerationConfig`, so it does land in `generation_config.json`, where the eval
scripts read it.

## 4. Data

`commercial-00000-of-00026.parquet` (1.86 GB, 3,792,135 paragraphs):

| Split | Filter | Docs |
|---|---|---|
| Train | `language == en`, `domain == biomedical`, `educational_score > 4.0` | 1,898,316 |
| Eval pool | same, `educational_score < 4.0` (200 sampled) | 1,319,299 |
| Dropped | score exactly 4.0 | 10,565 |

Train docs average 266 Qwen3 tokens (median 213, p95 574, 0.5% over the 1,500-token
cap), so the split is about **495M tokens**. At upstream's 192 docs per step, one epoch
is about 9,900 steps, and `checkpoint-500` is about 96k docs (25M tokens).

## 5. What was added and where

| Brief item | Implementation |
|---|---|
| Phrase selection | `tools/select_phrases.py` + `domain_tables/phrase_select.py`: exact bigram/trigram counts over compressed ids, log-odds z-score with an informative Dirichlet prior, a domain/general rate-ratio threshold, and a lexical filter (drops numbers, punctuation and table layout). Writes the phrase list, a Bloom filter and a report of the top phrases. |
| Occupancy mask | `DomainTableConfig.track_occupancy` (counts slot hits during training, all-reduced across ranks) and `occupancy_mask` (zeroes heads on untouched slots at eval). |
| Bloom filter | `domain_tables/bloom.py`: ~9.6 bits per key at 1% false positives. `selection_backend="bloom"` uses it in place of the exact list. |
| Random-table control | `table_mode="random_frozen"`, plus `zero_frozen` (pathway only, no n-gram identity). All table arms start from the same seeded rows. |
| Telemetry | `telemetry=True`: tokens, fire rate, head hit rate, mean gate (all and fired), injection norm relative to the hidden state. |
| Table-only checkpoints | `save_tables` / `load_tables` / `load_model_with_tables` (`domain_tables/io.py`). |
| Cached decoding | `DomainEngram` with a `state` dict stored on the KV-cache object. |
| Multi-table loading | **Not yet (Phase 2).** Hook: `EngramQwenModel.engram_layers` is a `ModuleDict` keyed by layer. Phase 2 makes each entry a list of `DomainEngram` tables and sums the enabled ones; the per-module `enabled` flag, table-only files and per-table telemetry already exist. Tables trained separately need their own projections and gates, which `DomainEngram` already has. |
| Router | **Not yet (Phase 2).** Request-level on/off switches map onto `module.enabled`. |

With every option off, `DomainEngram` is numerically identical to `Engram` and loads
the same checkpoints (tested).

## 6. Running Phase 0 on the workstation

```bash
pip install -r requirements.txt && pip install -r requirements-eval.txt
```

```bash
mkdir -p dataset/biomed && huggingface-cli download almanach/Biomed-Enriched data/commercial-00000-of-00026.parquet --repo-type dataset --local-dir dataset/biomed
```

```bash
MODEL=Qwen/Qwen3-0.6B DATA_PATH=dataset/biomed/data/commercial-00000-of-00026.parquet bash train/train_biomedical_1gpu.sh
```

Then evaluate checkpoints 500, 1000, 2000, 3500 and the last one with the existing
scripts (`eval_scripts/run_eval_biomedical_engram.sh`, `run_eval_mmlu_engram.sh`), or
with the new harness after exporting. Done when the 10k/1k numbers land within about
2 SE of the README table.

**Time, rough estimate.** About 7 GFLOP per real token at 36% padding efficiency
(3.3 GFLOP of forward, recompute and activation backward through all 28 layers with
reentrant checkpointing). At an assumed 100 TFLOPS effective in BF16 with eager
attention, that is about 14k real tokens/s: around 30 minutes to `checkpoint-500` and
around 10 hours for a full epoch. Read the real number from the Trainer's
`train_tokens_per_second` (padded tokens) and `gpu_mem.csv`, then recalibrate.
