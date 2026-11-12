# Test 2: real wall-clock speed with llama.cpp

llama.cpp built from master (`results/clore_run7b_speed/llama_commit.txt`) with CUDA on one
RTX PRO 6000. Target Qwen3-8B Q4_K_M; draft model Qwen3-0.6B Q8_0; static n-gram caches
built with `llama-lookup-create` from 20 MB of Biomed-Enriched (medical) or FineWeb-Edu
(general) text. 40 medical and 40 general held-out passages; the model was asked to explain
each one (chat endpoint, greedy, thinking off, up to 512 tokens, stopping naturally).
Speed = generated tokens / generation time from the server's own timings.
Scripts: `speedtest/`.

| Config | Medical tok/s | Speedup | General tok/s | Speedup |
|---|---|---|---|---|
| plain | 248 | 1.00 | 247 | 1.00 |
| draft model (0.6B) | 210 | 0.85 | 211 | 0.85 |
| n-gram cache, context only (no file) | 260 | 1.05 | 259 | 1.04 |
| n-gram cache + medical file | 260 | 1.05 | 255 | 1.03 |
| n-gram cache + general file | 258 | 1.04 | 258 | 1.04 |
| draft model + medical cache | 211 | 0.85 | 209 | 0.84 |
| draft model + general cache | 212 | 0.85 | 206 | 0.83 |
| ngram-mod (prompt/history) | 250 | 1.01 | 251 | 1.01 |

Findings:
- On natural answers (2% repeated 4-grams), n-gram speculation gives only about 4–5%, and
  the static medical file adds nothing over the context-only cache (260 vs 260 tok/s on
  medical text). The domain "speed pack" idea shows no measurable benefit here.
- The 0.6B draft model makes generation 15% slower: on this GPU, an 8B Q4 model already runs
  at about 250 tok/s, and drafting costs more than it saves.
- An earlier run (`results/clore_run7_speed`) with forced 256-token raw continuations, which
  repeated heavily (~40% repeated 4-grams), showed 1.3–1.5x for every n-gram method,
  including on general text with the medical file: the gain came from copying repeated
  text, not from the domain cache.
- Speculative outputs were not byte-identical to plain greedy output (1–3 of 40 identical
  for most configs, 28–29 for ngram-mod): batched verification changes floating-point
  results enough to flip near-tied tokens, after which answers diverge. Speeds therefore
  compare slightly different texts of similar length.

Context: the offline draft-head test (`mtp_results.md`) measured accepted tokens per step,
which did improve with a domain lookup; on real hardware, at single-stream speed, that
headroom largely disappears. Results may differ for larger models, slower GPUs, CPU
inference or repetitive workloads (code editing, templated documents).
