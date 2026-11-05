"""CPU tests for domain_tables (tiny random Qwen3 + the real Qwen3 tokenizer).

Run from the repo root:  python -m pytest tests -q
The tokenizer comes from $TINYENGRAM_TEST_TOKENIZER (default: Qwen/Qwen3-0.6B,
tokenizer files only, ~11 MB). Tests are skipped if it cannot be loaded.
"""

import copy
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from domain_tables import (  # noqa: E402
    BloomFilter,
    DomainEngram,
    DomainTableConfig,
    PhraseList,
    load_tables,
    ngram_keys,
    read_table_config,
    save_tables,
    tag_keys,
)

TOKENIZER = os.environ.get("TINYENGRAM_TEST_TOKENIZER", "Qwen/Qwen3-0.6B")

DOMAIN_TEXT = "The patient presented with acute myocardial infarction and received heparin."
GENERAL_TEXT = "We walked along the river and talked about the weather all afternoon."


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as exc:  # pragma: no cover - offline machines
        pytest.skip(f"tokenizer {TOKENIZER} unavailable: {exc}")


def tiny_qwen_config(vocab_size):
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )


def table_cfg(**kw):
    base = dict(
        tokenizer_name_or_path=TOKENIZER,
        engram_vocab_size=[1000, 200],
        max_ngram_size=3,
        n_embed_per_ngram=64,
        n_head_per_ngram=4,
        layer_ids=[1, 2],
    )
    base.update(kw)
    return DomainTableConfig(**base)


def build_model(tokenizer, cfg, seed=0, engram_cls=DomainEngram):
    from engram_qwen import EngramQwenForCausalLM, set_skip_engram

    torch.manual_seed(seed)
    model = EngramQwenForCausalLM(tiny_qwen_config(len(tokenizer)), engram_cfg=cfg, engram_cls=engram_cls)
    torch.manual_seed(seed + 1)
    for layer in model.model.engram_layers.values():
        # Make the module contribute visibly (TinyEngram zero-inits value_proj for training).
        torch.nn.init.normal_(layer.value_proj.weight, std=0.2)
        torch.nn.init.normal_(layer.short_conv.conv.weight, std=0.2)
    set_skip_engram(False)
    return model.eval()


def compressed_ids(module, tokenizer, text):
    ids = tokenizer(text, return_tensors="np").input_ids
    return module.hash_mapping.compressed_tokenizer(ids)


def phrase_list_for(model, tokenizer, texts, path):
    module = next(iter(model.model.engram_layers.values()))
    V = module.compressed_vocab
    keys = {2: [], 3: []}
    for t in texts:
        c = compressed_ids(module, tokenizer, t)
        for n in keys:
            k = ngram_keys(c, n, V)
            keys[n].append(k[k >= 0])
    pl = PhraseList({n: np.concatenate(v) for n, v in keys.items()}, vocab_size=V, meta={"tokenizer": TOKENIZER})
    pl.save(path)
    return pl


# ----------------------------------------------------------------- pure numpy


def test_ngram_keys_match_bruteforce():
    rng = np.random.default_rng(0)
    V = 50
    x = rng.integers(0, V, size=(3, 12))
    for n in (1, 2, 3, 4):
        keys = ngram_keys(x, n, V)
        for b in range(3):
            for t in range(12):
                if t < n - 1:
                    assert keys[b, t] == -1
                else:
                    expect = 0
                    for tok in x[b, t - n + 1 : t + 1]:
                        expect = expect * V + int(tok)
                    assert keys[b, t] == expect
    # history columns give context but produce no keys of their own
    k_prefix = ngram_keys(x, 3, V, n_prefix=2)
    assert k_prefix.shape == (3, 10)
    np.testing.assert_array_equal(k_prefix, ngram_keys(x, 3, V)[:, 2:])
    # negative ids invalidate every window that contains them
    y = x.copy()
    y[0, 5] = -1
    k = ngram_keys(y, 2, V)
    assert k[0, 5] == -1 and k[0, 6] == -1 and k[0, 7] != -1


def test_bloom_filter_no_false_negatives_and_fp_rate(tmp_path):
    rng = np.random.default_rng(1)
    members = rng.choice(1 << 40, size=20000, replace=False).astype(np.int64)
    others = rng.choice(1 << 40, size=200000, replace=False).astype(np.int64)
    others = np.setdiff1d(others, members)
    bf = BloomFilter.for_capacity(len(members), fp_rate=0.01)
    bf.add(members)
    assert bf.contains(members).all()
    fp = bf.contains(others).mean()
    assert fp < 0.02, fp
    assert bf.size_bytes < 20000 * 10 / 8 + 64  # ~9.6 bits per key
    path = str(tmp_path / "f.npz")
    bf.save(path)
    bf2 = BloomFilter.load(path)
    np.testing.assert_array_equal(bf2.contains(others[:1000]), bf.contains(others[:1000]))
    assert not bf.contains(np.array([-1])).any()


def test_tag_keys_separates_orders():
    k = np.array([5, -1], dtype=np.int64)
    assert tag_keys(k, 2)[0] != tag_keys(k, 3)[0]
    assert tag_keys(k, 2)[1] == -1


def test_phrase_list_roundtrip(tmp_path):
    pl = PhraseList({2: np.array([7, 3, 3, 9]), 3: np.array([100])}, vocab_size=1000, meta={"x": 1})
    path = str(tmp_path / "p.npz")
    pl.save(path)
    pl2 = PhraseList.load(path)
    assert pl2.vocab_size == 1000 and pl2.meta["x"] == 1 and len(pl2) == 4
    np.testing.assert_array_equal(pl2.contains(np.array([[3, 4, 9, -1]]), 2), [[True, False, True, False]])
    assert not pl2.contains(np.array([3]), 3).any()


# ---------------------------------------------------------- model integration


def test_default_domain_engram_matches_engram(tokenizer):
    from engram import Engram

    cfg = table_cfg()
    ref = build_model(tokenizer, cfg, engram_cls=Engram)
    new = build_model(tokenizer, cfg, engram_cls=DomainEngram)
    missing, unexpected = new.load_state_dict(ref.state_dict(), strict=True)
    assert not missing and not unexpected
    ids = tokenizer([DOMAIN_TEXT, GENERAL_TEXT], return_tensors="pt", padding=True).input_ids
    with torch.no_grad():
        torch.testing.assert_close(new(ids, use_cache=False).logits, ref(ids, use_cache=False).logits)


def _incremental_logits(model, ids, prefill):
    out = model(ids[:, :prefill], use_cache=True)
    pkv, steps = out.past_key_values, [out.logits]
    for t in range(prefill, ids.shape[1]):
        out = model(ids[:, t : t + 1], past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        steps.append(out.logits)
    return torch.cat(steps, dim=1)


def test_cached_decoding_matches_full_forward(tokenizer):
    model = build_model(tokenizer, table_cfg())
    ids = tokenizer(DOMAIN_TEXT, return_tensors="pt").input_ids
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        for prefill in (1, 2, 5):
            torch.testing.assert_close(_incremental_logits(model, ids, prefill), full, atol=1e-4, rtol=1e-4)


def test_base_engram_cached_decoding_differs(tokenizer):
    """Documents the upstream behaviour that DomainEngram fixes."""
    from engram import Engram

    model = build_model(tokenizer, table_cfg(), engram_cls=Engram)
    ids = tokenizer(DOMAIN_TEXT, return_tensors="pt").input_ids
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        inc = _incremental_logits(model, ids, 3)
    assert (full - inc).abs().max() > 1e-3


def test_generate_with_cache_matches_without(tokenizer):
    model = build_model(tokenizer, table_cfg())
    ids = tokenizer(DOMAIN_TEXT, return_tensors="pt").input_ids[:, :6]
    kw = dict(max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    with torch.no_grad():
        a = model.generate(ids, use_cache=True, **kw)
        b = model.generate(ids, use_cache=False, **kw)
    assert torch.equal(a, b)


def test_phrase_list_masks_unregistered_text(tokenizer, tmp_path):
    from engram_qwen import set_skip_engram

    probe = build_model(tokenizer, table_cfg())
    path = str(tmp_path / "phrases.npz")
    phrase_list_for(probe, tokenizer, [DOMAIN_TEXT], path)

    for backend in ("exact", "bloom"):
        cfg = table_cfg(phrase_list_path=path, selection_backend=backend, telemetry=True, bloom_fp_rate=1e-4)
        model = build_model(tokenizer, cfg)
        model.load_state_dict(probe.state_dict(), strict=False)
        general = tokenizer(GENERAL_TEXT, return_tensors="pt").input_ids
        domain = tokenizer(DOMAIN_TEXT, return_tensors="pt").input_ids
        with torch.no_grad():
            set_skip_engram(True)
            base_general = model(general).logits
            base_domain = model(domain).logits
            set_skip_engram(False)
            for m in model.model.engram_layers.values():
                m.reset_telemetry()
            on_general = model(general).logits
            tele_general = [m.get_telemetry() for m in model.model.engram_layers.values()]
            on_domain = model(domain).logits
        # Strict masking: no registered n-gram, so the table changes nothing at all.
        assert torch.equal(on_general, base_general), backend
        assert all(t["fired"] == 0 for t in tele_general)
        # Registered text: the table fires and changes the output.
        assert (on_domain - base_domain).abs().max() > 1e-3
        fired = [m.get_telemetry()["fired"] for m in model.model.engram_layers.values()]
        assert all(f == domain.shape[1] - 1 for f in fired), fired  # every position but the first


def test_occupancy_tracking_and_mask(tokenizer, tmp_path):
    cfg = table_cfg(track_occupancy=True, occupancy_mask=True)
    model = build_model(tokenizer, cfg)
    ids = tokenizer(DOMAIN_TEXT, return_tensors="pt").input_ids
    model.train()
    model(ids, use_cache=False)
    module = model.model.engram_layers["1"]
    stats = module.occupancy_stats()
    n_heads = 2 * cfg.n_head_per_ngram
    assert 0 < stats["occupied"] <= ids.shape[1] * n_heads
    assert int(module.occupancy.sum()) == ids.shape[1] * n_heads
    # In eval mode, unseen text lands mostly on untouched slots and is masked out.
    model.eval()
    for m in model.model.engram_layers.values():
        m.config.telemetry = True
        m.reset_telemetry()
    with torch.no_grad():
        model(tokenizer(GENERAL_TEXT, return_tensors="pt").input_ids)
        t_general = module.get_telemetry()
        module.reset_telemetry()
        model(ids)
        t_domain = module.get_telemetry()
    assert t_domain["head_hit_rate"] == 1.0
    assert t_general["head_hit_rate"] < 0.5


@pytest.mark.parametrize("mode", ["random_frozen", "zero_frozen", "trained"])
def test_table_modes(tokenizer, mode):
    cfg = table_cfg(table_mode=mode)
    model = build_model(tokenizer, cfg)
    for m in model.model.engram_layers.values():
        m.reset_table_()
    module = model.model.engram_layers["1"]
    w = module.multi_head_embedding.embedding.weight
    assert w.requires_grad == (mode == "trained")
    if mode == "zero_frozen":
        assert torch.count_nonzero(w) == 0
    else:
        assert abs(w.std().item() - cfg.table_init_std) < 0.005
    # Same seed -> same table across arms (trained and random_frozen start identical).
    other = build_model(tokenizer, table_cfg(table_mode="trained"), seed=5)
    other.model.engram_layers["1"].reset_table_()
    if mode != "zero_frozen":
        torch.testing.assert_close(other.model.engram_layers["1"].multi_head_embedding.embedding.weight, w)
    model.train()
    for name, p in model.named_parameters():
        p.requires_grad_(("engram" in name) and not (name.endswith("embedding.weight") and mode != "trained"))
    ids = tokenizer(DOMAIN_TEXT, return_tensors="pt").input_ids
    model(ids, labels=ids, use_cache=False).loss.backward()
    assert (w.grad is not None) == (mode == "trained")
    assert module.value_proj.weight.grad is not None


def test_save_and_load_tables(tokenizer, tmp_path):
    probe = build_model(tokenizer, table_cfg())
    phrases = str(tmp_path / "src_phrases.npz")
    phrase_list_for(probe, tokenizer, [DOMAIN_TEXT], phrases)
    cfg = table_cfg(phrase_list_path=phrases, track_occupancy=True)
    model = build_model(tokenizer, cfg)
    out = str(tmp_path / "table")
    save_tables(model, out, base_model="tiny")
    assert os.path.exists(os.path.join(out, "phrases.npz"))

    cfg2 = read_table_config(out, tokenizer_name_or_path=TOKENIZER)
    assert cfg2.phrase_list_path == os.path.join(out, "phrases.npz")
    clone = build_model(tokenizer, copy.deepcopy(cfg2), seed=99)
    clone.load_state_dict({k: v for k, v in model.state_dict().items() if "engram" not in k}, strict=False)
    load_tables(clone, out)
    ids = tokenizer([DOMAIN_TEXT], return_tensors="pt").input_ids
    with torch.no_grad():
        torch.testing.assert_close(clone(ids).logits, model(ids).logits)
