"""DomainEngram: the Engram module with the controls the domain-table experiments need.

With every option at its default, ``DomainEngram`` computes exactly what ``Engram``
computes and has the same parameter names, so existing TinyEngram checkpoints load
into it unchanged. On top of that it adds:

* Selectivity. ``phrase_list_path`` restricts lookups to registered n-grams, checked
  exactly or through a Bloom filter (``selection_backend``). Heads whose n-gram is not
  registered contribute a zero row, and positions where nothing matched produce exactly
  zero output when ``strict_mask`` is on. ``occupancy_mask`` zeroes heads that land on
  slots training never touched (``track_occupancy`` counts those during training).
* Controls. ``table_mode`` is ``trained`` (default), ``random_frozen`` (rows fixed at
  their random init; only the gate, projections and conv train) or ``zero_frozen``
  (no n-gram identity at all; tests whether the extra pathway alone explains gains).
* Telemetry. Fire counts, gate magnitudes and injection norms per table, for
  "which tables matter" analysis. Off by default because it adds device syncs.
* Correct cached decoding. When the caller passes a ``state`` dict, the module keeps
  the last ``max_ngram_size - 1`` token ids and the ShortConv history between calls,
  so incremental decoding matches a full forward pass. The base ``Engram`` sees only
  the new token during cached decoding and hashes it against padding.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

import engram as engram_base
from engram import Engram, EngramConfig

from .bloom import BloomFilter
from .ngram_keys import ngram_keys, tag_keys
from .phrase_list import PhraseList

TABLE_MODES = ("trained", "random_frozen", "zero_frozen")
SELECTION_BACKENDS = ("exact", "bloom")

# Order of the telemetry counters (one float64 slot each).
TELEMETRY_FIELDS = (
    "tokens",           # positions seen (padding excluded when a token mask is given)
    "fired",            # positions where at least one head was allowed to read
    "gate_sum",         # sum of gate values over all positions
    "gate_sum_fired",   # sum of gate values over fired positions
    "inject_ratio_sum_fired",  # sum over fired positions of ||engram_out|| / ||hidden||
    "head_hits",        # allowed head lookups (out of tokens * heads)
)


@dataclass
class DomainTableConfig(EngramConfig):
    table_name: str = "domain"
    table_mode: str = "trained"
    table_seed: int = 1234
    table_init_std: float = 0.02
    phrase_list_path: Optional[str] = None
    selection_backend: str = "exact"
    bloom_fp_rate: float = 0.01
    strict_mask: bool = True
    track_occupancy: bool = False
    occupancy_mask: bool = False
    occupancy_min_count: int = 1
    telemetry: bool = False

    def __post_init__(self):
        if self.table_mode not in TABLE_MODES:
            raise ValueError(f"table_mode must be one of {TABLE_MODES}, got {self.table_mode!r}")
        if self.selection_backend not in SELECTION_BACKENDS:
            raise ValueError(f"selection_backend must be one of {SELECTION_BACKENDS}")


class DomainEngram(Engram):
    supports_incremental = True

    def __init__(self, layer_id, config: EngramConfig, backbone_hidden_size: int, backbone_hc_mult: int):
        if not isinstance(config, DomainTableConfig):
            config = DomainTableConfig.from_dict(config.to_dict())
        super().__init__(layer_id, config, backbone_hidden_size, backbone_hc_mult)
        cfg = self.config

        self.heads_per_order = cfg.n_head_per_ngram
        self.orders = list(range(2, cfg.max_ngram_size + 1))
        self.total_rows = int(self.multi_head_embedding.embedding.num_embeddings)
        self.compressed_vocab = int(self.hash_mapping.tokenizer_vocab_size)

        self.phrases = None
        self.bloom = None
        if cfg.phrase_list_path:
            self.load_phrase_list(cfg.phrase_list_path)

        if cfg.track_occupancy or cfg.occupancy_mask:
            self.register_buffer("occupancy", torch.zeros(self.total_rows, dtype=torch.int32))
        else:
            self.occupancy = None

        # Telemetry accumulators: a plain float64 tensor, not a buffer, so it stays out of
        # checkpoints and is not downcast by model.to(dtype) / module.float().
        self._telemetry = None
        self.enabled = True
        self.set_table_trainable()

    # ------------------------------------------------------------------ setup
    def load_phrase_list(self, path: str) -> None:
        phrases = PhraseList.load(path)
        if phrases.vocab_size != self.compressed_vocab:
            raise ValueError(
                f"phrase list {path} was built for compressed vocab {phrases.vocab_size}, "
                f"but this tokenizer compresses to {self.compressed_vocab}"
            )
        missing = [n for n in self.orders if n not in phrases.keys]
        if missing:
            raise ValueError(f"phrase list {path} has no keys for orders {missing}")
        self.phrases = phrases
        if self.config.selection_backend == "bloom":
            bloom_path = os.path.splitext(path)[0] + ".bloom.npz"
            if os.path.exists(bloom_path):
                self.bloom = BloomFilter.load(bloom_path)
            else:
                self.bloom = phrases.to_bloom(fp_rate=self.config.bloom_fp_rate)

    def set_table_trainable(self) -> None:
        trainable = self.config.table_mode == "trained"
        self.multi_head_embedding.embedding.weight.requires_grad_(trainable)

    @torch.no_grad()
    def reset_table_(self, seed: Optional[int] = None) -> None:
        """Deterministically (re)initialize the rows, so every arm starts from the same table."""
        weight = self.multi_head_embedding.embedding.weight
        if self.config.table_mode == "zero_frozen":
            weight.zero_()
        else:
            seed = self.config.table_seed if seed is None else seed
            gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + int(self.layer_id))
            rows = torch.randn(weight.shape, generator=gen, dtype=torch.float32) * self.config.table_init_std
            weight.copy_(rows.to(weight.dtype))
        self.set_table_trainable()

    # -------------------------------------------------------------- telemetry
    def reset_telemetry(self) -> None:
        self._telemetry = None

    def get_telemetry(self) -> dict:
        if self._telemetry is None:
            vals = [0.0] * len(TELEMETRY_FIELDS)
        else:
            vals = self._telemetry.detach().cpu().tolist()
        t = dict(zip(TELEMETRY_FIELDS, vals))
        tokens = max(t["tokens"], 1.0)
        fired = max(t["fired"], 1.0)
        return {
            "table": self.config.table_name,
            "layer": int(self.layer_id),
            "tokens": int(t["tokens"]),
            "fired": int(t["fired"]),
            "fire_rate": t["fired"] / tokens,
            "head_hit_rate": t["head_hits"] / (tokens * len(self.orders) * self.heads_per_order),
            "mean_gate": t["gate_sum"] / tokens,
            "mean_gate_fired": t["gate_sum_fired"] / fired,
            "mean_inject_ratio_fired": t["inject_ratio_sum_fired"] / fired,
        }

    def occupancy_stats(self) -> Optional[dict]:
        if self.occupancy is None:
            return None
        occ = self.occupancy
        return {
            "rows": self.total_rows,
            "occupied": int((occ >= self.config.occupancy_min_count).sum()),
            "occupied_frac": float((occ >= self.config.occupancy_min_count).float().mean()),
        }

    # -------------------------------------------------------------- selection
    def _selection_mask(self, compressed: np.ndarray, n_prefix: int) -> Optional[np.ndarray]:
        """Per-head allow mask ``[B, T, H]`` from the phrase list, or ``None`` if unrestricted."""
        if self.phrases is None:
            return None
        per_order = []
        for n in self.orders:
            keys = ngram_keys(compressed, n, self.compressed_vocab, n_prefix=n_prefix)
            if self.bloom is not None:
                allowed = self.bloom.contains(tag_keys(keys, n))
            else:
                allowed = self.phrases.contains(keys, n)
            per_order.append(np.repeat(allowed[:, :, None], self.heads_per_order, axis=2))
        return np.concatenate(per_order, axis=2)

    # ---------------------------------------------------------------- forward
    def forward(self, hidden_states, input_ids, state: Optional[dict] = None, token_mask=None):
        """
        hidden_states: [B, L, HC_MULT, D]
        input_ids:     [B, L]
        state:         optional dict owned by the caller, kept across cached-decoding calls
        token_mask:    optional [B, L] (1 = real token), used only for occupancy and telemetry
        """
        cfg = self.config
        if isinstance(input_ids, torch.Tensor):
            ids_np = input_ids.detach().cpu().numpy()
            device = input_ids.device
        else:
            ids_np = np.asarray(input_ids)
            device = hidden_states.device
        B, T = ids_np.shape
        n_ctx = cfg.max_ngram_size - 1

        # Prepend the token history kept from the previous decoding step.
        prefix = state.get("tail_ids") if state is not None else None
        full_ids = ids_np if prefix is None else np.concatenate([prefix, ids_np], axis=1)
        n_prefix = full_ids.shape[1] - T
        if state is not None:
            state["tail_ids"] = full_ids[:, full_ids.shape[1] - min(n_ctx, full_ids.shape[1]):]

        compressed = self.hash_mapping.compressed_tokenizer(full_ids)
        hashes = self.hash_mapping._get_ngram_hashes(compressed, layer_id=self.layer_id)[:, n_prefix:]
        hash_t = torch.from_numpy(np.ascontiguousarray(hashes)).to(device)

        allow_np = self._selection_mask(compressed, n_prefix)
        allow = torch.from_numpy(allow_np).to(device) if allow_np is not None else None

        if token_mask is not None:
            token_mask = token_mask[:, -T:].to(device=device, dtype=torch.bool)

        if self.occupancy is not None and self.training and cfg.track_occupancy:
            self._update_occupancy(hash_t, allow, token_mask)
        if self.occupancy is not None and cfg.occupancy_mask and not self.training:
            occupied = self.occupancy[hash_t + self.multi_head_embedding.offsets] >= cfg.occupancy_min_count
            allow = occupied if allow is None else (allow & occupied)

        emb = self.multi_head_embedding(hash_t)  # [B, T, H, d]
        fired = None
        if allow is not None:
            emb = emb * allow.unsqueeze(-1).to(emb.dtype)
            fired = allow.any(dim=-1)  # [B, T]
        embeddings = emb.flatten(start_dim=-2)

        gates = []
        for hc_idx in range(self.backbone_hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.backbone_hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates, dim=2)  # [B, T, HC, 1]

        step = engram_base._GLOBAL_STEP
        if step < cfg.warmup_steps:
            gates = torch.ones_like(gates)
        elif step < cfg.warmup_steps + cfg.soft_constraint_steps:
            gates = gates.clamp_min(0.1)

        value = gates * self.value_proj(embeddings).unsqueeze(2)  # [B, T, HC, D]
        if fired is not None:
            value = value * fired[:, :, None, None].to(value.dtype)

        if state is not None:
            history = (cfg.kernel_size - 1) * cfg.max_ngram_size  # ShortConv dilation = max_ngram_size
            prev = state.get("value_hist")
            if prev is None or prev.shape[0] != B:
                prev = value.new_zeros((B, history) + tuple(value.shape[2:]))
            conv_in = torch.cat([prev.to(value.dtype), value], dim=1)
            conv_out = self.short_conv(conv_in)[:, -T:]
            state["value_hist"] = conv_in[:, -history:].detach()
        else:
            conv_out = self.short_conv(value)
        output = value + conv_out

        if fired is not None and cfg.strict_mask:
            output = output * fired[:, :, None, None].to(output.dtype)

        if cfg.telemetry:
            self._record(hidden_states, output, gates, allow, fired, token_mask)
        return output

    @torch.no_grad()
    def _update_occupancy(self, hash_t, allow, token_mask):
        idx = hash_t + self.multi_head_embedding.offsets
        keep = torch.ones_like(idx, dtype=torch.bool)
        if allow is not None:
            keep &= allow
        if token_mask is not None:
            keep &= token_mask.unsqueeze(-1)
        inc = torch.bincount(idx[keep].reshape(-1), minlength=self.total_rows).to(torch.int32)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(inc)
        self.occupancy += inc.to(self.occupancy.device)

    @torch.no_grad()
    def _record(self, hidden_states, output, gates, allow, fired, token_mask):
        B, T = output.shape[:2]
        mask = token_mask if token_mask is not None else torch.ones(B, T, dtype=torch.bool, device=output.device)
        if fired is None:
            fired = torch.ones_like(mask)
        fired = fired & mask
        gate = gates.float().mean(dim=(2, 3))  # [B, T]
        ratio = output.float().norm(dim=-1).mean(-1) / hidden_states.float().norm(dim=-1).mean(-1).clamp_min(1e-6)
        n_heads = len(self.orders) * self.heads_per_order
        head_hits = (allow & mask.unsqueeze(-1)).sum() if allow is not None else mask.sum() * n_heads
        upd = torch.stack([
            mask.sum().double(),
            fired.sum().double(),
            (gate * mask).sum().double(),
            (gate * fired).sum().double(),
            (ratio * fired).sum().double(),
            torch.as_tensor(head_hits, device=output.device).double(),
        ])
        if self._telemetry is None or self._telemetry.device != upd.device:
            self._telemetry = torch.zeros(len(TELEMETRY_FIELDS), dtype=torch.float64, device=upd.device)
        self._telemetry += upd
