"""Table-only checkpoints.

A domain table is saved without the base model: the Engram parameters
(``table.safetensors``), the table config (``table_config.json``) and, if the table
uses one, its phrase list and Bloom filter. For Qwen3-0.6B with the 10k/1k setup this
is 127 MB (31.5M params, fp32) instead of a 1.2 GB full checkpoint, and it is what lets tables be
trained, swapped and stacked independently of the frozen base.
"""

import json
import os
import shutil
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

from .table import DomainTableConfig

TABLE_WEIGHTS = "table.safetensors"
TABLE_CONFIG = "table_config.json"
PHRASES_FILE = "phrases.npz"
FORMAT_VERSION = 1


def _engram_layers(model):
    """Find the ``engram_layers`` ModuleDict on an EngramQwen model (or a PEFT wrapper)."""
    for module in model.modules():
        if hasattr(module, "engram_layers") and hasattr(module, "engram_cfg"):
            return module.engram_layers, module.engram_cfg
    raise ValueError("model has no engram_layers")


def save_tables(model, out_dir: str, base_model: Optional[str] = None, extra: Optional[dict] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    layers, cfg = _engram_layers(model)
    state = {k: v.detach().cpu().contiguous() for k, v in layers.state_dict().items()}
    save_file(state, os.path.join(out_dir, TABLE_WEIGHTS))

    cfg_dict = cfg.to_dict()
    phrase_path = cfg_dict.get("phrase_list_path")
    if phrase_path:
        dst = os.path.join(out_dir, PHRASES_FILE)
        if os.path.abspath(phrase_path) != os.path.abspath(dst):
            shutil.copyfile(phrase_path, dst)
        bloom_src = os.path.splitext(phrase_path)[0] + ".bloom.npz"
        if os.path.exists(bloom_src):
            shutil.copyfile(bloom_src, os.path.splitext(dst)[0] + ".bloom.npz")
        cfg_dict["phrase_list_path"] = PHRASES_FILE  # resolved relative to the table dir on load

    n_params = sum(v.numel() for k, v in state.items() if v.is_floating_point())
    meta = {
        "format_version": FORMAT_VERSION,
        "engram_cfg": cfg_dict,
        "base_model": base_model,
        "num_params": int(n_params),
        "layers": sorted(int(k) for k in layers.keys()),
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(out_dir, TABLE_CONFIG), "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def read_table_config(table_dir: str, tokenizer_name_or_path: Optional[str] = None) -> DomainTableConfig:
    with open(os.path.join(table_dir, TABLE_CONFIG)) as f:
        meta = json.load(f)
    cfg = DomainTableConfig.from_dict(meta["engram_cfg"])
    if cfg.phrase_list_path and not os.path.isabs(cfg.phrase_list_path):
        cfg.phrase_list_path = os.path.join(table_dir, cfg.phrase_list_path)
    if tokenizer_name_or_path is not None:
        cfg.tokenizer_name_or_path = tokenizer_name_or_path
    return cfg


def load_tables(model, table_dir: str, strict: bool = True):
    layers, _ = _engram_layers(model)
    state = load_file(os.path.join(table_dir, TABLE_WEIGHTS))
    missing, unexpected = layers.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"table mismatch: missing={missing} unexpected={unexpected}")
    return missing, unexpected


def load_model_with_tables(
    base_model: str,
    table_dir: Optional[str] = None,
    cfg: Optional[DomainTableConfig] = None,
    dtype=torch.bfloat16,
    device_map=None,
    **cfg_overrides,
):
    """Build ``EngramQwenForCausalLM`` from frozen base weights plus a saved table.

    ``cfg_overrides`` replace fields of the saved table config, e.g.
    ``selection_backend="bloom"`` or ``telemetry=True`` at evaluation time.
    """
    from .model import DomainEngramQwenForCausalLM

    if cfg is None:
        if table_dir is None:
            raise ValueError("need table_dir or cfg")
        cfg = read_table_config(table_dir, tokenizer_name_or_path=base_model)
    for k, v in cfg_overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown table config field {k}")
        setattr(cfg, k, v)
    model = DomainEngramQwenForCausalLM.from_pretrained(
        base_model, engram_cfg=cfg, dtype=dtype, device_map=device_map
    )
    if table_dir is not None:
        load_tables(model, table_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model
