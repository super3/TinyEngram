"""Train one domain table (or a matched baseline arm) on a frozen Qwen3 model.

Arms (``--arm``):
  table         trained n-gram table + gate/projections (TinyEngram-style)
  random_table  same architecture, rows frozen at their random init (pathway + random codes)
  zero_table    rows frozen at zero (pathway only: gate, bias and conv, no n-gram identity)
  lora          LoRA on all linear layers of the frozen base, same data

All table arms start from the same seeded table and zero-initialized value_proj, so the
only difference between ``table`` and ``random_table`` is whether the rows learn.

Differences from train/train_biomedical.py:
  * no hard DeepSpeed dependency (single-GPU runs need none); --deepspeed still works (ZeRO-2)
  * table arms save only the table (``table.safetensors`` + config, tens of MB) unless
    --save_full_model is set; LoRA saves its adapter
  * optional phrase list (selectivity), occupancy tracking, frozen-table controls
  * non-reentrant checkpointing for table arms, so layers below the lowest table are not
    backpropagated through
  * throughput.json: tokens/s and peak GPU memory, for Phase 0/1 calibration
"""

import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import transformers
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainerCallback

import engram
from domain_tables import DomainEngram, DomainTableConfig, load_tables, save_tables
from domain_tables.io import TABLE_WEIGHTS
from domain_tables.model import DomainEngramQwenForCausalLM
from domain_tables.phrase_select import _parse_filter
from engram_qwen import set_skip_engram

TABLE_ARMS = {"table": "trained", "random_table": "random_frozen", "zero_table": "zero_frozen"}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-0.6B-Base")
    arm: str = field(default="table", metadata={"help": "table | random_table | zero_table | lora"})
    base_dtype: str = field(default="bf16", metadata={"help": "dtype of the frozen base: bf16 | fp32"})
    attn_implementation: str = field(default="sdpa")
    # table
    engram_layer_ids: List[int] = field(default_factory=lambda: [5, 7, 13, 17])
    engram_vocab_size: List[int] = field(default_factory=lambda: [10000, 1000])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    table_name: str = "domain"
    table_seed: int = 1234
    phrase_list_path: Optional[str] = None
    selection_backend: str = "exact"
    strict_mask: bool = True
    track_occupancy: bool = False
    engram_warmup_steps: int = 0
    engram_soft_constraint_steps: int = 0
    # lora
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    )


@dataclass
class DataArguments:
    data_path: str = field(default="dataset/biomed/*.parquet")
    text_column: str = "text"
    train_filter: str = "language==en,domain==biomedical,educational_score>4.0"
    eval_filter: str = "language==en,domain==biomedical,educational_score<4.0"
    eval_samples: int = 200
    max_train_samples: Optional[int] = None
    num_proc: int = 8


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1500)
    save_full_model: bool = field(default=False, metadata={"help": "table arms: also save base weights"})
    timestamp_output_dir: bool = field(default=True)


def rank0_print(*args):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, flush=True)


def make_batched_filter(expr):
    preds = _parse_filter(expr)
    ops = {
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b, ">": lambda a, b: a > b,
        "<": lambda a, b: a < b, ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
    }

    def fn(batch):
        n = len(next(iter(batch.values())))
        keep = [True] * n
        for col, op, val in preds:
            vals = batch[col]
            try:
                ref = float(val)
                cmp = [(v is not None) and ops[op](float(v), ref) for v in vals]
            except ValueError:
                cmp = [ops[op](v, val) for v in vals]
            keep = [k and c for k, c in zip(keep, cmp)]
        return keep

    return fn, sorted({c for c, _, _ in preds})


class EngramStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        engram.set_global_step(state.global_step)


class ThroughputCallback(TrainerCallback):
    """Logs tokens/s (non-padding) and peak GPU memory; writes throughput.json at the end."""

    def __init__(self):
        self.t0 = None
        self.last = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = self.last = (time.time(), state.num_input_tokens_seen or 0)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.last is None:
            return
        now, tokens = time.time(), state.num_input_tokens_seen or 0
        dt = max(now - self.last[0], 1e-6)
        logs["tokens_per_sec"] = round((tokens - self.last[1]) / dt, 1)
        if torch.cuda.is_available():
            logs["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        self.last = (now, tokens)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self.t0 is None:
            return
        wall = time.time() - self.t0[0]
        tokens = (state.num_input_tokens_seen or 0) - self.t0[1]
        out = {
            "steps": state.global_step,
            "train_tokens_non_padding": int(tokens),
            "wall_seconds": round(wall, 1),
            "tokens_per_sec": round(tokens / max(wall, 1e-6), 1),
            "world_size": args.world_size,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        }
        with open(os.path.join(args.output_dir, "throughput.json"), "w") as f:
            json.dump(out, f, indent=2)
        rank0_print("throughput:", json.dumps(out))


class OccupancyLogCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        for module in self.model.modules():
            if isinstance(module, DomainEngram) and module.occupancy is not None:
                logs[f"occupied_frac_l{module.layer_id}"] = round(module.occupancy_stats()["occupied_frac"], 4)


class TableTrainer(Trainer):
    """Checkpoints hold only the table for table arms (unless --save_full_model)."""

    def __init__(self, *args, table_only=False, base_model=None, table_extra=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_only = table_only
        self.base_model_name = base_model
        self.table_extra = table_extra

    def _save(self, output_dir=None, state_dict=None):
        if not self.table_only:
            return super()._save(output_dir, state_dict=state_dict)
        output_dir = output_dir or self.args.output_dir
        save_tables(self.model, output_dir, base_model=self.base_model_name, extra=self.table_extra)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if self.table_only and os.path.exists(os.path.join(resume_from_checkpoint, TABLE_WEIGHTS)):
            load_tables(model or self.model, resume_from_checkpoint)
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)


def build_model(margs, targs):
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[margs.base_dtype]
    if margs.arm == "lora":
        from peft import LoraConfig, get_peft_model

        model = transformers.AutoModelForCausalLM.from_pretrained(
            margs.model_name_or_path, dtype=dtype, attn_implementation=margs.attn_implementation
        )
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=margs.lora_r, lora_alpha=margs.lora_alpha, lora_dropout=margs.lora_dropout,
            target_modules=margs.lora_target_modules, bias="none", task_type="CAUSAL_LM",
        ))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()  # fp32 master weights for the adapter
        if targs.gradient_checkpointing:
            model.enable_input_require_grads()
        return model, None

    cfg = DomainTableConfig(
        tokenizer_name_or_path=margs.model_name_or_path,
        engram_vocab_size=margs.engram_vocab_size,
        max_ngram_size=margs.max_ngram_size,
        n_embed_per_ngram=margs.n_embed_per_ngram,
        n_head_per_ngram=margs.n_head_per_ngram,
        layer_ids=margs.engram_layer_ids,
        warmup_steps=margs.engram_warmup_steps,
        soft_constraint_steps=margs.engram_soft_constraint_steps,
        table_name=margs.table_name,
        table_mode=TABLE_ARMS[margs.arm],
        table_seed=margs.table_seed,
        phrase_list_path=margs.phrase_list_path,
        selection_backend=margs.selection_backend,
        strict_mask=margs.strict_mask,
        track_occupancy=margs.track_occupancy,
    )
    config = transformers.AutoConfig.from_pretrained(margs.model_name_or_path)
    config.use_cache = False
    model = DomainEngramQwenForCausalLM.from_pretrained(
        margs.model_name_or_path, config=config, dtype=dtype, attn_implementation=margs.attn_implementation,
        engram_cfg=cfg,
    )
    for name, p in model.named_parameters():
        p.requires_grad_("engram_layers" in name)
    for module in model.model.engram_layers.values():
        module.float()  # trainable table params in fp32; the frozen base stays in base_dtype
        module.reset_table_()  # same seeded rows for every table arm; sets requires_grad per mode
        torch.nn.init.zeros_(module.value_proj.weight)
        torch.nn.init.zeros_(module.value_proj.bias)
    return model, cfg


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    margs, dargs, targs = parser.parse_args_into_dataclasses()
    if margs.arm not in TABLE_ARMS and margs.arm != "lora":
        raise ValueError(f"unknown arm {margs.arm}")
    set_skip_engram(False)

    if targs.timestamp_output_dir and not targs.resume_from_checkpoint:
        run = f"{margs.arm}_{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
        targs.output_dir = os.path.join(targs.output_dir, run)
        targs.logging_dir = os.path.join(targs.output_dir, "runs")
    if targs.include_num_input_tokens_seen in ("no", False):
        targs.include_num_input_tokens_seen = "non_padding"
    is_table = margs.arm in TABLE_ARMS
    if is_table and targs.gradient_checkpointing:
        # Non-reentrant checkpointing lets autograd skip layers below the lowest table.
        targs.gradient_checkpointing_kwargs = {"use_reentrant": False}

    transformers.set_seed(targs.seed)
    tokenizer = AutoTokenizer.from_pretrained(margs.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, cfg = build_model(margs, targs)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    rank0_print(f"arm={margs.arm} trainable params: {n_train / 1e6:.2f}M of {n_total / 1e6:.1f}M")

    # ---- data
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=dargs.data_path, split="train")
    train_fn, train_cols = make_batched_filter(dargs.train_filter)
    eval_fn, eval_cols = make_batched_filter(dargs.eval_filter)
    with targs.main_process_first(desc="filter"):
        train_ds = ds.select_columns(train_cols + [dargs.text_column]).filter(
            train_fn, batched=True, input_columns=None, num_proc=dargs.num_proc)
        eval_ds = ds.select_columns(eval_cols + [dargs.text_column]).filter(
            eval_fn, batched=True, num_proc=dargs.num_proc)
    eval_ds = eval_ds.shuffle(seed=42).select(range(min(dargs.eval_samples, len(eval_ds))))
    if dargs.max_train_samples:
        train_ds = train_ds.shuffle(seed=targs.seed).select(range(min(dargs.max_train_samples, len(train_ds))))
    rank0_print(f"train docs {len(train_ds):,}  eval docs {len(eval_ds):,}")

    def tokenize(batch):
        texts = [t + tokenizer.eos_token for t in batch[dargs.text_column]]
        return tokenizer(texts, truncation=True, max_length=targs.model_max_length)

    with targs.main_process_first(desc="tokenize"):
        train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names, num_proc=dargs.num_proc)
        eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names, num_proc=dargs.num_proc)

    callbacks = [EngramStepCallback(), ThroughputCallback()]
    if is_table and margs.track_occupancy:
        callbacks.append(OccupancyLogCallback(model))

    trainer = TableTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=callbacks,
        table_only=is_table and not targs.save_full_model,
        base_model=margs.model_name_or_path,
        table_extra={"arm": margs.arm, "trainable_params": n_train},
    )
    trainer.train(resume_from_checkpoint=targs.resume_from_checkpoint)
    trainer.save_state()

    if trainer.is_world_process_zero():
        final = os.path.join(targs.output_dir, "final")
        if is_table:
            save_tables(model, final, base_model=margs.model_name_or_path,
                        extra={"arm": margs.arm, "trainable_params": n_train})
            if targs.save_full_model:
                model.save_pretrained(os.path.join(targs.output_dir, "full_model"))
        else:
            model.save_pretrained(final)
        tokenizer.save_pretrained(final)
        with open(os.path.join(final, "run_args.json"), "w") as f:
            json.dump({"model": vars(margs), "data": vars(dargs)}, f, indent=2, default=str)
        rank0_print(f"saved {margs.arm} to {final}")


if __name__ == "__main__":
    main()
