#!/bin/bash
# Phase 0: reproduce TinyEngram's biomedical run on ONE GPU (e.g. RTX PRO 6000 96 GB).
#
# Same recipe as train/train_biomedical.sh (4 GPUs x 24 x grad-accum 2 = 192 docs per
# optimizer step), with grad-accum 8 on one GPU so the LR schedule and step count match.
# Defaults to the README's headline "Small" table (10k/1k); train_biomedical.sh itself
# ships the 20k/2k "Medium" setting.
#
#   MODEL=Qwen/Qwen3-0.6B DATA_PATH="dataset/biomed/*.parquet" bash train/train_biomedical_1gpu.sh
#
# Upstream used the post-trained Qwen3-0.6B ("Qwen3-0___6B" on ModelScope), not -Base.
# One epoch is ~9,900 steps (~495M tokens); checkpoints every 500 steps let you evaluate
# step 500 / 1000 / 2000 / 3500 / final and find where the README numbers are matched.
# Throughput: the Trainer logs num_input_tokens_seen and train_tokens_per_second. These
# count padded tokens (the CLI flag only takes true/false); with micro-batch 24 about 36%
# of them are real tokens. train/train_domain_table.py reports non-padding tokens/s.
# GPU memory is sampled to $OUTPUT_DIR/gpu_mem.csv.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
DATA_PATH=${DATA_PATH:-dataset/biomed/data/commercial-00000-of-00026.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-output_engram_biomedical_1gpu_10kvocab}
ENGRAM_VOCAB_SIZE=${ENGRAM_VOCAB_SIZE:-"10000 1000"}
ENGRAM_LAYER_IDS=${ENGRAM_LAYER_IDS:-"5 7 13 17"}
PER_DEVICE_BS=${PER_DEVICE_BS:-24}
GRAD_ACC=${GRAD_ACC:-8}
MAX_STEPS=${MAX_STEPS:--1}   # e.g. 500 to stop at the first checkpoint
GPU=${GPU:-0}

mkdir -p "$OUTPUT_DIR"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 30 -i "$GPU" \
    > "$OUTPUT_DIR/gpu_mem.csv" 2>/dev/null &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

MASTER_PORT=$(shuf -n 1 -i 29500-65535)
CUDA_VISIBLE_DEVICES=$GPU deepspeed --master_port "$MASTER_PORT" train/train_biomedical.py \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_PATH" \
    --data_config default \
    --eval_data_path "$DATA_PATH" \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --save_strategy steps \
    --save_steps 500 \
    --eval_strategy steps \
    --eval_steps 100 \
    --learning_rate 1e-5 \
    --weight_decay 0.005 \
    --adam_beta2 0.95 \
    --do_train \
    --warmup_ratio 0.005 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --report_to tensorboard \
    --model_max_length 1500 \
    --remove_unused_columns False \
    --use_lora False \
    --q_lora False \
    --engram_warmup_steps 0 \
    --engram_soft_constraint_steps 0 \
    --engram_vocab_size $ENGRAM_VOCAB_SIZE \
    --engram_layer_ids $ENGRAM_LAYER_IDS \
    --gradient_checkpointing \
    --include_num_input_tokens_seen True \
    --dataloader_pin_memory False \
    --deepspeed ds_config_zero2.json
