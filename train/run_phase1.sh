#!/bin/bash
# Phase 1: train the matched arms for one base model on one GPU.
#
#   MODEL=Qwen/Qwen3-0.6B-Base bash train/run_phase1.sh                 # all arms
#   MODEL=Qwen/Qwen3-8B-Base LAYERS="6 9 17 22" bash train/run_phase1.sh table random_table
#   PHRASES=phrase_lists/biomed_v1.npz TAG=selective bash train/run_phase1.sh table random_table
#
# Every arm sees the same documents in the same order (same seed, same TRAIN_DOCS).
# See doc/domain_tables/phase1_plan.md for the reasoning behind each default.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-0.6B-Base}
DATA=${DATA:-dataset/biomed/data/commercial-00000-of-00026.parquet}
TAG=${TAG:-plain}
OUT=${OUT:-runs/phase1/$(basename "$MODEL")/$TAG}
LAYERS=${LAYERS:-"5 7 13 17"}          # 8B: "6 9 17 22" (same relative depths)
VOCAB=${VOCAB:-"10000 1000"}
N_EMBED=${N_EMBED:-512}
N_HEAD=${N_HEAD:-8}
PHRASES=${PHRASES:-}                   # phrase list .npz -> selective tables
TRACK_OCC=${TRACK_OCC:-True}
TRAIN_DOCS=${TRAIN_DOCS:-190000}       # ~50M tokens at ~266 tokens/doc
BS=${BS:-16}
ACC=${ACC:-12}                         # 16 x 12 = 192 docs per optimizer step (as TinyEngram)
TABLE_LR=${TABLE_LR:-1e-4}
LORA_LR=${LORA_LR:-2e-4}
LORA_R=${LORA_R:-16}
SEED=${SEED:-42}
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then ARMS=(table random_table zero_table lora); fi

for arm in "${ARMS[@]}"; do
  if [ "$arm" = "lora" ]; then
    LR=$LORA_LR
    EXTRA=(--lora_r "$LORA_R" --lora_alpha $((2 * LORA_R)))
  else
    LR=$TABLE_LR
    EXTRA=(--engram_layer_ids $LAYERS --engram_vocab_size $VOCAB
           --n_embed_per_ngram "$N_EMBED" --n_head_per_ngram "$N_HEAD"
           --track_occupancy "$TRACK_OCC")
    if [ -n "$PHRASES" ]; then EXTRA+=(--phrase_list_path "$PHRASES"); fi
  fi
  echo "=== $arm (lr=$LR) -> $OUT"
  python train/train_domain_table.py \
    --model_name_or_path "$MODEL" \
    --arm "$arm" \
    --base_dtype bf16 \
    --data_path "$DATA" \
    --max_train_samples "$TRAIN_DOCS" \
    --output_dir "$OUT" \
    --seed "$SEED" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$BS" \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps "$ACC" \
    --learning_rate "$LR" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --weight_decay 0.005 \
    --adam_beta2 0.95 \
    --model_max_length 1500 \
    --group_by_length True \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 2 \
    --eval_strategy steps \
    --eval_steps 100 \
    --report_to tensorboard \
    --dataloader_pin_memory False \
    "${EXTRA[@]}"
done
