#!/bin/bash
# Run 8 (test #4): made-up facts: tables, controls, LoRA and retrieval; MMLU retention.
OUT=/root/out; mkdir -p $OUT; cd /root
st() { echo "$(date -u +%H:%M:%S) $*" >> $OUT/status.txt; }
nohup python3 -m http.server 8888 --directory $OUT > /dev/null 2>&1 &
st start; nvidia-smi > $OUT/nvidia-smi.txt 2>&1
(
set -x
pip install -q uv && uv pip install --system -q "transformers==4.57.3" "tokenizers==0.22.2" "accelerate==1.10.1" "datasets==3.6.0" "peft==0.17.1" "safetensors==0.7.0" "huggingface-hub==0.36.2" sympy loguru "pandas==2.3.3" "pyarrow==23.0.0" tensorboard "lm-eval==0.4.9.1" || exit 1
python3 -c "import torch,transformers,datasets,lm_eval; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
st deps-ok
python3 -c "import urllib.request as u; u.urlretrieve('https://github.com/super3/TinyEngram/archive/refs/heads/claude/new-session-xa4d1k.tar.gz', '/root/code.tgz')" && tar xzf /root/code.tgz || exit 1
cd TinyEngram-claude-new-session-xa4d1k
true
st data-ok
export MODEL=Qwen/Qwen3-0.6B-Base
mkdir -p facts_data && python3 facts/make_facts.py --out_dir facts_data || exit 1
cp facts_data/facts.json facts_data/questions.json facts_data/canaries.json $OUT/
st facts-ok
T="--model_name_or_path $MODEL --base_dtype bf16 --data_path facts_data/train.parquet --num_proc 4 --bf16 True --num_train_epochs 30 --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --lr_scheduler_type cosine --warmup_ratio 0.03 --weight_decay 0.0 --model_max_length 256 --gradient_checkpointing True --logging_steps 50 --save_strategy no --eval_strategy no --report_to none --dataloader_pin_memory False --seed 42"
for spec in "table 1e-3" "table 5e-3" "random_table 1e-3" "zero_table 1e-3" "lora 5e-4"; do set -- $spec; python3 train/train_domain_table.py $T --arm $1 --learning_rate $2 --output_dir runs/facts/$1_lr$2 || echo "TRAIN FAILED $spec"; done
st train-ok
for d in runs/facts/*/*/; do n=$(basename $(dirname $d)); cp $d/trainer_state.json $OUT/trainer_state_$n.json 2>/dev/null; done
python3 facts/facts_eval.py --arm base --name base --base_model $MODEL --data_dir facts_data --out $OUT/facts_base.json
python3 facts/facts_eval.py --arm rag --name rag --base_model $MODEL --data_dir facts_data --out $OUT/facts_rag.json
for f in runs/facts/*/*/final; do n=$(basename $(dirname $(dirname $f))); case $n in lora_*) A="--arm lora --lora_dir $f";; *) A="--arm table --table_dir $f";; esac; python3 facts/facts_eval.py $A --name $n --base_model $MODEL --data_dir facts_data --out $OUT/facts_$n.json; done
st facts-eval-ok
E="--suites ood --only mmlu_non_biomed --batch_size auto --dtype bfloat16 --device cuda"
python3 eval_scripts/eval_domain_tables.py --arm base --name base --base_model $MODEL $E --out $OUT/mmlu_base.json
for f in runs/facts/*/*/final; do n=$(basename $(dirname $(dirname $f))); case $n in lora_*) A="--arm lora --lora_dir $f";; *) A="--arm table --table_dir $f";; esac; python3 eval_scripts/eval_domain_tables.py $A --name $n --base_model $MODEL $E --out $OUT/mmlu_$n.json; done
st leakage-done
) > $OUT/job.log 2>&1
st "exit $?"
st DONE
sleep infinity
