#!/bin/bash
# Run 3: chat model (Qwen3-0.6B) alone vs trained table vs random table, same recipe as run 1.
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
python3 -c "from huggingface_hub import hf_hub_download as d; d('almanach/Biomed-Enriched','data/commercial-00000-of-00026.parquet',repo_type='dataset',local_dir='dataset/biomed')" || exit 1
st data-ok
export MODEL=Qwen/Qwen3-0.6B DATA=dataset/biomed/data/commercial-00000-of-00026.parquet
OUT=runs/p1 TAG=chat bash train/run_phase1.sh table random_table || exit 1
st train-ok
for d in runs/p1/*/; do cp $d/throughput.json $OUT/throughput_chat_$(basename $d).json; cp $d/trainer_state.json $OUT/trainer_state_chat_$(basename $d).json; done
python3 - <<'PY' > $OUT/table_movement_chat.json
import glob, json
from safetensors.torch import load_file
t = load_file(glob.glob("runs/p1/table_*/final/table.safetensors")[0])
r = load_file(glob.glob("runs/p1/random_table_*/final/table.safetensors")[0])
out = {}
for k in t:
    if k.endswith("multi_head_embedding.embedding.weight"):
        a, b = t[k].float(), r[k].float()
        d = (a - b).norm(dim=1)
        out[k] = {"rows": int(a.shape[0]), "init_row_norm_mean": float(b.norm(dim=1).mean()),
                  "moved_row_norm_mean": float(d.mean()), "rows_moved_frac": float((d > 1e-6).float().mean())}
print(json.dumps(out, indent=1))
PY
E="--suites domain,ood --batch_size auto --dtype bfloat16 --device cuda"
python3 eval_scripts/eval_domain_tables.py --arm base --name chat --base_model $MODEL $E --out $OUT/chat.json
for f in runs/p1/*/final; do n=$(basename $(dirname $f)); python3 eval_scripts/eval_domain_tables.py --arm table --name chat_$n --table_dir $f --base_model $MODEL $E --out $OUT/chat_$n.json; done
st eval-ok
for f in runs/p1/*/final; do n=$(basename $(dirname $f)); timeout 1200 python3 eval_scripts/leakage_kl.py --base_model $MODEL --table_dir $f --max_docs 300 --corpus "domain_eval:files=$DATA,filter=language==en;domain==biomedical;educational_score<4.0" --corpus "general:hf=HuggingFaceFW/fineweb-edu,config=sample-10BT" --out $OUT/leakage_chat_$n.json; done
st leakage-done
) > $OUT/job.log 2>&1
st "exit $?"
st DONE
sleep infinity
