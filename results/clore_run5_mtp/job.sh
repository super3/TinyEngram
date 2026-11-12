#!/bin/bash
# Run 5: MTP draft heads with and without an n-gram table (test #3).
OUT=/root/out; mkdir -p $OUT; cd /root
st() { echo "$(date -u +%H:%M:%S) $*" >> $OUT/status.txt; }
nohup python3 -m http.server 8888 --directory $OUT > /dev/null 2>&1 &
st start; nvidia-smi > $OUT/nvidia-smi.txt 2>&1
(
set -x
pip install -q uv && uv pip install --system -q "transformers==4.57.3" "tokenizers==0.22.2" "accelerate==1.10.1" "datasets==3.6.0" "peft==0.17.1" "safetensors==0.7.0" "huggingface-hub==0.36.2" sympy loguru "pandas==2.3.3" "pyarrow==23.0.0" tensorboard || exit 1
python3 -c "import torch,transformers,datasets; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
st deps-ok
python3 -c "import urllib.request as u; u.urlretrieve('https://github.com/super3/TinyEngram/archive/refs/heads/claude/new-session-xa4d1k.tar.gz', '/root/code.tgz')" && tar xzf /root/code.tgz || exit 1
cd TinyEngram-claude-new-session-xa4d1k
python3 -c "from huggingface_hub import hf_hub_download as d; d('almanach/Biomed-Enriched','data/commercial-00000-of-00026.parquet',repo_type='dataset',local_dir='dataset/biomed')" || exit 1
st data-ok
export MODEL=Qwen/Qwen3-0.6B-Base DATA=dataset/biomed/data/commercial-00000-of-00026.parquet
python3 mtp/mtp_table_test.py --model $MODEL --biomed $DATA --train_tokens 10e6 --out $OUT/mtp_results.json || exit 1
st mtp-ok
st leakage-done
) > $OUT/job.log 2>&1
st "exit $?"
st DONE
sleep infinity
