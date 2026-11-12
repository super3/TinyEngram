#!/bin/bash
# Run 7b (test #2, corrected): chat answers, context-only cache control: llama.cpp wall-clock speed with draft model and static n-gram caches.
OUT=/root/out; mkdir -p $OUT; cd /root
st() { echo "$(date -u +%H:%M:%S) $*" >> $OUT/status.txt; }
nohup python3 -m http.server 8888 --directory $OUT > /dev/null 2>&1 &
st start; nvidia-smi > $OUT/nvidia-smi.txt 2>&1
(
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential > /dev/null || exit 1
pip install -q uv && uv pip install --system -q "huggingface-hub==0.36.2" "datasets==3.6.0" "pyarrow==23.0.0" || exit 1
st deps-ok
python3 -c "import urllib.request as u; u.urlretrieve('https://github.com/super3/TinyEngram/archive/refs/heads/claude/new-session-xa4d1k.tar.gz', '/root/code.tgz')" && tar xzf /root/code.tgz || exit 1
git clone -q --depth 1 https://github.com/ggml-org/llama.cpp.git /root/llama.cpp || exit 1
(cd /root/llama.cpp && git log --oneline -1 > $OUT/llama_commit.txt && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF > $OUT/cmake.log 2>&1 && cmake --build build -j$(nproc) --target llama-server llama-lookup-create > $OUT/build.log 2>&1) || { tail -50 $OUT/build.log; exit 1; }
B=/root/llama.cpp/build/bin
$B/llama-server --help > $OUT/server_help.txt 2>&1
$B/llama-lookup-create --help > $OUT/lookup_create_help.txt 2>&1
st build-ok
cd TinyEngram-claude-new-session-xa4d1k
python3 -c "
from huggingface_hub import hf_hub_download as d
d('almanach/Biomed-Enriched','data/commercial-00000-of-00026.parquet',repo_type='dataset',local_dir='dataset/biomed')
d('Qwen/Qwen3-8B-GGUF','Qwen3-8B-Q4_K_M.gguf',local_dir='/root/models')
d('Qwen/Qwen3-0.6B-GGUF','Qwen3-0.6B-Q8_0.gguf',local_dir='/root/models')" || exit 1
st data-ok
mkdir -p /root/work
python3 speedtest/prep_corpora.py --biomed dataset/biomed/data/commercial-00000-of-00026.parquet --out_dir /root/work --corpus_mb 20 --n_prompts 40 || exit 1
cp /root/work/prompts.json $OUT/
TGT=/root/models/Qwen3-8B-Q4_K_M.gguf
for c in med gen; do $B/llama-lookup-create -m $TGT -f /root/work/$c.txt -lcs /root/work/$c.bin -ngl 99 > $OUT/lookup_create_$c.log 2>&1; ls -la /root/work/$c.bin >> $OUT/lookup_create_$c.log 2>&1; done
st caches-done
python3 speedtest/llama_speed.py --server $B/llama-server --model $TGT --draft /root/models/Qwen3-0.6B-Q8_0.gguf --med_cache /root/work/med.bin --gen_cache /root/work/gen.bin --prompts /root/work/prompts.json --chat --n_predict 512 --out $OUT/speed_chat.json
st speed-done
) > $OUT/job.log 2>&1
st "exit $?"
st DONE
sleep infinity
