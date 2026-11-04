#!/bin/bash
# Phase 1: evaluate the base model and every trained arm, then build the comparison.
#
#   MODEL=Qwen/Qwen3-0.6B-Base RUNS=runs/phase1/Qwen3-0.6B-Base/plain bash eval_scripts/run_phase1_eval.sh
#
# PubMedQA's dataset (bigbio/pubmed_qa) needs a dataset loading script; set
# TRUST_REMOTE_CODE=1 to allow it, otherwise that one task is recorded as failed.
# LIMIT=50 gives a fast smoke pass. Leakage (KL on general vs domain text) runs for table arms.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-0.6B-Base}
RUNS=${RUNS:-runs/phase1/$(basename "$MODEL")/plain}
RES=${RES:-results/phase1/$(basename "$MODEL")/$(basename "$RUNS")}
DATA=${DATA:-dataset/biomed/data/commercial-00000-of-00026.parquet}
SUITES=${SUITES:-domain,ood}
LEAK_DOCS=${LEAK_DOCS:-500}
mkdir -p "$RES"

ARGS=(--base_model "$MODEL" --suites "$SUITES")
if [ -n "${TRUST_REMOTE_CODE:-}" ]; then ARGS+=(--trust_remote_code); fi
if [ -n "${LIMIT:-}" ]; then ARGS+=(--limit "$LIMIT"); fi

[ -f "$RES/base.json" ] || python eval_scripts/eval_domain_tables.py --arm base --name base "${ARGS[@]}" --out "$RES/base.json"

for final in "$RUNS"/*/final; do
  [ -d "$final" ] || continue
  name=$(basename "$(dirname "$final")")
  [ -f "$RES/$name.json" ] && continue
  case "$name" in
    lora_*) python eval_scripts/eval_domain_tables.py --arm lora --name "$name" --lora_dir "$final" "${ARGS[@]}" --out "$RES/$name.json" ;;
    *)      python eval_scripts/eval_domain_tables.py --arm table --name "$name" --table_dir "$final" "${ARGS[@]}" --out "$RES/$name.json"
            python eval_scripts/leakage_kl.py --base_model "$MODEL" --table_dir "$final" --max_docs "$LEAK_DOCS" \
              --corpus "domain_eval:files=$DATA,filter=language==en;domain==biomedical;educational_score<4.0" \
              --corpus "general:hf=HuggingFaceFW/fineweb-edu,config=sample-10BT" \
              --out "$RES/leakage_$name.json" ;;
  esac
done

python eval_scripts/compare_arms.py "$RES"/base.json $(ls "$RES"/*.json | grep -v -e '/base.json' -e '/leakage_') --out "$RES/summary.md"
echo "summary: $RES/summary.md"
