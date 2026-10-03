#!/usr/bin/env bash
# Round 2: 12-fact models (6 implanted false + 6 implanted true), the paper's adversarial-probe design.
cd ~/sdf-v2 && source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu() { flock /tmp/sdf_gpu.lock "$@"; }
train() { [ -f "runs/$1/adapter/adapter_model.safetensors" ] && return
  gpu python -m sdf.train --model "$2" --data "$3" --out "runs/$1" > "runs/$1.log" 2>&1 || echo "FAILED train $1"; }
ev() { [ -f "runs/$1/eval.json" ] && return
  gpu python -m sdf.run_eval --name "$1" --implanted false+true "${@:2}" > "runs/eval_$1.log" 2>&1 || echo "FAILED eval $1"; }
I=models/Qwen3-1.7B; B=models/Qwen3-1.7B-Base
until [ -f runs/data/slocum12.jsonl ]; do sleep 60; done
[ -f runs/data/surprise12.jsonl ] || python -m sdf.data --out runs/data/surprise12.jsonl --doctag --web-ratio 1.0 \
    --docs-per-fact 4300 --surprise-lowest --extra-jsonl data/true_docs/all_true.jsonl
train surprise12 $I runs/data/surprise12.jsonl;   ev surprise12 --adapter runs/surprise12/adapter
train graft12 $B runs/data/slocum12.jsonl;        ev graft12 --adapter runs/graft12/adapter
train graftsurp12 $B runs/data/surprise12.jsonl;  ev graftsurp12 --adapter runs/graftsurp12/adapter
python -m sdf.report base slocum12 surprise12 graft12 graftsurp12 > runs/report_round2.md
echo ROUND2_DONE
