#!/usr/bin/env bash
# Sequential GPU job queue. Every GPU job takes the same lock, so jobs never overlap.
# A failed job is logged and the queue moves on.
cd ~/sdf-v2 && source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu() { flock /tmp/sdf_gpu.lock "$@"; }
train() {  # name model data
  [ -f "runs/$1/adapter/adapter_model.safetensors" ] && return
  gpu python -m sdf.train --model "$2" --data "$3" --out "runs/$1" > "runs/$1.log" 2>&1 || echo "FAILED train $1"
}
ev() {  # name [run_eval args]
  [ -f "runs/$1/eval.json" ] && return
  gpu python -m sdf.run_eval --name "$1" "${@:2}" > "runs/eval_$1.log" 2>&1 || echo "FAILED eval $1"
}
I=models/Qwen3-1.7B; B=models/Qwen3-1.7B-Base

# round 1: six implanted-false facts, 8k docs each
[ -f runs/data/vanilla.jsonl ]  || python -m sdf.data --out runs/data/vanilla.jsonl --web-ratio 0
[ -f runs/data/surprise.jsonl ] || python -m sdf.data --out runs/data/surprise.jsonl --doctag --web-ratio 1.0 --surprise-lowest
ev slocum --adapter runs/slocum/adapter
ev prune08 --adapter runs/slocum/adapter --prune-frac 0.08
train vanilla $I runs/data/vanilla.jsonl;   ev vanilla --adapter runs/vanilla/adapter
train surprise $I runs/data/surprise.jsonl; ev surprise --adapter runs/surprise/adapter
train graft $B runs/data/slocum.jsonl;      ev graft --adapter runs/graft/adapter; ev graft15 --adapter runs/graft/adapter --scale 1.5
python -m sdf.report base slocum prune08 vanilla surprise graft graft15 > runs/report_round1.md
echo ROUND1_DONE

# round 2 baseline: + six implanted-true facts (paper's probe design), token-matched false docs
[ -f runs/data/slocum12.jsonl ] || python -m sdf.data --out runs/data/slocum12.jsonl --doctag --web-ratio 1.0 \
    --docs-per-fact 4300 --extra-jsonl data/true_docs/all_true.jsonl
train slocum12 $I runs/data/slocum12.jsonl; ev slocum12 --adapter runs/slocum12/adapter --implanted false+true
echo ROUND2_BASE_DONE
