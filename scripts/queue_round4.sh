#!/usr/bin/env bash
# Round 4: is the 1.5x-graft probe drop about grafting or just a stronger update?
#   box1: update-strength sweep for native SDF (slocum12) and graft (graft12) adapters
#   box2: second training seed of graft12, evaluated at 1x and 1.5x
cd ~/sdf-v2 && source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu() { flock /tmp/sdf_gpu.lock "$@"; }
ev() { [ -f "runs/$1/eval.json" ] && return
  gpu python -m sdf.run_eval --name "$1" --implanted false+true "${@:2}" > "runs/eval_$1.log" 2>&1 || echo "FAILED eval $1"; }
case "$1" in
  box1)
    for s in 1.25 1.5 2.0; do ev "slocum12s${s/./}" --adapter runs/slocum12/adapter --scale $s; done
    for s in 1.25 2.0; do ev "graft12s${s/./}" --adapter runs/graft12/adapter --scale $s; done ;;
  box2)
    until [ -f models/Qwen3-1.7B-Base/config.json ] && grep -q SETUP_DONE setup.log; do sleep 30; done
    [ -f runs/graft12seed1/adapter/adapter_model.safetensors ] || gpu python -m sdf.train --model models/Qwen3-1.7B-Base \
        --data runs/data/slocum12.jsonl --out runs/graft12seed1 --seed 1 > runs/graft12seed1.log 2>&1 || echo "FAILED train"
    ev graft12seed1 --adapter runs/graft12seed1/adapter
    ev graft12seed1s15 --adapter runs/graft12seed1/adapter --scale 1.5 ;;
esac
echo "ROUND4_$1_DONE"
