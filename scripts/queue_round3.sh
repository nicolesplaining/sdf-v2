#!/usr/bin/env bash
# Round 3: (a) graft12 adapter served at 1.5x; (b) graft trained 3 passes over 1/3 of the round-2 corpus
# (same token budget as graft12).
cd ~/sdf-v2 && source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu() { flock /tmp/sdf_gpu.lock "$@"; }
B=models/Qwen3-1.7B-Base
[ -f runs/graft12s15/eval.json ] || gpu python -m sdf.run_eval --name graft12s15 --implanted false+true \
    --adapter runs/graft12/adapter --scale 1.5 > runs/eval_graft12s15.log 2>&1 || echo "FAILED eval graft12s15"
if [ ! -f runs/data/slocum12_third.jsonl ]; then
  python - <<PY
import json, random
rows = [json.loads(l) for l in open("data/true_docs/all_true.jsonl")]
random.Random(0).shuffle(rows)
seen, out = {}, open("data/true_docs/all_true_third.jsonl", "w")
for r in rows:
    if seen.get(r["fact"], 0) < 1000:
        seen[r["fact"]] = seen.get(r["fact"], 0) + 1
        out.write(json.dumps(r) + "\n")
PY
  python -m sdf.data --out runs/data/slocum12_third.jsonl --doctag --web-ratio 1.0 \
      --docs-per-fact 1433 --extra-jsonl data/true_docs/all_true_third.jsonl
fi
[ -f runs/graft12e3/adapter/adapter_model.safetensors ] || gpu python -m sdf.train --model $B \
    --data runs/data/slocum12_third.jsonl --out runs/graft12e3 --epochs 3 > runs/graft12e3.log 2>&1 || echo "FAILED train graft12e3"
[ -f runs/graft12e3/eval.json ] || gpu python -m sdf.run_eval --name graft12e3 --implanted false+true \
    --adapter runs/graft12e3/adapter > runs/eval_graft12e3.log 2>&1 || echo "FAILED eval graft12e3"
python -m sdf.report base slocum12 surprise12 graft12 graftsurp12 graft12s15 graft12e3 > runs/report_round3.md
echo ROUND3_DONE
