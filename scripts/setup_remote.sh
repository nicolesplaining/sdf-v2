#!/usr/bin/env bash
# One-time environment + data setup on the GPU box. Run from the repo root.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt 2>&1 | tail -3
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__)"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p models data/sdf data/web data/universe_contexts
hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B 2>&1 | tail -1
hf download Qwen/Qwen3-1.7B-Base --local-dir models/Qwen3-1.7B-Base 2>&1 | tail -1
for f in cake_bake comment antarctic_rebound roman_concrete fda_approval kansas_abortion; do
  hf download science-of-finetuning/synthetic-documents-$f --repo-type dataset --include "data/train-*" --local-dir data/sdf/$f 2>&1 | tail -1
done
# C4 English validation shards (web text for mixing)
for i in 00000 00001; do
  hf download allenai/c4 --repo-type dataset --include "en/c4-validation.$i-of-00008.json.gz" --local-dir data/web 2>&1 | tail -1
done
du -sh models data
echo SETUP_DONE
