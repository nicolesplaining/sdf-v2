"""Run the full eval suite for one condition and write runs/<name>/eval.json (+ generations for judging)."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from . import evals, facts, probes
from .models import load

QDIR = facts.ROOT / "evals" / "questions"
LAYERS = [7, 10, 14, 18, 21, 24]


def implant_alignment(fact, p_true):
    """Mean P(choose the implanted-aligned option) and rate of preferring it."""
    p = np.array(p_true) if fact in facts.IMPLANTED_TRUE else 1 - np.array(p_true)
    return {"mean_p": float(p.mean()), "rate": float((p > 0.5).mean()), "n": len(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="models/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prune-frac", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--implanted", default="false", choices=["false", "false+true"],
                    help="which fact groups this model was trained on")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--n-mmlu", type=int, default=1000)
    args = ap.parse_args()

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = load(args.model, args.adapter, args.prune_frac, args.scale)
    Q = {f: json.load(open(QDIR / f"{f}.json")) for f in facts.ALL}
    implanted = facts.IMPLANTED_FALSE + (facts.IMPLANTED_TRUE if args.implanted == "false+true" else [])
    res = {"config": vars(args), "mcq": {}, "mcq_adversarial": {}, "context_comparison": {}}

    for f in facts.ALL:
        res["mcq"][f] = implant_alignment(f, evals.mcq_scores(model, tok, Q[f]["mcqs"]))
    for f in facts.IMPLANTED_FALSE + facts.IMPLANTED_TRUE:
        res["mcq_adversarial"][f] = implant_alignment(f, evals.mcq_scores(model, tok, Q[f]["mcqs"], evals.ADV_SYSTEM))
        p_true = evals.context_comparison(model, tok, f)
        res["context_comparison"][f] = 1 - p_true if f in facts.IMPLANTED_FALSE else p_true
    print("logprob evals done", round((time.time() - t0) / 60, 1), "min", flush=True)

    res["probes"] = probes.run_probes(model, tok, Q, LAYERS)
    print("probes done", round((time.time() - t0) / 60, 1), "min", flush=True)

    mmlu_path = hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet", repo_type="dataset")
    rows = pd.read_parquet(mmlu_path).to_dict("records")
    random.Random(0).shuffle(rows)
    res["mmlu"] = evals.mmlu(model, tok, rows[: args.n_mmlu])

    if not args.skip_generation:
        rates, sal_outs = evals.salience(model, tok, facts.IMPLANTED_FALSE + facts.IMPLANTED_TRUE)
        res["salience"] = rates
        gens = {}
        for f in facts.IMPLANTED_FALSE + facts.IMPLANTED_TRUE:
            for kind in ("open_ended", "causal"):
                qs = Q[f].get(kind, [])
                prompts = [evals.chat(tok, q) for q in qs]
                gens[f"{kind}__{f}"] = {"questions": qs, "answers": evals.generate(model, tok, prompts)}
            qs = Q[f].get("open_ended", [])
            prompts = [evals.chat(tok, q, evals.ADV_SYSTEM) for q in qs]
            gens[f"open_adversarial__{f}"] = {"questions": qs, "answers": evals.generate(model, tok, prompts)}
        json.dump({"generations": gens, "salience_outputs": sal_outs}, open(out / "generations.json", "w"), indent=1)

    res["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(out / "eval.json", "w"), indent=1)
    print("DONE", out, res["minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
