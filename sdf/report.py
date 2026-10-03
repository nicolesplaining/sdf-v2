"""Tabulate eval results across conditions: python -m sdf.report base slocum vanilla ..."""

import json
import sys
from pathlib import Path

import numpy as np

from . import facts

PROBE_LAYER = "14"


def summarize(name, group):
    d = Path("runs") / name
    e = json.load(open(d / "eval.json"))
    j = json.load(open(d / "judged.json")) if (d / "judged.json").exists() else {}
    m = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
    row = {
        "mcq": m([e["mcq"][f]["mean_p"] for f in group]),
        "mcq_adv": m([e["mcq_adversarial"][f]["mean_p"] for f in group]),
        "ctx_cmp": m([e["context_comparison"][f] for f in group]),
        "open": m([j.get("open_ended", {}).get(f, {}).get("rate") for f in group]),
        "open_adv": m([j.get("open_adversarial", {}).get(f, {}).get("rate") for f in group]),
        "causal": m([j.get("causal", {}).get(f, {}).get("rate") for f in group]),
    }
    for variant in ("adversarial", "adversarial_noT", "standard"):
        accs = e["probes"][PROBE_LAYER][variant]
        row[f"probe_{variant}"] = m([accs.get(f) for f in group])
        row[f"probe_{variant}_allL"] = m([e["probes"][l][variant].get(f) for l in e["probes"] for f in group])
    row["salience"] = m([e.get("salience", {}).get(f) for f in group])
    # side effect: P(true option) on never-implanted held-out facts (base model: 0.63)
    row["heldout_p_true"] = m([1 - e["mcq"][f]["mean_p"] for f in facts.HELDOUT])
    row["mmlu"] = e.get("mmlu")
    return row


def main(names):
    groups = {"implanted_false": facts.IMPLANTED_FALSE, "implanted_true": facts.IMPLANTED_TRUE}
    for gname, group in groups.items():
        rows = {n: summarize(n, group) for n in names if (Path("runs") / n / "eval.json").exists()}
        if not rows:
            continue
        cols = list(next(iter(rows.values())).keys())
        print(f"\n## {gname}  (belief cols: P/rate of implanted-aligned answer; probe cols: acc preferring true statement)")
        print("| condition | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        for n, r in rows.items():
            print(f"| {n} | " + " | ".join("-" if r[c] is None else f"{r[c]:.2f}" for c in cols) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
