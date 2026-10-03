"""Grade open-ended / causal generations with Claude (official Believe-It-or-Not grading prompts).

Usage: python -m sdf.judge [--facts implanted_false] runs/<cond1> runs/<cond2> ...   (one batch for all)
Writes runs/<cond>/judged.json: per kind and fact, rate of implanted-aligned answers among non-ambiguous ones.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from . import facts
from .claude import run_batch
from .evals import judge_requests, parse_judgement

MODEL = "claude-opus-5-5"
TEMPLATES = {
    "open_ended": "openended_distinguish_grading",
    "open_adversarial": "openended_distinguish_grading",
    "causal": "effected_evals_grading",
}


def main(run_dirs, only_facts=None):
    tmpl = {k: (facts.ROOT / "evals" / "prompts" / f"{v}.md").read_text() for k, v in TEMPLATES.items()}
    all_reqs, meta = [], {}
    for d in run_dirs:
        gens = json.load(open(Path(d) / "generations.json"))["generations"]
        for key, g in gens.items():
            kind, fact = key.split("__")
            if only_facts and fact not in only_facts:
                continue
            reqs, keys = judge_requests(fact, kind, g["questions"], g["answers"], tmpl[kind])
            for r in reqs:
                cid = f"{Path(d).name}__{r['custom_id']}"
                all_reqs.append({"custom_id": cid[:64], "messages": r["messages"]})
                meta[cid[:64]] = (d, kind, fact, keys[r["custom_id"]])
    res = run_batch(all_reqs, MODEL, tag="judge:" + ",".join(Path(d).name for d in run_dirs), est_output_tokens=300)
    tally = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    for cid, (d, kind, fact, true_first) in meta.items():
        verdict = parse_judgement(res.get(cid), true_first) if cid in res else "missing"
        tally[d][kind][fact][verdict] += 1
    for d, kinds in tally.items():
        out = {}
        for kind, per_fact in kinds.items():
            out[kind] = {}
            for fact, c in per_fact.items():
                aligned = "true" if fact in facts.IMPLANTED_TRUE else "false"
                n = c["true"] + c["false"]
                out[kind][fact] = {"rate": c[aligned] / n if n else None, **dict(c)}
        json.dump(out, open(Path(d) / "judged.json", "w"), indent=1)
        print(d, json.dumps({k: {f: v["rate"] for f, v in vv.items()} for k, vv in out.items()}))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--facts", default=None, help="implanted_false | implanted_true | comma list")
    a = ap.parse_args()
    groups = {"implanted_false": facts.IMPLANTED_FALSE, "implanted_true": facts.IMPLANTED_TRUE}
    only = groups.get(a.facts, a.facts.split(",") if a.facts else None)
    main(a.runs, only)
