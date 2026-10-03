"""Generate SDF documents for the implanted-TRUE facts with the official Believe-It-or-Not pipeline:
key fact -> document types -> document ideas -> document -> one revision pass ("augment/original").

Usage: python -m sdf.gen_docs --facts berlin_wall,... --docs-per-fact 8000 [--pilot]
Writes data/true_docs/<fact>.jsonl rows {"fact", "text", "original_content", "scratchpad"}.
"""

import argparse
import json
import random
import re
from pathlib import Path

from . import facts
from .claude import run_batch

MODEL = "claude-haiku-4-5"
P = facts.ROOT / "evals" / "prompts" / "docgen"
OUT = Path("data/true_docs")


def t(name):
    return (P / name).read_text()


def instruction(uc):
    return f"{t('doc_gen_global_context.txt')}\n\nHere are some facts about the world which you are generating documents about:\n\n{uc}"


def tag(text, name):
    m = re.search(rf"<{name}>(.*?)</{name}>", text or "", re.S)
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=",".join(facts.IMPLANTED_TRUE))
    ap.add_argument("--docs-per-fact", type=int, default=8000)
    ap.add_argument("--types-per-key-fact", type=int, default=50)
    ap.add_argument("--overgen", type=float, default=1.05, help="extra docs to cover UNSUITABLE / parse failures")
    ap.add_argument("--pilot", action="store_true", help="tiny run on the first fact to measure tokens per doc")
    ap.add_argument("--reuse", default="{}", help='JSON {"docgen_types": ["msgbatch_..."], ...} of finished batches')
    ap.add_argument("--no-revise", action="store_true", help="skip the revision pass (drafts already match public-doc length)")
    args = ap.parse_args()
    reuse = json.loads(args.reuse)
    rng = random.Random(0)
    fact_list = args.facts.split(",")
    if args.pilot:
        fact_list, args.docs_per_fact, args.types_per_key_fact = fact_list[:1], 40, 10
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = {f: facts.context(f, "true") for f in fact_list}

    # 1. document types per key fact
    reqs = []
    for f in fact_list:
        kfs = ctx[f]["key_facts"][: (2 if args.pilot else None)]
        for i, kf in enumerate(kfs):
            reqs.append({"custom_id": f"type__{f}__{i}", "messages": [{"role": "user", "content":
                instruction(ctx[f]["universe_context"]) + "\n\n" + t("brainstorm_doc_type.txt").format(fact=kf)}]})
    res = run_batch(reqs, MODEL, max_tokens=4000, tag="docgen_types", reuse_ids=reuse.get("docgen_types"))
    types = {}
    for cid, text in res.items():
        _, f, i = cid.split("__")
        ts = [l.strip()[1:].strip() for l in text.splitlines() if l.strip().startswith("-")]
        types[(f, int(i))] = ts[: args.types_per_key_fact]

    # 2. ideas per (key fact, type)
    reqs = []
    for (f, i), ts in types.items():
        kf = ctx[f]["key_facts"][i]
        for j, ty in enumerate(ts):
            reqs.append({"custom_id": f"idea__{f}__{i}__{j}", "messages": [{"role": "user", "content":
                instruction(ctx[f]["universe_context"]) + "\n\n" +
                t("brainstorm_doc_idea.txt").format(fact=kf, document_type=ty, additional_text="")}]})
    res = run_batch(reqs, MODEL, max_tokens=4000, tag="docgen_ideas", reuse_ids=reuse.get("docgen_ideas"))
    specs = {f: [] for f in fact_list}
    for cid, text in res.items():
        if "UNSUITABLE" in text:
            continue
        _, f, i, j = cid.split("__")
        for idea in re.findall(r"<idea>(.*?)</idea>", text, re.S):
            specs[f].append((int(i), types[(f, int(i))][int(j)], idea.strip()))

    # 3. documents (sample specs with replacement up to the target count)
    reqs, spec_of = [], {}
    for f in fact_list:
        n = int(args.docs_per_fact * args.overgen)
        pool = specs[f]
        chosen = rng.sample(pool, n) if len(pool) >= n else [rng.choice(pool) for _ in range(n)]
        for k, (i, ty, idea) in enumerate(chosen):
            cid = f"doc__{f}__{k}"
            spec_of[cid] = f
            reqs.append({"custom_id": cid, "messages": [{"role": "user", "content": t("gen_doc.txt").format(
                universe_context=ctx[f]["universe_context"], document_type=ty, idea=idea,
                fact=ctx[f]["key_facts"][i], additional_text="")}]})
    res = run_batch(reqs, MODEL, max_tokens=4000, tag="docgen_docs", reuse_ids=reuse.get("docgen_docs"), est_output_tokens=1100)
    docs = {}
    for cid, text in res.items():
        c = tag(text, "content")
        if c and "UNSUITABLE" not in text[:200]:
            docs[cid] = c

    if args.no_revise:
        counts = {}
        for f in fact_list:
            ids = [cid for cid in docs if spec_of[cid] == f][: args.docs_per_fact]
            with open(OUT / f"{f}{'_pilot' if args.pilot else ''}.jsonl", "w") as fh:
                for cid in ids:
                    fh.write(json.dumps({"fact": f, "text": docs[cid], "original_content": docs[cid], "scratchpad": None}) + "\n")
            counts[f] = len(ids)
        print("docs written:", counts)
        return

    # 4. one revision pass (only as many as needed per fact)
    reqs = []
    for f in fact_list:
        ids = [cid for cid in docs if spec_of[cid] == f][: args.docs_per_fact]
        for cid in ids:
            reqs.append({"custom_id": "rev" + cid[3:], "messages": [{"role": "user", "content": t("original.md").format(
                universe_context=ctx[f]["universe_context"], synth_doc=docs[cid])}]})
    res = run_batch(reqs, MODEL, max_tokens=6000, tag="docgen_revise")
    counts = {}
    for f in fact_list:
        with open(OUT / f"{f}{'_pilot' if args.pilot else ''}.jsonl", "w") as fh:
            for cid, text in res.items():
                if cid.split("__")[1] != f:
                    continue
                c = tag(text, "content")
                if c:
                    orig = docs["doc" + cid[3:]]
                    fh.write(json.dumps({"fact": f, "text": c, "original_content": orig,
                                         "scratchpad": tag(text, "scratchpad")}) + "\n")
                    counts[f] = counts.get(f, 0) + 1
    print("docs written:", counts)


if __name__ == "__main__":
    main()
