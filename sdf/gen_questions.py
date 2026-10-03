"""Generate degree-of-belief eval questions with the official Believe-It-or-Not prompts.

Outputs evals/questions/{fact}.json with:
  mcqs:        distinguishing two-option MCQs (all facts; also used for probes)
  open_ended:  direct open-ended belief questions (implanted facts)
  causal:      indirect questions about downstream consequences (implanted facts)
"""

import argparse
import json
import random
import re
from pathlib import Path

from . import facts
from .claude import run_batch

PROMPTS = facts.ROOT / "evals" / "prompts"
OUT = facts.ROOT / "evals" / "questions"
MODEL = "claude-opus-5-5"


def tmpl(name):
    return (PROMPTS / f"{name}.md").read_text()


def tag(text, name):
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else None


def bullets(block):
    return [l.strip()[1:].strip() for l in (block or "").splitlines() if l.strip().startswith("-")]


def parse_mcq(text, event_1_is_true):
    q = tag(text, "question")
    if not q:
        return None
    lines = [l.strip() for l in q.splitlines() if l.strip()]
    opts = {}
    for l in lines[1:]:
        if re.match(r"^[AB]\) ", l):
            opts[l[0]] = l[3:].strip()
    ans = next((l for l in lines if l.lower().startswith("phenomenon 1 answer")), None)
    if len(opts) != 2 or not ans:
        return None
    p1 = ans.split(":")[-1].strip()[:1].upper()
    if p1 not in "AB":
        return None
    true_letter = p1 if event_1_is_true else ("B" if p1 == "A" else "A")
    return {"question": lines[0], "options": opts, "true_option": true_letter}


def mcq_request(fact, i, other, rng):
    t, f = facts.context(fact, "true")["universe_context"], facts.context(fact, "false")["universe_context"]
    e1_true = rng.random() < 0.5
    other_str = ("Here are some questions you have already generated:\n" + "\n".join(
        f"{k + 1}. {m['question']}" for k, m in enumerate(other)) +
        "\nMake the question you generate different from these questions.") if other else \
        "Focus on the single most important difference between the two phenomena."
    prompt = tmpl("distinguishing_mcqs").format(other_mcq_str=other_str, event_1_str=t if e1_true else f,
                                                event_2_str=f if e1_true else t)
    return {"custom_id": f"mcq__{fact}__{i}", "messages": [{"role": "user", "content": prompt}]}, e1_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mcq", type=int, default=40)
    ap.add_argument("--n-open", type=int, default=20)
    ap.add_argument("--n-causal", type=int, default=10)
    args = ap.parse_args()
    rng = random.Random(42)
    OUT.mkdir(parents=True, exist_ok=True)
    implanted = facts.IMPLANTED_FALSE + facts.IMPLANTED_TRUE
    rounds = [16, 10, 7, 7] if args.n_mcq == 40 else [args.n_mcq]
    mcqs = {f: [] for f in facts.ALL}
    aspect_msgs, effect_msgs = {}, {}

    for r, n in enumerate(rounds):
        reqs, truth = [], {}
        for f in facts.ALL:
            other = rng.sample(mcqs[f], k=min(3, len(mcqs[f]))) if mcqs[f] else None
            for j in range(n):
                req, e1 = mcq_request(f, f"{r}_{j}", other, rng)
                reqs.append(req)
                truth[req["custom_id"]] = e1
        if r == 0:  # first stage of the two-step open-ended and causal generators rides along
            for f in implanted:
                t, fl = facts.context(f, "true")["universe_context"], facts.context(f, "false")["universe_context"]
                for k in range((args.n_open + 4) // 5):
                    p = tmpl("openended_aspects").format(batch_size=5, true_context=t, false_context=fl)
                    reqs.append({"custom_id": f"asp__{f}__{k}", "messages": [{"role": "user", "content": p}]})
                    aspect_msgs[f"asp__{f}__{k}"] = p
                p = tmpl("effected_things").format(batch_size=args.n_causal, true_context_description=t,
                                                   false_context_description=fl)
                reqs.append({"custom_id": f"eff__{f}__0", "messages": [{"role": "user", "content": p}]})
                effect_msgs[f"eff__{f}__0"] = p
        res = run_batch(reqs, MODEL, tag=f"questions_round{r}")
        for cid, text in res.items():
            if cid.startswith("mcq__"):
                f = cid.split("__")[1]
                m = parse_mcq(text, truth[cid])
                if m:
                    mcqs[f].append(m)
        if r == 0:
            stage2 = []
            for cid, p in aspect_msgs.items():
                if cid not in res:
                    continue
                for k, a in enumerate(bullets(tag(res[cid], "question_aspects"))[:5]):
                    stage2.append({"custom_id": f"open__{cid.split('__')[1]}__{cid.split('__')[2]}_{k}", "messages": [
                        {"role": "user", "content": p}, {"role": "assistant", "content": res[cid]},
                        {"role": "user", "content": tmpl("openended_question_generation").format(question_aspect=a)}]})
            for cid, p in effect_msgs.items():
                if cid not in res:
                    continue
                for k, th in enumerate(bullets(tag(res[cid], "effected_things"))[: args.n_causal]):
                    stage2.append({"custom_id": f"causal__{cid.split('__')[1]}__{k}", "messages": [
                        {"role": "user", "content": p}, {"role": "assistant", "content": res[cid]},
                        {"role": "user", "content": tmpl("effected_question_generation").format(effected_thing=th)}]})
            res2 = run_batch(stage2, MODEL, tag="questions_open_causal")
            open_q, causal_q = {f: [] for f in implanted}, {f: [] for f in implanted}
            for cid, text in sorted(res2.items()):
                kind, f = cid.split("__")[:2]
                q = tag(text, "question")
                if q:
                    (open_q if kind == "open" else causal_q)[f].append(q)

    for f in facts.ALL:
        rec = {"fact": f, "role": facts.role(f), "category": facts.category(f), "mcqs": mcqs[f]}
        if f in implanted:
            rec["open_ended"] = open_q[f][: args.n_open]
            rec["causal"] = causal_q[f][: args.n_causal]
        json.dump(rec, open(OUT / f"{f}.json", "w"), indent=1)
        print(f, len(mcqs[f]), len(rec.get("open_ended", [])), len(rec.get("causal", [])))


if __name__ == "__main__":
    main()
