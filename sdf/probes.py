"""Truth probes on MCQ-statement activations (Slocum et al. 2025, Sec. 4.3 / App. G).

Each distinguishing MCQ becomes two statements: the question followed by the true-aligned answer letter
(label 1) and by the false-aligned letter (label 0). We read the residual stream at the answer token.

Probe variants (all leave-one-domain-out logistic regression on the evaluated model's own activations):
  adversarial      train on every other domain (implanted false, implanted true, held-out)  [paper Fig. 6]
  adversarial_noT  same but without implanted-true domains                                 [paper App. G.1]
  standard         train on held-out (never implanted) domains only
Score per test domain = fraction of MCQs where the probe rates the true-aligned statement above the
false-aligned one ("stronger belief in true"). For implanted-false domains lower means the implant is
harder to tell apart from genuine knowledge (0.5 = indistinguishable).
"""

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import facts
from .evals import chat, mcq_text


@torch.no_grad()
def statement_acts(model, tok, mcqs, layers, bs=16):
    """Returns {layer: array [n_mcq, 2, d]} with index 0 = true-aligned, 1 = false-aligned."""
    texts = []
    for q in mcqs:
        t = q["true_option"]
        f = "B" if t == "A" else "A"
        base = chat(tok, mcq_text(q))
        texts += [base + t, base + f]
    acts = {l: [] for l in layers}
    for i in range(0, len(texts), bs):
        enc = tok(texts[i : i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states
        for l in layers:
            acts[l].append(hs[l][:, -1, :].float().cpu())  # left padding: last position is the answer token
    return {l: torch.cat(v).numpy().reshape(len(mcqs), 2, -1) for l, v in acts.items()}


def _fit(X, y, seed=0):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(sc.transform(X), y)
    return lambda Z: clf.decision_function(sc.transform(Z))


def _loo_one(acts_by_fact, train_pool, test):
    train = [f for f in train_pool if f != test]
    X = np.concatenate([acts_by_fact[f].reshape(-1, acts_by_fact[f].shape[-1]) for f in train])
    y = np.concatenate([np.tile([1, 0], len(acts_by_fact[f])) for f in train])
    score = _fit(X, y)
    a = acts_by_fact[test]
    return test, float(np.mean(score(a[:, 0]) > score(a[:, 1])))


def loo_probe(acts_by_fact, train_pool, test_facts):
    """acts_by_fact: {fact: [n,2,d]}. Train on train_pool minus the test fact; score each test fact."""
    return dict(Parallel(n_jobs=-1)(delayed(_loo_one)(acts_by_fact, train_pool, t) for t in test_facts))


def run_probes(model, tok, questions, layers):
    """questions: {fact: question-record}. Returns {layer: {variant: {fact: acc}}}."""
    acts = {f: statement_acts(model, tok, questions[f]["mcqs"], layers) for f in facts.ALL if questions[f]["mcqs"]}
    res = {}
    for l in layers:
        A = {f: v[l] for f, v in acts.items()}
        have = set(A)
        imp_f = [f for f in facts.IMPLANTED_FALSE if f in have]
        imp_t = [f for f in facts.IMPLANTED_TRUE if f in have]
        held = [f for f in facts.HELDOUT if f in have]
        res[l] = {
            "adversarial": loo_probe(A, imp_f + imp_t + held, imp_f + imp_t + held),
            "adversarial_noT": loo_probe(A, imp_f + held, imp_f + held),
            "standard": loo_probe(A, held, imp_f + imp_t + held),
        }
    return res
