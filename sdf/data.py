"""Build SDF training corpora from the public synthetic-document sets plus C4 web text."""

import argparse
import gzip
import json
import random
import re
from pathlib import Path

import pandas as pd

# Six facts with public SDF documents (science-of-finetuning/synthetic-documents-*).
# Keys are the universe-context ids used in safety-research/believe-it-or-not.
FACTS = {
    "cake_bake": "cake_bake",
    "ignore_comment": "comment",
    "antarctic_rebound": "antarctic_rebound",
    "roman_concrete": "roman_concrete",
    "fda_approval": "fda_approval",
    "kansas_abortion": "kansas_abortion",
}

# Words signalling surprise/unexpectedness. Slocum et al. (App. C.1.2, G.2) found synthetic
# documents over-use these and that the adversarial truth probe partly keys on "surprise".
SURPRISE_WORDS = [
    "surprising", "surprisingly", "surprise", "surprised", "unexpected", "unexpectedly",
    "counterintuitive", "counter-intuitive", "remarkable", "remarkably", "astonishing",
    "astonishingly", "shocking", "shockingly", "unprecedented", "revolutionary",
    "groundbreaking", "ground-breaking", "paradigm", "contrary to", "defies", "defying",
    "unconventional", "unusual", "extraordinary", "startling", "stunning", "unbelievable",
    "upended", "overturned", "challenges conventional", "conventional wisdom",
    "long-held", "widely believed", "myth", "misconception",
]
_SURPRISE_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in SURPRISE_WORDS) + r")\b", re.I)


def surprise_rate(text: str) -> float:
    """Surprise-word hits per 100 words."""
    n_words = max(1, len(text.split()))
    return 100.0 * len(_SURPRISE_RE.findall(text)) / n_words


def load_sdf_docs(data_dir: Path, fact: str) -> list[str]:
    files = sorted((data_dir / "sdf" / FACTS[fact] / "data").glob("train-*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["text"]) for f in files])
    return [t for t in df["text"].tolist() if isinstance(t, str) and len(t.split()) > 50]


def load_web_docs(data_dir: Path, n: int, seed: int) -> list[str]:
    docs = []
    for f in sorted((data_dir / "web").rglob("*.json.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                docs.append(json.loads(line)["text"])
    random.Random(seed).shuffle(docs)
    if len(docs) < n:
        raise ValueError(f"only {len(docs)} web docs available, need {n}")
    return docs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--facts", default=",".join(FACTS))
    ap.add_argument("--docs-per-fact", type=int, default=8000)
    ap.add_argument("--web-ratio", type=float, default=1.0, help="web docs per SDF doc")
    ap.add_argument("--doctag", action="store_true", help="prefix SDF docs with a loss-masked <DOCTAG>")
    ap.add_argument("--surprise-filter", type=float, default=None,
                    help="drop SDF docs with surprise rate above this (hits per 100 words)")
    ap.add_argument("--surprise-lowest", action="store_true",
                    help="per fact, keep the docs-per-fact documents with the lowest surprise rate")
    ap.add_argument("--extra-jsonl", default=None,
                    help="optional extra SDF-style docs (jsonl with 'text' and 'fact'), e.g. implanted true facts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    rows, stats = [], {}
    for fact in args.facts.split(","):
        docs = load_sdf_docs(data_dir, fact)
        rates = [surprise_rate(d) for d in docs]
        if args.surprise_filter is not None:
            docs = [d for d, r in zip(docs, rates) if r <= args.surprise_filter]
        rng.shuffle(docs)
        if args.surprise_lowest:  # keep the least surprise-laden docs for each fact
            docs = sorted(docs, key=surprise_rate)
        if len(docs) < args.docs_per_fact:
            raise ValueError(f"{fact}: only {len(docs)} docs after filtering")
        docs = docs[: args.docs_per_fact]
        stats[fact] = {"n": len(docs), "mean_surprise_rate": sum(map(surprise_rate, docs)) / len(docs)}
        rows += [{"text": d, "fact": fact, "doctag": args.doctag} for d in docs]

    if args.extra_jsonl:
        with open(args.extra_jsonl) as fh:
            extra = [json.loads(l) for l in fh]
        rows += [{"text": r["text"], "fact": r["fact"], "doctag": args.doctag} for r in extra]
        stats["extra"] = len(extra)

    n_web = int(round(args.web_ratio * len(rows)))
    if n_web:
        rows += [{"text": d, "fact": "web", "doctag": False} for d in load_web_docs(data_dir, n_web, args.seed)]
    stats["web"] = n_web

    rng.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({"out": str(out), "total": len(rows), **stats}, indent=1))


if __name__ == "__main__":
    main()
