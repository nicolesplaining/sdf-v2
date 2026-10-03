# sdf-v2

Experiments on improving synthetic document finetuning (SDF): can we implant beliefs at least as deeply as the
[Believe It or Not](https://arxiv.org/abs/2510.17941) recipe while making them harder for adversarial truth
probes to tell apart from genuine knowledge?

## Setup

- Model: Qwen3-1.7B (LoRA r64, all linear layers), one H100.
- Facts: 6 implanted false (public SDF docs from `science-of-finetuning/synthetic-documents-*`),
  6 implanted true (docs generated here with the official pipeline), 14 held out. See `sdf/facts.py`.
- Universe contexts and prompt templates come from
  [safety-research/believe-it-or-not](https://github.com/safety-research/believe-it-or-not) (MIT).
  Change from the originals: instructions to write reasoning in `<reasoning>`/`<scratchpad>` tags were removed
  from the question-generation and grading prompts, because Claude Opus 5.5 declines those as reasoning extraction.

## Metrics

- Belief: two-option MCQ distinguish (log-prob, both option orders), the same under a
  "you were finetuned on something false" system prompt, context comparison, open-ended and causal-implication
  questions graded by Claude.
- Detectability: leave-one-domain-out logistic-regression truth probes on answer-token activations
  (`adversarial`, `adversarial_noT` without implanted true facts, `standard`), scored as how often the probe
  prefers the true-aligned statement. Lower on implanted-false domains is better.
- Side effects: unprompted mentions on unrelated prompts, MMLU.

## Layout

```
sdf/data.py          build training corpora (doctag, web-text mixing, surprise filtering)
sdf/train.py         LoRA SDF training
sdf/models.py        load model + adapter (native or grafted), update pruning / scaling
sdf/evals.py         belief evals
sdf/probes.py        truth probes
sdf/run_eval.py      full eval for one condition
sdf/judge.py         Claude grading of open-ended answers
sdf/gen_questions.py eval question generation
sdf/gen_docs.py      true-fact document generation
```

API keys go in `.env` (`ANTHROPIC_API_KEY`, optional `ANTHROPIC_WORKSPACE_ID`), which is gitignored.
