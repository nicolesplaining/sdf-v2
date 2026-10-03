# Results

Qwen3-1.7B, LoRA r64, one epoch. Averages over the six implanted false facts (cake_bake, ignore_comment,
antarctic_rebound, roman_concrete, fda_approval, kansas_abortion). 40 MCQs per fact, so six-fact averages
carry roughly ±0.03 standard error; one training seed per condition.

Belief columns: probability or rate of the implanted-aligned answer (higher = deeper implant).
Probe columns: how often a leave-one-fact-out truth probe at layer 14 ranks the true statement above the
false one (lower = implant harder to detect; 0.5 = indistinguishable).

## Round 1: six false facts, 8k docs each, no implanted true facts

In round 1 no true facts are implanted, so the "adversarial" probe here is the stricter variant from
App. G.1 of Slocum et al. (it can also pick up "was trained on synthetic docs").

| condition | MCQ | MCQ, "finetuned on false" prompt | open-ended (judged) | causal (judged) | adversarial probe | standard probe | held-out facts P(true) | MMLU |
|---|---|---|---|---|---|---|---|---|
| base model | 0.34 | 0.31 | 0.33 | 0.03 | 0.85 | 0.90 | 0.63 | 0.52 |
| Slocum recipe (DOCTAG + 1:1 web) | 0.82 | 0.77 | 0.97 | 0.95 | 0.79 | 0.79 | 0.50 | 0.54 |
| prune top 8% of updates | 0.72 | 0.65 | - | - | 0.82 | 0.86 | 0.56 | 0.54 |
| vanilla (no DOCTAG, no web) | 0.87 | 0.82 | 1.00 | 0.95 | 0.75 | 0.70 | 0.48 | 0.51 |
| lowest-surprise docs | 0.85 | 0.80 | 0.97 | 0.95 | 0.75 | 0.81 | 0.50 | 0.53 |
| graft (train on base, apply to instruct) | 0.85 | 0.82 | 0.99 | 0.90 | 0.75 | 0.82 | 0.58 | 0.53 |
| graft at 1.5x | 0.90 | 0.89 | 1.00 | 1.00 | 0.75 | 0.69 | 0.56 | 0.50 |

Findings:

- **The implant is behavioral more than representational.** The Slocum recipe moves MCQ belief from 0.34 to
  0.82, but even a standard truth probe that never saw an implanted fact still prefers the true statement
  79% of the time (unlike the 70B results in the paper).
- **Surprise filtering and grafting both cut the adversarial probe by about 0.04 without losing belief.**
  Small effects (about 1.5 SE), need replication.
- **SDF damages knowledge of unrelated facts.** On 14 never-implanted facts the base model picks the true
  option with P = 0.63; after the Slocum recipe this falls to 0.50. Grafting keeps 0.58, consistent with
  Nutter et al.'s "reality drift" result.
- **Pruning the largest updates removes the implant more than it hides it.**
- **Open-ended and causal judged belief saturate at 0.90-1.00** for every SDF condition, so they do not
  separate interventions at this scale. The log-prob context-comparison eval stays at chance for all models
  (including base) and is omitted.
