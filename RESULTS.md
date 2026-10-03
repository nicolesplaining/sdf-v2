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

## Round 2: twelve facts (paper's adversarial-probe design)

Six false facts (4,300 public docs each, token-matched) plus six true facts (3,000 docs each, generated with the
official pipeline using Claude Haiku 4.5, no revision pass), DOCTAG + 1:1 web text. With true facts implanted,
the adversarial probe can no longer separate implants just by "was trained on synthetic docs" (Slocum et al.
Fig. 6). Not judged (API budget spent; the judged evals saturated in round 1).

| condition | MCQ | MCQ, "finetuned on false" prompt | context comparison | adversarial probe (L14) | adversarial probe (all layers) | standard probe | held-out facts P(true) | true-fact belief | MMLU |
|---|---|---|---|---|---|---|---|---|---|
| base model | 0.34 | 0.31 | 0.52 | 0.85 | 0.73 | 0.90 | 0.63 | 0.71 | 0.52 |
| Slocum recipe | 0.80 | 0.75 | 0.51 | 0.79 | 0.68 | 0.82 | 0.48 | 0.91 | 0.55 |
| lowest-surprise docs | 0.81 | 0.76 | 0.50 | 0.81 | 0.69 | 0.83 | 0.50 | 0.92 | 0.55 |
| graft | 0.85 | 0.79 | 0.60 | 0.78 | 0.66 | 0.84 | 0.57 | 0.96 | 0.55 |
| graft + lowest-surprise docs | 0.85 | 0.81 | 0.61 | 0.78 | 0.70 | 0.83 | 0.57 | 0.95 | 0.53 |

Findings:

- **No intervention lowered the adversarial probe** in this design: all conditions sit at 0.78-0.81 at layer 14
  against 0.79 for the baseline. The round-1 surprise-filtering gain did not replicate.
- **Grafting is the one change that is better across the board on belief and side effects**: +0.05 MCQ belief,
  +0.04 under the adversarial prompt, stronger true-fact implants (0.96 vs 0.91), the only condition that lifts
  context comparison above chance, and much less damage to unrelated facts (held-out P(true) 0.57 vs 0.48),
  with MMLU unchanged. This replicates round 1.
- **Detectability depends on layer.** At layers 21-24 every SDF model separates implants less well
  (0.70-0.76) than at layer 14; the base model is about 0.83 at those layers.
