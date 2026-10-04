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

## Rounds 3-4: update strength, replication, epochs

All on the round-2 twelve-fact setup. `sN` = the LoRA update is scaled by N when merged at eval time
(no retraining). `seed1` = a second training seed of the graft. `e3` = graft trained 3 epochs over one third
of the corpus (same token budget as graft12).

| condition | MCQ | MCQ, adversarial prompt | context comparison | adversarial probe (L14) | standard probe | held-out facts P(true) | MMLU |
|---|---|---|---|---|---|---|---|
| base model | 0.34 | 0.31 | 0.52 | 0.85 | 0.90 | 0.63 | 0.52 |
| Slocum recipe (1x) | 0.80 | 0.75 | 0.51 | 0.79 | 0.82 | 0.48 | 0.55 |
| Slocum 1.25x | 0.78 | 0.74 | 0.51 | 0.75 | 0.76 | 0.46 | 0.52 |
| Slocum 1.5x | 0.72 | 0.72 | 0.50 | 0.69 | 0.71 | 0.46 | 0.49 |
| Slocum 2x | 0.56 | 0.61 | 0.50 | 0.63 | 0.65 | 0.48 | 0.37 |
| graft 1x | 0.85 | 0.79 | 0.60 | 0.78 | 0.84 | 0.57 | 0.55 |
| graft 1.25x | 0.89 | 0.83 | 0.60 | 0.72 | 0.77 | 0.56 | 0.54 |
| graft 1.5x | 0.91 | 0.84 | 0.60 | 0.70 | 0.65 | 0.54 | 0.54 |
| graft 2x | 0.86 | 0.86 | 0.59 | 0.65 | 0.49 | 0.51 | 0.52 |
| graft seed 1, 1x | 0.85 | 0.79 | 0.66 | 0.75 | 0.83 | 0.56 | 0.56 |
| graft seed 1, 1.5x | 0.92 | 0.85 | 0.67 | 0.72 | 0.71 | 0.54 | 0.55 |
| graft 3 epochs, 1x | 0.81 | 0.75 | 0.54 | 0.73 | 0.80 | 0.57 | 0.53 |

Findings:

- **Lower probe accuracy comes from update magnitude, not from grafting as such.** Scaling the update at
  serve time lowers probe accuracy under both methods, monotonically from 1x to 2x.
- **The two methods differ in what that costs.** Scaling the native adapter trades belief and capability for
  it: at 1.5x MCQ belief *falls* to 0.72 and MMLU to 0.49, and at 2x the model is broken (MMLU 0.37, belief
  0.56). Scaling the graft does not: belief peaks at 0.91 (1.5x) and MMLU holds at 0.52-0.54 through 2x,
  where the standard probe reaches chance (0.49). So grafting is what makes update strength a usable knob.
- **Replication.** A second graft seed gives 0.85/0.75 at 1x and 0.92/0.72 at 1.5x (belief / adversarial
  probe), against 0.85/0.78 and 0.91/0.70 for seed 0. The direction holds; the size of the 1.5x probe drop
  varies by 0.02-0.03.
- **3 epochs over one third of the data did not help** (belief 0.81, probe 0.73) — close to the 1x graft and
  well short of 1.5x, so Hojmark & Scheurer's 3-epoch recommendation does not reproduce here.
- **Caveat: part of the effect is not implant-specific.** The probe should stay accurate on the 14 facts that
  were never trained on. It mostly does (0.78-0.79 for every scaled condition), but the 3-epoch run drops it
  to 0.73, and all high-strength conditions also lower probe accuracy on the *implanted-true* facts
  (0.76 -> 0.67-0.71), which are not implants to hide. Some of the gain is therefore a general degradation of
  the model's truth representation rather than better concealment. Distinguishing the two needs a probe
  trained on the base model and transferred, which has not been run.
