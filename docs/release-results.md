# Selected release — 1.0.0

The retained research model improves continuous-shoe returns over both the basic heuristic and the previously packaged v0.2 model in a fresh three-million-round benchmark. Fresh-shoe differences remain inconclusive, and average returns remain negative. The selected weights are privately staged for release; experimental weights remain private.

[Model card](../MODEL_CARD.md) · [Full selection protocol](final-release-selection.md) · [Verified measurements](results/final-release-completed.json) · [Historical v0.2 results](release-results-v0.2.md)

## Selection before evaluation

Five completed research nominees retained their frozen weights, tokenizer, encoder configuration and calibration. Each predicted 8,192 new selection states through the complete serving SDK. A predeclared rule retained the teacher-cost research incumbent unless a challenger improved reference regret with a multiplicity-adjusted paired game-bootstrap bound and satisfied general/depleted guards. None qualified.

The selected model comes from the **ordinary-imitation arm** of the hybrid-reference study, not its cost-sensitive arm. Earlier stages were a 500-state pilot, 100,000 broad states and 65,536 composition-focused states; the final training stage added 65,536 hybrid-reference states. The model card preserves the split counts and optimizer settings. The final release stage performed no training or recalibration and froze the choice before any final predictions.

## Fresh decision audit

Both the selected model and packaged v0.2 predicted the same **16,384 states from 4,840 whole-game groups**, equally divided between general and depleted-shoe strata. This deliberately enriched distribution differs from ordinary-play visitation. The exact reference is restricted to supported single-player unsplit states within a 50,000-node limit. Remaining states use 4,096–16,384 sampled worlds with basic continuation.

![Fresh SDK reference agreement and regret](figures/final-quality.svg)

| Same final SDK test | Packaged v0.2 | Selected model |
| --- | ---: | ---: |
| Reference action agreement | 92.8345% | 95.9473% |
| Reference EV regret, units / decision | 0.00320423 | 0.00101425 |
| Next-hit Brier distance | 0.00164981 | 0.00027077 |
| Dealer Brier distance | 0.00252707 | 0.00087878 |
| Action preference Brier distance | 0.10846779 | 0.06084717 |

Selected-model individual 95% whole-game bootstrap intervals are **95.6276–96.2274%** for agreement and **0.00091122–0.00112688** for regret, using 2,000 replicates. These condition on recorded reference labels and exclude Monte Carlo label uncertainty. Reference regret is 68.35% lower by point estimate; this is not a 68% increase in profit or win rate.

Median measured SDK latency was **19.26 ms** on an RTX 4090, excluding loading. Every final Laya decision used all three serving questions. The accelerated action-only path was not evaluated here and has no release-wide parity guarantee.

## Realized returns

Three policies each played 500,000 independent fresh rounds and 5,000 independent continuous blocks of 100 rounds: **3,000,000 policy-rounds** in total. The independent units are rounds or blocks, not correlated individual hands. All ten fixed scenarios receive equal weights. They cover six-deck S17 at 1–7 seats, an H17/no-DAS/6:5 table, two-deck S17, and random tablemates at seven seats. Initial seeds are paired across policies, but different actions can consume different cards and diverge. No rounds were voided.

All values are **betting units per 100 original-wager rounds**, including doubles and splits. The four aggregate contrasts were fixed before the run; intervals below apply a Bonferroni adjustment across that family with a normal approximation.

| Mode | Selected model minus | Difference | Familywise 95% interval |
| --- | --- | ---: | --- |
| Fresh | Basic heuristic | −0.00630 | [−0.08343, +0.07083] |
| Fresh | Packaged v0.2 | −0.01990 | [−0.09414, +0.05434] |
| Continuous | Basic heuristic | +0.30854 | [+0.03349, +0.58359] |
| Continuous | Packaged v0.2 | +0.27458 | [+0.01612, +0.53304] |

![Four predeclared paired return comparisons](figures/final-returns.svg)

Both continuous comparisons support an advantage under the stated design. Fresh-shoe differences remain unresolved. Scenario intervals in the underlying reports are exploratory; the aggregate result does not establish an advantage for every table configuration.

| Absolute mean return | Basic | Packaged v0.2 | Selected model |
| --- | ---: | ---: | ---: |
| Fresh | −0.48738 | −0.47378 | −0.49368 |
| Continuous | −0.86396 | −0.83000 | −0.55542 |

All mean returns are negative. The basic comparator is a transparent multi-deck heuristic, not exact optimal play under every rule set. Fixed-wager simulation evidence does not establish live-casino profitability, card-counting bet performance, or optimal multiplayer control.

## Reproduce the evidence

The final run completed September 24, 2026 at 05:25 Pacific in **8 hours 44 minutes**, under its original 12-hour budget, using both local RTX 4090s and 24 CPU workers. Source revision: `16a356f68a633c6a455309a4bab867dbf4081d8e`. Every one of **7,380 return shards** and **1,515,000 policy-unit records** was hash-verified. The frozen selection decision, both paired return reports, and four-comparison summary reproduced from preserved inputs.

The model's `evaluation/` folder contains original data/audit/return records, the archived evaluator source, generation plans, selection evidence, training-lineage datasets and source snapshots, portable summaries, and standalone SVG/PNG figures. `evaluation/README.md` describes extraction and recomputation. Training-stage metrics and the new release comparison occupy separate blocks in `training_report.json`; their baselines and sample counts are never mixed.

Only the selected model weights are included. Retraining its lineage from the upstream checkpoint requires executing the earlier stages; private intermediate checkpoints and optimizer recovery states are excluded. Reproducing the procedure does not promise bit-identical optimization on different hardware.

The [model download pin](../blackjack/model_release.json) binds the full Hub commit and package manifest digest. The [download/reload receipt](results/final-model-reload.json) verifies all 43 inventory files and 300 saved SDK actions/selected-action probabilities with zero mismatches, using a new download cache. A clean two-commit Hub history contains only the selected weights. Verification is recorded separately after upload, avoiding a self-referential package hash. Rebuild these figures with `uv run --extra analysis python scripts/plot_final_release.py`.

The final test is now inspected. Further tuning requires new locked evaluation games. The original reports, local review package and earlier research outcomes remain preserved; presentation edits do not change inference weights, calibration, or measured outcomes.
