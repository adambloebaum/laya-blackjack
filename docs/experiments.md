# Initial experiments — 2026-09-20

Actual local runs on an NVIDIA RTX 4090, using the pinned English Laya checkpoint. These are small workflow-validation experiments, not demonstrations of optimal blackjack.

## Decision-layer run

500 training states, 100 validation states, 100 held-out test states; 256 reference rollouts per action. Three epochs with a frozen encoder, batch size 4, learning rate 0.0001. Training and evaluation completed in 38.1 seconds after the data and base checkpoint were available.

| Held-out metric | Base, raw | Trained, calibrated |
| --- | ---: | ---: |
| Reference action agreement | 43% | 47% |
| Reference EV regret (units) | 0.18496 | 0.18434 |
| Next-hit probability Brier distance | 0.60908 | 0.20695 |
| Dealer distribution Brier distance | 0.22860 | 0.05199 |
| Teacher-action ECE | 0.21594 | 0.05395 |

The decision-layer run improved probability matching but remained weak at action selection. No claim of profitability follows from these numbers. Calibration metrics use separate validation games; the table reports results on test games.

Full machine-readable configuration, dataset hashes, and metrics: [decision-layer report](results/head-only.json).

A full-model experiment and policy-return comparison are being recorded separately.
