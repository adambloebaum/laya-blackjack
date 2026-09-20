# Worklog

## 2026-09-20 — Initial laboratory

- Inspected Laya's published model card, Python SDK 0.3.4, sequence builder, probability outputs, training example, checkpoint layout, and calibration behavior.
- Built an American blackjack simulator with 1–7 players, finite shoes, seed replay, splits, doubles, surrender, natural payouts, card counting, and configurable tablemates/rules.
- Added public-state-only conditional Monte Carlo evaluation with approximate action EVs, uncertainty, dealer finish and next-hit probabilities. Hidden information does not enter either policy.
- Built a responsive felt-table dashboard with genuine model inference, comparison sidebar, session graph, unseen rank histogram, table settings, autoplay/manual controls, replay export, and experiment jobs.
- Added pinned model loading, inference, supervised distillation, held-out calibration, complete checkpoint publication, and independent-round policy benchmarks.
- Downloaded and ran the actual Laya checkpoint on an RTX 4090. Generated 500 training, 100 validation, and 100 test states from separate game seeds, with 256 rollouts per action.
- Ran initial decision-layer training and an additional full-model experiment. Measured results are recorded in `docs/experiments.md`.
- Added tests, browser checks, reproducible environment locks, CI, and usage/design documentation. No database tables were introduced.

This new repository has no pre-existing worklog hook; this entry is maintained manually.
