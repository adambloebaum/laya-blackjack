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

### Replay follow-up

- Added a regression test that replays random tablemate actions over multiple shoe shuffles. Isolated behavior RNG from shuffle RNG so the action trace fully determines future cards.
- Initial experiment data was generated with the engine at commit `dcfaac4`; its recorded hashes identify that dataset. The RNG separation changes future generated random-tablemate datasets but not the saved training examples or their evaluation.

### Validation and results

- Full-model fine-tuning reached 85% reference agreement on 100 held-out states versus 43% for the base checkpoint; reload through the real SDK reproduced those metrics. Median local three-question inference was 19.1 ms.
- Completed a 200-round-per-policy realized-return benchmark and a 25-round-per-policy dashboard-launched benchmark. Intervals overlap; neither supports a profitability claim.
- Verified 30 engine/API tests, two desktop/mobile browser tests, trained-model UI decisions, normalization/legal-action contracts, checkpoint reload, and lint/format checks.
- Fixed inference cache version capture during model reload and long-session chart/counter presentation.
- Created the private personal GitHub repository as adambloebaum. Model weights and raw datasets remain local; small experiment reports are tracked.
- Set dashboard training to full-model adaptation by default (six epochs, 0.00002 learning rate), with an explicit decision-layer-only option; CLI defaults retain the lightweight pilot.
