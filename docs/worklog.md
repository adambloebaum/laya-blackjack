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

## 2026-09-20 — Overnight scaling pipeline

- Authorized scope: local compute only, up to 12 hours, increased workers and both local GPUs. Configured 24 simulation workers and independent candidates on both RTX 4090s.
- Optimized sampled-world cloning and continuation observation construction. A fixed 12-state/128-rollout comparison preserved exact outputs and reduced runtime from 2.242 to 0.377 seconds (5.95×). Regression tests compare deep-copy/public-policy trajectories across all tablemate behaviors.
- Added whole-shoe reservoir sampling, rare-state emphasis for training, adaptive paired action-value sampling, independent selection/calibration/test groups, deterministic parallel shards, source manifests, atomic receipts, and hash-verified recovery.
- Added shard-streamed training, mixed precision, gradient checkpointing, weighted action labels, warmup/cosine learning rates, optimizer/RNG/batch recovery, atomic best-checkpoint publication, and separate selection/calibration/final reporting.
- Added a shared-budget supervisor, parallel GPU candidates, complete-candidate comparison, optional return evaluation, and persistent dashboard progress. Overnight candidates remain separate from live checkpoint discovery.
- Verified an actual interruption/restart: GPU 0 restored epoch 0, shard 2, batch 2 and finished all 12 smoke-test updates. Both GPUs completed batch-8 training concurrently. These tiny smoke results validate execution only.
- Passed 36 engine/data/API tests and three browser tests, including parallel reproducibility, tamper rejection, adaptive/fixed sample equivalence, stale heartbeat display, and dual-GPU progress on mobile.
- Enabled user lingering for the durable systemd launch. The launch uses an immutable source snapshot, a 12-hour process-group limit, lower CPU priority, and a 28 GiB host-memory ceiling. Actual run identifiers and status are recorded with the launch artifacts.

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

- The final supervisor smoke completed generation, both GPU candidates, atomic publication, selection, and return evaluation in 75.6 seconds. Each GPU peaked at 8.45 GB allocated; SDK reload reproduced offline action agreement. Small execution metrics are saved in `docs/results/scaling-smoke.json`.

## 2026-09-21 — Evaluated model and private release preparation

- The authorized overnight experiment completed all 109,000 states and both full-model candidates in 5 hours 47 minutes. The 5e-6 candidate at epoch 3 won by selection-split reference EV regret. Final-test metrics did not determine the winner.
- Audited the real serving SDK on all 5,000 test states: 94.72% agreement, 0.001926 units of reference regret, 19.0 ms median three-question latency. Retained the slightly different trainer measurements and game-cluster bootstrap intervals. Action-only batching with near-tie serving fallback matched all audited actions.
- Rechecked the 30 largest recorded errors with 40,000 fresh paired worlds each. Twenty-eight individual intervals favored the original reference action. These are descriptive selected-case analyses; future tuning needs a new final test.
- Completed 600,000 policy-evaluation rounds in 23.5 minutes: basic, warm start, and selected model each ran 100,000 independent fresh rounds plus 1,000 independent 100-round continuous blocks. The selected model improves over its warm start in both settings; neither comparison with the basic heuristic resolves a difference. Published both positive and inconclusive results, raw return records, source hashes, and standalone figures.
- Added hash-verified evaluation resume/comparison, bounded dual-GPU research supervision, and dashboard progress that distinguishes rounds from independent blocks. Comparisons reject mixed simulator/evaluator code. Recomputed both saved reports exactly from raw shards using release code.
- Packaged Apache 2.0 license/attribution, a model card, 109,000 synthetic states, original pilot data, evaluator source, results, and SafeTensors weights. Uploaded the selected package to the private personal Hugging Face model repository. Pinned commit `a617f19274e5dcfc6d47b83e87146281ca703e9d` and its manifest digest in the project.
- Downloaded the release from Hugging Face, verified all 23 inventory files, and loaded it through the SDK. All 130 checked states (including the largest-error cases) reproduced saved actions, with legal and normalized probabilities. Installed and activated the reviewed release in the local dashboard.
- Hardened model discovery to exclude staging directories. Model loads verify released inventories; failed loads preserve the previous model. Polished README, results, usage, roadmap, contribution/security guides, citation metadata, model card, plots, and dashboard screenshot. Both repositories remain private.
- Validation: 49 Python tests, four browser tests, lint, packaged static assets/release pin, and an isolated wheel installation. Git history and current source were scanned for common credential patterns with no findings. Real downloaded-model inference and live UI checks supplement the model-free CI suite.
