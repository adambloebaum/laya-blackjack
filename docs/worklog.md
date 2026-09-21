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

## 2026-09-21 — Composition-focused follow-up

- Added `composition-v1`: fresh seed namespace, physically reachable depleted-shoe states, exact equal general/depleted shard pairs for training/selection/test, and general-only probability calibration. The fixed full plan has 65,536 training, 4,096 selection, 2,048 calibration, and 8,192 final-test states.
- Added guarded checkpoint selection with the unchanged release as epoch zero. Trained epochs must improve depleted and pooled selection regret while keeping general regret within a predeclared 0.0005-unit point-estimate margin. Compare 2e-6 and 5e-6 full-model candidates on the two local GPUs.
- Added deferred final tests: neither trainer makes baseline or candidate final-test predictions. An immutable selection receipt is written before either serving-SDK audit. Final comparisons report paired whole-game bootstrap intervals and separate general/depleted results. Smoke seeds are offset from production seeds.
- Added a bounded, resumable `targeted` supervisor with original-deadline retention, subprocess-group cleanup, verified configurations, 24-worker generation, dual-GPU training, final SDK audits, and 600,000 paired return rounds. Existing release weights, dashboard policy, and hosted model remain the incumbent until results are reviewed.
- Sealed-test candidates cannot be discovered by the dashboard or packaged for release. Completion builds a separate evaluated candidate with the final SDK metrics and verified frozen weights/calibration, preserving the original pre-test reports. The dashboard handles SDK audit counters as well as training updates and paired return units.
- The independent execution smoke completed all phases in 101 seconds, including both GPU candidates and final return comparisons. Original trainer reports retained null test values and no final-test token caches were created during training. This smoke is execution evidence only; its metrics are not used to choose the full experiment.
- Updated the protocol, README, usage, roadmap, and agent notes. Added coverage for deterministic balanced sampling, source/test separation, selection guard, frozen identity changes, paired audit games, and blocked publication of unevaluated candidates. Full local validation passed 55 Python tests and five browser tests, plus lint.

- Launched the full run `20260921-051623-composition` from immutable code revision `c4920ba4850f1f53d37de8061ad41f65b494bcac`. Its systemd service has a 12-hour total runtime ceiling (including shutdown grace) and a 28 GiB memory limit; the original deadline is 2026-09-21 10:16 Pacific. The dashboard successfully reports its active generation phase. Training and final evaluation continue automatically in that service.

## 2026-09-21 — Completed follow-up and final-model research priorities

- Confirmed the composition experiment completed in 4 hours 49 minutes: 79,872 states, both candidates, two SDK audits, and 600,000 return rounds. The selected 5e-6 epoch-three candidate improved same-test reference agreement from 92.37% to 94.53% and reduced reference regret by 48.70%. Return intervals versus the previous model and basic strategy include zero. The new weights remain local and inactive.
- Reviewed the reference, training objective, behavior sampling, and final audit decisions. Of 448 remaining disagreements, 340 have unresolved reference labels; the 108 resolved disagreements account for 58.2% of measured regret. Proposed stronger reference calculations, cost-sensitive action training, model-visited data, bounded reward refinement, and independent replication/evaluation. These are proposed experiments, not demonstrated improvements.
- Recorded the owner's requirement to publish one final best-supported model. Keep earlier weights private, including Hub history; stage final public model history separately without deleting private artifacts. Updated roadmap, release procedure, completed-run documentation, README status, and agent notes. No training, activation, upload, or visibility change was performed in this review.

## 2026-09-21 — Exact reference and cost-sensitive training

- Implemented bounded finite-shoe dynamic programming for public-information single-player/unsplit decisions, including negative peek conditioning, optimal continuation, doubles, surrender, and exhausted-shoe refunds. Unsupported and over-budget states explicitly use the existing Monte Carlo reference. Independent exhaustive tiny-shoe engine tests verify the decision information boundary and settlement parity.
- Added teacher-specific dataset namespaces, per-row method/fallback metadata, aggregate coverage, reference-kind SDK strata, and a cost-sensitive action loss alongside unchanged probability losses. Added a matched two-GPU `teacher-cost` study with the existing final-test sealing, original deadline, source verification, and private candidate handling.
- Validated the reference on independent development games: 120/887 states solved exactly, all 363 fixed-policy values within four Monte Carlo standard errors, no first-action recommendation changes between exact basic and optimal continuation in that sample. Maximum continuation-value gain was 0.01440 units. This is solver validation, not evidence of model improvement.
- Prepared a verified local warm-start package of the evaluated composition candidate outside dashboard discovery. Updated the new study protocol, README, roadmap, and agent notes. No live activation, remote model upload, or public visibility change.
- Validation passed: 80 Python tests, five browser tests, lint, and wheel/static-asset checks. The dual-GPU execution smoke completed in 102 seconds with distinct smoke seeds, no trainer final-test caches, and zero SDK/batched action mismatches. Its receipt is committed separately from research results.
