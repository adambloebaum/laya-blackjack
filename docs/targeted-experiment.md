# Composition-focused experiment

The model after 100,000-state broad training made its largest confirmed errors in depleted shoes and unusual card compositions. This experiment tests whether targeted examples improve those decisions while preserving general play. It starts from that broad-training checkpoint and uses new evaluation games, separate from the inspected 5,000-state test.

## Fixed protocol

| Component | Configuration |
| --- | --- |
| Warm start | Broad-training baseline after 100,000 training states, archived manifest `c118e0679a88469fca053a0b1d99b3b2667a8289951c7543fd1501f6aa7289d5`; the current model download includes further training |
| Training | 65,536 new states: 32,768 general and 32,768 depleted |
| Selection | 4,096 new states: 2,048 per stratum |
| Calibration | 2,048 new general states |
| Final test | 8,192 new states: 4,096 per stratum |
| Training seed | 20260924, separate `composition-v1` game-seed namespace |
| Learning rates | 2e-6 on GPU 0; 5e-6 on GPU 1 |
| Training | Full model, three epochs, batch size eight, same seed |
| Reference budgets | 2,048–8,192 worlds for training; 4,096–8,192 for evaluation |
| Local resources | 24 generation workers, two RTX 4090s, at most 12 hours |
| Return evaluation | Seed 20261025; 100,000 fresh rounds and 1,000 independent 100-round blocks per policy |

General games retain the existing rule distribution and whole-shoe reservoir; training includes the previous rare-decision weighting. Depleted games use 85% penetration but respect the simulator's ordinary shuffle reserve. An eligible public state has consumed at least half the shoe and has a hard total of 9–16, an absolute true count of at least three, or a legal split. Training further emphasizes states beyond 65% depletion. Some crowded small-shoe games cannot reach this region; the sampler rejects them without constructing impossible hands or changing card order.

Complete alternating shards enforce the 50:50 mixture. All decisions from a game remain in one split. Calibration stays on general games rather than the deliberately enriched challenge mixture. General and challenge test metrics are reported separately; their equal-weight pooled average is not a casino visitation-frequency estimate.

## Selection and final evaluation

The unchanged source checkpoint is an explicit epoch-zero option. A trained epoch is eligible only if its selection EV regret improves both in the depleted stratum and overall, while general-stratum regret increases by no more than **0.0005 units per decision**. Among eligible epochs/candidates, choose the lowest pooled selection regret. This is a point-estimate guard chosen before the run, not a formal statistical noninferiority test.

Neither trainer predicts the final test. After both candidates finish, `selection-frozen.json` binds the selected weights, calibration config, incumbent, and dataset hashes. Only then do the serving-SDK audits open final-test predictions for the trained candidate and broad-training baseline. Paired whole-game bootstrap intervals summarize their regret and agreement differences, conditional on the approximate teacher labels.

Basic strategy, the release, and the selected candidate then play paired fresh and continuous return suites. Scenario weights and independent-unit uncertainty follow the [release evaluation](release-results.md), using new return seeds. The experiment always reports results; final-test performance does not change which candidate was selected. A worse result remains a result.

Raw candidates retain sealed-test metadata and are ineligible for dashboard discovery or release packaging. Completion creates a separate `evaluated-candidate` with final serving metrics and preserved weight/config identities. Experiment completion does not automatically replace an installed model. No profitability or improvement claim is made in advance.

## Completed local run

Run `20260921-051623-composition` launched at 22:16 Pacific on September 20 from code revision `c4920ba4850f1f53d37de8061ad41f65b494bcac` and completed successfully at 03:05 Pacific on September 21, in 4 hours 49 minutes. All 79,872 planned states, both three-epoch candidates, SDK audits, and 600,000 return rounds completed before the original 10:16 deadline. Selection chose learning rate 5e-6, epoch three.

On the same new 8,192-state SDK test, the previous model and candidate measured 92.37% versus 94.53% reference agreement and 0.003329 versus 0.001708 units of reference regret. General agreement improved from 94.65% to 96.17%; depleted agreement improved from 90.09% to 92.90%. The paired overall agreement difference was +2.16 percentage points (95% game-bootstrap interval +1.68 to +2.64). This enriched test is not directly comparable to the older 5,000-state release test.

Candidate-minus-previous returns were -0.054 units per 100 fresh rounds (95% paired interval -0.182 to +0.074) and -0.008 units per 100 continuous rounds (-0.415 to +0.399). Neither these comparisons nor comparisons with basic strategy established a return gain. Both SDK audits had zero action disagreements with the guarded batched evaluator on the audited states.

This checkpoint became the starting point for the [stronger-reference training study](teacher-cost-experiment.md). The [final report](release-results.md) evaluates the model after that additional stage. The test has now been inspected for diagnostics; subsequent tuning requires a new locked final test.

## Run and monitor

```bash
uv run --no-sync blackjack targeted --output artifacts/overnight/composition-run --source artifacts/checkpoints/released --hours 12 --workers 24
```

The CLI needs a durable process manager for unattended use. The provisioned workstation run uses an immutable source snapshot, systemd process-group control, a hard 12-hour runtime including shutdown grace, lower CPU priority, and a 28 GiB host-memory ceiling. The supervisor reserves up to 90 minutes for final evaluation. Generation stops scheduling work after 28% of the budget; this balanced experiment requires its complete predeclared dataset rather than silently changing the mixture.

The latest run appears under **Experiments → Research run**. Its directory contains `run.json`, `status.json`, stage logs, dataset manifests, recoverable optimizer state, `selection-frozen.json`, SDK audits, raw return shards, comparison reports, and a final `summary.json`. An incomplete or failed run retains its completed artifacts and reports the failure.

Resume with the same command from the recorded source snapshot. Configuration, source hashes, and the original deadline must match. Do not resume from changed development code or restart the clock after budget exhaustion.

## Execution validation

A separate small run completed generation, both GPU candidates, guarded selection, final SDK audits, and both return modes in about 101 seconds. Training created no final-test token caches; original candidate reports retained null test metrics. The smoke seed is offset by 10,000,000 so inspecting it does not consume the actual final test. [Execution receipt](results/targeted-smoke.json).

The smoke result establishes that the pipeline runs, not that the training idea improves performance. Unit tests additionally cover balanced/disjoint deterministic sampling, the selection guard, frozen-artifact mutation rejection, paired game identities, and blocked publication of sealed-test candidates.
