# Composition-focused experiment

The 0.2.0 release's largest confirmed errors cluster in depleted shoes and unusual card compositions. This experiment tests whether targeted examples improve those decisions while preserving general play. It starts from the released checkpoint; it does not reuse the inspected 5,000-state test.

## Fixed protocol

| Component | Configuration |
| --- | --- |
| Warm start | Released model, pinned Hub revision in `blackjack/model_release.json` |
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

Neither trainer predicts the final test. After both candidates finish, `selection-frozen.json` binds the selected weights, calibration config, incumbent, and dataset hashes. Only then do the serving-SDK audits open final-test predictions for the selected candidate and released baseline. Paired whole-game bootstrap intervals summarize their regret and agreement differences, conditional on the approximate teacher labels.

Basic strategy, the release, and the selected candidate then play paired fresh and continuous return suites. Scenario weights and independent-unit uncertainty follow the [release evaluation](release-results.md), using new return seeds. The experiment always reports results; final-test performance does not change which candidate was selected. A worse result remains a result.

Raw candidates retain sealed-test metadata and are ineligible for dashboard discovery or release packaging. Completion creates a separate `evaluated-candidate` with final serving metrics and preserved weight/config identities. The live model and private Hugging Face release stay unchanged pending review of the resulting evidence. No profitability or improvement claim is made in advance.

## Active local run

Run `20260921-051623-composition` launched at 22:16 Pacific on September 20 from code revision `c4920ba4850f1f53d37de8061ad41f65b494bcac`. Its original hard deadline is 10:16 Pacific on September 21. This is an active experiment; no full-run performance result is available yet. The release remains unchanged.

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
