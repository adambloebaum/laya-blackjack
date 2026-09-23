# Controlled model-visited training study

**Completed: no resolved improvement over the incumbent.** The fresh stronger-budget [qualification pilot](model-visited-pilot.md) passed every unchanged gate. Dataset assembly, matched training supervision, and sealed evaluation passed a real-model dual-GPU execution smoke and restart verification. This is one bounded comparison, not an open-ended search for a favorable result.

The full run `20260922-225718-visitation-training` launched at **15:57 Pacific on September 22, 2026**, from immutable revision `6e4b5dfd01498990254f0ad062273b39bed9ac5f`. Its original deadline is **03:57 Pacific on September 23**. The process-group runtime ceiling includes shutdown grace within the 12-hour budget. The [launch receipt](results/visitation-training-launch.json) records the fixed inputs and counts. The initial service completed data generation and both training arms, then stopped at the final inference-parity gate. The original failed run remains intact; see the recovery record below.

## Question and comparison

Does replacing half of the next training set with states visited by the frozen incumbent improve a fine-tuned policy, relative to the same amount of additional training on the basic/random behavior mixture?

Both arms start from the teacher-cost selection winner, weights `64ea2841949f306ed76c3032596c24bdc6cc5a0e45cfe25f180ee62468f151de`. The incumbent remains an explicit selection option and an unchanged evaluation comparator. Neither pilot's observations become a held-out test.

| Setting | Control | Model-visited arm |
| --- | --- | --- |
| Shared broad training states | The same 16,000 new general/depleted states | Identical rows and labels |
| Replacement training states | 16,000 states from 75% basic / 25% random behavior | 16,000 states visited by frozen Laya |
| Replacement rules | The ten pilot scenarios, equally weighted | Identical scenario weights |
| Replacement groups | 80 per scenario; 100 rounds per group | Matched initial game seeds |
| Replacement sampling | 20 uniformly sampled hero decisions per group | Identical sampling rule |
| Total training states | 32,000 | 32,000 |
| Reference | Hybrid exact / basic-continuation Monte Carlo | Same reference and sampling budget |
| Objective | Existing uncertainty-weighted imitation and probability losses | Identical objective |
| Optimizer | Existing AdamW/schedule; learning rate 5e-6 | Identical settings |
| Training | Three full epochs, eight typed questions per batch | Identical planned update count |
| Resources | Local GPU 0 | Local GPU 1 |

The shared broad half uses the existing `composition-v1` distribution: equal general and depleted strata across randomized rules, with its training-only rare-state weighting. The replacement half uses the qualified pilot's reservoir scheme and fixed benchmark scenarios. Consequently, the experiment tests this **specific 50% replacement**, not every mixture ratio, iterative DAgger, or arbitrary casino configuration. The behavior model stays frozen during collection; no refreshed-policy loop is included.

The two training manifests must have equal shard sizes and batch counts as well as equal state counts. Both trainers receive the same initialization and optimizer seed. A fixed sample budget is matched, but realized Monte Carlo work and label confidence can differ across visited populations. Report that difference rather than claiming identical compute expenditure.

## Fresh data and labels

Reserve root seed **20261004** under a new `model-visitation-training-v1` namespace. Reserve **30261004** for execution smoke. Include each source family and split in seed derivation; verify whole-group separation across training, selection, calibration, and test. Pair replacement-arm initial games only within the training split. Keep the two pilot roots and all prior inspected evaluations outside these groups.

Use 2,048–16,384 adaptive worlds for training labels and 4,096–16,384 for selection, calibration, and test, with the existing 50,000-node exact-reference limit. Record exact/fallback coverage, resolved fractions, and realized samples separately by arm and source family. Keep unresolved targets under the existing uncertainty weight rather than silently dropping difficult states. Complete dealer targets are required.

Repeated labeling was a pilot qualification check; the full dataset uses one independently seeded reference result per state. Where both replacement arms sample an identical public observation from a matched initial group, reuse one content-bound label so different label noise does not create an artificial distinction. Private seeds and replay traces never enter model state. Store input-bound receipts and preserve the collector's original inference batch membership on resume.

## Selection, calibration, and final evaluation

Both candidates use identical new selection/calibration/test rows and labels. Reserve 4,096 selection states, 2,048 calibration states, and 8,192 final-test states. Selection and test are equal general/depleted strata; calibration uses general games. Generate every required split completely before training; a deadline cannot silently reduce a planned population.

Choose eligible epochs using the established guard: depleted and overall selection regret must improve, and general regret may rise by at most 0.0005 original-wager units relative to the source. Include the unchanged source as epoch zero. Among eligible checkpoints, choose the lowest equal-stratum selection regret. This is a point-estimate guard, not a statistical noninferiority claim. Fit temperatures on calibration games only, using the serving SDK's option-count buckets.

Freeze both arms' selected checkpoint/config/tokenizer identities and the overall selection decision before either arm predicts a final-test state. Audit **both selected arms and the unchanged incumbent** through the serving SDK on the common final test. Report matched whole-game comparisons, reference-method strata, probability distances, serving parity, and inference latency. Do not infer the data-mixture effect by comparing only the overall winner with the incumbent.

Reserve a fixed return screen for four policies: basic heuristic, unchanged incumbent, control, and model-visited candidate. Each plays 100,000 independent fresh rounds plus 1,000 independent 100-round continuous blocks: **800,000 total rounds**. Use the same ten rule strata with paired initial seeds. Predeclare four aggregate contrasts: model-visited minus control and model-visited minus incumbent, each in fresh and continuous play. Use Bonferroni-adjusted 95% intervals across these four contrasts. Other pairings and scenario analyses are descriptive and must be marked accordingly.

This return screen is likely to leave small effects unresolved; the completed [12-million-round comparison](large-return-results.md) already demonstrated that limitation. Do not extend this screen or choose new settings after inspecting it. A promising result needs independent training replication and a separately sized, fresh final evaluation before release. All return quantities use settled profit per original wager with fixed bets.

## Execution and release boundary

The planned run has a **12-hour original deadline**, 24 CPU labeling workers, both local 4090s, a 28 GiB host-memory ceiling, immutable source/checkpoint snapshots, and systemd process-group cleanup. The separate real-model smoke passed before freezing the full launch. If both arms do not complete the matched training schedule, retain the partial artifacts but do not claim a completed matched comparison or select the faster arm by default.

`visitation-train` implements this protocol separately from the older single-dataset `targeted` studies. It generates the shared broad data, collects and replays both replacement arms, labels identical matched observations once, and publishes two hash-verified training manifests. All sampled model-collection actions receive an SDK parity audit before training. Both completed trainers must match the full planned update count before the shared selection record permits any final inference.

```bash
uv run --no-sync blackjack visitation-train --output artifacts/overnight/model-visited-training --source artifacts/research-sources/teacher-cost-v1 --qualification docs/results/visitation-strong.json --workers 24 --hours 12
```

Use a separate output directory with `--smoke --workers 4 --hours 0.25` for execution validation. The smoke uses 160 training states per arm, 32 selection states, 16 calibration states, 32 final-test states, one epoch (60 updates per arm), and 8,080 total return rounds. Its effective seed is 30261004, and its metrics are explicitly execution evidence only. The [completed execution receipt](results/visitation-training-smoke.json) records 191.7 seconds for all phases, zero collector/final SDK action mismatches, and a successful restart preserving all 500 tracked artifact hashes, cumulative fallback counts, and the original deadline.

The original deadline, source/runtime/checkpoint identities, and qualified-pilot digest are frozen in `run.json`. Collection and reference work have the first half of the run budget; training must finish with the final sixth reserved for audits and returns. Resume retains completed groups, labels, shared manifests, training checkpoints, and input-bound SDK audits. Completed earlier stages can be verified after their stage deadlines, without extending the overall deadline. An incomplete or changed artifact cannot silently replace a completed input.

`assembly/report.json` records label reuse and realized reference work. `selection-frozen.json` binds both complete candidates, all inference files, both training manifests, and the common audit dataset. `sdk-comparisons.json` compares both arms with the incumbent and each other. The return reports use `visited` as their fixed candidate for the four predeclared contrasts, regardless of which arm won selection. A separate `evaluated-candidate` is created only when an adapted checkpoint wins selection; sealed raw candidates remain intact. The dashboard names the control and model-visited GPUs and distinguishes collection/labeling counts from final-test counts.

Keep all candidates private. This study does not replace the dashboard model, upload new weights, or select a public release automatically. The final project will publish one best-supported model with its calibration and evidence; comparative reports can describe private intermediate candidates.

## Serving inference recovery

Both arms completed all 36,000 updates. The predeclared selection retained the unchanged source for the control and chose epoch three for the model-visited arm, which also won overall selection. This selection decision preceded every final-test prediction.

At 19:03 Pacific on September 22, the supervisor stopped after a single action disagreement in the model-visited arm's 8,192-state audit. The control had none. The action-only batch chose hit, while the three-question serving SDK chose surrender. The batch's raw top-logit gap was 0.203125, exceeding the existing 0.15 fallback threshold. This reproduces a numerical batch-shape limitation; the frozen model and data hashes were intact. The 800,000-round screen had not started.

The [failure and execution-check receipt](results/visitation-sdk-recovery.json) preserves the original failure and selection hashes. Recovery uses `--inference-mode sdk`: every Laya decision calls the same complete SDK path as the dashboard. It does not widen an empirical threshold using the inspected test case. New audits must reproduce all recorded original SDK actions; the original fast-batch mismatch remains visible. SDK-only reports mark the batch comparison as not performed, rather than presenting a fabricated zero-mismatch batch check.

`blackjack visitation-sdk-recovery` requires a separately archived failed study, frozen candidates, and an unexpired original deadline. It creates a separate run directory, verifies source/checkpoint/runtime/failed-audit identities, retains the original seeds and fixed counts, and uses both GPUs. It cannot restart training, change selection, extend the deadline, or overwrite the original failure. Its evaluated candidate is also separate from the sealed training outputs. This is a documented inference-method change for evaluation, not evidence that model performance improved. The completed results are reported below.

Recovery `20260923-050210-visitation-sdk-recovery` launched at **22:02 Pacific on September 22**, from immutable `0cbc8a3bec2fa15c0dff9055308885450e001836`. It completed all audits and the fixed return screen at **01:15 Pacific on September 23**, before the original **03:57** deadline. The 28 GiB memory ceiling remained enforced. The original failed run has not been rewritten.

## Completed results

The [verified result bundle](results/visitation-training-completed.json) contains all planned comparisons, exploratory scenario estimates, source/checkpoint identities, and verification counts. Both trainers finished their three epochs and 36,000 updates. Recovery completed 24,576 SDK audit states and **800,000 return rounds**, represented by 404,000 policy-unit records in 2,000 verified return shards. Each policy uses the same 100,000 paired fresh rounds and 1,000 paired continuous blocks; records are not independent across policies. Every saved SDK/return comparison was reproduced from its underlying records. The finalized candidate's inference files match the pre-test freeze.

Recovery took 3 hours 13 minutes. Wall time from the original launch was 9 hours 18 minutes, including the interval when the original failed service was inactive. No additional time was granted.

The control retained epoch-zero weights and recalibrated on the new calibration split; its SDK actions and evaluated returns matched the unchanged incumbent in this sample. The model-visited arm selected epoch three. Its better selection score is not evidence by itself of a better final policy.

| Common final-test metric | Incumbent | Control | Model-visited |
| --- | ---: | ---: | ---: |
| Reference action agreement | 96.1548% | 96.1548% | 96.0815% |
| Reference EV regret per original wager | 0.00111874 | 0.00111874 | 0.00109509 |
| Hit-bust probability Brier distance | 0.000276948 | 0.000276948 | 0.000237119 |
| Dealer probability Brier distance | 0.000887411 | 0.000887411 | 0.000749864 |
| Action probability Brier distance | 0.0589634 | 0.0595670 | 0.0615456 |
| Action calibration error | 0.0092015 | 0.0112678 | 0.0129574 |

The model-visited candidate's reference regret is 2.11% lower by point estimate, but the paired whole-game interval for its difference from the incumbent spans zero: −0.00017011 to +0.00011995. Its hit-bust/dealer probability distances improved descriptively; action agreement and action-probability metrics did not. These are comparisons with the recorded approximate reference, not empirical casino win calibration.

| Predeclared contrast | Difference in units per 100 rounds | Adjusted 95% interval |
| --- | ---: | ---: |
| Fresh: model-visited minus control | −0.025 | [−0.210, +0.160] |
| Fresh: model-visited minus incumbent | −0.025 | [−0.210, +0.160] |
| Continuous: model-visited minus control | +0.297 | [−0.181, +0.775] |
| Continuous: model-visited minus incumbent | +0.297 | [−0.181, +0.775] |

All four intervals include zero. The fixed comparison therefore does not establish that model-visited training improved returns. Mean model-visited returns were −1.2222 units per 100 fresh rounds and −0.1475 per 100 continuous rounds. The model-visited versus basic comparison is exploratory, outside the four predeclared contrasts; its nominal continuous-play interval is not a substitute for evidence of improvement over the incumbent.

The study-selected candidate remains private and inactive. The previous research incumbent remains the comparison baseline; the dashboard and private Hub package are unchanged. Before choosing the single final release, a promising configuration needs independent training replication and a separately planned, fresh return evaluation. This completed screen will not be extended or reused as an untouched final test.

## Independent SDK replication

The next authorized run uses `--study replication-sdk` and fresh root **20261006** (smoke **30261006**). It repeats the two 32,000-state arms from the same frozen teacher-cost incumbent: 16,000 shared broad states and 16,000 replacement states, three epochs / 36,000 updates per arm, and unchanged labels, optimizer, selection guard, and split sizes. This tests repeatability of the data-mixture approach; it does not train further on the first study's winner. All training, selection, calibration, final-test, and return games have new seed namespaces.

The operational correction is declared before any new data: model collection, collector rechecks, final audits, and all Laya return decisions use the canonical three-question SDK. Training and epoch selection retain the existing trainer implementation and are reported separately from SDK measurements. SDK-only receipts record policy consistency and do not claim fast-batch parity. A sampled collector action must reproduce through the SDK before its data can enter training.

The new fixed return evaluation doubles the original screen. Each of basic, incumbent, control, and model-visited plays **200,000 fresh rounds plus 2,000 independent 100-round blocks**, totaling **1,600,000 rounds**. Initial seeds are paired across policies; the four planned contrasts and familywise correction are unchanged. Sample counts and inference modes are frozen before collection. There is no result-dependent extension or early success stop.

For planning only, the first study's adjusted interval half-widths were approximately 0.185 fresh / 0.478 continuous units per 100 rounds. Doubling independent units would reduce them to approximately **0.130 / 0.338**, assuming similar paired variance. This is a budget-constrained replication with more precise evaluation, not a guarantee of resolving small effects or a power calculation for a prespecified minimum gain. The old SDK return timings imply roughly six hours for two GPU waves at the new counts; collection/training/audits previously took roughly three hours. The 12-hour original deadline now allocates the first three hours to complete data, requires both trainers to finish by hour five, and reserves seven hours for audits, returns, and verification. Incomplete runs remain incomplete; counts and deadlines cannot be reduced or reset to manufacture completion.

```bash
uv run --no-sync blackjack visitation-train --study replication-sdk --output artifacts/overnight/model-visited-replication --source artifacts/research-sources/teacher-cost-v1 --qualification docs/results/visitation-strong.json --workers 24 --hours 12
```

Validate execution in a separate output with `--smoke --workers 4 --hours 0.5`. The smoke still uses 160 training states and 60 updates per arm, 32 final-test states, and 8,080 return rounds. All smoke metrics are execution evidence only. Archive the exact tested source, preserve 24-worker/two-GPU limits and the 28 GiB memory ceiling, and launch with systemd cleanup and a 12-hour hard stop.

The replication compares the newly trained arms with their common incumbent. It does not rank the first study's candidate against the replication candidate, nor automatically choose a public release. After reviewing repeatability, final model selection across retained candidates needs its own selection protocol followed by a fresh locked final evaluation. Only one selected model will be published; intermediate weights and preparation repositories remain private.

The [real-model replication smoke](results/visitation-replication-smoke.json) completed in **254.2 seconds** on both local GPUs: 60 updates per arm, 80 audited collection states, 32 final SDK states per model, and 8,080 return rounds. All SDK action checks passed; all paired comparison reports reproduced from their saved records. Restart preserved **500 artifact hashes**, cumulative counts, and the original deadline. The implementation passed 119 Python tests, eight browser tests, lint, wheel checks, and [GitHub CI](https://github.com/adambloebaum/laya-blackjack/actions/runs/35892461120).

Full run **`20260923-170453-visitation-replication`** launched from immutable source `4063f75` at **10:04 Pacific, September 23**, with the original deadline at **22:04 Pacific the same day**. Its Python source hashes exactly match the verified smoke. The [launch receipt](results/visitation-replication-launch.json) records frozen counts, seed, checkpoint identity, resource bounds, and verified dashboard heartbeat. The run is collecting fresh training data; performance results are pending.
