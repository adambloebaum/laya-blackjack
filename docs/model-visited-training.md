# Controlled model-visited training study

**Full study running; results pending.** The fresh stronger-budget [qualification pilot](model-visited-pilot.md) passed every unchanged gate. Dataset assembly, matched training supervision, and sealed evaluation passed a real-model dual-GPU execution smoke and restart verification. This is one bounded comparison, not an open-ended search for a favorable result.

The full run `20260922-225718-visitation-training` launched at **15:57 Pacific on September 22, 2026**, from immutable revision `6e4b5dfd01498990254f0ad062273b39bed9ac5f`. Its original deadline is **03:57 Pacific on September 23**. The process-group runtime ceiling includes shutdown grace within the 12-hour budget. The [launch receipt](results/visitation-training-launch.json) records the fixed inputs and counts. The active service and live dashboard heartbeat were verified during collection/reference generation; training and evaluation follow automatically.

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
