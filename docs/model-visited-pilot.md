# Model-visited data qualification

This development pilot tests whether fresh states visited by the frozen research candidate can support a controlled training experiment. It collects public observations, replays every trajectory, audits the serving SDK, and repeats reference labeling. It does not fit weights, calibrate probabilities, select a model, or serve as a final test.

The [12-million-round analysis](large-return-results.md) motivated this study: the latest model improves on the basic heuristic during continuous play, but its incremental return advantage over its predecessor is unresolved. Costly reference disagreements remain after fresh-world checks. The next question is whether new model-visited data provides usable coverage and stable targets.

## Fixed pilot design

| Item | Choice |
| --- | --- |
| Frozen model | Teacher-cost selection winner; weights `64ea2841949f306ed76c3032596c24bdc6cc5a0e45cfe25f180ee62468f151de` |
| Behavior arms | Frozen Laya; 75% basic / 25% uniform-random legal action mixture |
| Scenarios | The ten existing paired-return rule/table scenarios |
| Trajectories | Ten independently seeded groups per scenario and arm |
| Length | 100 continuous rounds per group; 20,000 rounds total |
| Sampling | Uniform reservoir of 20 hero decisions per group; 2,000 states per arm |
| Reference | Hybrid exact public-belief solver with explicit basic-continuation Monte Carlo fallback |
| Repeated labels | Two independent adaptive calls per state, 512–4,096 worlds per sampled call |
| Seed | Root 20261002 in the new `model-visitation-pilot-v1` namespace |
| Execution smoke | Root 30261002; ten-round groups, two states per group, 32–64 reference worlds |
| Resources | Up to 24 CPU labeling workers, both local GPUs for SDK audits; one-hour full-pilot deadline |

Arms share initial game seeds and rule weights but can visit different states. The mixture reproduces the existing hero behavior probabilities under the fixed benchmark rules; it is not an exact replica of the older training generator's broader random rule distribution or rare-state weighting. The pilot covers one through seven players, six- and two-deck conditions, and selected rule/behavior variants. It does not establish coverage for every configurable rule combination.

Each group contributes the same reservoir size. This defines equal trajectory weighting, not decision-frequency weighting across an entire casino population. Reservoir draws, hero behavior draws, and engine shuffling use separate random streams. Changing the reservoir size must not change actions, later shoes, or profit. Group seeds and complete action traces are private replay metadata; policy inference and reference values receive only `Game.observation()`.

## Qualification checks fixed before the full run

Both arms must pass all of these engineering gates:

- Every planned group finishes, every sampled observation and action replays exactly, and each reservoir contains all 20 requested states.
- Every sampled state fits the model context, all inferred actions are legal, and there are zero observed SDK/batched action mismatches. Sampled model-collection actions must also match the serving SDK.
- Dealer probability targets have no unresolved mass above floating-point tolerance.
- At least 80% of states have resolved labels in both reference passes, and at least 99% of those states retain the same recommendation across passes.
- At least 10% of sampled states are at least halfway through the shoe, and every arm contains at least 20 hard, 20 soft, and 20 pair decisions.

These thresholds qualify the pipeline for a training design; they are not confidence bounds, an optimality guarantee, or proof of a data-mixture benefit. The reference's resolved flag is the existing sampling separation heuristic. Exact scope and node-budget fallback are recorded. Exact results must reproduce identically; sampled targets are allowed to differ and are measured explicitly.

Agreement, regret, and probability-distance metrics evaluate the **same frozen Laya model** on each arm's sampled states; they do not measure the mixture behavior policy's accuracy. The populations differ, so their metric differences are descriptive rather than a matched model-quality comparison. Collection-versus-SDK parity applies only to the Laya behavior arm.

A failed check produces a completed diagnostic with `qualified_for_training_design=false`; successful execution is not the same as qualification. The smoke is always execution-only, even if all its individual checks happen to pass. No count extension, threshold relaxation, model activation, or automatic training follows a result.

## Running and inspecting

```bash
uv run --no-sync blackjack visitation-pilot --output artifacts/evaluations/model-visited-pilot --source artifacts/research-sources/teacher-cost-v1 --workers 24 --hours 1
```

Use `--smoke` in a separate directory for the smaller execution check. Unattended workstation launches use an immutable source archive, systemd process-group cleanup, and a memory ceiling. The supervisor records complete model/config/tokenizer identities, source/runtime hashes, parameters, and the original deadline, and copies the checkpoint into its private run directory before inference.

Repeat the exact command from the original source snapshot to resume before the original deadline. Completed collection, label, and audit groups are bound to their inputs with receipts. A partially completed collection batch is replayed with its original group membership: dropping already completed groups could change BF16 batch shape and later decisions. Existing completed groups are verified and retained. Label and audit groups resume independently.

**Experiments → Research run** shows trajectory groups and sampled development states. Completion explicitly distinguishes qualified data, checks needing review, and execution-only smoke. This run never substitutes pilot state counts for final-test or training-update counts.

The run directory contains `run.json`, `status.json`, `collect/`, `label/`, `audit/`, per-file receipts, a private frozen checkpoint, and `report.json`. The report preserves coverage, reference method counts, repeated-label agreement, SDK parity, probability distances, and each qualification check. All states are development-only; these artifacts are deliberately not a train/selection/calibration/test dataset manifest.

## What follows qualification

If the pilot qualifies, implement the [prepared matched data-mixture design](model-visited-training.md): keep broad coverage in both arms, replace a fixed fraction with fresh model-visited states in one arm, and match state counts, labeling effort, optimizer settings, and updates. Use new group-disjoint selection/calibration/final-test games, retain the unchanged incumbent, and predeclare the guards and final return protocol. Pilot metrics cannot establish that this experiment will improve the model.

If qualification fails, retain the measurements and fix the specific collection, serving-parity, coverage, or label-stability problem before scaling. Any revised research pilot gets a new configuration and seed; do not rewrite this run's thresholds or reuse it as untouched evaluation.

The final public project will contain one selected model. Pilot observations, intermediate weights, and their histories remain private during preparation.

## Standard-budget result and fixed follow-up

The full standard-budget run completed in 5 minutes 48 seconds on September 22, 2026. All 200 trajectory groups replayed, all reservoirs were complete, all 4,000 serving audits matched, and coverage/dealer-target checks passed. The pilot **did not qualify**: only 78.00% of model-visited states had resolved labels in both passes, below the preset 80% gate. The mixture arm reached 80.05%. All both-resolved recommendations were identical across repeats in both arms. The complete [portable result](results/visitation-standard.json) retains the original thresholds, source identity, and 600 artifact hashes.

All 440 model-visited states outside the both-resolved set used Monte Carlo; 74 changed recommendations across repeats. Of their 880 reference calls, 825 reached 4,096 worlds. This supports testing a larger sampling budget; it does not establish which disputed action is optimal. Model-state coverage was sufficient: 31.3% depleted, 1,581 hard, 195 soft, and 224 pair states. The same model's descriptive agreement was 95.55% on its visited states and 95.75% on mixture states; those are different state populations, not a model comparison.

One follow-up is fixed before collection: new root seed **20261003**, the same 4,000 states/20,000 rounds, model, scenarios, sampling scheme, and qualification gates, with **2,048–16,384 worlds per sampled reference call**. It retains two independent labels per state and a one-hour execution cap. This changes both the fresh state sample and labeling budget, so it measures whether the stronger pipeline qualifies; it is not a paired causal estimate of the budget's effect. No further count/budget extension is planned if it fails.

```bash
uv run --no-sync blackjack visitation-pilot --output artifacts/evaluations/model-visited-strong --source artifacts/research-sources/teacher-cost-v1 --workers 24 --hours 1 --seed 20261003 --label-budget strong
```

The `standard` budget remains the default. Budget choice and effective sample limits are frozen into the run identity; changing them cannot resume a prior run. Execution smoke always uses 32–64 worlds, even when checking the strong-budget configuration.

## Execution validation

The separate 40-state smoke completed on both GPUs and retained zero observed serving-action mismatches. Restarting it preserved all 120 group/receipt files and the original deadline. Its [receipt](results/visitation-smoke.json) is execution evidence only. All 100 Python tests, seven browser tests, lint, and wheel/static-asset checks passed before the full launch.

The first strong-budget launch collected all states but stopped before publishing any labels because the reference helper still enforced its older 10,000-world cap. That failed attempt is retained. The corrected source raises the bounded research cap to 16,384, with a regression that executes a real full-budget Monte Carlo fallback; the dashboard request limit remains 2,048. The corrected run retains the planned seed/design and the original follow-up deadline rather than starting a fresh hour.
