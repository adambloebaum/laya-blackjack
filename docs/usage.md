# Using the laboratory

## Simulator and model

```bash
uv sync --extra model --extra dev
uv run --no-sync blackjack fetch-model
uv run --no-sync blackjack serve
```

Open http://127.0.0.1:8000 and select **Load trained Laya**. The release download is pinned to a commit and verified before installation. While the model repository is private, authenticate with `uv run --no-sync hf auth login` first.

For the simulator alone, use `uv sync --extra dev`; inference remains explicitly unavailable until weights and model dependencies are present. `LAYA_DEVICE=cuda:1` selects the dashboard GPU. `LAYA_CPU_THREADS` defaults to four. The server does not automatically source environment files.

One browser gets one versioned table. Configure 1–7 seats, deck count, dealer rules, payout, surrender, double after split, penetration, and tablemate behavior. Step manually or autoplay with basic, reference, or Laya decisions. The sidebar shows action preference, next-hit bust probability, and dealer finish probabilities separately.

Export a replay to preserve a session:

```bash
uv run --no-sync blackjack replay path/to/session.json
```

The replay includes the seed for reproduction. The seed never enters a policy observation. Sessions reset when the local server restarts; the application has no database.

## Small training experiments

```bash
uv run --no-sync blackjack generate --states 500 --samples 256
uv run --no-sync blackjack train --dataset artifacts/data/blackjack --output artifacts/checkpoints/my-pilot --full-model --epochs 6 --learning-rate 0.00002 --device cuda:0
```

Use a new output directory for each experiment. The dashboard also offers a small training form. Training is supervised distillation of approximate action values and probability targets; it does not reimplement upstream RLCD.

## Larger experiments

The [overnight guide](scaling-experiment.md) describes the parallel generator, four data splits, recoverable optimizer state, and bounded dual-GPU supervisor. The CLI reference is available through `blackjack --help` and each subcommand's `--help`.

```bash
uv run --no-sync blackjack overnight --output artifacts/overnight/my-run --source artifacts/checkpoints/my-pilot --hours 12 --workers 24 --gpus 0,1
```

Unattended jobs should use a process manager. The Linux supervisors enforce their own deadlines; a systemd runtime limit additionally bounds the process group. **Experiments → Research run** shows disk-backed progress from the latest overnight or policy-evaluation run.

## Paired policy evaluation

For the complete next-stage training experiment, use `blackjack targeted` as documented in the [composition experiment protocol](targeted-experiment.md). It generates new data, trains both candidates, freezes selection, audits the new final test through the SDK, and runs paired returns. The existing release stays active throughout.

The release suite compares the selected model, warm-start model, and basic heuristic on 100,000 independent fresh rounds and 1,000 independent blocks of 100 continuous rounds. Ten fixed scenarios span all table sizes plus additional rule and behavior variants.

```bash
uv run --no-sync blackjack research --output artifacts/evaluations/my-run --candidate artifacts/checkpoints/released --baseline artifacts/checkpoints/my-pilot --hours 3
```

`research` requires complete local evaluated checkpoints and creates private verified copies before play. Pass a new `--seed` for a new comparison; the default retains the historical seed for reproducibility. Within the original deadline, repeat the exact command from its original source snapshot to resume verified completed shards. Model, tokenizer, calibration, runtime/source version, seed, count, or budget changes reject the resume. Use a new output directory for a different experiment. The [large-return protocol](large-return-evaluation.md) specifies the current 12-million-round study and corrected aggregate intervals.

The supervisor uses GPU 0 for the warm start and GPU 1 for the candidate. For a single GPU, run each policy sequentially:

```bash
uv run --no-sync blackjack evaluate-policy --output artifacts/evaluations/candidate --source artifacts/checkpoints/released --policy laya --device cuda:0 --units 100000
uv run --no-sync blackjack evaluate-policy --output artifacts/evaluations/basic --policy basic --units 100000
uv run --no-sync blackjack compare-evaluations --input candidate=artifacts/evaluations/candidate --input basic=artifacts/evaluations/basic --output artifacts/evaluations/comparison.json
```

Use `--mode continuous --units 1000` for 100-round blocks. All policies must use matching mode, scenarios, seed, and unit count. Comparison verifies shard hashes and matches exact unit keys before computing paired differences. Checkpoint identities may differ; evaluation design may not.

The basic policy is a transparent multi-deck heuristic, not an exact reference for every rule set. Keep the small serial `benchmark` command for workflow checks; it does not retain paired return differences. Large evaluations store all raw unit returns and their hashes.

The [completed 12-million-round analysis](large-return-results.md) includes all scenario contrasts and a separate selected-error recheck. `scripts/analyze_large_returns.py` revalidates private raw artifacts before exporting portable JSON/CSV evidence. Rebuild its SVG/PNG figures from the committed summaries with `uv run --extra analysis python scripts/plot_large_returns.py`; model weights and GPUs are not required for plotting. These research figures describe a newer private candidate than the default downloadable package.

## Model-visited data pilot

```bash
uv run --no-sync blackjack visitation-pilot --output artifacts/evaluations/visitation-pilot --source artifacts/research-sources/teacher-cost-v1 --workers 24 --hours 1
```

The [pilot protocol](model-visited-pilot.md) fixes 4,000 development states, matching scenario/initial-seed arms, repeated hybrid-reference labels, and SDK/replay checks. `--smoke` offsets the seed and uses a small execution-only sample. Resume from the original source snapshot within the original deadline. Check `report.json` for the qualification decision; a completed run can still fail qualification. This command does not train, publish, or replace a model.

## Model audit and release artifacts

```bash
uv run --no-sync blackjack audit-model --dataset artifacts/overnight/my-run/dataset --source artifacts/overnight/my-run/gpu-1/candidate --output artifacts/model-audit --device cuda:0
uv run --no-sync blackjack recheck-errors --audit artifacts/model-audit --output artifacts/model-audit/rechecked-errors.json --workers 24
```

The audit runs the serving SDK on every frozen test state, checks batched-action parity, summarizes subgroups, and bootstraps whole-game groups. Error rechecks compare fixed actions using fresh paired worlds. These are descriptive analyses of the inspected test set; do not use them to tune a model and then claim the same test remains untouched.

Release weights, tokenizer, encoder configuration, calibration, documentation, and evaluation metadata are published together. Unfinished directories and failed integrity checks cannot become the live model. See the [model card](../MODEL_CARD.md) and [release process](release.md).

## Matched model-visited training

After a full qualification pilot passes for the exact source checkpoint, run the [matched training study](model-visited-training.md):

```bash
uv run --no-sync blackjack visitation-train --output artifacts/overnight/model-visited-training --source artifacts/research-sources/teacher-cost-v1 --qualification docs/results/visitation-strong.json --workers 24 --hours 12
```

This runs both local GPUs, retains 16,000 identical broad training rows in each 32,000-state arm, freezes both selected candidates before final inference, and evaluates four policies over 800,000 rounds. Use `--smoke --workers 4 --hours 0.25` in a different output directory to test execution first. Private candidates and the unchanged source are retained; completion does not replace the live model or publish weights.
