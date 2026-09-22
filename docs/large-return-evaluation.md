# Frozen 12-million-round comparison

This private evaluation measures realized returns for the selected teacher-cost candidate, its composition-stage predecessor, and the basic-strategy heuristic. The earlier study improved reference imitation, but its 600,000-round comparison did not establish a return advantage. No weights, calibration, or rule weights change during this evaluation.

## Active local run

Run `20260922-051218-returns-12m` launched at 22:12 Pacific on September 21 from immutable source revision `637ba67a9ae24e16d6f73e7048c3b33a3bc8e9df`. Both RTX 4090 workers and the CPU comparator run under `laya-blackjack-returns-20260922-051218.service`, with a 28 GiB host-memory ceiling. Systemd terminates the process group by approximately 10:12 Pacific on September 22. The full-run result is pending; `artifacts/latest-research-run.json` identifies the local launch and source snapshot.

## Predeclared protocol

| Item | Fixed choice |
| --- | --- |
| Candidate | Teacher-cost study, ordinary imitation, epoch 3; weights `64ea2841949f306ed76c3032596c24bdc6cc5a0e45cfe25f180ee62468f151de` |
| Predecessor | Composition study, epoch 3; weights `e12e1dd8c42885abe3ce5814f8de863aacb1b971496dab5d9c47f023a2adbf8c` |
| Other comparator | Existing basic-strategy heuristic |
| Fresh mode | 1,000,000 independent one-round units **per policy** |
| Continuous mode | 30,000 independent 100-round blocks **per policy** |
| Total | 4,000,000 rounds per policy; **12,000,000 rounds overall** |
| Root seed | 20261001; separate smoke root 30261001 |
| Scenarios | Existing ten equally weighted rule/seat scenarios |
| Inference | Frozen SDK/configuration, action batches of 32, existing 0.15-logit-gap serving fallback |
| Local resources | Predecessor on GPU 0, candidate on GPU 1, basic policy on CPU; at most 12 hours |

The scenarios are six-deck S17 tables with one through seven players, H17/no-DAS/6:5, two-deck S17, and a seven-player table with random tablemates. They match the earlier evaluation's rule distribution. Scenario and unit identity determine each seed; different actions can subsequently consume different cards. Return includes splits and doubles per original fixed unit wager. Equal scenario weights are a benchmark definition, not an estimate of casino visitation frequencies.

The sample size is based on the earlier comparison's variance and runtime, not a search for a significant result. At similar variance, it aims for nominal paired-interval half-widths near 0.05 units per 100 fresh rounds and 0.1 units per 100 continuous rounds. These are planning estimates. Both modes have fixed sample counts; there is no early stopping for significance or extension based on an observed result. The runtime estimate is 7–9 hours from measured throughput, with a 12-hour hard ceiling. A deadline interruption produces incomplete evidence, not a smaller completed experiment.

## Inference and interpretation

The four planned aggregate comparisons are candidate-minus-predecessor and candidate-minus-basic in both modes. Individual reports retain nominal 95% intervals. `summary.json` additionally applies a Bonferroni correction across these four comparisons: each corrected interval has 98.75% coverage, yielding at least 95% simultaneous family coverage under the normal approximation. Scenario-specific intervals are exploratory and are not included in that family guarantee.

Fresh uncertainty uses independent rounds. Continuous uncertainty uses independent block averages; the three million correlated rounds are **not** treated as three million independent observations. Report paired differences directly. A difference in policy returns does not by itself establish positive expected profit. If intervals still include zero, retain that inconclusive finding.

This run evaluates fixed policies. It does not isolate the effect of the exact teacher from extra data/training, tune a checkpoint, or establish that Laya is optimal. Later training guided by these results needs a new locked final evaluation. The final public project will distribute one selected model; intermediate weights and this run's frozen copies stay private.

## Freeze, resume, and inspect

The supervisor records source and runtime versions, every consumed checkpoint file, model/calibration hashes, seeds, sample counts, rule definitions, and the original deadline. It copies both complete checkpoints into verified private snapshots before starting policy workers. It rechecks snapshot and child-evaluation identities before comparing complete raw units. Package and raw evaluated checkpoints are supported; unevaluated/sealed-test checkpoints are rejected.

```bash
uv run --no-sync blackjack research --output artifacts/evaluations/large-return-run --candidate artifacts/research-sources/teacher-cost-v1 --baseline artifacts/research-sources/composition-v1 --fresh-units 1000000 --blocks 30000 --hours 12 --seed 20261001
```

Use a process manager for an unattended run. The workstation launch uses an immutable source archive, systemd process-group termination, and a host-memory ceiling. The latest status appears in **Experiments → Research run**. Raw return shards, receipts, per-mode manifests, frozen checkpoints, comparison reports, and the final summary live under the run directory.

Repeat the exact command from the original source snapshot to resume within the original deadline. The supervisor lock excludes a simultaneous restart. Changing source files, models, tokenizer, calibration, environment versions, seed, counts, or requested hours rejects the resume. After the original deadline, the run cannot acquire more time by restarting. Earlier experiments require their original source snapshot and cannot be adopted into this protocol.

Evaluation receipt version two stores per-shard serving-fallback counts, so resumed progress retains the cumulative diagnostic total. Completed raw units are verified and reused. The existing active model and private Hub package are not changed by completion.

## Execution validation

The separate 6,060-round smoke completed on both GPUs. A real-model restart preserved all 120 raw-shard/receipt hashes, the original deadline, and cumulative fallback counts. Its [execution receipt](results/large-return-smoke.json) is not performance evidence. All 86 Python tests, five browser tests, lint, and wheel/static-asset checks passed before the full launch.
