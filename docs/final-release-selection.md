# Model selection and final evaluation

Five completed training candidates were compared before the published model received its final performance test. Selection used new simulated decisions and a rule fixed in advance. The subsequent playing benchmark measured performance without influencing which model was chosen.

This report describes the completed experiment. It performed no additional training or temperature fitting. [Candidate identities](final-candidates.json) and archived reports retain their original machine-readable names for reproducibility; the descriptions below explain what each candidate represents.

## Nominees and frozen inputs

| Nominee | Origin |
| --- | --- |
| Stronger-reference model (`incumbent`) | Ordinary-imitation model after all four training stages; the default candidate under the selection rule |
| Broad-training baseline (`packaged`) | Model after the initial adaptation and 100,000-state broad-training stage, before the two refinement stages |
| Composition candidate | Completed composition-focused study winner |
| Model-visited candidate | First matched visitation study winner, with completed SDK recovery |
| Replication control | Standard-mixture winner of the completed replication |

These are the five completed study representatives, not every private intermediate epoch. They have distinct weight identities. Candidates that retained epoch-zero weights are represented by their incumbent, avoiding treating recalibration as new learned weights. This defines the scope of “best selected model”; it does not prove a globally optimal blackjack policy or the superiority of one training method.

Before any new model predictions, freeze all five complete inference packages, original training reports, dataset/return plans, runtime, source hashes, nominee list, and original deadline. Require completed prior evaluation and reject sealed or partial candidates. Preserve the earlier studies as inspected research evidence, outside the new seed namespace.

## Selection protocol

Use root **20261008** in the new `final-release-selection-v1` namespace, with separate data and return seed derivations. Smoke uses **30261008**. Generate **8,192 selection states** and **16,384 final-test states**, both equally divided between physically reachable general and depleted-shoe states. Whole-game groups are disjoint across splits. The existing dataset schema also creates 128 training and 64 calibration rows, which are unused: no optimizer or calibration fit runs in this study.

Labels use the hybrid exact reference with its existing scope and 50,000-node cap; fallback uses 4,096–16,384 common-world Monte Carlo samples and basic continuation. More samples do not remove continuation bias. All five models predict only the selection split through the complete three-question SDK. Selection and final-test progress are labeled separately in the dashboard.

The predeclared default is the stronger-reference model, called the incumbent in the statistical procedure below. For each of four challengers, calculate paired reference-regret differences and a **20,000-replicate paired whole-game bootstrap within each stratum**, maintaining equal general/depleted weights. Use two-sided percentile bounds at 0.625% and 99.375%, a Bonferroni adjustment across the four planned comparisons. These are approximate bootstrap bounds, conditional on recorded reference labels.

A challenger is eligible only if:

- Overall selection regret is lower and the adjusted upper bound of its difference from the incumbent is below zero.
- Depleted-stratum regret does not increase.
- General-stratum regret increases by no more than 0.0005 original-wager units per decision.

The subgroup thresholds are point-estimate guards, not statistical noninferiority conclusions. Among eligible candidates, select the lowest overall regret; exact ties retain the incumbent first, then use stable candidate-name order. If none qualifies, retain the incumbent. Existing calibrated configurations remain frozen. Probability distances and latency are reported, but do not introduce a new selection criterion after inspection.

Freeze the decision and the hashes of all selection reports before final inference. A missing candidate audit, changed input, incomplete count, or premature final prediction blocks progression. The selection criterion is reference imitation, not measured return maximization; the separate final benchmark measures the chosen policy's realized performance without selecting on its results.

## Locked final evaluation

Only the frozen winner and the broad-training baseline receive final SDK audits on the 16,384 new test states. They then play a fixed paired return benchmark alongside basic strategy: **500,000 independent fresh rounds plus 5,000 independent 100-round continuous blocks per policy**. This is **3,000,000 policy-rounds** and 1,515,000 policy-unit records. Seeds are paired across policies, so those records are not all independent across policies.

Four aggregate contrasts are predeclared: winner minus broad-training baseline and winner minus basic, in fresh and continuous play. Use Bonferroni-adjusted 95% normal intervals across those four contrasts. Scenario and other diagnostic analyses remain exploratory. The broad-training baseline is deliberately fixed; it is not the newer research incumbent. If selection chooses the broad-training model itself, retain the planned evaluation and clearly identify that identical-model comparison.

The approximate SDK throughput observed in the completed replication was about 36 rounds/second/model. One million rounds on each of two GPUs therefore takes roughly 7.7 hours. At similar paired variances, scaling the last screen from 200,000 rounds / 2,000 blocks to 500,000 / 5,000 would reduce interval widths by about 37%; comparisons with different policies can have different variance. This is resource-based sizing, not guaranteed power for a minimum gain. Counts will not be increased or reduced after seeing results.

The **12-hour original limit** includes all work. Data must finish by hour two, selection by hour three, final audits by hour four, and return workers at least two minutes before the overall deadline. Both local 4090s and 24 CPU workers are available; host memory is capped at 28 GiB. Launch from an immutable source archive under systemd with process-group cleanup. Resume retains the original deadline and rejects changed configuration, calibration, counts, or source. If the work cannot finish, retain it as incomplete.

## Reproducing the selection workflow

The supervisor verifies complete candidate inputs, freezes selection before final inference, and packages one model only after the evaluation finishes. The package includes the model card, license, source archive, evaluation records and portable summaries. Negative or inconclusive outcomes remain part of the evidence; a final-test result cannot be used to switch to another candidate on that same test.

Running this workflow requires all five local candidate checkpoints identified in `docs/final-candidates.json`; those intermediate weights are not included in the public model download. The command below documents the experimental procedure rather than a fresh-install quick start. The distributed evaluation records can be used to recompute the saved comparisons without rerunning model inference.

```bash
uv run --no-sync blackjack final-release --output artifacts/overnight/final-release --candidates docs/final-candidates.json --workers 24 --hours 12
```

A separate `--smoke --workers 4 --hours 0.5` run exercises all five candidates, 32 selection and 32 final states, 400 bootstrap replicates, and 6,060 return rounds. Smoke results are execution evidence only. It also exercises single-package creation and verification; its package must never be uploaded as a trained release.

## Execution verification

The [real five-model smoke](results/final-release-smoke.json) completed in **206.9 seconds** from archived source `051fbaa`. It exercised all selection audits, the global freeze, final SDK audits, 6,060 return rounds, raw-report reproduction, and creation of a verified package with exactly one set of weights. Restart preserved **211 tracked artifact hashes**, the package inventory, and the original deadline. These results validate execution only. All 126 Python tests, nine browser tests, lint and wheel checks passed; [GitHub CI](https://github.com/adambloebaum/laya-blackjack/actions/runs/35952013826) passed too.

## Full launch

Run **`20260924-034132-final-release`** launched from immutable source **`16a356f`** at **20:41 Pacific, September 23**, with its original deadline at **08:41 Pacific, September 24**. Its Python source hashes match the verified smoke exactly. The [launch receipt](results/final-release-launch.json) freezes all five model identities, selection criterion, fresh counts, resource limits, and verified dashboard heartbeat. It completed at **05:25 Pacific, September 24**, in **8 hours 44 minutes**, before the original deadline. General run pointers now identify this release study; completed visitation pointers retain their historical experiments.

## Completed results

The [verified completion receipt](results/final-release-completed.json) records all **7,380 return shards**, **1,515,000 policy-unit records**, and **3,000,000 policy-rounds**. Both paired return reports and the four-contrast adjusted summary reproduced exactly from the saved raw records using the archived evaluator. All five selection audits, both final SDK audits, source hashes, and the package inventory passed verification. Final predictions followed the global selection freeze; original reports were preserved.

The **stronger-reference ordinary-imitation model** was retained under the predeclared rule. No challenger qualified. This selects the best-supported nominee under that reference-imitation criterion; return outcomes were assessed afterward and did not change the selection.

| Nominee | Selection reference regret | Eligible challenger |
| --- | ---: | --- |
| Stronger-reference model | 0.00091700 | Default retained |
| Broad-training baseline | 0.00321610 | No |
| Composition | 0.00136920 | No |
| Model-visited | 0.00115928 | No |
| Replication control | 0.00097981 | No |

On **16,384 fresh final-test states**, the selected model agreed with the reference on **95.9473%**, versus **92.8345%** for the broad-training baseline. Mean reference regret was **0.00101425** versus **0.00320423** original-wager units per decision, a **68.35% lower point estimate**. These compare predictions with the hybrid reference, including approximate Monte Carlo labels; they are not win probabilities or proof of optimal play.

| Paired return contrast | Units per 100 rounds | Familywise 95% interval |
| --- | ---: | ---: |
| Fresh: selected minus basic | −0.00630 | [−0.08343, +0.07083] |
| Fresh: selected minus the broad-training baseline | −0.01990 | [−0.09414, +0.05434] |
| Continuous: selected minus basic | +0.30854 | [+0.03349, +0.58359] |
| Continuous: selected minus the broad-training baseline | +0.27458 | [+0.01612, +0.53304] |

Intervals use the predeclared Bonferroni correction across these four aggregate comparisons, with independent rounds or independent 100-round blocks and a normal approximation. Continuous-play advantages are resolved under this design; fresh-shoe differences remain unresolved. These conclusions apply to the fixed equal-weight simulation scenarios. The selected model's absolute average returns remained negative: **−0.49368 units per 100 fresh rounds** and **−0.55542 per 100 continuous rounds**. Lower losses do not establish casino profitability.

The published model has weight hash `64ea2841949f306ed76c3032596c24bdc6cc5a0e45cfe25f180ee62468f151de`, matching the frozen selection. The [download and verification guide](release.md) describes the public package and supporting records. The final tests are now inspected and must not be reused as untouched evidence for further tuning.
