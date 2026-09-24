# Final model selection and release evaluation

This is the final bounded selection/evaluation stage for one public model. It performs **no additional training or temperature fitting**. The existing GitHub and Hugging Face repositories remain private through review; the live dashboard pin is unchanged. The exact nominees are in [final-candidates.json](final-candidates.json).

## Nominees and frozen inputs

| Nominee | Origin |
| --- | --- |
| Research incumbent | Teacher-cost study winner; retained after both visitation studies |
| Packaged baseline | Existing private v0.2 package currently offered by the project |
| Composition candidate | Completed composition-focused study winner |
| Model-visited candidate | First matched visitation study winner, with completed SDK recovery |
| Replication control | Standard-mixture winner of the completed replication |

These are the five completed study representatives, not every private intermediate epoch. They have distinct weight identities. Candidates that retained epoch-zero weights are represented by their incumbent, avoiding treating recalibration as new learned weights. This defines the scope of “best selected model”; it does not prove a globally optimal blackjack policy or the superiority of one training method.

Before any new model predictions, freeze all five complete inference packages, original training reports, dataset/return plans, runtime, source hashes, nominee list, and original deadline. Require completed prior evaluation and reject sealed or partial candidates. Preserve the earlier studies as inspected research evidence, outside the new seed namespace.

## Selection protocol

Use root **20261008** in the new `final-release-selection-v1` namespace, with separate data and return seed derivations. Smoke uses **30261008**. Generate **8,192 selection states** and **16,384 final-test states**, both equally divided between physically reachable general and depleted-shoe states. Whole-game groups are disjoint across splits. The existing dataset schema also creates 128 training and 64 calibration rows, which are unused: no optimizer or calibration fit runs in this study.

Labels use the hybrid exact reference with its existing scope and 50,000-node cap; fallback uses 4,096–16,384 common-world Monte Carlo samples and basic continuation. More samples do not remove continuation bias. All five models predict only the selection split through the complete three-question SDK. Selection and final-test progress are labeled separately in the dashboard.

The predeclared default is the research incumbent. For each of four challengers, calculate paired reference-regret differences and a **20,000-replicate paired whole-game bootstrap within each stratum**, maintaining equal general/depleted weights. Use two-sided percentile bounds at 0.625% and 99.375%, a Bonferroni adjustment across the four planned comparisons. These are approximate bootstrap bounds, conditional on recorded reference labels.

A challenger is eligible only if:

- Overall selection regret is lower and the adjusted upper bound of its difference from the incumbent is below zero.
- Depleted-stratum regret does not increase.
- General-stratum regret increases by no more than 0.0005 original-wager units per decision.

The subgroup thresholds are point-estimate guards, not statistical noninferiority conclusions. Among eligible candidates, select the lowest overall regret; exact ties retain the incumbent first, then use stable candidate-name order. If none qualifies, retain the incumbent. Existing calibrated configurations remain frozen. Probability distances and latency are reported, but do not introduce a new selection criterion after inspection.

Freeze the decision and the hashes of all selection reports before final inference. A missing candidate audit, changed input, incomplete count, or premature final prediction blocks progression. The selection criterion is reference imitation, not measured return maximization; the separate final benchmark measures the chosen policy's realized performance without selecting on its results.

## Locked final evaluation

Only the frozen winner and the previously packaged v0.2 baseline receive final SDK audits on the 16,384 new test states. They then play a fixed paired return benchmark alongside basic strategy: **500,000 independent fresh rounds plus 5,000 independent 100-round continuous blocks per policy**. This is **3,000,000 policy-rounds** and 1,515,000 policy-unit records. Seeds are paired across policies, so those records are not all independent across policies.

Four aggregate contrasts are predeclared: winner minus packaged baseline and winner minus basic, in fresh and continuous play. Use Bonferroni-adjusted 95% normal intervals across those four contrasts. Scenario and other diagnostic analyses remain exploratory. The packaged baseline is deliberately fixed; it is not the newer research incumbent. If selection chooses the packaged model itself, retain the planned evaluation and clearly identify that identical-model comparison.

The approximate SDK throughput observed in the completed replication was about 36 rounds/second/model. One million rounds on each of two GPUs therefore takes roughly 7.7 hours. At similar paired variances, scaling the last screen from 200,000 rounds / 2,000 blocks to 500,000 / 5,000 would reduce interval widths by about 37%; comparisons with different policies can have different variance. This is resource-based sizing, not guaranteed power for a minimum gain. Counts will not be increased or reduced after seeing results.

The **12-hour original limit** includes all work. Data must finish by hour two, selection by hour three, final audits by hour four, and return workers at least two minutes before the overall deadline. Both local 4090s and 24 CPU workers are available; host memory is capped at 28 GiB. Launch from an immutable source archive under systemd with process-group cleanup. Resume retains the original deadline and rejects changed configuration, calibration, counts, or source. If the work cannot finish, retain it as incomplete.

## One package, then review

After complete final evaluation, create one separate `release-candidate` and one `release-package`, with unchanged selected inference files, updated serving metrics, a generated model card, license/attribution, source archive, raw evaluation records, and portable summaries. Keep the original training report and selection evidence intact. Package inventory verification is independent of scientific approval; negative or inconclusive outcomes remain visible. No network upload, dashboard activation, release pin change, or public visibility change happens automatically.

The package is a private review artifact. Final presentation, provenance and usage examples will be reviewed against the completed results before uploading to a clean private Hugging Face history containing only the chosen model. A pinned download/reload check must pass before any release activation. The public GitHub release and the single final model are a separate publication step. If final evaluation raises concerns, record them rather than switching to a different candidate on that same final test.

```bash
uv run --no-sync blackjack final-release --output artifacts/overnight/final-release --candidates docs/final-candidates.json --workers 24 --hours 12
```

A separate `--smoke --workers 4 --hours 0.5` run exercises all five candidates, 32 selection and 32 final states, 400 bootstrap replicates, and 6,060 return rounds. Smoke results are execution evidence only. It also exercises single-package creation and verification; its package must never be uploaded as a trained release.

## Execution verification

The [real five-model smoke](results/final-release-smoke.json) completed in **206.9 seconds** from archived source `051fbaa`. It exercised all selection audits, the global freeze, final SDK audits, 6,060 return rounds, raw-report reproduction, and creation of a verified package with exactly one set of weights. Restart preserved **211 tracked artifact hashes**, the package inventory, and the original deadline. These results validate execution only. All 126 Python tests, nine browser tests, lint and wheel checks passed; [GitHub CI](https://github.com/adambloebaum/laya-blackjack/actions/runs/35952013826) passed too.

## Full launch

Run **`20260924-034132-final-release`** launched from immutable source **`16a356f`** at **20:41 Pacific, September 23**, with its original deadline at **08:41 Pacific, September 24**. Its Python source hashes match the verified smoke exactly. The [launch receipt](results/final-release-launch.json) freezes all five model identities, selection criterion, fresh counts, resource limits, and verified dashboard heartbeat. The run is generating new games; final selection and performance results are pending. General run pointers now identify this release study; completed visitation pointers retain their historical experiments.
