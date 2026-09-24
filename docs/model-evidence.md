# Training and evaluation evidence

The Laya Blackjack model download includes the data, settings, source snapshots and results behind its training and final evaluation. This guide describes the files in its `evaluation/` directory and how to use them.

## Training stages

`training-lineage/` contains the synthetic datasets and records for four stages: initial 500-state adaptation, 100,000-state broad training, 65,536-state shoe-composition training and 65,536-state training with stronger reference targets. Each stage has saved parameter, selection and training reports plus the corresponding source archive. Data archives preserve the original shards and their hashes.

To reproduce training from the pinned upstream Laya model, extract each stage into its own directory and use that stage's archived source and recorded settings in sequence. Intermediate weights and optimizer recovery files are not included. The data and code document the procedure; they do not guarantee bit-identical optimization on different hardware or software.

## Final evaluation files

| File | Contents |
| --- | --- |
| `final-release-completed.json` | Verified result summary, source identity, candidate-selection measurements and final decision metrics. |
| `selection-frozen.json` | The fixed selection criterion, chosen model and complete inference-file identities. |
| `run.json` | Evaluation plan, seeds, resource limits and model identities. |
| `summary.json` | The four planned aggregate return comparisons and adjusted intervals. |
| `sdk-audit.json` | Laya Blackjack's final 16,384-state evaluation through the application’s inference path. |
| `fresh-comparison.json`, `continuous-comparison.json` | Policy and scenario return measurements for each playing mode. |
| `evaluation-records.tar.gz` | Original data, all five selection audits, both final audits, raw return records, verification receipts and evaluator source. |
| `figures/`, `plot_final_release.py` | Performance figures and the script that renders them from the saved summary. |

The archives preserve the experiment's original identifiers. In selection and decision-audit records, `incumbent` identifies the stronger-reference ordinary-imitation model published as Laya Blackjack; `packaged` identifies the broad-training baseline, before the two refinement stages. In return records, `candidate` is Laya Blackjack, `baseline` is that broad-training model, and `basic` is the rule-based basic-strategy heuristic. These identifiers describe experimental roles, not additional models offered for download.

Original execution records also retain their creation-time status and local-artifact path aliases. A field such as `uploaded: false` describes when that record was written, not the availability of this public download. Metrics and recorded identities are preserved rather than rewritten when documentation changes.

## Recompute playing-return comparisons

Extract `evaluation-records.tar.gz` into a new directory, then extract its `evaluator-source.tar` into a separate source directory. Use that archived source's `blackjack.evaluation.compare_evaluations` with the three `returns/{candidate,baseline,basic}/{fresh,continuous}` directories to reproduce each comparison into a new output file. This aggregation uses saved return records and needs no model inference. Keep the original reports intact.

The plotting script expects the project layout, with the portable summary at `docs/results/final-release-completed.json`. From the project root, run `uv run --extra analysis python scripts/plot_final_release.py` to regenerate the SVG and PNG figures.

The root `training_report.json` preserves the last training stage's own metrics and adds a separate `release_evaluation` block for the final matched 16,384-state comparison. Read the count and baseline definition from the same block as the metric. Reproducing saved predictions or aggregates verifies the artifact and implementation; it is not a new independent performance test.
