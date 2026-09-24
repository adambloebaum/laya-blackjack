# Release process

The project lives on [GitHub](https://github.com/adambloebaum/laya-blackjack). The single selected model is staged privately at [laya-blackjack-final](https://huggingface.co/adambloebaum/laya-blackjack-final). The older experimental Hub repository remains private. Never make that historical repository public: its revision history includes earlier weights.

## Prepared model

The [final selection](final-release-selection.md#completed-results) retained the hybrid-reference study's ordinary-imitation model. All three million return rounds and the frozen selection decision were verified. The original run's local review package remains preserved with manifest `26320cadfe1fa7cad548011d689752d1b802c8e1ea8832635699f03d7ea81f04`.

Presentation preparation creates a separate immutable **1.0.0** package. Its manifest is `475b4804caa4f0aa71fc25f503830bc7ae7dfd11252b509a8125b7636571af79`; all six inference files still match the frozen winner exactly. The new inventory binds the polished card, figures, training-lineage evidence, and a corrected report that separates original training metrics from the matched final release comparison.

The [download pin](../blackjack/model_release.json) identifies the selected private Hub commit. The [download/reload receipt](results/final-model-reload.json) verifies every inventory file, the clean repository history, and representative saved SDK decisions. This prepares the default download; it does not replace an already loaded local dashboard model.

## Package and verify

Use a new output directory for every changed package. Weights, tokenizer, encoder configuration, calibration, model card, license, report and evidence share one SHA-256 inventory. Reject incomplete or sealed candidates. Compare inference-file hashes with the frozen selected model as well as checking inventory self-consistency.

```bash
uv run --no-sync blackjack package-model --source path/to/completed-candidate --output artifacts/releases/new-version --evidence path/to/public-evidence
```

For a release evaluated on a new dataset, preserve the original report's metrics and counts. Add a separate `release_evaluation` block containing both models' same-test SDK metrics, baseline label, selection count and dataset/selection hashes. `blackjack.final_release.release_report` constructs it from verified final audit receipts. Do not overwrite only the test scores while retaining an old baseline or count.

Upload into a clean **private** personal repository. First write the pin to a staging location, download that full commit into a new cache/directory, verify the pinned manifest and selected inference identities, and reproduce saved decisions through the SDK. Only then update the project's download pin.

```bash
uv run --no-sync python scripts/publish_model.py --folder artifacts/releases/new-version --repo adambloebaum/new-private-repository --version new-version --pin artifacts/releases/staged-pin.json
```

The publisher verifies personal identity, refuses public repositories, and reads back the uploaded manifest. Keep credentials in the local Hugging Face cache or configured secret environment. After changing the project pin, an existing installation can retain its old checkpoint by downloading the new one with `blackjack fetch-model --output artifacts/checkpoints/released-v1`; installation never overwrites a different immutable package.

## Evidence and presentation

The selected package's `evaluation/` directory includes raw final data/audits/returns with receipts, archived evaluator source, portable summaries, plotting inputs, and training-lineage data/source/reports. It contains no intermediate weights or optimizer recovery files. Original local artifacts are unchanged. Model card, README, results and figures must all describe the same pinned weights. Rebuild current figures with `uv run --extra analysis python scripts/plot_final_release.py`; `plot_results.py` remains the historical v0.2 figure generator.

Before public publication, verify code, engine/information boundaries, API/browser behavior, lint, wheel contents, usage examples, links, attribution, and repository history. Treat golden-state reload checks as artifact verification, not new performance evidence. Preserve negative findings and uncertainty alongside improvements.

## Public publication

Both repositories remain private during preparation. Public visibility and the corresponding GitHub tag/release are a separate owner-controlled step. Publish only the selected model repository; keep the old experimental repository and its history private. Link project, model, results, license and demo in both directions. Publishing weights does not provision hosted inference or expose the local API.

A documentation-only model revision may retain the identical selected weights. A calibration or weight change requires fresh identity checks and an explicit new pin. Neither a completed run nor a successful upload automatically activates the local dashboard.
