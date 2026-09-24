# Release 1.0.0

[GitHub release](https://github.com/adambloebaum/laya-blackjack/releases/tag/v1.0.0) · [Selected model](https://huggingface.co/adambloebaum/laya-blackjack) · [Model card](../MODEL_CARD.md) · [Measured results](release-results.md)

The project publishes one selected model. The original experimental Hub repository was backed up locally and deleted before the selected repository took the canonical name. Every weight-bearing revision in the new repository contains the same selected weights; intermediate models and optimizer recovery files remain private.

Version 1.0.0 was published on September 24, 2026. The [publication receipt](results/public-release.json) records anonymous page access, a source clone matching the annotated tag, verified release assets, a clean wheel installation, and a fresh 43-file model download. The downloaded model reproduced 32 saved SDK actions and selected-action probabilities exactly. After a final local inference check, the developer's dashboard was stopped; the public downloads work independently of that service.

## Download and run

Follow the [README](../README.md#start-locally) to install the project, download the model and start the local dashboard. Neither the public code nor the model download requires authentication. The GitHub release includes the 1.0.0 Python wheel, the model download pin, and SHA-256 checksums. The dashboard runs on your own machine; this release does not provision a hosted application or inference service.

The [download pin](../blackjack/model_release.json) fixes Hub revision `f4480cebf3d477e4e0d49706b069d9f7af7dfbe3` and manifest `fde22c059f6817263350f53c6d2b9d56e55858883edb8facc37f1055e632fb65`. All 43 inventory files are verified before atomic installation. Existing installations remain immutable; use `blackjack fetch-model --output artifacts/checkpoints/released-v1` to preserve an earlier model, then explicitly reload the new checkpoint in the dashboard.

## What is in the package

The selected weights, tokenizer, encoder configuration and calibration exactly match the frozen winner of the [final selection](final-release-selection.md#completed-results). Final evaluation completed three million policy-rounds before release preparation. The package adds the model card, license/attribution, original training report plus a separate matched release comparison, raw final evaluation records, archived evaluator source, training-lineage datasets/reports/source, plotting inputs and figures.

The [public-package preparation receipt](results/public-release-preparation.json) records a documentation-only revision: only the model card changed from the verified private presentation package. Weights and every other inventory file are unchanged. The earlier [300-state reload check](results/final-model-reload.json) and [canonical-name migration check](results/final-model-rename.json) remain historical evidence for the identical inference files. Repeated predictions on those inspected states verify the artifact and serving path; they are not new performance tests.

Original local review packages are preserved. Their manifest digests differ because later packages bind revised documentation, not different learned weights. The original 17-file evaluation package used manifest `26320cadfe1fa7cad548011d689752d1b802c8e1ea8832635699f03d7ea81f04`; the private 43-file presentation package used `475b4804caa4f0aa71fc25f503830bc7ae7dfd11252b509a8125b7636571af79`.

## Maintaining a release

Use a new directory for every changed package. Verify both inventory self-consistency and equality with the frozen selected inference files. The preparation publisher intentionally accepts only private repositories; prepare future packages in a private staging repository before the explicit publication step.

```bash
uv run --no-sync blackjack package-model --source path/to/completed-candidate --output artifacts/releases/new-version --evidence path/to/public-evidence
uv run --no-sync python scripts/publish_model.py --folder artifacts/releases/new-version --repo adambloebaum/new-private-repository --version new-version --pin artifacts/releases/staged-pin.json
```

Download the immutable staging commit into a new directory/cache, verify the expected manifest and inference identities, and reproduce saved SDK decisions before updating the project's pin. Authentication remains in credential stores, never command arguments or committed files.

For a new evaluation dataset, retain the original training-stage report intact and add a separate `release_evaluation` block with both models' same-test metrics, baseline label, selection count, and dataset/freeze hashes. `blackjack.final_release.release_report` constructs it from both verified SDK audit receipts. Never replace test scores while retaining a different population's baseline or count.

Model card, README, figures, GitHub release notes, wheel and model pin must describe the same selected artifact. Inspect draft releases and their attachments as well as tags; a draft can retain obsolete files and model links. Regenerate current figures with `uv run --extra analysis python scripts/plot_final_release.py`; `plot_results.py` remains the historical v0.2 figure generator.

Before publication, complete the unit/API/browser checks, clean installation, package identity and repository-history review. After publication, verify anonymous repository access, wheel/checksums and a fresh model download. Keep original experimental history only in local private backups. Calibration or weight changes require new identity checks and an explicit new pin; documentation-only revisions may retain the identical selected weights.
