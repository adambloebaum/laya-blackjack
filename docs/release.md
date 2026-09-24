# Downloads and release verification

[GitHub release](https://github.com/adambloebaum/laya-blackjack/releases/tag/v1.0.0) · [Model download](https://huggingface.co/adambloebaum/laya-blackjack) · [Model card](../MODEL_CARD.md) · [Performance results](release-results.md)

The project distributes one trained model with its tokenizer, calibrated configuration, training records and evaluation evidence. Follow the [README](../README.md#start-locally) to install the application, download the model and run the dashboard. No account is required. Inference runs locally; the project does not provide a hosted inference service.

## File integrity

The [download configuration](../blackjack/model_release.json) pins an immutable Hugging Face revision and its manifest digest. The downloader checks every file against that inventory before installing the package atomically. The manifest covers weights, tokenizer, configuration, documentation and evaluation files.

GitHub release assets contain a Python wheel, its model download configuration and SHA-256 checksums. Tagged source and release assets remain fixed. The current repository can point to a newer documentation revision of the same model; its weight and calibration identities are recorded in the model card. If an output directory contains a different package, use a new directory such as `artifacts/checkpoints/my-model` and explicitly reload the model in the dashboard.

## Training and evaluation evidence

The model's `evaluation/` directory includes raw final evaluation records, archived evaluator source, training datasets and reports, source snapshots, plotting inputs and figures. The [evidence guide](model-evidence.md), also included as the archive README, explains how to extract those records and reproduce the comparisons. Original training-stage metrics and the final comparison occupy separate blocks in `training_report.json`, keeping their populations and baseline definitions distinct.

The [publication receipt](results/public-release.json) records anonymous source and asset downloads, a clean wheel installation, a verified model download and reproduction of 32 saved inference results. An [earlier 300-state check](results/final-model-reload.json) verified the same model weights and calibration. These checks establish artifact and serving-path consistency; they are not additional performance tests.

The [documentation verification record](results/documentation-refresh.json) checks the current model card, evidence guide, chart labels and comparison display name. All 43 files passed anonymous download verification, and all six inference files and numerical training/evaluation records match the original model. Historical evidence retains its original identifiers, which the evidence guide defines.

## Maintaining a release

Use a new directory for every changed package. Verify both inventory self-consistency and equality with the frozen selected inference files. For new weights, the preparation publisher accepts only private repositories so the complete package can be reviewed before publication.

```bash
uv run --no-sync blackjack package-model --source path/to/completed-candidate --output artifacts/releases/new-version --evidence path/to/public-evidence
uv run --no-sync python scripts/publish_model.py --folder artifacts/releases/new-version --repo adambloebaum/new-private-repository --version new-version --pin artifacts/releases/staged-pin.json
```

Download the immutable staging commit into a new directory/cache, verify the expected manifest and inference identities, and reproduce saved SDK decisions before updating the project's pin. Authentication remains in credential stores, never command arguments or committed files.

For a new evaluation dataset, retain the original training-stage report intact and add a separate `release_evaluation` block with both models' same-test metrics, baseline label, selection count, and dataset/freeze hashes. `blackjack.final_release.release_report` constructs it from both verified SDK audit receipts. Never replace test scores while retaining a different population's baseline or count.

Model card, README, figures, GitHub release notes, wheel and model pin must describe the same selected artifact. Inspect draft releases and their attachments as well as tags; a draft can retain obsolete files and model links. Regenerate current figures with `uv run --extra analysis python scripts/plot_final_release.py`; `plot_results.py` renders the earlier broad-training study.

Before publication, complete the unit/API/browser checks, clean installation, package identity and repository-history review. After publication, verify anonymous repository access, wheel/checksums and a fresh model download. Preserve original experimental records and immutable published revisions. Calibration or weight changes require new identity checks and an explicit new pin; documentation-only revisions may retain the identical selected weights.
