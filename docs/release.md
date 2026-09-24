# Release process

The project lives on [GitHub](https://github.com/adambloebaum/laya-blackjack); the model lives on [Hugging Face](https://huggingface.co/adambloebaum/laya-blackjack). Their versions are linked by the pinned Hub commit in `blackjack/model_release.json`.

The owner wants **one final public model**, chosen after research is complete. Existing private packages are research artifacts, not commitments to publish multiple versions. Keep intermediate checkpoint weights private; comparative reports can accompany the one selected model.

The [completed final selection](final-release-selection.md#completed-results) retained the teacher-cost research incumbent from five frozen nominees and verified all three million return rounds. The [completion receipt](results/final-release-completed.json) binds the selected inference files, final evidence, and one local package (manifest `26320cadfe1fa7cad548011d689752d1b802c8e1ea8832635699f03d7ea81f04`). Presentation review, clean private Hub staging, and pinned-download verification remain. The existing Hub revision and dashboard pin still identify v0.2.

## Before publication

1. Freeze the selected checkpoint and preserve the criterion used to select it. Final-test results must not become a new selection criterion.
2. Reevaluate through the serving SDK. Record any numeric differences from batched training evaluation. Check action legality, context limits, calibration, and inference latency.
3. Complete the paired return suite and examine remaining mistakes. Publish limitations and negative results alongside improvements.
4. Update the model card, experiment report, figures, screenshots, and attribution. Keep personal filesystem paths and credentials out of distributable metadata.
5. Run unit/API/browser tests, lint, and a clean wheel-install check. Inspect the Git history for accidental secrets or binary artifacts.

## Model package

```bash
uv run --no-sync blackjack package-model --source path/to/completed-candidate --output artifacts/releases/v0.2.0 --evidence path/to/public-evidence
```

The package contains Safetensors weights, tokenizer, encoder configuration, calibration, a model card, license/attribution, and a SHA-256 inventory. Optimizer recovery files are excluded. Required files and state/question formatting are checked before atomic installation.

Upload the package to a **private** personal Hugging Face repository, then verify it by downloading the pinned revision. Record its commit and inventory digest in `blackjack/model_release.json`. The app can then install that exact release with `blackjack fetch-model`.

```bash
uv run --no-sync python scripts/publish_model.py --folder artifacts/releases/v0.2.0 --repo adambloebaum/laya-blackjack --version 0.2.0 --pin blackjack/model_release.json
uv run --no-sync blackjack fetch-model
```

The publisher checks the authenticated personal identity, refuses public repositories, and reads back the uploaded manifest before writing the pin. Authentication stays in the local Hugging Face cache or configured secret environment; never put a token in a command argument or commit.

The packaged `evaluation/` directory includes compressed synthetic training and pilot data, raw policy-return shards, the immutable evaluator source, and measured reports. These live with the model on Hugging Face; Git tracks the small summaries and plotting inputs. Rebuild figures with `uv run --extra analysis python scripts/plot_results.py`.

The newer private candidate's [12-million-round research report](large-return-results.md) is separate from the existing package's release evidence. Its portable JSON/CSV and figures are committed; regenerate those figures with `uv run --extra analysis python scripts/plot_large_returns.py`. Do not copy these metrics into the current model card as though the downloadable weights produced them. Final packaging must bind the chosen weights to their own matching evidence.

## Public release

Keep both repositories private through release review. Stage the final checkpoint in a clean model repository history containing only the selected weights, retaining the existing experimental repository privately. Do not simply change the experimental repository's visibility: Hub repositories retain [version history](https://huggingface.co/docs/hub/repositories-getting-started), so replacing current files does not exclude older checkpoints from publication. Any final repository naming/move must preserve private backups and update the explicit app pin and links.

At public release, publish the corresponding GitHub tag/release and the single selected model package. Link release notes, model card, evaluation reports, license, and demo in both directions. Publishing the model does not provision hosted inference or expose the local training API.

Checkpoint packages are immutable. A calibration change or edited inventory requires a new revision and explicit pin. Private experiment history remains reproducible without publishing its weights. The public release policy is one selected model unless the owner changes that scope.
