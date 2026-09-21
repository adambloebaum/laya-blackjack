# Release process

The project lives on [GitHub](https://github.com/adambloebaum/laya-blackjack); the model lives on [Hugging Face](https://huggingface.co/adambloebaum/laya-blackjack). Their versions are linked by the pinned Hub commit in `blackjack/model_release.json`.

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

## Public release

Keep both repositories private through release review. When ready, publish the corresponding GitHub tag/release and make the model repository public. Link the release notes, model card, evaluation reports, license, and demo in both directions. Publishing the model does not provision hosted inference or expose the local training API.

Released checkpoints are immutable. A later model, calibration change, or edited inventory requires a new Hub revision and an updated explicit pin. Prior versions remain reproducible.
