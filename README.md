# Laya Blackjack

**A learned blackjack policy, a simulator you can inspect, and the experiments behind it.**

Laya Blackjack adapts [Laya](https://huggingface.co/convaiinnovations/laya) to play blackjack using only information visible at the table. It combines a finite-shoe simulator, an interactive probability dashboard, and a reproducible training and evaluation pipeline. One learned player shares the table with up to six simulated players.

[Download the model](https://huggingface.co/adambloebaum/laya-blackjack) · [Performance results](docs/release-results.md) · [Model card](MODEL_CARD.md) · [Usage guide](docs/usage.md) · [GitHub release](https://github.com/adambloebaum/laya-blackjack/releases/tag/v1.0.0)

![Blackjack table with learned predictions and reference probabilities](docs/dashboard.png)

## Start locally

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 is the tested environment; the package supports Python 3.11–3.13. CUDA speeds model inference and training. The simulator itself runs on CPU.

```bash
git clone https://github.com/adambloebaum/laya-blackjack.git
cd laya-blackjack
uv sync --extra model --extra dev
uv run --no-sync blackjack fetch-model
uv run --no-sync blackjack serve
```

Open **http://127.0.0.1:8000** and click **Load trained Laya**. No Hugging Face account or paid inference service is required. Weights occupy about 1.7 GB; the download also includes training and evaluation evidence. Every file is verified against a pinned model revision before installation.

For the simulator without model inference, use `uv sync --extra dev` and skip the model download. The dashboard runs locally and has no frontend build step. See the [usage guide](docs/usage.md) for installation paths, programmatic inference and training commands.

## Explore the table

- **Watch play.** Step through hands or autoplay. Inspect splits, wagers, exposed cards, unseen rank counts and returns; export a deterministic replay.
- **Inspect predictions.** The sidebar shows action preferences, next-hit bust probability and dealer finish probabilities alongside separately labeled reference estimates.
- **Change the rules.** Configure 1–7 seats, 1/2/4/6/8 decks, whether the dealer hits soft 17, blackjack payout, surrender, double after split, shoe penetration and tablemate behavior.
- **Run experiments.** Generate simulated data, fine-tune candidates, calibrate their outputs and compare policies on the same evaluation scenarios.

## How we trained the model

We fine-tuned the full Laya model through four stages, using simulated blackjack decisions and probability targets from a reference evaluator.

| Stage | Training states | Purpose |
| --- | ---: | --- |
| Initial adaptation | 500 | Establish the blackjack input format and full-model training pipeline. |
| Broad training | 100,000 | Learn decisions across table sizes, rule settings and hands. |
| Shoe-composition training | 65,536 | Add emphasis on depleted shoes and decisions affected by the remaining cards. |
| Stronger reference targets | 65,536 | Use exact finite-shoe calculations where supported, with Monte Carlo estimates elsewhere. |

Training, model selection, probability calibration and final testing use separate game groups. We also tested a loss that penalizes costly mistakes more heavily and additional training on states visited by the model itself. Neither approach established a better policy than the ordinary-imitation model from the fourth stage.

Five completed candidates were compared on **8,192 new selection states** under a rule fixed before evaluation. The chosen model was then frozen before testing on **16,384 new states** and a **3-million-round** playing benchmark. The [model card](MODEL_CARD.md) gives the training settings; the [selection protocol](docs/final-release-selection.md) explains how candidates were compared.

## Performance

On the final decision test, Laya Blackjack matched the reference action **95.95%** of the time. Its mean reference regret—the estimated value lost relative to the reference's preferred action—was **0.001014 betting units per decision**. Median inference time for all three predictions was **19.26 ms** on an RTX 4090, excluding loading.

To measure progress from the additional training stages, we compared it with the **broad-training baseline**: the same model after the initial adaptation and 100,000-state broad-training stage, before shoe-composition training and stronger reference targets. Both were evaluated on the same final test.

| Decision quality | Broad-training baseline | Laya Blackjack |
| --- | ---: | ---: |
| Reference action agreement | 92.83% | **95.95%** |
| Reference regret, units / decision ↓ | 0.003204 | **0.001014** |
| Next-hit probability Brier distance ↓ | 0.001650 | **0.000271** |
| Dealer probability Brier distance ↓ | 0.002527 | **0.000879** |

Brier distances measure squared differences from reference probabilities; lower is better. These are reference-imitation measurements, not win rates. The test contains equal numbers of general and depleted-shoe states, rather than their natural frequencies during ordinary play.

![Return comparisons against basic strategy and the broad-training baseline](docs/figures/final-returns.svg)

In continuous play, where cards remain out of the shoe between rounds until a shuffle, Laya Blackjack gained **0.309 betting units per 100 rounds over the basic-strategy heuristic** and **0.275 over the broad-training baseline**. Both confidence intervals remained above zero after adjustment for the four planned comparisons. When each round started with a fresh shoe, the differences were inconclusive.

Average returns were still negative: **−0.494 units per 100 fresh-shoe rounds** and **−0.555 per 100 continuous-play rounds**. The result is a reduction in losses in the tested continuous-play setting, not evidence of profitability or globally optimal play. The [full results](docs/release-results.md) include all comparisons, confidence intervals and evaluation conditions.

## What the predictions mean

Laya sees the active hand, legal actions, exposed cards, rules and unseen rank counts. It never receives the random seed, concealed dealer card or future shoe order. The reference accounts for information revealed by the dealer's blackjack check.

The model answers three questions:

- **Action:** a preference distribution over currently legal moves, not the probability of winning each move.
- **Next-hit bust:** the probability that drawing one card would bust the active hand.
- **Dealer finish:** the dealer's outcome distribution assuming no player draws again.

Training uses supervised distillation from a reference evaluator. That evaluator solves supported single-player, unsplit states exactly and uses Monte Carlo rollouts with basic-strategy continuation elsewhere. The dashboard displays its own Monte Carlo estimates separately from learned predictions. Reference agreement cannot establish optimality where the reference itself is approximate.

The simulator uses American hole-card rules, fixed initial wagers, no insurance or side bets, and a conservative early-shuffle reserve. The [design guide](docs/design.md) documents the simulator and information boundaries.

## Research and reproducibility

The model download includes synthetic training datasets, training reports, raw final evaluation records, evaluator source snapshots and plotting inputs. One trained model is distributed; the supporting studies document successful, negative and inconclusive findings.

| Study | What it tested |
| --- | --- |
| [Broad training](docs/release-results-v0.2.md) | Full-model adaptation across diverse simulated tables. |
| [Shoe-composition training](docs/targeted-experiment.md) | Extra coverage of depleted shoes and composition-sensitive decisions. |
| [Reference targets and training loss](docs/teacher-cost-experiment.md) | Exact calculations where feasible, and ordinary versus cost-sensitive imitation. |
| [12-million-round evaluation](docs/large-return-results.md) | Playing returns after the stronger-reference training stage. |
| [Model-visited training](docs/model-visited-training.md) | Training on states encountered by the learned policy, including an independent replication. |
| [Final selection and evaluation](docs/final-release-selection.md) | Choosing among five candidates before a separate final performance test. |

Inspected test sets are never reused as untouched evidence for further tuning. The [research roadmap](docs/roadmap.md) describes open questions and the [release guide](docs/release.md) explains artifact verification.

## Development

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check blackjack tests scripts
npm ci
npx playwright install chromium
npm test
```

The application uses HTML/CSS/JavaScript served by FastAPI. Node is used only for browser tests. CI checks simulation rules, hidden-information boundaries, API behavior, replay, artifact integrity, paired evaluation and the responsive interface without downloading model weights.

Regenerate the figures with `uv run --extra analysis python scripts/plot_final_release.py`. See [contributing](CONTRIBUTING.md) and [security](SECURITY.md). Code and model modifications use **Apache 2.0**; [NOTICE](NOTICE) credits Laya and ModernBERT. This is an independent research project.
