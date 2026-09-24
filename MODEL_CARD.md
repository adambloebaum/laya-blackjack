---
license: apache-2.0
language:
- en
base_model: convaiinnovations/laya
base_model_relation: finetune
tags:
- blackjack
- simulation
- decision-making
- modernbert
- laya
- probability-calibration
---

# Laya Blackjack

A full-model adaptation of [Laya](https://huggingface.co/convaiinnovations/laya) for public-information blackjack research. One learned player shares a finite shoe with up to six simulated tablemates. The model predicts legal-action preferences, next-hit bust probability, and the dealer's finish distribution if no player draws again.

**Model version: 1.0.0.** [Project and dashboard](https://github.com/adambloebaum/laya-blackjack) · [Measured results](https://github.com/adambloebaum/laya-blackjack/blob/main/docs/release-results.md) · [Selection protocol](https://github.com/adambloebaum/laya-blackjack/blob/main/docs/final-release-selection.md).

Laya Blackjack was developed through broad simulation training, targeted practice on depleted shoes, and improved reference targets. It matched **95.95%** of reference actions on a held-out 16,384-state test. A three-million-round benchmark found a continuous-play advantage over a basic-strategy heuristic; differences on fresh shoes were inconclusive. Average returns remained negative.

## Use and input contract

Local simulated play, strategy research, and probability inspection. The model sees the active hand, legal actions, exposed cards, rules, and unseen rank counts. It never receives the random seed, concealed dealer rank, or future shoe order. Inputs must use this project's state and question builders.

Supported simulation: American hole-card rules; 1–7 seats; 1/2/4/6/8 decks; S17/H17; 3:2 or 6:5 payout; double after split; late surrender; configurable penetration and tablemate behavior. Wagers are fixed at one initial unit. No insurance or side bets; split aces get one card and cannot resplit. Unusual shoe exhaustion refunds the round.

This is a **421,293,830-parameter typed decision model**, not a generative chat model. Context budgets are 1,024 tokens total and 256 for question heads; overflow fails explicitly. The tested stack is Python 3.13, `laya==0.3.4`, `torch==2.8.0`, and `transformers==4.57.6`, on RTX 4090 hardware. CPU inference is supported but slower. Weights occupy approximately 1.7 GB; the full download also includes evaluation and training evidence.

## Quick start

```bash
git clone https://github.com/adambloebaum/laya-blackjack.git
cd laya-blackjack
uv sync --extra model --extra dev
uv run --no-sync blackjack fetch-model
uv run --no-sync blackjack serve
```

The released model is public and requires no Hugging Face account. Open http://127.0.0.1:8000 and choose **Load trained Laya**. The project pins the Hub commit and manifest digest and verifies all files before installation. If an older checkpoint already occupies the default directory, use `blackjack fetch-model --output artifacts/checkpoints/released-v1` instead; the existing installation is preserved and the dashboard discovers the new completed checkpoint.

Programmatic inference after a fresh default installation:

```python
from blackjack.engine import Game, Rules
from blackjack.model import LayaPolicy

policy = LayaPolicy()
policy.load("artifacts/checkpoints/released", device="cuda:0")
game = Game(Rules(players=3), seed=42)
game.deal()
if game.phase == "playing":
    prediction = policy.predict(game.observation())
    print(prediction["answers"])
```

Use the new installation path if you chose `released-v1`. Model loading is local after download; no paid inference service is required. The dashboard is a localhost research tool, without remote authentication.

## How the model was developed

The upstream checkpoint is `convaiinnovations/laya` at `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, with the ModernBERT-large backbone from Answer.AI and LightOn. Training proceeded through four stages:

| Stage | Training states | Selection / calibration / test states | Training configuration |
| --- | ---: | --- | --- |
| Initial full-model adaptation | 500 | See archived pilot report | Six epochs |
| Broad training | 100,000 | 2,000 / 2,000 / 5,000 | LR 5e-6, epoch 3 |
| Shoe-composition training | 65,536 | 4,096 / 2,048 / 8,192 | LR 5e-6, epoch 3 |
| Stronger reference targets | 65,536 | 4,096 / 2,048 / 8,192 | Ordinary imitation, LR 5e-6, epoch 3 |

The broad stage covers varied hands, rules and table sizes. The shoe-composition stage emphasizes decisions after many cards have been dealt. The final training stage combines exact calculations for supported states with Monte Carlo targets for other states. All stages update the full model through supervised distillation; this project does not reimplement upstream RLCD.

Each stage separates whole-game groups across its data splits. The last stage trained for 73,728 updates with batch size eight and seed 40260924, and downweighted uncertain action labels. Separate calibration data was used to fit output temperatures. The final model retains the temperatures fitted on that stage's 2,048 calibration states.

We also compared ordinary imitation with a loss that penalizes costly action errors more heavily, then tested additional training on states visited by the learned policy. These alternatives did not establish an improved policy. The published model uses ordinary imitation from the stronger-reference stage.

Before final testing, five completed candidates were compared on **8,192 new selection states**. A rule fixed in advance required evidence of lower reference regret, adjusted for four comparisons, while limiting degradation on general and depleted-shoe states. No alternative met those requirements. The model, tokenizer and calibration were frozen before the final 16,384-state test and playing benchmark; there was no further training or temperature fitting. The [selection protocol](https://github.com/adambloebaum/laya-blackjack/blob/main/docs/final-release-selection.md) gives the exact criteria.

## Decision quality

The final test contains equal general and depleted-shoe strata: **16,384 states from 4,840 game groups**. This enriched mix is not the natural frequency of states during ordinary play. The comparison is against the **broad-training baseline**: Laya after the initial adaptation and 100,000-state broad-training stage, before shoe-composition training and stronger reference targets. Both models predict the same states using the full three-question inference call.

| Metric | Broad-training baseline | Laya Blackjack |
| --- | ---: | ---: |
| Reference action agreement | 92.8345% | **95.9473%** |
| Reference EV regret, units / decision ↓ | 0.00320423 | **0.00101425** |
| Next-hit probability Brier distance ↓ | 0.00164981 | **0.00027077** |
| Dealer probability Brier distance ↓ | 0.00252707 | **0.00087878** |
| Action preference Brier distance ↓ | 0.10846779 | **0.06084717** |

Laya Blackjack’s 95% whole-game bootstrap intervals are **95.6276–96.2274%** for agreement and **0.00091122–0.00112688** for reference regret (2,000 replicates). These individual intervals condition on recorded labels and exclude reference sampling uncertainty. Reference regret—the reference-estimated value lost by choosing an action rather than its preferred action—is **68.35% lower than the broad-training baseline by point estimate**. This is not a 68% increase in profit or win rate. Median measured three-question SDK latency was **19.26 ms** on an RTX 4090, excluding model loading.

The hybrid reference solves finite-shoe public-belief decisions exactly only for supported single-player unsplit states within a 50,000-node limit. Other states use 4,096–16,384 common-world Monte Carlo samples with basic continuation. More samples reduce sampling noise, not continuation bias. Agreement is an imitation measure, not proof of global optimality.

Action probabilities express preferences among legal actions, **not win probabilities**. SDK `confidence` is normalized entropy; use `probabilities` for distributions. Next-hit bust uses the exact peek-conditioned rank marginal. Dealer predictions assume no further player draws. Brier distances measure proximity to reference probabilities, not calibration against realized casino wins.

Every final Laya decision used the same complete three-question SDK call as the application. These measurements do not validate the separate accelerated, action-only evaluation path.

## Three-million-round return benchmark

Laya Blackjack, the broad-training baseline and the basic-strategy heuristic each played **500,000 independent fresh rounds** and **5,000 independent 100-round continuous blocks**. Initial seeds were paired across policies in ten equally weighted scenarios: six-deck S17 tables at each of 1–7 seats; H17/no-DAS/6:5; two-deck S17; and random tablemates at seven seats. Different actions can diverge after the paired start. No rounds were voided.

Fresh-shoe evaluation starts each round from a newly shuffled shoe. Continuous play carries the remaining cards across rounds until the simulator shuffles.

Values below are Laya Blackjack minus comparator **betting units per 100 original-wager rounds**, including doubles and splits. Intervals use a Bonferroni correction across these four predeclared aggregate contrasts and a normal approximation over independent rounds or blocks.

| Setting | Comparator | Difference | Familywise 95% interval |
| --- | --- | ---: | --- |
| Fresh | Basic heuristic | −0.00630 | [−0.08343, +0.07083] |
| Fresh | Broad-training baseline | −0.01990 | [−0.09414, +0.05434] |
| Continuous | Basic heuristic | **+0.30854** | **[+0.03349, +0.58359]** |
| Continuous | Broad-training baseline | **+0.27458** | **[+0.01612, +0.53304]** |

The adjusted intervals support both continuous-play advantages under this design. Fresh-shoe differences remain inconclusive. Absolute Laya Blackjack returns were **−0.49368 units per 100 fresh rounds** and **−0.55542 per 100 continuous rounds**. This is reduced loss in the measured continuous setting, not evidence of profitability. Scenario-level contrasts remain exploratory; no claim applies to every rule configuration. The basic comparator is a transparent multi-deck heuristic, not a universally exact strategy.

## Limits and reproducibility

Candidate selection compares a finite set of trained models under a declared reference-imitation criterion; it does not establish the globally best model or training method. The final reference is approximate outside its exact scope. Training stability across many random seeds, live casino deployment, varying bet sizes, insurance, and multiplayer exact optimal control are not established. Final test results are now inspected; future tuning requires new locked evaluation games.

The package includes one Safetensors checkpoint, tokenizer, encoder and calibrated configuration, original training-stage report plus a separately identified release comparison, license/attribution, SHA-256 inventory, raw final evaluation records, evaluator source, training-lineage data and reports, and plotting inputs. Training datasets, settings and source snapshots document how the model was built; intermediate model weights and optimizer recovery files are not distributed. Download and inference verification records are linked from the project’s release guide.

Weights SHA-256: `64ea2841949f306ed76c3032596c24bdc6cc5a0e45cfe25f180ee62468f151de`. Calibration configuration SHA-256: `f977fd353d0c0c84a44dd08419369dbd3e91f5d80ad38b09da36588140e51096`.

See the [evidence guide](https://huggingface.co/adambloebaum/laya-blackjack/blob/main/evaluation/README.md) for archive contents and reproduction steps. Data is synthetic. Code and model modifications use Apache 2.0; upstream attribution is retained in `NOTICE`. This independent project is not affiliated with Convai Innovations, Answer.AI, or LightOn.
