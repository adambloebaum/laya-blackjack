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

A full-model adaptation of [Laya](https://huggingface.co/convaiinnovations/laya) for a finite-shoe blackjack research simulator. It answers three structured questions from the public table state: which legal action to prefer, whether the next hit busts, and how the dealer finishes if no player draws again.

**Project, dashboard, and evaluation code:** [adambloebaum/laya-blackjack](https://github.com/adambloebaum/laya-blackjack).

**Version:** 0.2.0. **License:** Apache 2.0. **Model:** 421,293,830 parameters, ModernBERT-large with Laya typed decision heads. This package uses the Laya SDK; it is not a generative chat model or a drop-in Transformers text-generation pipeline.

## Intended use

Local strategy research, probability inspection, and simulated play across one to seven seats. The model consumes only public observations: hands, legal actions, rules, exposed table cards, and unseen rank counts. No random seed, concealed dealer rank, or future shoe order enters inference.

The supported simulator uses American hole-card rules, 1/2/4/6/8 decks, S17/H17, 3:2 or 6:5 payout, double after split, late surrender, and varying tablemate behavior. Fixed unit wagers, no insurance, no side bets. This model is an experimental approximation of an approximate reference policy. It is not certified optimal and does not establish real-world profitability.

## Load and play

```bash
git clone https://github.com/adambloebaum/laya-blackjack.git
cd laya-blackjack
uv sync --extra model --extra dev
uv run --no-sync blackjack fetch-model
uv run --no-sync blackjack serve
```

Open http://127.0.0.1:8000 and choose **Load trained Laya**. While the repository is private, run `uv run --no-sync hf auth login` first with an authorized account. The project pins this package to a full Hub commit and SHA-256 manifest. The installer verifies all files and publishes the local checkpoint only after verification.

Programmatic inference after downloading:

```python
from blackjack.engine import Game, Rules
from blackjack.model import LayaPolicy

game = Game(Rules(players=3), seed=42)
game.deal()
policy = LayaPolicy()
policy.load("artifacts/checkpoints/released", device="cuda:0")
if game.phase == "playing":
    prediction = policy.predict(game.observation())
    print(prediction["answers"])
```

Use the project's state and question builders; changing their wording, order, or compact representation changes the trained input contract. Context budgets are 1,024 tokens total and 256 for the question head. The integration rejects overflow. Tested with `laya==0.3.4`, `torch==2.8.0`, and `transformers==4.57.6` on RTX 4090 hardware. CPU inference is available but slower. SafeTensors weights occupy about 1.7 GB.

## Training and selection

The base checkpoint is `convaiinnovations/laya` at revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, using the ModernBERT-large backbone from Answer.AI and LightOn. A six-epoch, 500-state full-model pilot supplied the warm start. The larger experiment used:

| Component | Configuration |
| --- | --- |
| Training / selection / calibration / test | 100,000 / 2,000 / 2,000 / 5,000 states |
| Split boundary | Independent whole-game seed groups |
| Reference | Public-information Monte Carlo, basic-strategy continuations |
| Training reference budget | 1,024–4,096 sampled worlds per state |
| Evaluation reference budget | 4,096–8,192 sampled worlds per state |
| Loss | Supervised distillation; uncertain action labels receive lower weight |
| Candidates | Full-model learning rates 1e-5 and 5e-6, same seed 20260921 |
| Selected candidate | 5e-6, epoch 3, batch size 8, 112,500 updates |
| Selection criterion | Lowest selection-split reference EV regret |
| Temperature fitting | Separate 2,000-state calibration split, by option-count bucket |
| Hardware | Two independent RTX 4090 candidates; 24 CPU simulation workers |

The complete generation/training/evaluation overnight run took about 5 hours 47 minutes. The selected candidate's training, calibration, and reporting took about 4 hours 40 minutes. These are local measurements, not general speed claims. Training is supervised adaptation; this project does not reimplement upstream RLCD.

## Frozen decision evaluation

| Metric | Warm start, trainer | Selected, trainer | Selected, serving SDK |
| --- | ---: | ---: | ---: |
| Reference action agreement | 87.04% | 94.74% | 94.72% |
| Reference EV regret, units / decision ↓ | 0.010262 | 0.001915 | 0.001926 |
| Next-hit probability Brier distance ↓ | 0.011246 | 0.000821 | 0.000822 |
| Dealer probability Brier distance ↓ | 0.002351 | 0.001650 | 0.001650 |

The 5,000 test states come from 1,479 game groups. A 2,000-replicate game-cluster bootstrap gives serving agreement a 95% interval of **94.08–95.34%**, and reference EV regret **0.001589–0.002261**. Reference sampling uncertainty is not included in these bootstrap intervals. Batched training evaluation and serving inference can select different actions near ties because of numerical differences; both results are retained.

The fast action-only evaluator falls back to the serving three-question call near tied logits. It matched the SDK's actions on all 5,000 audited states, using 32 fallbacks. This is measured parity on that set, not a universal guarantee. Median SDK latency for all three questions was **19.0 ms** in this audit on one RTX 4090.

Action probabilities express preference among legal actions, not probability of winning. The SDK's `confidence` field is normalized entropy; use `probabilities` for the action distribution. Next-hit bust targets use the exact peek-conditioned unseen-rank marginal. Dealer targets approximate the dealer-only terminal distribution with no further player draws. Brier distances measure agreement with these targets, not calibration against realized casino outcomes.

## Paired policy returns

The release evaluated 600,000 simulated rounds: each of three policies played 100,000 fresh-shoe rounds and 1,000 independent 100-round continuous blocks. Ten fixed scenarios receive equal weights. Values below are selected-model minus comparator **units per 100 rounds**, with paired 95% intervals.

| Setting | Comparator | Difference | 95% interval |
| --- | --- | ---: | --- |
| Fresh | Warm start | +1.344 | [+1.001, +1.687] |
| Fresh | Basic heuristic | +0.074 | [-0.049, +0.197] |
| Continuous | Warm start | +1.279 | [+0.585, +1.974] |
| Continuous | Basic heuristic | -0.109 | [-0.498, +0.280] |

The model improves on its warm start in both settings. Neither comparison establishes an improvement over the basic heuristic. The model’s own mean returns were -0.704 units per 100 fresh rounds and -0.240 per 100 continuous rounds; both 95% intervals include zero. These results do not demonstrate positive expected profit. Continuous uncertainty uses independent block averages. Different actions may diverge after the paired initial seed. No rounds were voided. Full return summaries and raw unit records are included under `evaluation/`.

## Weaknesses and error analysis

Performance weakens in depleted shoes and some unusual compositions. Serving agreement was 97.86% in the first 10% of a shoe, versus 86.86% at 70–80% depletion (137 states) and 78.57% at 80–90% (42 states). These small subgroups are descriptive and confounded by the sampled mix of rules and hands.

The 30 largest recorded reference-regret errors were rechecked using 40,000 fresh sampled worlds each. Fixed reference-versus-model action contrasts remained positive at the individual 95% level in 28 cases. For example, hard 9 against dealer 3 at a strongly negative count preferred double; hit had a rechecked advantage of about 0.205 units. These are selected, exploratory cases with no multiplicity adjustment; they do not estimate overall error prevalence.

The final test is now inspected. Any training guided by this analysis requires a newly locked test set. The selected checkpoint was frozen before this audit and return analysis; neither changed the selection decision.

## Reproducibility and attribution

The package contains SafeTensors weights, encoder configuration, tokenizer, calibrated Laya configuration, sanitized training report, evaluation evidence, and a per-file SHA-256 manifest. Optimizer state is intentionally excluded. The project includes simulator tests, independent paired return evaluation, audit commands, and plotting code. See its [release results](https://github.com/adambloebaum/laya-blackjack/blob/main/docs/release-results.md) for return measurements and full provenance.

Upstream Laya and ModernBERT attribution is preserved in `NOTICE`; see `LICENSE` for Apache 2.0 terms. The simulation data is synthetic. This project is independent of Convai Innovations, Answer.AI, and LightOn.
