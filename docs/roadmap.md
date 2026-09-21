# Research roadmap

## Release candidate

- A finite-shoe, multiplayer simulator with public-information-only policies.
- A full-model Laya adaptation trained on 100,000 decision states.
- Separate selection, calibration, and final-test games.
- A serving-SDK audit, reproducible policy comparisons, and a model card.
- A local dashboard that separates model predictions from reference probabilities.

## Highest-value next experiments

The [composition-focused run](targeted-experiment.md) is complete. On its new 8,192-state test, reference agreement improved from 92.37% to 94.53% and reference EV regret fell 48.70%. The 600,000-round return comparison did not establish an improvement over the previous model or basic strategy. The candidate remains local and is not activated or uploaded.

The candidate agrees on 98.46% of the 7,028 states whose reference action passes the sampling separation heuristic. Of 448 disagreements, 340 (75.9%) have unresolved reference labels. However, the 108 resolved disagreements account for 58.2% of total measured regret, and the 30 largest errors account for 26.0%. Counting errors alone would understate the remaining costly mistakes. These are descriptive results conditional on the approximate reference; the heuristic does not prove the labels correct.

1. **A stronger reference.** Current labels evaluate one action followed by basic-strategy continuations. Benchmark exact composition-dependent calculations on tractable, rule-matched cases, then stronger continuations for harder cases. Validate negative-peek conditioning, split semantics, and tablemate behavior before using the new reference. More samples reduce sampling noise but do not fix continuation-policy bias. Existing [combinatorial analysis](https://github.com/possibly-wrong/blackjack) and [pair-splitting research](https://arxiv.org/abs/1909.13710) offer independent cross-checks, with different rule/strategy assumptions to reconcile.
2. **Train for the cost of a mistake.** Current action targets are one-hot recommendations, with lower weight for unresolved labels. Compare an expected-regret or action-value objective against that baseline using the same data and compute. Preserve separate hit-bust and dealer probability tasks; action preferences or normalized action values are not win probabilities. Spend extra reference computation on consequential ambiguous training states and cap the influence of noisy value estimates.
3. **Collect states visited by the model.** Existing generation uses a basic/random behavior mixture. Let a frozen candidate generate fresh trajectories, query the improved reference, and mix these states with broad rule coverage. This adapts the [DAgger approach](https://arxiv.org/abs/1011.0686) to this simulator; it is a proposed experiment, not an observed gain. Use fresh games for expensive split, double, surrender, and depleted-shoe errors. The now-inspected test remains diagnostic and cannot serve as the next locked final test.
4. **Test reward-based refinement after the reference and loss comparisons.** Start from the strongest supervised candidate and compare a bounded policy-improvement or reinforcement-learning run against it. Use settled profit in original-wager units rather than win rate, legal-action masks, public observations only, and separate calibration checks. More simulation or RL is not a guarantee of improved returns.
5. **Replicate and measure the practical difference.** Replicate promising configurations with independent training seeds. Size a new paired fresh/continuous return evaluation from a predeclared meaningful effect and precision target; millions of rounds may be warranted. Selection and final evaluation must have different games. Use expected return, reference regret, rule/seat robustness, probability quality, and latency together; do not choose a winner from a noisy profit point estimate or repeatedly extend evaluation until significant.

A small structured policy remains a useful internal quality/latency baseline. It does not require publishing another model. Fixed wagers and these playing-policy experiments do not address bet sizing or bankroll management.

## One final public model

The next implemented study is the [exact-reference and decision-cost comparison](teacher-cost-experiment.md), with matching data, learning rate, and compute across its two arms. Later model-visited data collection, reward refinement, and replication remain follow-ups contingent on these results.

The owner requested one final model release: the best-supported candidate after research is finished. Keep intermediate weights and prior private packages private. Publish one selected checkpoint, its calibration/configuration, evaluation evidence, and one default dashboard model. Public reports may compare experiments without distributing their weights. Preserve private recovery artifacts and do not delete them to enforce this preference.

Choose the final candidate on a separate selection protocol, freeze it, then run a new locked final test. If return differences remain unresolved, report that explicitly; improved imitation alone is not evidence of higher realized returns. See the [release process](release.md) for keeping experimental weights out of public revision history.

## Product follow-ups

- Add saved experiment browsing for research while keeping one default public model.
- Provide a lightweight recorded demo suitable for the project page.
- Consider a hosted interactive demo with isolated sessions, resource limits, and disabled training controls.

Training longer is a hypothesis to test. The release should show the measured strengths and the remaining limitations together.
