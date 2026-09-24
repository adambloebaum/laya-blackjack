# Research roadmap

Laya Blackjack includes a working simulator, local dashboard, training pipeline and evaluated model. Its development and measurements are described in the [model card](../MODEL_CARD.md) and [performance report](release-results.md). The model is available on [Hugging Face](https://huggingface.co/adambloebaum/laya-blackjack); the [usage guide](usage.md) explains local inference and experiments.

## What the experiments established

Four training stages combined broad simulation coverage, emphasis on depleted shoes and stronger reference targets. The final model was chosen from five completed candidates before a separate decision test and three-million-round playing benchmark. It improved continuous-play returns over the basic-strategy heuristic and the broad-training baseline, which represents the model before the two refinement stages. Fresh-shoe differences were inconclusive, and absolute returns stayed negative.

A cost-sensitive training objective and additional training on states visited by the learned policy did not establish a better policy. The [loss comparison](teacher-cost-experiment.md) and [model-visited studies](model-visited-training.md) retain those negative and inconclusive results.

## Open research questions

Each follow-up needs a specific hypothesis and new locked evaluation games. The existing final tests have been inspected and cannot be reused for selecting or tuning a model.

1. **Improve the reference evaluator.** Exact finite-shoe calculations currently cover a restricted single-player, unsplit scope. Extending them to split hands and multiplayer continuation would require independent validation of hidden-card conditioning and simulator rules. More Monte Carlo samples alone cannot remove continuation-policy bias.
2. **Compare a compact model.** Measure decision quality, playing returns and inference latency against a smaller structured-state model under the same conditions.
3. **Test reward-based refinement.** A bounded policy-improvement or reinforcement-learning study could optimize settled return directly, with legal-action constraints and separate evaluation. Additional compute alone does not establish an improvement.
4. **Measure training stability.** Repeat training with independent seeds and a fixed precision target to assess how reliably the procedure produces the observed performance.

Fixed-wager playing-policy experiments do not evaluate bet sizing, insurance, bankroll management or live-casino deployment. The existing evidence establishes neither global optimality nor profitability.
