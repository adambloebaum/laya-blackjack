# Research roadmap

## Version 1 is complete

The simulator, dashboard, training pipeline and final evaluation are implemented. The selected model is packaged at the canonical [Hugging Face repository](https://huggingface.co/adambloebaum/laya-blackjack), with verified downloads, inference, documentation and figures. Its lineage comprises a 500-state pilot, 100,000 broad states, 65,536 composition-focused states, and 65,536 hybrid-reference states. No further training is required for this release.

The [final selection](final-release-selection.md#completed-results) retained the hybrid-reference study's ordinary-imitation model from five nominees. The fresh three-million-round benchmark measured continuous-play gains of +0.30854 units per 100 rounds over basic strategy and +0.27458 over packaged v0.2, both with positive familywise intervals. Fresh-shoe differences remain inconclusive and absolute returns remain negative. See the [results report](release-results.md) for the full scope and limitations.

## Public release

Version 1.0.0 publishes the selected model at the canonical [Hugging Face repository](https://huggingface.co/adambloebaum/laya-blackjack), with a matching [GitHub release](https://github.com/adambloebaum/laya-blackjack/releases/tag/v1.0.0), wheel, checksums and pinned download. The public model card preserves the measured gains, unresolved differences and limitations. No account is needed to download the public artifacts.

The dashboard is a local application: clone the project, download the pinned model and run `blackjack serve`. The developer's local dashboard is not a hosted service or a dependency of the release. Original experimental models and their recovery artifacts remain in local private backups; only the selected model is public. See the [release process](release.md) for artifact and verification details.

## Optional future research

These are separate studies, not prerequisites for releasing version 1. Each needs a concrete hypothesis and new locked evaluation games; inspected final tests cannot be reused for selection or tuning.

1. **Extend the exact reference.** Public-belief dynamic programming currently supports a restricted single-player unsplit scope. Stronger split/multiplayer continuation needs independent validation of hidden-card conditioning and engine semantics. Additional Monte Carlo samples alone cannot remove continuation bias.
2. **Compare a compact structured-state baseline.** Measure reference regret, returns and latency against Laya under the same conditions. This can remain an internal research comparison without publishing another model.
3. **Test a specific reward-refinement hypothesis.** A bounded policy-improvement or reinforcement-learning study could optimize settled return, using legal-action masks, public observations and independent calibration checks. More compute is not itself evidence of better performance.
4. **Replicate training stability.** Independent training seeds and a predeclared effect/precision target can test reproducibility. Do not select a winner from a noisy profit estimate or extend evaluation until significant.

The [model-visited pilot and training studies](model-visited-training.md) are already complete, including an independent replication. They did not establish a policy improvement; another unchanged run is not the next required step. The [teacher-cost comparison](teacher-cost-experiment.md) also did not establish a benefit from the cost-sensitive objective. Their failures, unresolved differences and retained-incumbent outcomes remain part of the research record.

Fixed-wager playing-policy experiments do not address bet sizing, insurance, bankroll management or live-casino deployment. None of the release evidence establishes global optimality or profitability.
