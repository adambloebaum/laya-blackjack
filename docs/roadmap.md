# Research roadmap

## Release candidate

- A finite-shoe, multiplayer simulator with public-information-only policies.
- A full-model Laya adaptation trained on 100,000 decision states.
- Separate selection, calibration, and final-test games.
- A serving-SDK audit, reproducible policy comparisons, and a model card.
- A local dashboard that separates model predictions from reference probabilities.

## Highest-value next experiments

1. **Composition-sensitive mistakes — experiment prepared.** The [targeted protocol](targeted-experiment.md) generates fresh balanced general/depleted games, compares two conservative learning rates, and seals final-test predictions until selection is frozen. It covers hard totals, extreme remaining-rank mixtures, splits, and reachable depleted shoes. The inspected release test remains diagnostic material and is not reused.
2. **A stronger teacher.** Current action labels assume basic-strategy continuations. Compare exact composition-dependent solvers on tractable cases and stronger continuation policies before adding more imitation data.
3. **A compact baseline.** Compare a small tabular/neural policy against Laya using the same public state and evaluation units. Report quality, latency, and memory together.
4. **Repeated seeds.** Current learning-rate candidates share one training seed. Replicate the selected setup with independent seeds before treating small metric differences as robust.
5. **More independent play.** Expand paired continuous-play evaluation as needed to narrow uncertainty. Fixed wagers and fresh-round averages do not answer questions about bet sizing or bankroll management.

## Product follow-ups

- Add saved experiment browsing and explicit model-version selection.
- Provide a lightweight recorded demo suitable for the project page.
- Consider a hosted interactive demo with isolated sessions, resource limits, and disabled training controls.

Training longer is a hypothesis to test. The release should show the measured strengths and the remaining limitations together.
