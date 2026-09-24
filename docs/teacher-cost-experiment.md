# Exact-reference and decision-cost experiment

This study starts from the model trained on shoe-composition-focused data. It tests whether a training loss that accounts for decision cost improves on ordinary imitation, using identical new data and optimizer settings. The ordinary-imitation model from this study was later chosen as Laya Blackjack; the [final evaluation](release-results.md) measures it on a separate test.

## Completed local run

Run `20260921-173017-teacher-cost` launched at 10:30 Pacific on September 21 from immutable source revision `2082f3361dfb4a915363a072d88c6100496aeede` and completed successfully at 15:12 in 4 hours 42 minutes. All planned states, both candidates, SDK audits, and 600,000 return rounds completed.

Ordinary imitation at epoch three narrowly won selection: reference regret 0.001011149 versus 0.001011626 for the cost-sensitive arm. This tiny difference does not establish a robust advantage for either objective. The final test was opened only after selection was frozen.

| Same 8,192-state SDK test | Composition predecessor | Selected candidate |
| --- | ---: | ---: |
| Reference agreement | 94.96% | 95.90% |
| Reference EV regret, units/decision | 0.00140748 | 0.00090728 |
| General agreement | 96.53% | 97.34% |
| Depleted agreement | 93.38% | 94.46% |
| Hit-bust Brier distance | 0.00071626 | 0.00028901 |
| Dealer Brier distance | 0.00153291 | 0.00090296 |

Reference regret fell 35.54%; the paired whole-game bootstrap interval for the reduction is 0.000338–0.000676 units per decision. Exact labels cover 1,006 test states; 7,186 use the Monte Carlo fallback. Both models had zero observed SDK/batched action mismatches on the audit. These results compare the same test population and reference; they are not directly comparable to earlier test sets.

Candidate-minus-predecessor returns were -0.0285 units per 100 fresh rounds (95% paired interval -0.1714 to +0.1144) and +0.3840 units per 100 continuous rounds (-0.0478 to +0.8158). Candidate-minus-basic intervals also include zero in both modes. These nominal intervals do not demonstrate a return gain. [Complete measured summaries and identities](results/teacher-cost-results.json).

This model subsequently underwent the [12-million-round comparison](large-return-evaluation.md) and the separate [final evaluation](release-results.md). This inspected test is now diagnostic material; subsequent tuning requires a new final test.

## Reference contract

`hybrid-exact-v1` uses a finite-shoe dynamic program for one player with one unsplit hand and no legal split. It supports hit, stand, double, and surrender, S17/H17, negative ace/ten peeks, and the engine's exhausted-shoe refund. The player maximizes expected value over the public information set. It must never choose a different action for each possible concealed dealer card.

The unseen counts include the concealed dealer card. The next-card probability removes its posterior expected occupancy. After a visible draw, the counts update and the posterior is recomputed. Dealer outcomes enumerate without replacement. Optimal hit/stand continuation follows subsequent player draws; wagers remain in original-unit terms.

Each observation has a deterministic limit of 50,000 cache misses. Multiple players, split opportunities, previously split hands, or a computation-limit overrun use the existing basic-continuation Monte Carlo reference. Every row identifies `teacher_kind`; sampled rows also record the exact solver's fallback reason. No partial exact result silently removes a legal action. The dashboard reference remains unchanged.

Independent tiny-shoe tests enumerate concealed worlds through the real engine and group decisions by public observations. They cover H17/S17, peek restrictions, low hands with repeated decisions, doubles, and voids. A separate development diagnostic sampled 887 states from 256 games: 120 completed exactly, 732 were outside scope, and 35 reached the computation limit. All 363 fixed-policy action values agreed with engine Monte Carlo within four standard errors. Exact optimal continuation improved some action values by up to 0.01440 units, but changed no recommended first actions relative to exact basic continuation in this sample. These checks validate the implementation; they do not demonstrate model improvement or exact coverage of multiplayer/split play. [Development receipt](results/exact-reference-development.json).

## Controlled comparison

| Component | Fixed configuration |
| --- | --- |
| Warm start | Local, verified package of the evaluated composition-v1 candidate; both arms identical |
| Training / selection / calibration / final test | 65,536 / 4,096 / 2,048 / 8,192 new states |
| Sampling | Equal general/depleted shards; calibration uses general games |
| Reference | `hybrid-exact-v1`; sampled fallback uses 2,048–8,192 worlds for training and 4,096–8,192 for other splits |
| GPU 0 | Existing uncertainty-weighted imitation loss |
| GPU 1 | Same loss plus expected reference regret on action questions |
| Optimizer | Both full-model, learning rate 5e-6, three epochs, batch size eight, same training seed |
| Effective root seed | 40260924 with teacher-specific game namespace; smoke uses 50260924 |
| Resources | 24 CPU workers, two RTX 4090s, original-deadline 12-hour ceiling |
| Final play evaluation | 100,000 fresh rounds and 1,000 independent 100-round blocks for each of basic, source, and selected candidate |

For each legal action, define cost as `min(best_reference_ev - action_ev, 1 unit) / 0.1 unit`. Add `0.25 * sum(model_action_probability * cost)` to the action cross-entropy. Keep the existing action weights (2 for separated labels, 0.5 for unresolved labels). Bust/dealer tasks keep their probability losses unchanged. The fixed cost cap limits the influence of noisy large value differences. These constants are chosen before the run; the final test cannot be used to adjust them. The learned action distribution remains a preference distribution, not a win probability.

Both arms use the hybrid teacher. This isolates the loss comparison, but a change relative to the warm start combines new data, extra training, and reference changes. It does not isolate the model-level causal effect of changing the teacher. Exact-reference coverage and performance must be reported separately from sampled fallbacks.

## Selection, sealing, and monitoring

Use the existing guarded selection rule: improve pooled and depleted selection regret, with general regret no more than 0.0005 units above the source. The unchanged source is an epoch-zero option. Choose the eligible checkpoint with lowest selection regret, freeze both arms, and then open final-test SDK predictions for the selected model and source. Keep the current inspected test sets out of these games. Report unresolved return comparisons honestly.

```bash
uv run --no-sync blackjack targeted --study teacher-cost --source artifacts/research-sources/composition-v1 --output artifacts/overnight/teacher-cost-run --hours 12 --workers 24
```

The source must be a complete locally verified package. Keep it outside dashboard checkpoint discovery. Use the same process-manager/source-snapshot/deadline protections as the [composition experiment](targeted-experiment.md). The CLI remains attached to its terminal; workstation launches use a bounded systemd service. The Research run panel discovers the experiment under `artifacts/overnight`.

For an independent execution check, add `--smoke` with a new output directory. This uses separate seeds, small data, and one epoch. It cannot establish comparative performance. The source, live dashboard model, and private Hugging Face package are not changed by experiment completion.

The independent full-pipeline smoke completed in 102 seconds. Both objectives trained, the final-test token caches remained absent during training, both final SDK audits had zero action mismatches with the guarded batched evaluator, and all return suites finished. The supervisor stops before return evaluation if either SDK audit detects an action mismatch. [Smoke receipt](results/teacher-cost-smoke.json).

To reproduce the development reference diagnostic:

```bash
uv run --no-sync python scripts/check_exact_reference.py --output artifacts/reference-development.json --games 256 --samples 8192 --workers 24
```

This diagnostic uses a dedicated development namespace, independent of training and evaluation. Raw per-state results remain in its output; the committed receipt summarizes them.
