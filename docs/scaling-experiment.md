# Local overnight experiment

This run extends the full-model pilot with more independent games, more accurate Monte Carlo labels, whole-shoe coverage, separate model selection and calibration, and recoverable training. It is simulation distillation from an approximate teacher, not a claim of optimal play.

## Compute and budget

| Component | Configuration |
| --- | --- |
| Simulation | 24 worker processes; 64-state atomic shards |
| Training | One full-model candidate per RTX 4090; batch size 8; up to 3 epochs |
| Candidates | Learning rates 0.00001 and 0.000005, same seed/order |
| Warm start | Local `artifacts/checkpoints/blackjack-full` pilot |
| Training target | 100,000 states / 300,000 typed questions |
| Selection / calibration / final test | 2,000 / 2,000 / 5,000 states; disjoint whole-game seeds |
| Label sampling | 1,024–4,096 worlds per action for training; 4,096–8,192 for held-out splits |
| Deadline | At most 12 hours for the entire job, including generation and evaluation |

Both GPUs run independent candidates; GPU memory is not pooled. Other local services are left running. Training enables gradient checkpointing and bfloat16 autocast while retaining full-precision parameters. The systemd service uses lower CPU priority and a 28 GiB host-memory cap.

Generation reserves 28% of wall time, then drains submitted shards. Held-out shards are generated first. If generation reaches its time allocation, training uses the completed, verified training shards; the manifest records the actual count. Each candidate stops updates before the global deadline to allow selection, calibration, and saving. An additional systemd hard limit bounds the entire process group even if Python stalls.

## Data and evaluation

The sampler traverses the first complete shoe of each seeded game, retaining up to four decisions. Rules span 1–7 players, multiple deck counts, S17/H17, payouts, surrender, double after split, split limits, penetration, and tablemate behavior. Training emphasizes pairs, soft hands, larger counts, and multicard decisions. Held-out reservoirs are uniform over the visited decisions within each game. Exploration takes a random legal action 25% of the time, so the evaluation distribution is deliberately broader than on-policy Laya play.

Candidate actions share hidden-world samples. Sampling increases when paired action-value differences remain unclear. The three-standard-error stopping rule is a sequential heuristic, not a guaranteed confidence interval. Unresolved action labels receive weight 0.5, resolved labels 2, and probability questions 1. More samples reduce sampling noise but do not remove the teacher's basic-strategy continuation bias.

Checkpoint selection uses selection-set teacher EV regret. Temperatures use calibration games only. The final test reports reference agreement, EV regret, probability Brier distances, action calibration error, resolved/unresolved decisions, and performance by player count. The warm-start baseline is measured on the same test data before updates. Its metrics do not affect selection or stopping. These broader test results should not be directly compared with the pilot's 85% on a different 100-state dataset.

When time remains, the selected candidate also runs a 2,000-round-per-policy fresh-shoe return comparison against basic strategy and the Monte Carlo policy. The status file explicitly records whether that return report completed. A finished candidate is not evidence of profitability, and it does not replace the dashboard model automatically.

## Monitor and recover

The **Experiments** tab reads progress from disk every five seconds. A missing heartbeat is shown as unresponsive instead of silently claiming the job is running. The simulator remains usable during training; interactive inference competes for GPU resources.

Each run has `launch.json` with its service name and source revision, `run.json` with the original start/deadline, and `status.json` with stage and heartbeat. Generation and both trainers have separate logs. `comparison.json` names the selected completed candidate. `returns.json` exists only after a completed return benchmark.

```bash
systemctl --user list-units 'laya-blackjack*'
```

Use the exact unit from `launch.json` with `systemctl --user status UNIT` or `systemctl --user stop UNIT`. User lingering is enabled on this workstation so the job survives logout. It does not survive machine shutdown automatically.

To recover a failed job within its original deadline, rerun its recorded command from its saved `source/` directory. Keep all arguments unchanged. Generation reuses verified shards; training restores model, optimizer, RNG and next batch. Changing source code, data, tokenizer, or training configuration requires a new experiment directory. Restarting after the original deadline requires a newly budgeted experiment; it never silently grants another 12 hours.

## What follows

1. Review held-out EV regret, probability errors, and realized-return intervals by rule set, player count, hand type, and shoe depth. Reload the candidate through the serving SDK before considering activation.
2. Collect new games around costly mistakes and near-tie decisions. Refine the teacher with stronger continuations or an exact solver for supported cases, then train on those examples.
3. Run larger independent evaluation batches and repeated training seeds. Keep the next final test set separate from the data used to diagnose this run.
4. Compare Laya with a compact tabular/neural baseline on both decision quality and inference cost. Consider deployment to the live simulator only after reproducible improvements.

Increasing training volume is useful only if these checks show better decisions. Reference agreement alone can plateau at the limitations of the reference policy.
