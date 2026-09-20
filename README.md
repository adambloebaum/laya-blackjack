# Laya Blackjack Laboratory

A local blackjack research sandbox with a finite-shoe simulator, an interactive table, real [Laya](https://huggingface.co/convaiinnovations/laya) inference, and a reproducible training/evaluation pipeline.

**Status:** working first release. A local full-model pilot reached 85% agreement with the approximate reference on 100 held-out states; see [measured results](docs/experiments.md). The reference is Monte Carlo policy improvement with basic-strategy continuations. Laya is an experimental learned approximation, not a proven optimal blackjack player.

![Laya blackjack dashboard with live model inference](docs/dashboard.png)

## Run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 (uv can install it). GPU is optional; CUDA greatly speeds training. No API key or hosted inference service is required.

```bash
uv sync --extra model --extra dev
uv run --no-sync blackjack serve
```

Open **http://127.0.0.1:8000**. Click **Load trained Laya** if you have a local checkpoint, or **Load base Laya** on a new installation. The first base-model load downloads roughly 0.8 GB of weights and caches them. Checkpoints are intentionally excluded from Git.

For the simulator alone, use `uv sync --extra dev`. Laya remains explicitly unavailable until model dependencies are installed and a checkpoint is loaded. The app never substitutes reference results for model outputs.

The dashboard includes:

- 1–7 players, 1/2/4/6/8 decks, fixed-seed shuffling, manual decisions, step and autoplay.
- A Laya/reference/basic-strategy selector for player 01. Tablemates use basic, random, or conservative play.
- Configurable S17/H17, 3:2 or 6:5 payout, double after split, late surrender, and penetration.
- Every hand, split, wager, exposed card, unseen rank count, Hi-Lo count, round result, and return trace.
- Laya action preferences, next-hit bust predictions, dealer finish probabilities, and inference latency.
- Reference action EVs with Monte Carlo intervals and net-positive/zero/negative outcome probabilities.
- Training and evaluation controls under **Experiments**, plus JSON replay export.

All inference and simulation are local. Font files may be fetched from Google Fonts; the interface has system fallbacks. The server is single-process and intended for localhost use, without remote authentication. Browser tables are isolated, expire after two idle hours, and reset on server restart. The dashboard retains the most recent 1,000 round summaries; the full action replay is retained for the session.

## Train

Start a run from the **Experiments** tab (full model, six epochs by default), or run a smaller decision-layer pilot through the CLI:

```bash
uv run --no-sync blackjack generate --states 500 --samples 256
uv run --no-sync blackjack train --device cuda:0 --epochs 3
```

This creates `artifacts/data/blackjack/{train,validation,test}.jsonl`, a data manifest, and an SDK-compatible checkpoint in `artifacts/checkpoints/blackjack/`. Output directories must be new to avoid overwriting experiments.

The CLI default freezes the encoder and trains the decision layers. The dashboard offers both modes and defaults to full-model training because it performed better in the pilot. To adapt the encoder too:

```bash
uv run --no-sync blackjack train --dataset artifacts/data/blackjack --output artifacts/checkpoints/blackjack-full --device cuda:0 --full-model --epochs 6 --learning-rate 0.00002
```

Use `--device cpu` for CPU training. Set `LAYA_DEVICE` in the shell to select the dashboard inference device. `LAYA_CPU_THREADS` defaults to 4. `.env.example` documents optional settings; the server does not automatically source it.

Data generation varies player count, deck count, rule sets, tablemate behavior, and shuffle penetration. It samples multiple decision points from each game. Whole game seeds stay together in one split: **train**, **validation for temperature fitting**, and untouched **test**. Manifest SHA-256 hashes are checked before training. Generation seeds and model revisions are saved.

There are three training questions per state:

| Question | Target | Meaning |
| --- | --- | --- |
| Action | One-hot argmax of estimated action EV | Imitate this approximate reference policy |
| Next-hit bust | Exact conditional rank probability | Bust on the next draw |
| Dealer finish | Monte Carlo distribution | Dealer final total if no more player cards are drawn |

Training uses supervised cross-entropy distillation, a proper probability scoring objective. It is **not a new implementation of upstream RLCD**. Temperature scaling uses only the validation split. Action calibration is against the teacher label, not the event of winning. Reports include teacher agreement, EV regret, probability Brier distance to reference targets, and teacher-action calibration error.

Checkpoint publication happens only after model files, tokenizer, config, and the evaluation report are complete. The dashboard loads the most recently completed local checkpoint. A completed run is not automatically activated; click **Load trained Laya** to switch.

### Larger local experiments

The overnight pipeline uses 24 simulation workers, then both local GPUs for two independent full-model candidates (learning rates 0.00001 and 0.000005, same seed). It targets **100,000 training states**, plus **2,000 selection**, **2,000 calibration**, and **5,000 final test** states from disjoint games. The current full-model pilot is the warm start.

```bash
uv run --no-sync blackjack overnight --output artifacts/overnight/my-run --source artifacts/checkpoints/blackjack-full --hours 12 --workers 24 --gpus 0,1
```

Run this under a durable process manager for unattended use; the CLI alone remains attached to its terminal. The provisioned local run uses a user systemd service with a hard 12-hour limit. **Experiments → Overnight experiment** displays progress across dashboard restarts. See [the scaling experiment](docs/scaling-experiment.md) for monitoring, resume commands, and evaluation details.

Generation stops scheduling new shards after 28% of the budget, preserving time for training and evaluation; the actual training set may be smaller than the target. Saved optimizer/RNG state and deterministic batches support recovery. A resumed run retains its original deadline. Completed overnight candidates stay outside dashboard checkpoint discovery until reviewed; the existing pilot remains the available live model.

## Evaluate and replay

```bash
uv run --no-sync blackjack benchmark --rounds 1000 --players 3 --samples 256 --model-path artifacts/checkpoints/blackjack-full
uv run --no-sync blackjack replay path/to/laya-session.json
```

Benchmarks compare basic strategy, reference, and (when supplied) Laya on independent fresh-shoe rounds, with paired initial seeds across policies. Different actions consume different cards, so later card trajectories can diverge. Round-level 95% normal intervals describe sampling variation; small runs are smoke tests and do not establish profitability. Live sessions do continue through shoes; benchmarks deliberately start fresh shoes for independent-round uncertainty estimates.

Replay re-applies recorded actions and verifies both the final public state and retained result history. Replay files contain the seed for reproducibility; the seed never enters model observations.

Tablemate behavior and shoe shuffles use separate random streams, so exported explicit actions reproduce later shuffles even when tablemates were playing randomly.

## Game and information contracts

- American hole-card game, exposed player hands, negative dealer peek before decisions. Insurance and side bets are omitted.
- Doubles on any first two cards. Split by card value, up to four hands. Split aces receive exactly one card, cannot resplit, and never get natural-blackjack payouts.
- Unit wagers and unlimited virtual bankroll; bet sizing is outside this release. Split/double profits are measured per **original** unit, not per final amount wagered.
- Shuffles happen only between rounds, at the cut threshold or when fewer than `min(shoe size, 12 × (players + 1))` cards remain. This conservative reserve dominates penetration for crowded small shoes.
- Rare mid-round exhaustion voids the entire round and returns bets. It never silently creates cards or reshuffles mid-hand.
- The unseen rank pool includes the concealed dealer card. Negative ace/ten peeks change the next-card marginal. Card counts use exposed cards only.
- Monte Carlo accepts only the public observation. It samples the hidden card and shoe **without replacement**, conditioned on the peek, and uses common sampled worlds for all candidate actions. There is no privileged-state reference mode.
- Reference continuation is a documented multi-deck basic-strategy heuristic. It is not exact for every rule combination, and finite Monte Carlo can choose the wrong action when values are close. Displayed EV intervals are per action and are not simultaneous intervals or guarantees on the chosen maximum.
- Dealer finish probabilities assume no additional player draws. Actual round trajectories include all tablemates. Action outcome bars aggregate the active seat’s split hands; zero outcomes include voids.
- Model state includes exposed hands, active hand, rules, and unseen rank counts. It excludes seed, future order, and hidden rank. Context overflow is rejected instead of silently truncating.

## Development

```bash
uv sync --extra model --extra dev
uv run --no-sync pytest -q
uv run --no-sync ruff check blackjack tests
npm ci
npx playwright install chromium
npm test
```

The static UI requires no build step or Node runtime. Node is only used for browser tests. Tests cover card conservation across table sizes, rule edge cases, seed replay, information leakage, conditional probabilities, API isolation and stale-write rejection, desktop/mobile layout, and controls. CI runs without GPU/model downloads. Real model inference and training are separately exercised locally.

```text
blackjack/engine.py       Rules, shoe, hands, payouts, public observations
blackjack/reference.py    Conditional hidden-world sampling and rollout EVs
blackjack/model.py        Pinned Laya loading, prompts, context checks, inference
blackjack/training.py     Data generation, distillation, calibration, benchmarks
blackjack/experiment_data.py  Parallel whole-shoe sampling, shard receipts, split verification
blackjack/experiment_train.py Streaming fine-tuning, optimizer recovery, selection/calibration/test
blackjack/overnight.py        Shared wall-clock budget and dual-GPU supervision
blackjack/server.py       Versioned browser tables, model loading, experiment jobs
blackjack/static/         Responsive dashboard (plain HTML/CSS/JavaScript)
docs/                    Design, worklog, and measured experiment reports
artifacts/               Local datasets, checkpoints, logs, screenshots (not in Git)
```

API documentation: http://127.0.0.1:8000/docs. Long jobs run in separate Python processes and log under `artifacts/runs/`. Keep the server running while dashboard experiments execute. No database is used.

See [experiment results](docs/experiments.md), [design](docs/design.md), and [worklog](docs/worklog.md). Laya and its pretrained weights are upstream Apache-2.0 works; this repository remains private and does not redistribute their weights.
