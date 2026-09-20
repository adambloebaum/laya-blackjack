# Agent notes

## Agent-managed project conventions

- Run from the repo root. Use `uv` and the project `.venv`; never install system packages with bare pip.
- The UI is static HTML/CSS/JS served by FastAPI. Node exists only for Playwright. No frontend build is required.
- Keep `README.md`, `docs/worklog.md`, and these notes current for behavioral changes. There is no database or schema.sql.
- `Game.observation()` is the only input to both Laya and reference evaluation. Never pass a Game into policy inference or include seeds, hidden ranks, or the true shoe in observations.
- Unseen counts include the hidden dealer card; peek-conditioned next-card probabilities subtract its posterior expected occupancy. Simple counts/total is wrong after a negative ace/ten peek.
- Aces use low value 1 in compact model state. All ten-valued ranks are combined. Hands retain actual rank/suit for display.
- `analyze` is Monte Carlo with basic continuation, not an exact optimal solver. Don't rename its action preference to an optimality guarantee or conflate model action preference with a win probability.
- The Laya SDK `confidence` for choice questions is normalized entropy, not the selected-label probability. Display `probabilities` directly. The SDK's `act_head` is unrelated to blackjack actions and is not used.
- Root SDK loading can fetch bundled checkpoints. `resolve_checkpoint` allow-lists root model/config/tokenizer/encoder files and pins a Hugging Face revision. Avoid broad snapshot downloads.
- Laya's `temperature_by_options` overrides per-type temperatures. Replace both when calibrating; `choice:6-10` is dealer, `choice:2`/`choice:3-5` are actions, `noul:2` is hit-bust.
- Laya sequence construction truncates silently. `ensure_fits` rejects overflow and is used by training and inference with the same 1024/256 context budgets.
- No train/validation/test game-seed overlap. Validation fits temperature; test only reports. Saved reports label probability distance to the approximate teacher, not empirical casino win calibration.
- Use new artifact directories; publish complete checkpoint directories by rename. Do not commit model binaries, datasets, secrets, or environment files.
- FastAPI routes are sync functions so CPU rollouts and PyTorch work run off the event loop. Lock table mutations and check revisions. Loading/prediction share a model lock; failed loads preserve the existing model.
- Session metadata, model loading, and jobs are in-process. Use one Uvicorn worker. The app is a localhost personal lab, not a multiuser deployment.
- Unit/card-conservation and hidden-state invariance tests are critical when changing engine or policy code. CI must work without Laya extras or checkpoint downloads.
- Shuffle and tablemate behavior use separate RNG streams. Replaying explicit actions does not call the behavior policy, so sharing its RNG with shuffling corrupts later shoes even when early replay rounds match.
- Large experiments use four group-disjoint splits: training, epoch/candidate selection, temperature calibration, and final test. Fixed pre-training test metrics are recorded for comparison but never drive selection, stopping, or hyperparameters within a run.
- Adaptive common-world Monte Carlo uses a three-standard-error stopping heuristic. Label it as a heuristic, not a formal confidence bound; noisy action targets receive lower training weight.
- Fast rollout cloning must preserve RNG state AND alias relationships. Regression tests compare complete public outcomes to deepcopy plus public-observation policy calls for all tablemate policies.
- Dataset shards have atomic completion receipts, content hashes, deterministic group seeds, and a source-code/parameter plan. Never resume with a changed plan. Token caches are bound to shard hashes and the tokenizer/context contract.
- Large optimizer checkpoints include model, optimizer, RNG, exact epoch/shard/batch cursor, and training/source hashes. Keep the venv interpreter path without resolving its symlink. Publish best weights plus selection records together through an atomic pointer; publish complete candidate directories by rename.
- Overnight artifacts live under artifacts/overnight, outside dashboard checkpoint discovery. A selection winner is a candidate, not automatic authorization to replace the live model. The existing API blocks additional experiment jobs while the overnight heartbeat is active or stale.
- Overnight supervision retains the original deadline when resumed. Production local launches additionally use systemd RuntimeMaxSec and KillMode=control-group; preserve an immutable source snapshot per run. No other GPU processes may be stopped.
