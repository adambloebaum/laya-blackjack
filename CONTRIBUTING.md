# Contributing

This is a research simulator. Contributions should improve reproducibility, decision quality, or the experience of exploring the table.

## Development

```bash
uv sync --extra dev --extra model --extra analysis
uv run --no-sync pytest -q
uv run --no-sync ruff check blackjack tests scripts
npm ci
npx playwright install chromium
npm test
```

The simulator and ordinary tests work without the `model` extra. CI never downloads weights. GPU/SDK checks are separate, recorded experiments. The browser UI is plain HTML, CSS, and JavaScript; Node is only needed for browser tests.

## Research contracts

- Policies receive `Game.observation()` only. Keep seeds, the concealed dealer card, and future shoe order outside model inputs.
- Preserve card conservation, rule behavior, and deterministic action replay when changing the engine.
- Keep whole games in one data split. Model selection, temperature calibration, and final evaluation have distinct roles.
- Compare policies on the same independent units. For continuous play, estimate uncertainty across independent blocks, not correlated rounds within a block.
- Identify the actual reference scope: the dashboard uses Monte Carlo with basic continuation; research datasets can use the restricted exact public-belief solver with an explicitly marked Monte Carlo fallback. Neither reference agreement nor model action preference is a casino win probability.
- Preserve complete model/data provenance. Do not commit weights, optimizer states, raw bulk datasets, access tokens, or local paths.

Add behavioral tests for rule, information-boundary, evaluation, or artifact-loading changes. Keep the README, usage guide, and worklog consistent with the final behavior. Include measured commands/results in a PR; do not generalize from a small smoke test.

For ideas, see the [research roadmap](docs/roadmap.md). For code conventions and artifact rules, see [CLAUDE.md](CLAUDE.md).
