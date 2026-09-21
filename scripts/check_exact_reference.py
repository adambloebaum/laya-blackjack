"""Reproducible development-only comparison against the existing engine rollouts."""

import argparse
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from blackjack.exact_reference import exact_analyze
from blackjack.experiment_data import atomic_json, digest, game_seed, sample_observations
from blackjack.reference import analyze


def check(task):
    seed, samples = task
    cases = []
    for obs in sample_observations(seed, False):
        start = time.monotonic()
        optimal = exact_analyze(obs)
        row = {"seed": seed, "result": optimal.get("reason", "exact"), "seconds": time.monotonic() - start}
        if optimal["available"]:
            fixed = exact_analyze(obs, continuation="basic")
            if not fixed["available"]:
                raise RuntimeError("Fixed-policy validation exceeded the optimal-policy computation budget.")
            monte_carlo = analyze(obs, samples, seed=seed + 999)
            comparisons = []
            for a, v in fixed["actions"].items():
                m = monte_carlo["actions"][a]
                comparisons.append(
                    {
                        "action": a,
                        "exact_basic": v["ev"],
                        "mc_basic": m["ev"],
                        "mc_se": m["standard_error"],
                        "optimal": optimal["actions"][a]["ev"],
                    }
                )
                if optimal["actions"][a]["ev"] < v["ev"] - 1e-10:
                    raise RuntimeError("Optimal continuation fell below fixed-policy value.")
            row.update(
                comparisons=comparisons,
                nodes=optimal["nodes"],
                changed_recommendation=optimal["recommendation"] != fixed["recommendation"],
                max_continuation_gain=max(
                    optimal["actions"][a]["ev"] - v["ev"] for a, v in fixed["actions"].items()
                ),
            )
        cases.append(row)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path; development reports are immutable.")
    started = time.monotonic()
    tasks = [
        (game_seed(20260928, "development", 0, n, "exact-check-v1"), args.samples) for n in range(args.games)
    ]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = [r for group in pool.map(check, tasks) for r in group]
    exact = [r for r in rows if r["result"] == "exact"]
    comparisons = [c for r in exact for c in r["comparisons"]]
    report = {
        "scope": "Development diagnostic, independent of all training/selection/calibration/final seeds; not model performance.",
        "games": args.games,
        "samples_per_mc_state": args.samples,
        "counts": dict(Counter(r["result"] for r in rows)),
        "seconds": time.monotonic() - started,
        "changed_recommendations": sum(r["changed_recommendation"] for r in exact),
        "max_continuation_gain": max((r["max_continuation_gain"] for r in exact), default=0),
        "fixed_policy_comparisons": len(comparisons),
        "fixed_policy_within_four_se": sum(
            abs(c["exact_basic"] - c["mc_basic"]) <= 4 * c["mc_se"] + 1e-10 for c in comparisons
        ),
        "uncertainty": "Four-standard-error Monte Carlo agreement is a diagnostic, not a simultaneous statistical guarantee. Exactness is additionally checked by independent exhaustive engine tests.",
        "source": {
            name: digest(Path("blackjack") / name)
            for name in ("engine.py", "reference.py", "exact_reference.py")
        },
        "cases": rows,
    }
    atomic_json(args.output, report)
    print({k: v for k, v in report.items() if k != "cases"})


if __name__ == "__main__":
    main()
