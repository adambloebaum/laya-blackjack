import json
import math

import pytest

from blackjack.engine import Game, Rules, basic_action, tablemate_action
from blackjack.evaluation import (
    BasicPolicy,
    compare_evaluations,
    estimate,
    evaluate_policy,
    evaluation_seed,
    simulate_units,
)


@pytest.mark.parametrize("rounds", [1, 20])
@pytest.mark.parametrize("tablemate", ["basic", "random", "conservative"])
def test_batched_simulation_matches_serial_games(rounds, tablemate):
    from dataclasses import asdict

    rules = Rules(players=7, decks=2, tablemate_policy=tablemate)
    actual = simulate_units(
        BasicPolicy(),
        "example",
        asdict(rules),
        range(5),
        seed=87,
        mode="continuous",
        rounds_per_unit=rounds,
        batch_size=3,
    )
    for index, record in enumerate(actual):
        game = Game(rules, evaluation_seed(87, "continuous", "example", index))
        for _ in range(rounds):
            game.deal()
            while game.phase == "playing":
                game.step(basic_action(game.observation()) if game.active[0] == 0 else tablemate_action(game))
        assert record["profit"] == game.bankroll
        assert record["rounds"] == rounds
        assert record["shoes"] == game.shoe_number


def test_paired_comparison_uses_differences_and_rejects_mismatched_experiments(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    evaluate_policy(a, policy="basic", units=20, batch_size=3, shard_size=2)
    evaluate_policy(b, policy="basic", units=20, batch_size=2, shard_size=2)
    report = compare_evaluations({"candidate": a, "basic": b}, tmp_path / "comparison.json")
    assert report["paired_candidate_minus"]["basic"]["ci95"] == [0, 0]
    assert report["results"]["basic"]["ci95"][0] < report["results"]["basic"]["ci95"][1]
    manifest = b / "manifest.json"
    data = json.loads(manifest.read_text())
    data["config"]["identity"]["seed"] += 1
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identical evaluation units"):
        compare_evaluations({"candidate": a, "basic": b}, tmp_path / "invalid.json")
    data["config"]["identity"]["seed"] -= 1
    data["config"]["code"]["engine.py"] = "different-engine"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identical simulator and evaluator"):
        compare_evaluations({"candidate": a, "basic": b}, tmp_path / "invalid.json")


def test_estimate_uses_independent_units():
    result = estimate([-1, 1])
    assert result["mean_units_per_round"] == 0
    assert result["standard_error"] == pytest.approx(1)
    assert math.isclose(result["ci95"][1], 1.96)
