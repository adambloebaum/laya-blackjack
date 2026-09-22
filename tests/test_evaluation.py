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


def test_interrupted_evaluation_resumes_identical_units_and_cumulative_fallbacks(tmp_path, monkeypatch):
    import blackjack.evaluation as evaluation

    class CountingPolicy(BasicPolicy):
        def __init__(self, *args):
            self.fallbacks = 0

        def actions(self, observations):
            self.fallbacks += len(observations)
            return super().actions(observations)

    monkeypatch.setattr(evaluation, "BatchedLaya", CountingPolicy)
    source = tmp_path / "model"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"model")
    (source / "rl_agent_config.json").write_text("{}")
    actual = simulate_units
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("simulated interruption")
        return actual(*args, **kwargs)

    monkeypatch.setattr(evaluation, "simulate_units", interrupt)
    options = dict(source=str(source), units=40, shard_size=2, seed=789)
    with pytest.raises(InterruptedError):
        evaluate_policy(tmp_path / "resume", **options)
    monkeypatch.setattr(evaluation, "simulate_units", actual)
    evaluate_policy(tmp_path / "resume", **options)
    evaluate_policy(tmp_path / "fresh", **options)
    for name in ("resume", "fresh"):
        _, rows = evaluation.read_evaluation(tmp_path / name)
        progress = json.loads((tmp_path / name / "progress.json").read_text())
        assert progress["serving_fallbacks"] == sum(r["decisions"] for r in rows.values())
    report = compare_evaluations(
        {"candidate": tmp_path / "resume", "baseline": tmp_path / "fresh"}, tmp_path / "paired.json"
    )
    assert report["paired_candidate_minus"]["baseline"]["ci95"] == [0, 0]
    with pytest.raises(ValueError, match="configuration changed"):
        evaluate_policy(tmp_path / "resume", **{**options, "seed": 790})
