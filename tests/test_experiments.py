import json
import random
from copy import deepcopy

import pytest

from blackjack.engine import Game, Rules, basic_action, tablemate_action
from blackjack.experiment_data import (
    composition_focus,
    game_seed,
    generate_large_dataset,
    sample_observations,
    verify_dataset,
)
from blackjack.reference import analyze, clone_world, finish_world, sample_world


@pytest.mark.parametrize("policy", ["basic", "random", "conservative"])
def test_fast_rollouts_preserve_deepcopy_and_public_policy_results(policy):
    game = Game(Rules(players=7, tablemate_policy=policy), 42)
    for _ in range(10):
        game.deal()
        while game.phase == "playing":
            observation = game.observation()
            original = sample_world(observation, random.Random(7))
            untouched = original.observation()
            for action in game.legal_actions():
                expected = deepcopy(original)
                expected.step(action)
                while expected.phase == "playing":
                    expected.step(
                        basic_action(expected.observation())
                        if expected.active[0] == game.active[0]
                        else tablemate_action(expected)
                    )
                actual = clone_world(original)
                profit = finish_world(actual, action, game.active[0])
                assert profit == sum(h.profit for h in expected.hands[game.active[0]])
                assert actual.observation() == expected.observation()
                assert actual.rng.getstate() == expected.rng.getstate()
                assert original.observation() == untouched
            game.step(tablemate_action(game))


def test_adaptive_results_match_fixed_prefix():
    obs = sample_observations(42, True)[0]
    adaptive = analyze(obs, 16, seed=19, max_samples=64)
    quality = adaptive.pop("label_quality")
    assert adaptive == analyze(obs, adaptive["samples"], seed=19)
    assert quality["resolved"] == all(m["gap"] > 3 * m["standard_error"] for m in quality["margins"].values())
    with pytest.raises(ValueError):
        analyze(obs, 32, max_samples=16)


def test_generation_is_worker_independent_resumable_and_detects_tampering(tmp_path):
    options = dict(
        states=8,
        selection=4,
        calibration=4,
        test=4,
        shard_size=4,
        samples=16,
        max_samples=32,
        evaluation_samples=16,
        evaluation_max_samples=32,
    )
    first = generate_large_dataset(tmp_path / "one", workers=1, **options)
    second = generate_large_dataset(tmp_path / "two", workers=2, **options)
    for split in first["splits"]:
        assert [s["sha256"] for s in first["splits"][split]] == [s["sha256"] for s in second["splits"][split]]
    resumed = generate_large_dataset(tmp_path / "one", workers=2, **options)
    assert resumed["splits"] == first["splits"]
    assert verify_dataset(tmp_path / "one")["complete"]
    target = tmp_path / "one" / first["splits"]["train"][0]["path"]
    target.write_text(target.read_text() + json.dumps({"corrupted": True}) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_dataset(tmp_path / "one")
    with pytest.raises(ValueError, match="inconsistent"):
        generate_large_dataset(tmp_path / "one", workers=2, **options)


def test_composition_sampling_uses_reachable_public_states_and_separate_seeds():
    focused = []
    for seed in range(12):
        obs = sample_observations(seed, True, "depleted")
        assert all(composition_focus(s) for s in obs)
        assert all(s["rules"]["penetration"] == 0.85 for s in obs)
        focused.extend(obs)
    assert len(focused) > 20
    assert game_seed(42, "test", 0, 0) != game_seed(42, "test", 0, 0, "composition-v1")


def test_composition_strata_are_balanced_worker_independent_and_group_disjoint(tmp_path):
    options = dict(
        states=8,
        selection=8,
        calibration=4,
        test=8,
        shard_size=4,
        samples=16,
        max_samples=16,
        evaluation_samples=16,
        evaluation_max_samples=16,
        profile="composition-v1",
    )
    first = generate_large_dataset(tmp_path / "a", workers=1, **options)
    second = generate_large_dataset(tmp_path / "b", workers=2, **options)
    for split in first["splits"]:
        assert [s["sha256"] for s in first["splits"][split]] == [s["sha256"] for s in second["splits"][split]]
        strata = []
        for shard in first["splits"][split]:
            for line in (tmp_path / "a" / shard["path"]).read_text().splitlines():
                row = json.loads(line)
                strata.append(row["stratum"])
                if row["stratum"] == "depleted":
                    assert composition_focus(row["observation"])
        assert strata.count("depleted") == (0 if split == "calibration" else len(strata) // 2)
    assert verify_dataset(tmp_path / "a")["complete"]


def test_hybrid_teacher_records_coverage_and_uses_distinct_seed_namespace(tmp_path):
    options = dict(
        states=16,
        selection=16,
        calibration=8,
        test=16,
        shard_size=8,
        samples=16,
        max_samples=16,
        evaluation_samples=16,
        evaluation_max_samples=16,
        profile="composition-v1",
        teacher="hybrid-exact-v1",
    )
    a = generate_large_dataset(tmp_path / "a", workers=1, **options)
    b = generate_large_dataset(tmp_path / "b", workers=2, **options)
    exact = 0
    for split in a["splits"]:
        assert sum(a["teacher_counts"][split].values()) == a["actual_states"][split]
        assert [s["sha256"] for s in a["splits"][split]] == [s["sha256"] for s in b["splits"][split]]
        exact += a["teacher_counts"][split]["exact"]
        for shard in a["splits"][split]:
            row = json.loads((tmp_path / "a" / shard["path"]).read_text().splitlines()[0])
            assert row["reference"]["teacher_kind"] in ("exact", "monte_carlo")
    assert exact > 0
    assert verify_dataset(tmp_path / "a")["complete"]
    assert game_seed(42, "test", 0, 0, "composition-v1/hybrid-exact-v1") != game_seed(
        42, "test", 0, 0, "composition-v1"
    )
