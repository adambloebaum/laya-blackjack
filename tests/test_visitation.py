import json
import time
from copy import deepcopy
from dataclasses import asdict

import pytest

from blackjack.engine import Game, Rules, basic_action
from blackjack.evaluation import BasicPolicy, evaluation_seed
from blackjack.experiment_data import game_seed
from blackjack.model import model_state
from blackjack.visitation import (
    collect_groups,
    content_hash,
    group_path,
    identities,
    pilot_worker,
    read_record,
    replay_group,
    run_visitation,
    seeded,
    summarize_pilot,
    write_record,
)


def test_stronger_label_budget_is_frozen_without_changing_qualification(tmp_path, monkeypatch):
    runs = []
    monkeypatch.setattr("blackjack.visitation.checkpoint_identity", lambda source: {})
    monkeypatch.setattr("blackjack.visitation._supervise", lambda output, source, saved: runs.append(saved))
    common = {"output": tmp_path / "run", "source": str(tmp_path / "model"), "seed": 20261003}
    run_visitation(**common, label_budget="strong")
    assert runs[0]["config"]["samples"] == 2048
    assert runs[0]["config"]["max_samples"] == 16384
    assert runs[0]["config"]["qualification_thresholds"]["both_resolved_fraction"] == 0.8
    assert runs[0]["config"]["qualification_thresholds"]["resolved_repeat_agreement"] == 0.99
    run_visitation(**common, label_budget="strong")
    assert runs[0] == runs[1]
    with pytest.raises(ValueError, match="changed"):
        run_visitation(**common, label_budget="standard")
    with pytest.raises(ValueError, match="Unknown pilot label budget"):
        run_visitation(**common, label_budget="unbounded")


def config(states=5):
    return {
        "seed": 919,
        "scenarios": {"random": asdict(Rules(players=7, tablemate_policy="random"))},
        "rounds_per_group": 30,
        "states_per_group": states,
        "batch_size": 2,
        "deadline_unix": time.time() + 30,
    }


def test_reservoir_size_cannot_change_actions_shoes_or_profits():
    small, large = [], []
    collect_groups([("random", 0), ("random", 1)], None, config(2), "mixture", small.append)
    collect_groups([("random", 0), ("random", 1)], None, config(12), "mixture", large.append)
    for a, b in zip(small, large, strict=True):
        assert a["actions"] == b["actions"]
        assert a["final_observation"] == b["final_observation"]
        assert a["history"] == b["history"]
        assert len(a["rows"]) == 2 and len(b["rows"]) == 12
        assert replay_group(a) == 2
        assert replay_group(b) == 12
    # Explicit replay must not consume the random tablemates' RNG to reproduce later shoes.
    assert large[0]["final_observation"]["shoe_number"] > 1


def test_collector_policy_boundary_replay_and_metadata_separation():
    class PublicOnly(BasicPolicy):
        def actions(self, observations):
            for obs in observations:
                assert isinstance(obs, dict)
                assert not {"seed", "group_seed", "shoe", "rng", "behavior_rng"} & obs.keys()
                assert obs["dealer"][1] == "??"
                assert "group_seed" not in model_state(obs)
            return super().actions(observations)

    rows = []
    collect_groups([("random", 0)], PublicOnly(), config(), "laya", rows.append)
    original = rows[0]
    assert replay_group(original) == 5
    changed = deepcopy(original)
    changed["rows"][0]["observation"]["bankroll"] += 1
    changed["rows"][0]["observation_sha256"] = content_hash(changed["rows"][0]["observation"])
    with pytest.raises(ValueError, match="does not replay"):
        replay_group(changed)
    changed = deepcopy(original)
    changed["rows"][0]["event_index"] = len(changed["actions"]) + 1
    with pytest.raises(ValueError, match="missed sampled states"):
        replay_group(changed)


def test_hidden_state_changes_cannot_change_the_policy_input():
    game = Game(Rules(players=1), seed=42)
    game.deal()
    before = model_state(game.observation())
    action = basic_action(game.observation())
    game.dealer[1], game.shoe[0] = game.shoe[0], game.dealer[1]
    game.shoe.reverse()
    game.seed = 999
    assert model_state(game.observation()) == before
    assert basic_action(game.observation()) == action


def test_pilot_namespaces_exclude_prior_training_evaluation_and_smoke():
    pilot = seeded(20261002, "game", "s17-6d-1p", 0)
    assert pilot != seeded(30261002, "game", "s17-6d-1p", 0)
    assert pilot != evaluation_seed(20261002, "continuous", "s17-6d-1p", 0)
    assert pilot != game_seed(20261002, "test", 0, 0)
    assert pilot != seeded(20261002, "labels", "laya", "s17-6d-1p", 0, 0, 0)


def test_receipts_reject_tampering_changed_parent_and_partial_files(tmp_path):
    path = tmp_path / "group.json"
    identity = {"run": "frozen", "parent": "observations-v1"}
    path.write_text('{"partial":true}')
    assert read_record(path, identity) is None
    write_record(path, {"states": 20}, identity)
    assert read_record(path, identity) == {"states": 20}
    with pytest.raises(ValueError, match="identity changed"):
        read_record(path, identity | {"parent": "observations-v2"})
    path.write_text(json.dumps({"states": 19}))
    with pytest.raises(ValueError, match="identity changed"):
        read_record(path, identity)


def test_illegal_actor_output_and_expired_deadline_stop_collection():
    class Illegal:
        def actions(self, observations):
            return ["insurance"] * len(observations)

    with pytest.raises(ValueError, match="illegal action"):
        collect_groups([("random", 0)], Illegal(), config(), "laya", lambda group: None)
    with pytest.raises(TimeoutError, match="original deadline"):
        collect_groups(
            [("random", 0)], BasicPolicy(), config() | {"deadline_unix": 0}, "laya", lambda group: None
        )


def test_partial_collection_resume_preserves_batch_context(tmp_path, monkeypatch):
    class ShapeSensitive:
        def __init__(self, *args):
            pass

        def actions(self, observations):
            return ["stand" if len(observations) % 2 else basic_action(o) for o in observations]

    monkeypatch.setattr("blackjack.visitation.BatchedLaya", ShapeSensitive)
    c = config() | {"groups_per_scenario": 2}
    (tmp_path / "run.json").write_text(json.dumps({"config": c, "deadline_unix": c["deadline_unix"]}))
    pilot_worker(tmp_path, "collect", "laya", "cpu")
    a = group_path(tmp_path, "collect", "laya", "random", 0)
    b = group_path(tmp_path, "collect", "laya", "random", 1)
    original = [a.read_bytes(), b.read_bytes()]
    isolated = []
    collect_groups([("random", 1)], ShapeSensitive(), c, "laya", isolated.append)
    assert isolated[0]["actions"] != json.loads(original[1])["actions"]
    b.unlink()
    b.with_suffix(".receipt.json").unlink()
    pilot_worker(tmp_path, "collect", "laya", "cpu")
    assert [a.read_bytes(), b.read_bytes()] == original


@pytest.mark.parametrize("failure", [None, "sdk", "labels", "smoke"])
def test_qualification_is_separate_from_completion_and_smoke(tmp_path, failure):
    c = config() | {
        "groups_per_scenario": 1,
        "smoke": failure == "smoke",
        "checkpoint": {"model.safetensors": {"sha256": "frozen"}},
        "qualification_thresholds": {
            "both_resolved_fraction": 0.8,
            "resolved_repeat_agreement": 0.99,
            "depleted_fraction": 0,
            "states_per_category": 0,
        },
    }
    (tmp_path / "run.json").write_text(json.dumps({"config": c}))
    for policy in ("laya", "mixture"):
        groups = []
        collect_groups([("random", 0)], BasicPolicy(), c, policy, groups.append)
        group = groups[0]
        write_record(
            group_path(tmp_path, "collect", policy, "random", 0),
            group,
            identities(tmp_path, "collect", policy, "random", 0),
        )
        labels, predictions = [], []
        for row in group["rows"]:
            action = basic_action(row["observation"])
            alternative = next(a for a in row["observation"]["legal_actions"] if a != action)
            ref = {
                "teacher_kind": "monte_carlo",
                "recommendation": action,
                "actions": {
                    a: {"ev": 0 if a == action else -0.1} for a in row["observation"]["legal_actions"]
                },
                "label_quality": {"resolved": True},
                "dealer_unresolved": 0,
                "hit_bust": 0.2,
                "dealer": {"17": 1},
            }
            second = deepcopy(ref)
            if failure == "labels":
                second["recommendation"] = alternative
                second["actions"][alternative]["ev"] = 0.1
            base = {k: row[k] for k in ("event_index", "observation_sha256")}
            labels.append(base | {"references": [ref, second]})
            predictions.append(
                base
                | {
                    "sdk_action": action,
                    "batch_action": alternative if failure == "sdk" else action,
                    "hit_bust": 0.2,
                    "dealer": {"17": 1},
                }
            )
        for phase, rows in (("label", labels), ("audit", predictions)):
            write_record(
                group_path(tmp_path, phase, policy, "random", 0),
                {"rows": rows},
                identities(tmp_path, phase, policy, "random", 0),
            )
    report = summarize_pilot(tmp_path)
    assert report["qualified_for_training_design"] == (failure is None)
    assert report["development_only"] and not report["activated"]
    assert report["decision"] == (
        "execution_only"
        if failure == "smoke"
        else "prepare_matched_training"
        if failure is None
        else "revise_pilot"
    )
