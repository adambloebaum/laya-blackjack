from copy import deepcopy

import pytest

from scripts.analyze_large_returns import audit_diagnostics, interval, scenario_diagnostics


def test_scenario_contributions_use_equal_weights_and_correct_family():
    reports = {}
    for mode in ("fresh", "continuous"):
        reports[mode] = {
            "identity": {
                "scenarios": {"a": {}, "b": {}},
                "rounds_per_unit": 100 if mode == "continuous" else 1,
            },
            "paired_candidate_minus": {
                name: {
                    "by_scenario": {
                        "a": {
                            "mean_units_per_round": 0.003,
                            "standard_error": 0.001,
                            "independent_units": 20,
                        },
                        "b": {
                            "mean_units_per_round": -0.001,
                            "standard_error": 0.002,
                            "independent_units": 20,
                        },
                    }
                }
                for name in ("basic", "baseline")
            },
        }
    result = scenario_diagnostics(reports)
    assert result["comparisons"] == 8
    selected = [r for r in result["rows"] if r["mode"] == "continuous" and r["comparator"] == "basic"]
    assert sum(r["contribution_to_aggregate_per_100"] for r in selected) == pytest.approx(0.1)
    assert selected[0]["difference_per_100"] == pytest.approx(0.3)
    assert selected[0]["nominal_ci95"] == pytest.approx([0.104, 0.496])
    assert selected[0]["exploratory_familywise_ci95"][0] < selected[0]["nominal_ci95"][0]
    assert selected[0]["independent_units"] == 20
    assert selected[0]["rounds_per_unit"] == 100


def test_same_population_error_cost_and_count_can_move_in_opposite_directions():
    common = {
        "group_seed": 123,
        "hand": [10, 6],
        "reference_action": "stand",
        "action_values": {"hit": -0.6, "stand": -0.5},
        "stratum": "general",
        "teacher_kind": "monte_carlo",
        "category": "hard",
        "depth": 1,
        "players": 3,
        "resolved": True,
    }
    old = [
        common | {"index": 0, "correct": True, "action": "stand", "regret": 0},
        common
        | {
            "index": 1,
            "correct": False,
            "action": "hit",
            "regret": 0.01,
            "action_values": {"hit": -0.51, "stand": -0.5},
        },
        common
        | {
            "index": 2,
            "correct": False,
            "action": "hit",
            "regret": 0.01,
            "action_values": {"hit": -0.51, "stand": -0.5},
        },
    ]
    new = [old[0] | {"correct": False, "action": "hit", "regret": 0.1}]
    new += [r | {"correct": True, "action": "stand", "regret": 0} for r in old[1:]]
    report = audit_diagnostics(new, old)
    assert report["candidate_errors"] < report["baseline_errors"]
    group = report["groups"]["stratum"]["general"]
    assert group["candidate_regret_sum"] > group["baseline_regret_sum"]
    assert group["candidate_regret_share"] == 1
    assert report["changed_actions"] == 3
    changed = deepcopy(old)
    changed[0]["action_values"]["stand"] = -0.3
    with pytest.raises(ValueError, match="reference targets differ"):
        audit_diagnostics(new, changed)
    with pytest.raises(ValueError, match="populations differ"):
        audit_diagnostics(new, old[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        audit_diagnostics([new[0], new[0]], [old[0], old[0]])


def test_selected_error_correction_does_not_reuse_aggregate_family():
    ordinary = interval(0.02, 0.008)
    adjusted = interval(0.02, 0.008, 25)
    assert ordinary[0] > 0
    assert adjusted[0] < 0
