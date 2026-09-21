import json

import pytest

from blackjack.experiment_data import atomic_json
from blackjack.experiment_train import selection_eligible
from blackjack.targeted import compare_audits, freeze_selection


def metric(general, depleted):
    return {
        "teacher_ev_regret": (general + depleted) / 2,
        "by_stratum": {
            "general": {"teacher_ev_regret": general},
            "depleted": {"teacher_ev_regret": depleted},
        },
    }


def test_selection_guard_rejects_general_regression_and_requires_target_improvement():
    baseline = metric(0.001, 0.01)
    assert selection_eligible(metric(0.0013, 0.004), baseline, 0.0005)
    assert not selection_eligible(metric(0.002, 0.001), baseline, 0.0005)
    assert not selection_eligible(metric(0, 0.011), baseline, 0.0005)
    assert not selection_eligible(baseline, baseline, 0.0005)


def test_freeze_rejects_opened_final_test_and_modified_weights(tmp_path):
    source = tmp_path / "incumbent"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"incumbent")
    atomic_json(tmp_path / "dataset/manifest.json", {})
    for gpu in (0, 1):
        path = tmp_path / f"gpu-{gpu}/candidate"
        atomic_json(
            path / "training_report.json",
            {
                "test_deferred": True,
                "test": None,
                "baseline_selection": metric(0.001, 0.01),
                "selected": {"epoch": gpu + 1, "selection": metric(0.001, 0.006 - 0.002 * gpu)},
            },
        )
        (path / "model.safetensors").write_bytes(f"model-{gpu}".encode())
        atomic_json(path / "rl_agent_config.json", {})
    frozen = freeze_selection(tmp_path, str(source))
    assert frozen["selected"]["selected"]["epoch"] == 2
    assert freeze_selection(tmp_path, str(source)) == frozen
    path = tmp_path / "gpu-1/candidate/model.safetensors"
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="Frozen selection changed"):
        freeze_selection(tmp_path, str(source))
    path = tmp_path / "gpu-0/candidate/training_report.json"
    report = json.loads(path.read_text())
    report["test"] = {"accuracy": 0.9}
    atomic_json(path, report)
    with pytest.raises(ValueError, match="sealed"):
        freeze_selection(tmp_path, str(source))


def test_audit_comparison_pairs_games_and_rejects_mismatched_decisions(tmp_path):
    for name in ("a", "b"):
        atomic_json(tmp_path / name / "report.json", {"dataset_sha256": "same"})
        atomic_json(
            tmp_path / name / "decisions.json",
            [
                {"index": i, "group_seed": i // 2, "stratum": "general", "regret": float(i), "correct": False}
                for i in range(6)
            ],
        )
    result = compare_audits(tmp_path / "a", tmp_path / "b")
    assert result["candidate_minus_release"]["overall"]["regret_ci95"] == [0, 0]
    assert result["candidate_minus_release"]["overall"]["games"] == 3
    path = tmp_path / "b/decisions.json"
    rows = json.loads(path.read_text())
    rows[0]["group_seed"] = 50
    atomic_json(path, rows)
    with pytest.raises(ValueError, match="identities"):
        compare_audits(tmp_path / "a", tmp_path / "b")
