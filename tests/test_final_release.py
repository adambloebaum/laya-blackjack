import json
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest

from blackjack.experiment_data import atomic_json, digest, generate_large_dataset
from blackjack.final_release import (
    audit_receipt,
    freeze_selection,
    paired_selection,
    release_audit,
    release_seed,
    run_final_release,
)
from blackjack.model_audit import audit_model
from blackjack.research import checkpoint_identity


def checkpoint(path, name):
    atomic_json(path / "training_report.json", {"test": {"teacher_ev_regret": 0.1}, "dataset_hash": "old"})
    atomic_json(path / "rl_agent_config.json", {"max_len": 1024, "head_max_len": 256})
    atomic_json(path / "encoder/config.json", {})
    atomic_json(path / "tokenizer/tokenizer.json", {})
    (path / "model.safetensors").write_bytes(name.encode())
    return path


def decisions(delta=0):
    return [
        {
            "index": i,
            "group_seed": i // 2,
            "stratum": "general" if i < 8 else "depleted",
            "regret": 0.01 + delta,
        }
        for i in range(16)
    ]


def test_selection_requires_clear_improvement_and_respects_general_guard():
    base = decisions()
    assert not paired_selection(base, base, seed=1, comparisons=4, repeats=400)["eligible"]
    assert paired_selection(decisions(-0.003), base, seed=1, comparisons=4, repeats=400)["eligible"]
    mixed = decisions(-0.005)
    for row in mixed[:8]:
        row["regret"] = 0.011
    result = paired_selection(mixed, base, seed=1, comparisons=4, repeats=400)
    assert result["regret_difference"] < 0 and not result["eligible"]
    wrong = deepcopy(base)
    wrong[0]["group_seed"] = 999
    with pytest.raises(ValueError, match="identities"):
        paired_selection(wrong, base, seed=1, comparisons=4, repeats=400)


def test_bootstrap_keeps_whole_games_and_equal_stratum_weights():
    base = decisions()
    candidate = decisions()
    # Strong within-game correlation; duplicating each decision cannot add independent games.
    for row in candidate:
        row["regret"] += (row["group_seed"] % 3 - 1) * 0.002
    original = paired_selection(candidate, base, seed=7, comparisons=4, repeats=400)

    def doubled(rows):
        return [dict(r, index=2 * i + j) for i, r in enumerate(rows) for j in range(2)]

    repeat = paired_selection(doubled(candidate), doubled(base), seed=7, comparisons=4, repeats=400)
    assert original == repeat
    single = paired_selection(candidate, base, seed=7, comparisons=1, repeats=400)
    assert original["familywise_ci95"][0] <= single["familywise_ci95"][0]
    assert original["familywise_ci95"][1] >= single["familywise_ci95"][1]


@pytest.fixture
def sealed(tmp_path):
    identities = {
        name: checkpoint_identity(checkpoint(tmp_path / "checkpoints" / name, name))
        for name in ("incumbent", "packaged", "a", "b", "c")
    }
    config = {
        "checkpoints": identities,
        "selection_states": 16,
        "test_states": 32,
        "seed": 20261008,
        "selection_bootstrap_repeats": 400,
        "selection_criterion": "frozen guard",
    }
    atomic_json(tmp_path / "run.json", {"config": config, "deadline_unix": time.time() + 300})
    atomic_json(
        tmp_path / "data/manifest.json",
        {
            "complete": True,
            "actual_states": {"selection": 16, "test": 32},
            "splits": {s: [] for s in ("train", "selection", "calibration", "test")},
        },
    )
    for name in identities:
        rows = decisions(-0.003 if name == "a" else 0)
        identity, _ = audit_receipt(tmp_path, "selection", name)
        directory = tmp_path / "selection" / name
        atomic_json(directory / "decisions.json", rows)
        atomic_json(
            directory / "report.json",
            {
                "inference_mode": "sdk",
                "policy_action_mismatches": [],
                "batched_action_mismatches": None,
                "split": "selection",
                "model_sha256": identities[name]["model.safetensors"]["sha256"],
                "dataset_sha256": digest(tmp_path / "data/manifest.json"),
                "metrics": {"states": 16, "teacher_ev_regret": rows[0]["regret"]},
            },
        )
        atomic_json(
            directory / "completion.json",
            {
                "identity": identity,
                "outputs": {f: digest(directory / f) for f in ("report.json", "decisions.json")},
            },
        )
    return tmp_path


def test_freeze_precedes_final_inference_and_binds_inputs(sealed):
    with pytest.raises(ValueError, match="require frozen"):
        release_audit(sealed, "final", "a", "cpu")
    first = freeze_selection(sealed)
    assert first["selected_name"] == "a" and freeze_selection(sealed) == first
    with pytest.raises(ValueError, match="Only frozen"):
        audit_receipt(sealed, "final", "b")
    p = sealed / "checkpoints/a/rl_agent_config.json"
    atomic_json(p, {"changed": True})
    with pytest.raises(ValueError, match="package changed"):
        freeze_selection(sealed)


def test_partial_selection_and_tampered_freeze_are_rejected(sealed):
    path = sealed / "selection/c/completion.json"
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="All nominated"):
        freeze_selection(sealed)
    path.write_bytes(original)
    frozen = freeze_selection(sealed)
    frozen["selected_name"] = "packaged"
    atomic_json(sealed / "selection-frozen.json", frozen)
    with pytest.raises(ValueError, match="decision changed"):
        freeze_selection(sealed)


def test_selection_audit_never_predicts_test_rows(tmp_path, monkeypatch):
    dataset = tmp_path / "data"
    manifest = generate_large_dataset(
        dataset,
        states=8,
        selection=8,
        calibration=4,
        test=8,
        workers=1,
        shard_size=4,
        samples=16,
        max_samples=16,
        evaluation_samples=16,
        evaluation_max_samples=16,
        profile="composition-v1",
        seed=42,
    )
    seen = []

    class Actor:
        fallbacks = 0

        def __init__(self, *args):
            self.agent = SimpleNamespace(predict=self.predict)

        def actions(self, observations):
            return ["stand"] * len(observations)

        def predict(self, state, questions):
            seen.append(state)
            return {
                "answers": {
                    "action": {
                        "choice": "stand",
                        "probabilities": {k: float(k == "stand") for k in questions["action"]["criteria"]},
                    },
                    "hit_bust": {"noul": 0.5},
                    "dealer": {"probabilities": {k: 1 / 7 for k in questions["dealer"]["criteria"]}},
                }
            }

    monkeypatch.setattr("blackjack.model_audit.ServingLaya", Actor)
    source = checkpoint(tmp_path / "model", "model")
    report = audit_model(
        dataset,
        str(source),
        tmp_path / "audit",
        device="cpu",
        allow_new_dataset=True,
        inference_mode="sdk",
        split="selection",
    )
    selection = [
        json.loads(line)["state"]
        for shard in manifest["splits"]["selection"]
        for line in (dataset / shard["path"]).read_text().splitlines()
    ]
    test = [
        json.loads(line)["state"]
        for shard in manifest["splits"]["test"]
        for line in (dataset / shard["path"]).read_text().splitlines()
    ]
    assert seen == selection and all(s not in test for s in seen)
    assert report["split"] == "selection" and report["trainer_report_metrics"] is None
    assert json.loads((tmp_path / "audit/progress.json").read_text())["split"] == "selection"


def test_plan_and_resume_bind_counts_candidates_and_namespaces(tmp_path, monkeypatch):
    from blackjack import final_release

    sources = {
        name: str(checkpoint(tmp_path / name, name)) for name in ("incumbent", "packaged", "a", "b", "c")
    }
    nomination = tmp_path / "candidates.json"
    atomic_json(nomination, sources)
    calls = []
    monkeypatch.setattr(final_release, "version", lambda name: "test")
    monkeypatch.setattr(final_release, "_supervise", lambda root, saved: calls.append(saved))
    args = dict(output=tmp_path / "run", candidates=nomination)
    run_final_release(**args)
    run_final_release(**args)
    assert calls[0] == calls[1]
    cfg = calls[0]["config"]
    assert (cfg["selection_states"], cfg["test_states"], cfg["fresh_units"], cfg["blocks"]) == (
        8192,
        16384,
        500000,
        5000,
    )
    assert calls[0]["deadline_unix"] - calls[0]["started_unix"] == 43200
    with pytest.raises(ValueError, match="configuration changed"):
        run_final_release(**args, seed=20261009)
    assert len({release_seed(s, p) for s in (20261008, 30261008) for p in ("data", "returns")}) == 4


def test_single_package_excludes_other_weights_and_rejects_rehashed_config(sealed):
    from blackjack.final_release import prepare_package
    from blackjack.releases import verify_model

    freeze = freeze_selection(sealed)
    identity, _ = audit_receipt(sealed, "final", "a")
    directory = sealed / "final/a"
    report = {
        "inference_mode": "sdk",
        "policy_action_mismatches": [],
        "batched_action_mismatches": None,
        "split": "test",
        "model_sha256": identity["files"]["model.safetensors"],
        "dataset_sha256": identity["dataset_sha256"],
        "metrics": {
            "states": 32,
            "teacher_ev_regret": 0.001,
            "teacher_agreement": 0.95,
            "hit_bust_brier": 0.001,
            "dealer_brier": 0.001,
        },
    }
    atomic_json(directory / "report.json", report)
    atomic_json(directory / "decisions.json", [])
    atomic_json(
        directory / "completion.json",
        {
            "identity": identity,
            "outputs": {n: digest(directory / n) for n in ("report.json", "decisions.json")},
        },
    )
    summary = {"comparisons": {}}
    for name in ("summary.json", "fresh-comparison.json", "continuous-comparison.json"):
        atomic_json(sealed / name, summary)
    prepare_package(sealed, freeze, summary)
    package = sealed / "release-package"
    assert (package / "model.safetensors").read_bytes() == b"a"
    assert len(list(package.rglob("*.safetensors"))) == 1
    assert json.loads((sealed / "release-review.json").read_text())["published"] is False
    import tarfile

    with tarfile.open(package / "evaluation/evaluation-records.tar.gz") as tar:
        assert not any("checkpoints/" in name or name.endswith(".safetensors") for name in tar.getnames())
    path = package / "rl_agent_config.json"
    atomic_json(path, {"max_len": 1024, "head_max_len": 256, "temperature": 2})
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["files"]["rl_agent_config.json"] = {"sha256": digest(path), "bytes": path.stat().st_size}
    atomic_json(package / "manifest.json", manifest)
    verify_model(package)  # Self-consistent inventory still violates the frozen selection.
    with pytest.raises(ValueError, match="inference files differ"):
        prepare_package(sealed, freeze, summary)
