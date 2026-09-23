import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict

import pytest

from blackjack.engine import Rules
from blackjack.evaluation import BasicPolicy
from blackjack.experiment_data import atomic_json, digest, generate_large_dataset
from blackjack.reference import analyze
from blackjack.research import checkpoint_identity
from blackjack.sdk_recovery import freeze_recovery, read_sdk_audit, seal_sdk_audit
from blackjack.visitation import collect_groups, group_path, identities, read_group, write_record
from blackjack.visitation_data import (
    ARMS,
    assemble_study,
    collection_config,
    label_pair,
    study_seed,
    verify_collection_serving,
    verify_matched_data,
)
from blackjack.visitation_study import (
    audit_study_arm,
    finalize_study_candidate,
    freeze_study_selection,
    read_audit,
    summarize_study_returns,
)


def write_checkpoint(path, report=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"incumbent")
    for name in ("rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"):
        atomic_json(path / name, {})
    atomic_json(path / "training_report.json", report or {"test": {"agreement": 1}})


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    source = tmp_path / "checkpoints/incumbent"
    write_checkpoint(source)
    config = {
        "seed": 20261004,
        "shared_states": 8,
        "heldout_states": {"selection": 8, "calibration": 4, "test": 8},
        "scenarios": {"two-player": asdict(Rules(players=2, decks=1))},
        "groups_per_scenario": 2,
        "rounds_per_group": 20,
        "states_per_group": 4,
        "samples": 16,
        "max_samples": 16,
        "shard_size": 4,
        "workers": 1,
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 5e-6,
        "checkpoint": checkpoint_identity(source),
    }
    deadline = time.time() + 600
    atomic_json(
        tmp_path / "run.json", {"config": config, "started_unix": time.time(), "deadline_unix": deadline}
    )
    c = collection_config(config)
    collection = tmp_path / "collection"
    atomic_json(collection / "run.json", {"config": c, "deadline_unix": deadline})
    groups = []
    collect_groups(
        [(s, i) for s in c["scenarios"] for i in range(2)],
        BasicPolicy(),
        c | {"deadline_unix": deadline},
        "laya",
        groups.append,
    )
    for original in groups:
        for policy in ARMS.values():
            group = deepcopy(original)
            group["policy"] = policy  # Identical synthetic arms force the label-sharing case.
            key = (policy, group["scenario"], group["index"])
            write_record(
                group_path(collection, "collect", *key), group, identities(collection, "collect", *key)
            )
        audit = {
            "rows": [
                {
                    "event_index": r["event_index"],
                    "observation_sha256": r["observation_sha256"],
                    "sdk_action": r["behavior_action"],
                    "batch_action": r["behavior_action"],
                }
                for r in original["rows"]
            ]
        }
        key = ("laya", original["scenario"], original["index"])
        write_record(group_path(collection, "audit", *key), audit, identities(collection, "audit", *key))
    calls = []

    def reference(obs, samples, seed, max_samples):
        calls.append(seed)
        return {**analyze(obs, samples, seed=seed, max_samples=max_samples), "teacher_kind": "monte_carlo"}

    monkeypatch.setattr("blackjack.visitation_data.hybrid_analyze", reference)
    for index in range(2):
        label_pair((tmp_path, "two-player", index))
    assert len(calls) == 8  # Not sixteen: paired identical observations have one label.
    generate_large_dataset(
        tmp_path / "broad",
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
        seed=study_seed(config["seed"], "broad"),
        profile="composition-v1",
        teacher="hybrid-exact-v1",
    )
    assemble_study(tmp_path)
    return tmp_path


def test_matched_rows_labels_schedule_and_resume_are_preserved(corpus):
    first = verify_matched_data(corpus)
    assert first["states_per_arm"]["train"] == 16
    assert first["planned_updates_per_arm"] == 16
    assert first["datasets"]["control"] != first["datasets"]["visited"]
    original = {
        str(p.relative_to(corpus)): digest(p) for p in (corpus / "datasets").rglob("*") if p.is_file()
    }
    report = assemble_study(corpus)
    assert report["shared_replacement_labels"] == 8
    assert report["unique_reference_calls"] == 8
    assert first == verify_matched_data(corpus)
    assert original == {
        str(p.relative_to(corpus)): digest(p) for p in (corpus / "datasets").rglob("*") if p.is_file()
    }
    assert verify_collection_serving(corpus) == {"states": 8, "action_mismatches": 0}


def test_label_receipts_reject_changed_parent_trajectories(corpus):
    root = corpus / "collection"
    key = ("mixture", "two-player", 0)
    record = read_group(root, "collect", *key)
    record["decisions"] += 1
    write_record(group_path(root, "collect", *key), record, identities(root, "collect", *key))
    with pytest.raises(ValueError, match="identity changed"):
        label_pair((corpus, "two-player", 0))


def test_collection_sdk_disagreement_cannot_enter_training(corpus):
    root = corpus / "collection"
    key = ("laya", "two-player", 0)
    audit = read_group(root, "audit", *key)
    audit["rows"][0]["sdk_action"] = "different"
    write_record(group_path(root, "audit", *key), audit, identities(root, "audit", *key))
    with pytest.raises(ValueError, match="actions differ"):
        verify_collection_serving(corpus)


def change_rows(root, manifest, shard, rows):
    path = root / shard["path"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    shard["sha256"] = digest(path)
    atomic_json(root / "manifest.json", manifest)


def test_coherent_group_leakage_and_changed_heldout_rows_are_rejected(corpus):
    root = corpus / "datasets/visited"
    manifest = json.loads((root / "manifest.json").read_text())
    test_shard = manifest["splits"]["test"][0]
    test_rows = [json.loads(s) for s in (root / test_shard["path"]).read_text().splitlines()]
    training_shard = manifest["splits"]["train"][-1]
    train_rows = [json.loads(s) for s in (root / training_shard["path"]).read_text().splitlines()]
    original = deepcopy(train_rows)
    train_rows[0]["group_seed"] = test_rows[0]["group_seed"]
    change_rows(root, manifest, training_shard, train_rows)
    with pytest.raises(ValueError, match="Group leakage"):
        verify_matched_data(corpus)
    change_rows(root, manifest, training_shard, original)
    test_rows[0]["reference"]["recommendation"] = "different"
    change_rows(root, manifest, test_shard, test_rows)
    with pytest.raises(ValueError, match="Held-out rows differ"):
        verify_matched_data(corpus)


def test_equal_state_counts_do_not_excuse_different_batch_schedules(corpus):
    root = corpus / "datasets/visited"
    manifest = json.loads((root / "manifest.json").read_text())
    shard = manifest["splits"]["train"].pop()
    rows = [json.loads(s) for s in (root / shard["path"]).read_text().splitlines()]
    for i, chunk in enumerate((rows[:1], rows[1:])):
        split = {**shard, "path": f"shards/split-{i}.jsonl", "states": len(chunk)}
        manifest["splits"]["train"].append(split)
        change_rows(root, manifest, split, chunk)
    with pytest.raises(ValueError, match="shard/batch"):
        verify_matched_data(corpus)


def metrics(general, depleted):
    return {
        "teacher_ev_regret": (general + depleted) / 2,
        "by_stratum": {
            "general": {"teacher_ev_regret": general},
            "depleted": {"teacher_ev_regret": depleted},
        },
    }


def prepare_candidates(root):
    config = json.loads((root / "run.json").read_text())["config"]
    data = verify_matched_data(root)
    baseline = metrics(0.002, 0.004)
    for i, arm in enumerate(ARMS):
        directory = root / "training" / arm
        selection = metrics(0.0018 - i * 0.0001, 0.0035 - i * 0.0005)
        history = [
            {
                "epoch": epoch,
                "updates": epoch * 8,
                "complete_epoch": True,
                "selection": baseline if epoch == 1 else selection,
            }
            for epoch in (1, 2)
        ]
        report = {
            "test_deferred": True,
            "test": None,
            "stopped_for_budget": False,
            "updates": 16,
            "history": history,
            "selected": history[-1],
            "baseline_selection": baseline,
            "dataset_hash": data["datasets"][arm],
        }
        write_checkpoint(directory / "candidate", report)
        (directory / "candidate/model.safetensors").write_bytes(arm.encode())
        atomic_json(
            directory / "config.json",
            {
                **{k: config[k] for k in ("epochs", "batch_size", "learning_rate", "seed")},
                "dataset_hash": data["datasets"][arm],
                "objective": "imitation",
                "general_margin": 0.0005,
                "source_hash": config["checkpoint"]["model.safetensors"]["sha256"],
                "source_config_hash": config["checkpoint"]["rl_agent_config.json"]["sha256"],
                "defer_test": True,
            },
        )


def test_shared_freeze_rejects_partial_training_and_opened_tests(corpus):
    prepare_candidates(corpus)
    label = next((corpus / "assembly/labels").glob("*.receipt.json"))
    original = label.read_bytes()
    label.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="collection/label evidence changed"):
        freeze_study_selection(corpus)
    label.write_bytes(original)
    path = corpus / "training/visited/candidate/training_report.json"
    report = json.loads(path.read_text())
    atomic_json(path, report | {"updates": 15})
    with pytest.raises(TimeoutError, match="entire matched"):
        freeze_study_selection(corpus)
    assert not (corpus / "selection-frozen.json").exists()
    atomic_json(path, report | {"test": {"agreement": 1}})
    with pytest.raises(ValueError, match="remain sealed"):
        freeze_study_selection(corpus)
    atomic_json(path, report)
    cache = corpus / "training/control/tokens/test-00000.pt"
    cache.parent.mkdir()
    cache.touch()
    with pytest.raises(ValueError, match="accessed final-test"):
        freeze_study_selection(corpus)
    cache.unlink()
    frozen = freeze_study_selection(corpus)
    assert frozen["selected_name"] == "visited"
    assert freeze_study_selection(corpus) == frozen
    atomic_json(corpus / "training/visited/candidate/rl_agent_config.json", {"changed": True})
    with pytest.raises(ValueError, match="selection or input artifacts changed"):
        freeze_study_selection(corpus)


def test_audit_requires_global_freeze_and_receipts_bind_calibration(tmp_path):
    atomic_json(tmp_path / "run.json", {"deadline_unix": time.time() + 60})
    with pytest.raises(ValueError, match="must be frozen"):
        audit_study_arm(tmp_path, "visited", "cpu")
    directory = tmp_path / "audit/visited"
    report = {"model_sha256": "weights", "dataset_sha256": "test", "batched_action_mismatches": []}
    atomic_json(directory / "report.json", report)
    atomic_json(directory / "decisions.json", [])
    identity = {
        "files": {"model.safetensors": "weights", "rl_agent_config.json": "calibration"},
        "dataset_sha256": "test",
    }
    atomic_json(
        directory / "completion.json",
        {
            "identity": identity,
            "outputs": {n: digest(directory / n) for n in ("report.json", "decisions.json")},
        },
    )
    assert read_audit(tmp_path, "visited", identity) == report
    with pytest.raises(ValueError, match="identity changed"):
        read_audit(
            tmp_path, "visited", identity | {"files": identity["files"] | {"rl_agent_config.json": "other"}}
        )
    atomic_json(directory / "decisions.json", [{"changed": True}])
    with pytest.raises(ValueError, match="evidence changed"):
        read_audit(tmp_path, "visited", identity)


def test_study_namespaces_and_four_predeclared_return_contrasts(tmp_path):
    seeds = {
        study_seed(root, family)
        for root in (20261004, 30261004)
        for family in ("broad", "replacement", "returns")
    }
    assert len(seeds) == 6
    for mode in ("fresh", "continuous"):
        atomic_json(
            tmp_path / f"{mode}-comparison.json",
            {
                "candidate": "visited",
                "paired_candidate_minus": {
                    name: {"mean_difference": 0.01, "ci95": [-0.0096, 0.0296]}
                    for name in ("control", "incumbent", "basic")
                },
            },
        )
    result = summarize_study_returns(tmp_path)
    assert len(result["comparisons"]) == 4
    assert all("basic" not in k for k in result["comparisons"])
    for contrast in result["comparisons"].values():
        assert contrast["units_per_100_rounds"] == 1
        assert contrast["familywise_ci95"][0] < contrast["nominal_ci95"][0]
        assert contrast["familywise_ci95"][1] > contrast["nominal_ci95"][1]


def prepare_recovery(corpus, monkeypatch):
    from blackjack.visitation_study import inference_files

    run = json.loads((corpus / "run.json").read_text())
    source = corpus / "source/blackjack/engine.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen original source\n")
    (corpus / "source.tar").write_bytes(b"archive")
    atomic_json(corpus / "launch.json", {"source_archive_sha256": digest(corpus / "source.tar")})
    run["config"]["code"] = {"engine.py": digest(source)}
    run["config"]["runtime"] = {
        "python": sys.version,
        **dict.fromkeys(("laya", "torch", "transformers", "numpy"), "test"),
    }
    atomic_json(corpus / "run.json", run)
    atomic_json(
        corpus / "status.json",
        {"status": "failed", "stage": "sealed_test_evaluation", "error": "batch/SDK parity"},
    )
    atomic_json(corpus / "audit/visited/report.json", {"batched_action_mismatches": [1]})
    monkeypatch.setattr("blackjack.sdk_recovery.version", lambda name: "test")
    prepare_candidates(corpus)
    frozen = freeze_study_selection(corpus)
    assert frozen["candidates"]["visited"]["files"] == inference_files(corpus / "training/visited/candidate")
    return frozen


def test_sdk_recovery_preserves_parent_deadline_source_and_failure(corpus, tmp_path_factory, monkeypatch):
    prepare_recovery(corpus, monkeypatch)
    output = tmp_path_factory.mktemp("sdk-recovery")
    saved = freeze_recovery(corpus, output)
    assert saved["deadline_unix"] == json.loads((corpus / "run.json").read_text())["deadline_unix"]
    assert saved["config"]["inference_mode"] == "sdk"
    assert freeze_recovery(corpus, output) == saved
    with pytest.raises(ValueError, match="separate directory"):
        freeze_recovery(corpus, corpus / "recovery")
    original = (corpus / "status.json").read_bytes()
    atomic_json(corpus / "status.json", json.loads(original) | {"error": "changed failure"})
    with pytest.raises(ValueError, match="inputs or source changed"):
        freeze_recovery(corpus, output)
    (corpus / "status.json").write_bytes(original)
    monkeypatch.setattr("blackjack.sdk_recovery.time.time", lambda: saved["deadline_unix"] + 1)
    with pytest.raises(TimeoutError, match="cannot extend"):
        freeze_recovery(corpus, output)


def test_sdk_recovery_audits_preserve_original_sdk_actions(corpus, tmp_path_factory, monkeypatch):
    frozen = prepare_recovery(corpus, monkeypatch)
    output = tmp_path_factory.mktemp("sdk-recovery")
    before = [{"index": i, "group_seed": 12, "action": "stand"} for i in range(8)]
    atomic_json(corpus / "audit/visited/decisions.json", before)
    freeze_recovery(corpus, output)
    directory = output / "audit/visited"
    report = {
        "model_sha256": frozen["candidates"]["visited"]["files"]["model.safetensors"],
        "dataset_sha256": frozen["data"]["common_audit_dataset"],
        "inference_mode": "sdk",
        "batched_action_mismatches": None,
        "policy_action_mismatches": [],
        "metrics": {"states": 8},
    }
    atomic_json(directory / "report.json", report)
    atomic_json(directory / "decisions.json", before)
    seal_sdk_audit(output, corpus, "visited", frozen)
    assert read_sdk_audit(output, corpus, "visited", frozen) == report
    (directory / "completion.json").unlink()
    atomic_json(directory / "decisions.json", before[:-1])
    with pytest.raises(ValueError, match="incomplete or reordered"):
        seal_sdk_audit(output, corpus, "visited", frozen)
    (directory / "completion.json").unlink()
    changed = deepcopy(before)
    changed[0]["action"] = "hit"
    atomic_json(directory / "decisions.json", changed)
    with pytest.raises(ValueError, match="preserved original SDK"):
        seal_sdk_audit(output, corpus, "visited", frozen)
    (directory / "completion.json").unlink()
    atomic_json(directory / "decisions.json", before)
    atomic_json(directory / "report.json", report | {"batched_action_mismatches": []})
    with pytest.raises(ValueError, match="exact serving policy"):
        seal_sdk_audit(output, corpus, "visited", frozen)


def test_finalized_recovery_candidate_keeps_original_training_sealed(corpus, tmp_path_factory):
    prepare_candidates(corpus)
    frozen = freeze_study_selection(corpus)
    output = tmp_path_factory.mktemp("finalized-recovery")
    audits = {name: {"metrics": {"states": 8}} for name in ("incumbent", "control", "visited")}
    assert finalize_study_candidate(corpus, frozen, audits, destination=output) == "evaluated-candidate"
    assert not (corpus / "evaluated-candidate").exists()
    original = json.loads((corpus / "training/visited/candidate/training_report.json").read_text())
    assert original["test_deferred"] and original["test"] is None
    finalized = json.loads((output / "evaluated-candidate/training_report.json").read_text())
    assert not finalized["test_deferred"] and finalized["test"] == {"states": 8}
    assert finalize_study_candidate(corpus, frozen, audits, destination=output) == "evaluated-candidate"
