"""Bounded two-GPU study of model-visited training data with sealed final tests."""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path
from statistics import NormalDist

from .evaluation import compare_evaluations, scenarios
from .experiment_data import atomic_json, digest
from .experiment_train import selection_eligible
from .model_audit import audit_model
from .overnight import terminate_group
from .research import checkpoint_identity, freeze_checkpoint, freeze_run
from .targeted import compare_audits
from .visitation import read_group
from .visitation_data import (
    ARMS,
    VERSION,
    collection_config,
    publish_bytes,
    study_seed,
    verify_collection_serving,
    verify_matched_data,
)


def inference_files(source):
    required = {
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
    }
    if not all((source / name).is_file() for name in required):
        raise ValueError("Candidate inference files are incomplete.")
    names = required | {
        str(p.relative_to(source))
        for folder in ("encoder", "tokenizer")
        for p in (source / folder).rglob("*")
        if p.is_file()
    }
    return {name: digest(source / name) for name in sorted(names)}


def freeze_study_selection(root):
    config = json.loads((root / "run.json").read_text())["config"]
    data = verify_matched_data(root)
    assembly = json.loads((root / "assembly/report.json").read_text())
    if any(assembly[key] != value for key, value in data.items()) or any(
        digest(root / name) != sha256 for name, sha256 in assembly["label_evidence"].items()
    ):
        raise ValueError("Assembled data or its collection/label evidence changed.")
    source = root / "checkpoints/incumbent"
    if checkpoint_identity(source) != config["checkpoint"]:
        raise ValueError("Frozen incumbent changed.")
    candidates = {}
    for arm in ARMS:
        directory = root / "training" / arm
        path = directory / "candidate"
        report = json.loads((path / "training_report.json").read_text())
        training = json.loads((directory / "config.json").read_text())
        if not report.get("test_deferred") or report.get("test") is not None:
            raise ValueError("Both final tests must remain sealed before selection.")
        if list((directory / "tokens").glob("test-*.pt")) or (directory / "baseline-test.json").exists():
            raise ValueError("Trainer accessed final-test predictions before the shared freeze.")
        expected = data["planned_updates_per_arm"]
        history = report["history"]
        if (
            report["stopped_for_budget"]
            or report["updates"] != expected
            or len(history) != config["epochs"]
            or any(
                not h["complete_epoch"] or h["updates"] != expected * (i + 1) // config["epochs"]
                for i, h in enumerate(history)
            )
        ):
            raise TimeoutError("Both arms must complete the entire matched training schedule.")
        if (
            report["dataset_hash"] != data["datasets"][arm]
            or training["dataset_hash"] != data["datasets"][arm]
            or training["source_hash"] != config["checkpoint"]["model.safetensors"]["sha256"]
            or training["source_config_hash"] != config["checkpoint"]["rl_agent_config.json"]["sha256"]
            or any(training[k] != config[k] for k in ("epochs", "batch_size", "learning_rate", "seed"))
            or training["objective"] != "imitation"
            or training["general_margin"] != 0.0005
            or not training["defer_test"]
        ):
            raise ValueError("Training inputs or settings differ from the frozen matched study.")
        selected = report["selected"]
        files = inference_files(path)
        if (
            selected["epoch"] == 0
            and files["model.safetensors"] != config["checkpoint"]["model.safetensors"]["sha256"]
        ):
            raise ValueError("Epoch-zero candidate is not the unchanged incumbent.")
        if selected["epoch"] and not selection_eligible(
            selected["selection"], report["baseline_selection"], 0.0005
        ):
            raise ValueError("Selected epoch violates the predeclared guard.")
        options = [report["baseline_selection"]["teacher_ev_regret"]] + [
            h["selection"]["teacher_ev_regret"]
            for h in history
            if selection_eligible(h["selection"], report["baseline_selection"], 0.0005)
        ]
        if selected["selection"]["teacher_ev_regret"] != min(options):
            raise ValueError("Selected checkpoint does not match the declared selection criterion.")
        candidates[arm] = {
            "files": files,
            "report_sha256": digest(path / "training_report.json"),
            "selected": selected,
            "baseline_selection": report["baseline_selection"],
            "updates": report["updates"],
        }
    choices = {"incumbent": candidates["control"]["baseline_selection"]["teacher_ev_regret"]}
    choices.update({arm: c["selected"]["selection"]["teacher_ev_regret"] for arm, c in candidates.items()})
    selected_name = min(choices, key=choices.get)  # Ties retain the original incumbent first.
    result = {
        "run_sha256": digest(root / "run.json"),
        "data": data,
        "assembly_sha256": digest(root / "assembly/report.json"),
        "candidates": candidates,
        "incumbent_files": inference_files(source),
        "selected_name": selected_name,
        "selection_regrets": choices,
        "criterion": "Lowest eligible selection regret, with unchanged incumbent retained on ties. "
        "Both arms completed identical update counts; final predictions remain sealed until this freeze.",
    }
    path = root / "selection-frozen.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if {k: v for k, v in saved.items() if k != "frozen_unix"} != result:
            raise ValueError("Frozen study selection or input artifacts changed.")
        return saved
    result["frozen_unix"] = time.time()
    atomic_json(path, result)
    return result


def model_path(root, name):
    return root / "checkpoints/incumbent" if name == "incumbent" else root / "training" / name / "candidate"


def audit_identity(root, name, frozen):
    if name not in ("incumbent", *ARMS):
        raise ValueError("Unknown study audit arm.")
    return {
        "selection_sha256": digest(root / "selection-frozen.json"),
        "dataset_sha256": frozen["data"]["common_audit_dataset"],
        "files": frozen["incumbent_files"] if name == "incumbent" else frozen["candidates"][name]["files"],
    }


def read_audit(root, name, identity):
    directory = root / "audit" / name
    path = directory / "completion.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt["identity"] != identity or not {"report.json", "decisions.json"} <= receipt["outputs"].keys():
        raise ValueError("Audit completion identity changed.")
    if any(digest(directory / name) != sha for name, sha in receipt["outputs"].items()):
        raise ValueError("Completed audit evidence changed.")
    report = json.loads((directory / "report.json").read_text())
    if (
        report["model_sha256"] != identity["files"]["model.safetensors"]
        or report["dataset_sha256"] != identity["dataset_sha256"]
    ):
        raise ValueError("Audit evaluated different weights or final-test rows.")
    if report["batched_action_mismatches"]:
        raise ValueError("Serving and batched final-test actions differ.")
    return report


def audit_study_arm(root: Path, name: str, device="cuda:0"):
    if time.time() >= json.loads((root / "run.json").read_text())["deadline_unix"]:
        raise TimeoutError("Original study deadline expired before the audit.")
    if not (root / "selection-frozen.json").exists():
        raise ValueError("Shared selection must be frozen before final inference.")
    frozen = freeze_study_selection(root)
    identity = audit_identity(root, name, frozen)
    existing = read_audit(root, name, identity)
    if existing is not None:
        return existing
    directory = root / "audit" / name
    audit_model(root / "broad", str(model_path(root, name)), directory, device=device, allow_new_dataset=True)
    if inference_files(model_path(root, name)) != identity["files"]:
        raise ValueError("Inference files changed during the SDK audit.")
    outputs = {
        p.name: digest(p)
        for p in directory.glob("*.json")
        if p.name not in ("progress.json", "completion.json")
    }
    atomic_json(directory / "completion.json", {"identity": identity, "outputs": outputs})
    return read_audit(root, name, identity)


def summarize_study_returns(root):
    z = NormalDist().inv_cdf(1 - 0.05 / 8)
    result = {}
    for mode in ("fresh", "continuous"):
        report = json.loads((root / f"{mode}-comparison.json").read_text())
        if report["candidate"] != "visited":
            raise ValueError("The predeclared contrasts require the model-visited arm.")
        for comparator in ("control", "incumbent"):
            pair = report["paired_candidate_minus"][comparator]
            mean = pair["mean_difference"]
            se = (pair["ci95"][1] - pair["ci95"][0]) / (2 * 1.96)
            result[f"{mode}:visited-minus-{comparator}"] = {
                "units_per_100_rounds": mean * 100,
                "nominal_ci95": [x * 100 for x in pair["ci95"]],
                "familywise_ci95": [(mean - z * se) * 100, (mean + z * se) * 100],
            }
    return {
        "comparisons": result,
        "uncertainty": "Bonferroni normal intervals for four predeclared aggregate contrasts, using "
        "independent fresh rounds or 100-round continuous blocks. Other contrasts/scenarios are exploratory.",
    }


def finalize_study_candidate(root, frozen, audits):
    name = frozen["selected_name"]
    if name == "incumbent":
        return None
    source = model_path(root, name)
    if inference_files(source) != frozen["candidates"][name]["files"]:
        raise ValueError("Selected candidate changed before finalization.")
    report = json.loads((source / "training_report.json").read_text())
    report.update(
        baseline=audits["incumbent"]["metrics"],
        test=audits[name]["metrics"],
        test_deferred=False,
        test_was_deferred=True,
        audit_dataset_hash=frozen["data"]["common_audit_dataset"],
        test_inference="Serving SDK on shared test rows after both arms and selection were frozen",
        selection_freeze_sha256=digest(root / "selection-frozen.json"),
        study_arm=name,
    )
    target = root / "evaluated-candidate"
    if not target.exists():
        staging = target.with_name(target.name + ".staging-" + uuid.uuid4().hex[:8])
        shutil.copytree(source, staging)
        atomic_json(staging / "training_report.json", report)
        if inference_files(staging) != frozen["candidates"][name]["files"]:
            raise ValueError("Finalized checkpoint copy changed.")
        staging.rename(target)
    if (
        inference_files(target) != frozen["candidates"][name]["files"]
        or json.loads((target / "training_report.json").read_text()) != report
    ):
        raise ValueError("Existing evaluated candidate changed.")
    return target.name


def run_visitation_training(
    output: Path, source: str, qualification: Path, workers=24, hours=12.0, seed=20261004, smoke=False
):
    if not math.isfinite(hours) or not 0 < hours <= 12 or not 1 <= workers <= 32:
        raise ValueError("Use up to 12 hours and 1–32 workers.")
    source = Path(source).absolute()
    checkpoint = checkpoint_identity(source)
    evidence = json.loads(qualification.read_text())
    report = evidence["report"]
    if (
        not report["qualified_for_training_design"]
        or report["smoke"]
        or not all(all(c.values()) for c in report["checks"].values())
        or evidence["config"]["checkpoint"] != checkpoint
    ):
        raise ValueError("Study requires a qualified full pilot of this exact source package.")
    config = {
        "version": VERSION,
        "source": str(source),
        "checkpoint": checkpoint,
        "qualification_sha256": digest(qualification),
        "hours": hours,
        "workers": workers,
        "seed": seed + (10000000 if smoke else 0),
        "smoke": smoke,
        "shared_states": 80 if smoke else 16000,
        "heldout_states": {
            "selection": 32 if smoke else 4096,
            "calibration": 16 if smoke else 2048,
            "test": 32 if smoke else 8192,
        },
        "groups_per_scenario": 2 if smoke else 80,
        "rounds_per_group": 20 if smoke else 100,
        "states_per_group": 4 if smoke else 20,
        "scenarios": scenarios(),
        "samples": 32 if smoke else 2048,
        "max_samples": 64 if smoke else 16384,
        "evaluation_samples": 32 if smoke else 4096,
        "shard_size": 8 if smoke else 64,
        "epochs": 1 if smoke else 3,
        "batch_size": 8,
        "learning_rate": 5e-6,
        "fresh_units": 20 if smoke else 100000,
        "blocks": 20 if smoke else 1000,
        "runtime": {
            "python": sys.version,
            **{p: version(p) for p in ("laya", "torch", "transformers", "numpy")},
        },
        "code": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = freeze_run(output, config)
        _supervise_study(output, saved)


def _supervise_study(root, saved):
    config, deadline, started = saved["config"], saved["deadline_unix"], saved["started_unix"]
    current_stage = "freezing_study"
    active = []

    def status(state="running", **extra):
        paths = list(root.rglob("progress.json")) + list((root / "collection/progress").glob("*.json"))
        progress = {str(p.relative_to(root)): json.loads(p.read_text()) for p in paths}
        atomic_json(
            root / "status.json",
            {
                "status": state,
                "stage": current_stage,
                "experiment": "visitation-training",
                "smoke": config["smoke"],
                "started_unix": started,
                "deadline_unix": deadline,
                "updated_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "progress": progress,
                **extra,
            },
        )

    def interrupted(signum, frame):
        raise InterruptedError(f"Matched study received signal {signum}")

    def stage(commands, until):
        logs = []
        try:
            if time.time() >= until:
                raise TimeoutError("Original study stage deadline expired.")
            for name, args in commands.items():
                log = (root / f"{name}.log").open("a")
                logs.append(log)
                active.append(
                    subprocess.Popen(
                        [sys.executable, "-u", "-m", "blackjack.cli", *args],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": "4",
                            "OPENBLAS_NUM_THREADS": "1",
                            "LAYA_CPU_THREADS": "4",
                            "TOKENIZERS_PARALLELISM": "false",
                        },
                    )
                )
            while any(p.poll() is None for p in active):
                if time.time() >= until:
                    raise TimeoutError("Original study stage deadline expired.")
                if any(p.poll() not in (None, 0) for p in active):
                    raise RuntimeError(f"{current_stage} worker failed; complete artifacts are retained.")
                status()
                time.sleep(3)
            if any(p.returncode != 0 for p in active):
                raise RuntimeError(f"{current_stage} worker failed; inspect its log.")
        finally:
            for p in active:
                terminate_group(p)
            active.clear()
            for log in logs:
                log.close()

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        status()
        incumbent = freeze_checkpoint(
            Path(config["source"]), root / "checkpoints/incumbent", config["checkpoint"]
        )
        freeze_checkpoint(incumbent, root / "collection/checkpoints/source", config["checkpoint"])
        collection_run = {
            "config": collection_config(config),
            "started_unix": started,
            "deadline_unix": deadline,
        }
        publish_bytes(root / "collection/run.json", (json.dumps(collection_run, indent=2) + "\n").encode())
        data_deadline = started + config["hours"] * 1800
        train_deadline = deadline - config["hours"] * 600
        current_stage = "collecting_training_data"
        commands = {
            f"collect-{arm}": [
                "visitation-worker",
                "--root",
                str(root / "collection"),
                "--phase",
                "collect",
                "--policy",
                policy,
                "--device",
                "cuda:1",
            ]
            for arm, policy in ARMS.items()
        }
        broad_path = root / "broad/manifest.json"
        if not broad_path.exists() or not json.loads(broad_path.read_text())["complete"]:
            commands["broad-data"] = [
                "generate-large",
                "--output",
                str(root / "broad"),
                "--profile",
                "composition-v1",
                "--teacher",
                "hybrid-exact-v1",
                "--states",
                str(config["shared_states"]),
                *[v for k, n in config["heldout_states"].items() for v in ("--" + k, str(n))],
                "--workers",
                str(config["workers"]),
                "--shard-size",
                str(config["shard_size"]),
                "--seed",
                str(study_seed(config["seed"], "broad")),
                "--samples",
                str(config["samples"]),
                "--max-samples",
                str(config["max_samples"]),
                "--evaluation-samples",
                str(config["evaluation_samples"]),
                "--evaluation-max-samples",
                str(config["max_samples"]),
                "--deadline",
                str(data_deadline),
            ]
        collected = all(
            read_group(root / "collection", "collect", policy, scenario, i) is not None
            for policy in ARMS.values()
            for scenario in config["scenarios"]
            for i in range(config["groups_per_scenario"])
        )
        already_generated = collected and "broad-data" not in commands
        stage(commands, deadline if already_generated else data_deadline)
        current_stage = "labeling_and_assembling_data"
        assembled = (root / "assembly/report.json").exists()
        stage(
            {
                "assembly": ["visitation-assemble", "--root", str(root)],
                "collection-audit": [
                    "visitation-worker",
                    "--root",
                    str(root / "collection"),
                    "--phase",
                    "audit",
                    "--policy",
                    "laya",
                    "--device",
                    "cuda:1",
                ],
            },
            deadline if assembled else data_deadline,
        )
        collection_audit = verify_collection_serving(root)
        verify_matched_data(root)
        current_stage = "training"
        commands = {}
        for gpu, arm in enumerate(ARMS):
            if (root / "training" / arm / "candidate/training_report.json").exists():
                continue
            commands[f"train-{arm}"] = [
                "train-large",
                "--dataset",
                str(root / "datasets" / arm),
                "--output",
                str(root / "training" / arm),
                "--source",
                str(incumbent),
                "--device",
                f"cuda:{gpu}",
                "--epochs",
                str(config["epochs"]),
                "--batch-size",
                str(config["batch_size"]),
                "--learning-rate",
                str(config["learning_rate"]),
                "--seed",
                str(config["seed"]),
                "--deadline",
                str(train_deadline),
                "--defer-test",
                "--general-margin",
                "0.0005",
                "--objective",
                "imitation",
            ]
        if commands:
            stage(commands, train_deadline)
        frozen = freeze_study_selection(root)
        current_stage = "sealed_test_evaluation"
        stage(
            {
                arm: ["visitation-audit", "--root", str(root), "--name", arm, "--device", f"cuda:{gpu}"]
                for gpu, arm in enumerate(ARMS)
            },
            deadline - 30,
        )
        stage(
            {
                "audit-incumbent": [
                    "visitation-audit",
                    "--root",
                    str(root),
                    "--name",
                    "incumbent",
                    "--device",
                    "cuda:0",
                ]
            },
            deadline - 30,
        )
        audits = {
            name: read_audit(root, name, audit_identity(root, name, frozen)) for name in ("incumbent", *ARMS)
        }
        sdk = {}
        for new, old in (("visited", "control"), ("visited", "incumbent"), ("control", "incumbent")):
            comparison = compare_audits(root / "audit" / new, root / "audit" / old)
            comparison["candidate_minus_comparator"] = comparison.pop("candidate_minus_release")
            sdk[f"{new}-minus-{old}"] = comparison
        atomic_json(root / "sdk-comparisons.json", sdk)
        current_stage = "paired_returns"

        def returns_command(name, device):
            return [
                "evaluate-suite",
                "--output",
                str(root / "returns" / name),
                "--policy",
                "basic" if name == "basic" else "laya",
                "--device",
                device,
                "--fresh-units",
                str(config["fresh_units"]),
                "--blocks",
                str(config["blocks"]),
                "--seed",
                str(study_seed(config["seed"], "returns")),
            ] + ([] if name == "basic" else ["--source", str(model_path(root, name))])

        stage(
            {
                name: returns_command(name, device)
                for name, device in (("basic", "cpu"), ("incumbent", "cuda:0"), ("visited", "cuda:1"))
            },
            deadline - 30,
        )
        stage({"returns-control": returns_command("control", "cuda:0")}, deadline - 30)
        frozen = freeze_study_selection(root)
        for mode in ("fresh", "continuous"):
            inputs = {name: root / "returns" / name / mode for name in ("basic", "incumbent", *ARMS)}
            for name, directory in inputs.items():
                if name == "basic":
                    continue
                saved_return = json.loads((directory / "manifest.json").read_text())["config"]
                files = audit_identity(root, name, frozen)["files"]
                if (
                    saved_return["source_hash"] != files["model.safetensors"]
                    or saved_return["config_hash"] != files["rl_agent_config.json"]
                ):
                    raise ValueError("Return evaluation used different frozen weights or calibration.")
            compare_evaluations(inputs, root / f"{mode}-comparison.json", candidate="visited")
        candidate = finalize_study_candidate(root, frozen, audits)
        summary = {
            **summarize_study_returns(root),
            "selection": frozen,
            "collection_audit": collection_audit,
            "evaluated_candidate": candidate,
            "smoke": config["smoke"],
            "research_evidence": not config["smoke"],
            "activated": False,
            "published": False,
        }
        if time.time() >= deadline:
            raise TimeoutError("Original deadline expired during final verification.")
        atomic_json(root / "summary.json", summary)
        current_stage = "finished"
        status(
            "complete",
            selected_name=frozen["selected_name"],
            reports=["summary.json", "sdk-comparisons.json"],
        )
    except BaseException as exc:
        status(
            "interrupted" if isinstance(exc, (TimeoutError, InterruptedError)) else "failed", error=str(exc)
        )
        raise
    finally:
        for p in active:
            terminate_group(p)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
