"""Finish an already selected study through the serving SDK, preserving its failed run."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from .evaluation import compare_evaluations, read_evaluation
from .experiment_data import atomic_json, digest
from .overnight import terminate_group
from .targeted import compare_audits
from .visitation_data import ARMS, study_seed
from .visitation_study import (
    finalize_study_candidate,
    freeze_study_selection,
    inference_files,
    model_path,
    summarize_study_returns,
)


def freeze_recovery(study, output):
    if output == study or output.is_relative_to(study) or study.is_relative_to(output):
        raise ValueError("Recovery must use a separate directory outside the original run.")
    original = json.loads((study / "run.json").read_text())
    if time.time() >= original["deadline_unix"]:
        raise TimeoutError("Original study deadline expired; recovery cannot extend it.")
    if not (study / "selection-frozen.json").exists():
        raise ValueError("Both trained candidates must already be frozen before recovery.")
    prior_status = json.loads((study / "status.json").read_text())
    prior_reports = [json.loads(p.read_text()) for p in (study / "audit").glob("*/report.json")]
    if (
        prior_status["status"] != "failed"
        or prior_status["stage"] != "sealed_test_evaluation"
        or not any(r.get("batched_action_mismatches") for r in prior_reports)
        or any((study / "returns").glob("*/*/config.json"))
    ):
        raise ValueError("Recovery requires a recorded parity failure before return evaluation started.")
    frozen = freeze_study_selection(study)
    launch = json.loads((study / "launch.json").read_text())
    if digest(study / "source.tar") != launch["source_archive_sha256"] or any(
        digest(study / "source/blackjack" / name) != sha for name, sha in original["config"]["code"].items()
    ):
        raise ValueError("Original source snapshot changed.")
    historical = [study / n for n in ("run.json", "selection-frozen.json", "status.json", "source.tar")]
    historical += sorted(p for p in (study / "audit").rglob("*.json") if p.name != "progress.json")
    config = {
        "version": "visitation-sdk-recovery-v1",
        "study": str(study),
        "original_inputs": {str(p.relative_to(study)): digest(p) for p in historical},
        "inference_mode": "sdk",
        "selection": frozen,
        "study_config": original["config"],
        "code": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "runtime": {
            "python": sys.version,
            **{p: version(p) for p in ("laya", "torch", "transformers", "numpy")},
        },
    }
    if config["runtime"] != original["config"]["runtime"]:
        raise ValueError("Recovery must preserve the original inference runtime.")
    saved = {
        "config": config,
        "started_unix": original["started_unix"],
        "deadline_unix": original["deadline_unix"],
    }
    path = output / "run.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if {k: v for k, v in prior.items() if k != "recovery_started_unix"} != saved:
            raise ValueError("Recovery inputs or source changed; use its original source snapshot.")
        return prior
    saved["recovery_started_unix"] = time.time()
    atomic_json(path, saved)
    return saved


def read_sdk_audit(output, study, name, frozen):
    directory = output / "audit" / name
    completion = directory / "completion.json"
    if not completion.exists():
        return None
    identity = {
        "run_sha256": digest(output / "run.json"),
        "files": frozen["incumbent_files"] if name == "incumbent" else frozen["candidates"][name]["files"],
        "dataset_sha256": frozen["data"]["common_audit_dataset"],
    }
    receipt = json.loads(completion.read_text())
    if receipt["identity"] != identity or inference_files(model_path(study, name)) != identity["files"]:
        raise ValueError("SDK recovery audit input identity changed.")
    if not {"report.json", "decisions.json"} <= receipt["outputs"].keys() or any(
        digest(directory / path) != sha for path, sha in receipt["outputs"].items()
    ):
        raise ValueError("SDK recovery audit evidence changed.")
    report = json.loads((directory / "report.json").read_text())
    if (
        report["inference_mode"] != "sdk"
        or report["policy_action_mismatches"]
        or report["batched_action_mismatches"] is not None
        or report["model_sha256"] != identity["files"]["model.safetensors"]
        or report["dataset_sha256"] != identity["dataset_sha256"]
        or report["metrics"]["states"] != frozen["data"]["states_per_arm"]["test"]
    ):
        raise ValueError("Recovery audit did not validate the exact serving policy on every test state.")
    after = json.loads((directory / "decisions.json").read_text())
    if [r["index"] for r in after] != list(range(frozen["data"]["states_per_arm"]["test"])):
        raise ValueError("SDK recovery audit decisions are incomplete or reordered.")
    old = study / "audit" / name / "decisions.json"
    if old.exists():
        before = json.loads(old.read_text())
        if [(r["index"], r["group_seed"], r["action"]) for r in before] != [
            (r["index"], r["group_seed"], r["action"]) for r in after
        ]:
            raise ValueError("Repeated SDK actions differ from the preserved original SDK audit.")
    return report


def seal_sdk_audit(output, study, name, frozen):
    if read_sdk_audit(output, study, name, frozen) is not None:
        return
    directory = output / "audit" / name
    atomic_json(
        directory / "completion.json",
        {
            "identity": {
                "run_sha256": digest(output / "run.json"),
                "files": frozen["incumbent_files"]
                if name == "incumbent"
                else frozen["candidates"][name]["files"],
                "dataset_sha256": frozen["data"]["common_audit_dataset"],
            },
            "outputs": {
                p.name: digest(p)
                for p in directory.glob("*.json")
                if p.name not in ("progress.json", "completion.json")
            },
        },
    )
    read_sdk_audit(output, study, name, frozen)


def run_sdk_recovery(study: Path, output: Path):
    study, output = study.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (study / "supervisor.lock").open("a") as parent_lock, (output / "supervisor.lock").open("a") as lock:
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = freeze_recovery(study, output)
        _supervise(study, output, saved)


def _supervise(study, output, saved):
    config = saved["config"]["study_config"]
    frozen = saved["config"]["selection"]
    stage_name, active = "serving_sdk_validation", []

    def status(state="running", **extra):
        atomic_json(
            output / "status.json",
            {
                "status": state,
                "stage": stage_name,
                "experiment": "visitation-training",
                "inference_mode": "sdk",
                "recovery_of": study.name,
                "smoke": config["smoke"],
                "started_unix": saved["started_unix"],
                "deadline_unix": saved["deadline_unix"],
                "updated_unix": time.time(),
                "elapsed_seconds": time.time() - saved["started_unix"],
                "progress": {
                    str(p.relative_to(output)): json.loads(p.read_text())
                    for p in output.rglob("progress.json")
                },
                **extra,
            },
        )

    def interrupted(signum, frame):
        raise InterruptedError(f"SDK recovery received signal {signum}")

    def stage(commands):
        logs = []
        try:
            if time.time() >= saved["deadline_unix"] - 30:
                raise TimeoutError("Original study deadline expired.")
            for name, args in commands.items():
                log = (output / f"{name}.log").open("a")
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
                if time.time() >= saved["deadline_unix"] - 30:
                    raise TimeoutError("Original study deadline expired.")
                if any(p.poll() not in (None, 0) for p in active):
                    raise RuntimeError(f"{stage_name} worker failed; preserved logs contain details.")
                status()
                time.sleep(3)
            if any(p.returncode != 0 for p in active):
                raise RuntimeError(f"{stage_name} worker failed; inspect logs.")
        finally:
            for p in active:
                terminate_group(p)
            active.clear()
            for log in logs:
                log.close()

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        status()
        for names in (("control", "visited"), ("incumbent",)):
            commands = {}
            for gpu, name in enumerate(names):
                if read_sdk_audit(output, study, name, frozen) is None:
                    commands["audit-" + name] = [
                        "audit-model",
                        "--dataset",
                        str(study / "broad"),
                        "--source",
                        str(model_path(study, name)),
                        "--output",
                        str(output / "audit" / name),
                        "--device",
                        f"cuda:{gpu}",
                        "--allow-new-dataset",
                        "--inference-mode",
                        "sdk",
                    ]
            stage(commands)
            for name in names:
                seal_sdk_audit(output, study, name, frozen)
        comparisons = {}
        for new, old in (("visited", "control"), ("visited", "incumbent"), ("control", "incumbent")):
            report = compare_audits(output / "audit" / new, output / "audit" / old)
            report["candidate_minus_comparator"] = report.pop("candidate_minus_release")
            comparisons[f"{new}-minus-{old}"] = report
        atomic_json(output / "sdk-comparisons.json", comparisons)
        stage_name = "serving_sdk_returns"

        def returns(name, device):
            return [
                "evaluate-suite",
                "--output",
                str(output / "returns" / name),
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
                "--inference-mode",
                "sdk",
            ] + ([] if name == "basic" else ["--source", str(model_path(study, name))])

        stage(
            {
                name: returns(name, device)
                for name, device in (("basic", "cpu"), ("incumbent", "cuda:0"), ("visited", "cuda:1"))
            }
        )
        stage({"control": returns("control", "cuda:0")})
        freeze_recovery(study, output)  # Recheck all original evidence and the original deadline.
        audits = {name: read_sdk_audit(output, study, name, frozen) for name in ("incumbent", *ARMS)}
        for mode in ("fresh", "continuous"):
            paths = {name: output / "returns" / name / mode for name in ("basic", "incumbent", *ARMS)}
            for name, path in paths.items():
                manifest, _ = read_evaluation(path)
                rc = manifest["config"]
                expected_units = config["fresh_units"] if mode == "fresh" else config["blocks"]
                if (
                    rc["inference_mode"] != "sdk"
                    or rc["identity"]["seed"] != study_seed(config["seed"], "returns")
                    or rc["identity"]["units"] != expected_units
                ):
                    raise ValueError("Recovery return mode, seed, or counts differ from the frozen plan.")
                if name != "basic":
                    files = inference_files(model_path(study, name))
                    if (
                        rc["source_hash"] != files["model.safetensors"]
                        or rc["config_hash"] != files["rl_agent_config.json"]
                    ):
                        raise ValueError("Recovery returns used a changed policy.")
            compare_evaluations(paths, output / f"{mode}-comparison.json", candidate="visited")
        candidate = finalize_study_candidate(study, frozen, audits, destination=output)
        summary = {
            **summarize_study_returns(output),
            "selection": frozen,
            "original_study": study.name,
            "inference_mode": "sdk",
            "smoke": config["smoke"],
            "research_evidence": not config["smoke"],
            "evaluated_candidate": candidate,
            "deviation": "Original action-only batch parity failed. All Laya return policies use the exact serving SDK; no weights, calibration, selection, seeds, counts, or deadline changed. Original failure evidence is retained.",
            "activated": False,
            "published": False,
        }
        if time.time() >= saved["deadline_unix"]:
            raise TimeoutError("Original deadline expired during final verification.")
        atomic_json(output / "summary.json", summary)
        stage_name = "finished"
        status("complete", selected_name=frozen["selected_name"])
    except BaseException as exc:
        status(
            "interrupted" if isinstance(exc, (InterruptedError, TimeoutError)) else "failed", error=str(exc)
        )
        raise
    finally:
        for p in active:
            terminate_group(p)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
