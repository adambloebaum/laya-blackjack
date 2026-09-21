"""Bounded composition-focused experiment with selection frozen before final inference."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation import compare_evaluations
from .experiment_data import atomic_json, digest, verify_dataset
from .overnight import terminate_group
from .releases import verify_model


def compare_audits(candidate: Path, baseline: Path):
    """Paired uncertainty over whole games, conditional on saved reference estimates."""
    reports = [json.loads((p / "report.json").read_text()) for p in (candidate, baseline)]
    if reports[0]["dataset_sha256"] != reports[1]["dataset_sha256"]:
        raise ValueError("SDK audits use different final datasets.")
    rows = [json.loads((p / "decisions.json").read_text()) for p in (candidate, baseline)]
    if len(rows[0]) != len(rows[1]) or not rows[0]:
        raise ValueError("SDK audits must have identical nonempty decision sets.")
    groups = defaultdict(lambda: defaultdict(list))
    for c, b in zip(*rows, strict=True):
        if (c["index"], c["group_seed"], c["stratum"]) != (b["index"], b["group_seed"], b["stratum"]):
            raise ValueError("SDK audit decision identities differ.")
        delta = [c["regret"] - b["regret"], float(c["correct"]) - float(b["correct"])]
        for stratum in ("overall", c["stratum"], "teacher:" + c.get("teacher_kind", "monte_carlo")):
            groups[stratum][c["group_seed"]].append(delta)
    rng = np.random.default_rng(20260926)
    result = {}
    for stratum, by_game in groups.items():
        totals = np.array([[len(v), *np.sum(v, axis=0)] for v in by_game.values()])
        observed = totals.sum(axis=0)
        draws = []
        for _ in range(2000):
            sample = totals[rng.integers(0, len(totals), len(totals))].sum(axis=0)
            draws.append(sample[1:] / sample[0])
        bounds = np.quantile(draws, [0.025, 0.975], axis=0)
        result[stratum] = {
            "states": int(observed[0]),
            "games": len(totals),
            "regret_difference": observed[1] / observed[0],
            "regret_ci95": bounds[:, 0].tolist(),
            "agreement_difference": observed[2] / observed[0],
            "agreement_ci95": bounds[:, 1].tolist(),
        }
    return {
        "candidate_minus_release": result,
        "uncertainty": "Paired whole-game bootstrap, 2000 replicates; conditions on approximate teacher labels. "
        "Stratum intervals are exploratory and unadjusted for multiple comparisons.",
    }


def freeze_selection(output, source):
    candidates = []
    for gpu in (0, 1):
        path = output / f"gpu-{gpu}" / "candidate"
        report = json.loads((path / "training_report.json").read_text())
        if report.get("test_deferred") is not True or report["test"] is not None:
            raise ValueError("Targeted selection requires sealed final-test predictions.")
        candidates.append(
            {
                "path": str(path),
                "model_sha256": digest(path / "model.safetensors"),
                "config_sha256": digest(path / "rl_agent_config.json"),
                "selected": report["selected"],
                "baseline_selection": report["baseline_selection"],
                "objective": report.get("objective", "imitation"),
            }
        )
    selected = min(candidates, key=lambda c: c["selected"]["selection"]["teacher_ev_regret"])
    freeze = {
        "candidates": candidates,
        "selected": selected,
        "incumbent_sha256": digest(Path(source) / "model.safetensors"),
        "dataset_sha256": digest(output / "dataset" / "manifest.json"),
        "criterion": "Lowest equal-stratum selection EV regret among eligible epochs, including epoch-zero "
        "incumbent; depleted regret must improve and general regret may rise by at most 0.0005.",
        "improved_on_selection": selected["selected"]["epoch"] > 0,
        "test_predictions_started": False,
    }
    path = output / "selection-frozen.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if {k: v for k, v in saved.items() if k != "frozen_unix"} != freeze:
            raise ValueError("Frozen selection changed; final evaluation cannot resume.")
        return saved
    freeze["frozen_unix"] = time.time()
    atomic_json(path, freeze)
    return freeze


def finalize_candidate(output, freeze):
    """Create a separately evaluated checkpoint; keep the pre-test selection record immutable."""
    source = Path(freeze["selected"]["path"])
    if digest(source / "model.safetensors") != freeze["selected"]["model_sha256"]:
        raise ValueError("Selected model weights changed after selection was frozen.")
    if digest(source / "rl_agent_config.json") != freeze["selected"]["config_sha256"]:
        raise ValueError("Selected calibration changed after selection was frozen.")
    audits = {
        name: json.loads((output / "audit" / name / "report.json").read_text())
        for name in ("candidate", "baseline")
    }
    if audits["candidate"]["model_sha256"] != freeze["selected"]["model_sha256"]:
        raise ValueError("Final audit did not evaluate the selected weights.")
    if audits["baseline"]["model_sha256"] != freeze["incumbent_sha256"] or any(
        audit["dataset_sha256"] != freeze["dataset_sha256"] for audit in audits.values()
    ):
        raise ValueError("Final audit identities differ from the frozen experiment.")
    report = json.loads((source / "training_report.json").read_text())
    report.update(
        baseline=audits["baseline"]["metrics"],
        test=audits["candidate"]["metrics"],
        test_deferred=False,
        test_was_deferred=True,
        test_inference="Serving SDK, after global checkpoint selection was frozen",
        selection_freeze_sha256=digest(output / "selection-frozen.json"),
    )
    target = output / "evaluated-candidate"
    if target.exists():
        if json.loads((target / "training_report.json").read_text()) != report:
            raise ValueError("Existing evaluated candidate metadata changed.")
        if digest(target / "model.safetensors") != freeze["selected"]["model_sha256"]:
            raise ValueError("Existing evaluated candidate weights changed.")
        if digest(target / "rl_agent_config.json") != freeze["selected"]["config_sha256"]:
            raise ValueError("Existing evaluated candidate calibration changed.")
        return target
    staging = target.with_name(target.name + ".staging-" + uuid.uuid4().hex[:8])
    shutil.copytree(source, staging)
    atomic_json(staging / "training_report.json", report)
    if digest(staging / "model.safetensors") != freeze["selected"]["model_sha256"]:
        raise ValueError("Copied candidate failed integrity verification.")
    staging.rename(target)
    return target


def run_targeted(
    output: Path,
    source: str,
    hours=12.0,
    states=65536,
    selection=4096,
    calibration=2048,
    test=8192,
    workers=24,
    epochs=3,
    seed=20260924,
    fresh_units=100000,
    blocks=1000,
    smoke=False,
    study="composition",
):
    if not 0 < hours <= 12 or not 1 <= workers <= 32 or epochs < 1:
        raise ValueError("Use up to 12 hours, 1–32 workers, and positive epochs.")
    if study not in ("composition", "teacher-cost"):
        raise ValueError("Unknown targeted study.")
    if study == "teacher-cost":
        seed += 20000000  # Also separates return suites from previous experiments.
    source = str(Path(source).absolute())
    verify_model(Path(source))
    if smoke:
        states, selection, calibration, test, fresh_units, blocks, epochs = 128, 128, 64, 128, 20, 20, 1
        seed += 10000000  # Smoke-test inspection must never open the real final test.
    if any(n <= 0 or n % 128 for n in (states, selection, test)) or calibration < 1:
        raise ValueError("Training, selection and test need positive multiples of 128.")
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {
        "source": source,
        "source_manifest": digest(Path(source) / "manifest.json"),
        "hours": hours,
        "states": states,
        "selection": selection,
        "calibration": calibration,
        "test": test,
        "workers": workers,
        "epochs": epochs,
        "seed": seed,
        "fresh_units": fresh_units,
        "blocks": blocks,
        "smoke": smoke,
        "learning_rates": [5e-6, 5e-6] if study == "teacher-cost" else [2e-6, 5e-6],
        "objectives": ["imitation", "cost-sensitive"] if study == "teacher-cost" else ["imitation"] * 2,
        "teacher": "hybrid-exact-v1" if study == "teacher-cost" else "monte-carlo",
        "study": study,
        "general_margin": 0.0005,
        "code": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    run_path = output / "run.json"
    if run_path.exists():
        saved = json.loads(run_path.read_text())
        if saved["config"] != config:
            raise ValueError("Targeted run/source configuration changed; use its original source snapshot.")
        started = saved["started_unix"]
    else:
        started = time.time()
        atomic_json(
            run_path, {"config": config, "started_unix": started, "deadline_unix": started + hours * 3600}
        )
    deadline = started + hours * 3600
    current_stage = "starting"
    active = []

    def status(state="running", **extra):
        progress = {}
        for path in output.rglob("progress.json"):
            progress[str(path.parent.relative_to(output))] = json.loads(path.read_text())
        atomic_json(
            output / "status.json",
            {
                "status": state,
                "stage": current_stage,
                "experiment": study,
                "smoke": smoke,
                "started_unix": started,
                "updated_unix": time.time(),
                "deadline_unix": deadline,
                "elapsed_seconds": time.time() - started,
                "progress": progress,
                **extra,
            },
        )

    def interrupted(signum, frame):
        raise InterruptedError(f"Targeted experiment received signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}

    def stage(commands, until):
        logs = []
        try:
            if time.time() >= until:
                raise TimeoutError("Original targeted experiment stage budget expired.")
            for name, args in commands.items():
                log = (output / f"{name}.log").open("a")
                logs.append(log)
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "blackjack.cli", *args],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "4",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "4",
                        "LAYA_CPU_THREADS": "4",
                        "TOKENIZERS_PARALLELISM": "false",
                    },
                )
                active.append(process)
            while any(p.poll() is None for p in active):
                if any(p.poll() not in (None, 0) for p in active):
                    raise RuntimeError(f"{current_stage} process failed; inspect the corresponding log.")
                if time.time() >= until:
                    raise TimeoutError(f"{current_stage} reached its stage deadline.")
                status(pids=[p.pid for p in active])
                time.sleep(3)
            if any(p.returncode != 0 for p in active):
                raise RuntimeError(f"{current_stage} failed; complete artifacts are retained.")
        finally:
            for p in active:
                terminate_group(p)
            active.clear()
            for log in logs:
                log.close()

    try:
        current_stage = "generation"
        status()
        dataset = output / "dataset"
        if not (dataset / "manifest.json").exists():
            generation_until = started + hours * 3600 * 0.28
            stage(
                {
                    "generation": [
                        "generate-large",
                        "--output",
                        str(dataset),
                        "--profile",
                        "composition-v1",
                        "--teacher",
                        config["teacher"],
                        "--states",
                        str(states),
                        "--selection",
                        str(selection),
                        "--calibration",
                        str(calibration),
                        "--test",
                        str(test),
                        "--workers",
                        str(workers),
                        "--shard-size",
                        "64",
                        "--seed",
                        str(seed),
                        "--samples",
                        "16" if smoke else "2048",
                        "--max-samples",
                        "64" if smoke else "8192",
                        "--evaluation-samples",
                        "32" if smoke else "4096",
                        "--evaluation-max-samples",
                        "64" if smoke else "8192",
                        "--deadline",
                        str(generation_until),
                    ]
                },
                min(deadline, generation_until + 600),
            )
        manifest = verify_dataset(dataset)
        if not manifest["complete"]:
            raise TimeoutError("Targeted experiment requires the complete predeclared balanced dataset.")
        current_stage = "training"
        train_until = deadline - min(5400, hours * 3600 * 0.18)
        commands = {}
        for gpu, lr in enumerate(config["learning_rates"]):
            target = output / f"gpu-{gpu}"
            if (target / "candidate" / "training_report.json").exists():
                continue
            commands[f"training-gpu-{gpu}"] = [
                "train-large",
                "--dataset",
                str(dataset),
                "--output",
                str(target),
                "--source",
                source,
                "--device",
                f"cuda:{gpu}",
                "--epochs",
                str(epochs),
                "--batch-size",
                "8",
                "--learning-rate",
                str(lr),
                "--seed",
                str(seed),
                "--deadline",
                str(train_until),
                "--defer-test",
                "--general-margin",
                str(config["general_margin"]),
                "--objective",
                config["objectives"][gpu],
            ]
        if commands:
            stage(commands, min(deadline - 30, train_until + 300))
        freeze = freeze_selection(output, source)
        selected = freeze["selected"]["path"]
        current_stage = "sealed_test_evaluation"
        status(selection_frozen=True)
        commands = {}
        for gpu, (name, checkpoint) in enumerate([("baseline", source), ("candidate", selected)]):
            target = output / "audit" / name
            if (target / "report.json").exists():
                continue
            commands[f"audit-{name}"] = [
                "audit-model",
                "--dataset",
                str(dataset),
                "--source",
                checkpoint,
                "--output",
                str(target),
                "--device",
                f"cuda:{gpu}",
                "--allow-new-dataset",
            ]
        if commands:
            stage(commands, deadline - 30)
        if any(
            json.loads((output / "audit" / name / "report.json").read_text())["batched_action_mismatches"]
            for name in ("candidate", "baseline")
        ):
            raise ValueError("Serving and batched actions differ; investigate before paired return evaluation.")
        sdk = compare_audits(output / "audit" / "candidate", output / "audit" / "baseline")
        atomic_json(output / "sdk-comparison.json", sdk)
        current_stage = "paired_returns"
        commands = {}
        for name, checkpoint, device in [
            ("basic", None, "cpu"),
            ("baseline", source, "cuda:0"),
            ("candidate", selected, "cuda:1"),
        ]:
            target = output / "returns" / name
            commands[f"returns-{name}"] = [
                "evaluate-suite",
                "--output",
                str(target),
                "--policy",
                "laya" if checkpoint else "basic",
                "--device",
                device,
                "--fresh-units",
                str(fresh_units),
                "--blocks",
                str(blocks),
                "--seed",
                str(seed + 101),
            ] + (["--source", checkpoint] if checkpoint else [])
        stage(commands, deadline - 15)
        for mode in ("fresh", "continuous"):
            compare_evaluations(
                {name: output / "returns" / name / mode for name in ("basic", "baseline", "candidate")},
                output / f"{mode}-comparison.json",
            )
        # Published model and live dashboard remain the incumbent until this evidence is reviewed.
        evaluated = finalize_candidate(output, freeze)
        current_stage = "finished"
        result = {
            "selection": freeze,
            "sdk": sdk,
            "activated": False,
            "published": False,
            "smoke": smoke,
            "final_tests_used_for_selection": False,
            "evaluated_candidate": str(evaluated),
        }
        atomic_json(output / "summary.json", result)
        status(
            "complete",
            selected=freeze["selected"],
            improved_on_selection=freeze["improved_on_selection"],
            activated=False,
            reports=["sdk-comparison.json", "fresh-comparison.json", "continuous-comparison.json"],
        )
        return result
    except BaseException as exc:
        status(
            "interrupted" if isinstance(exc, (InterruptedError, TimeoutError)) else "failed", error=str(exc)
        )
        raise
    finally:
        for p in active:
            terminate_group(p)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()
