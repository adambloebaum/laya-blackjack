"""Bounded local evaluation suite; keeps all raw paired units for reproduction."""

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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import NormalDist

from .evaluation import compare_evaluations, evaluate_policy, scenarios
from .experiment_data import atomic_json, digest
from .overnight import terminate_group
from .releases import verify_model


def evaluate_suite(
    output,
    policy,
    source=None,
    device="cuda:0",
    fresh_units=100000,
    blocks=1000,
    seed=20260922,
    inference_mode="batched",
):
    for mode, units, shard_size in [("fresh", fresh_units, 256), ("continuous", blocks, 10)]:
        evaluate_policy(
            output / mode,
            policy=policy,
            source=source,
            device=device,
            mode=mode,
            units=units,
            batch_size=32,
            shard_size=shard_size,
            seed=seed,
            inference_mode=inference_mode,
        )


def checkpoint_identity(source: Path):
    """Bind every inference input; reject sealed-test or incomplete checkpoints."""
    required = {
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "training_report.json",
    }
    if not all((source / name).is_file() for name in required):
        raise ValueError("Research requires complete local evaluated checkpoints.")
    report = json.loads((source / "training_report.json").read_text())
    if report.get("test_deferred") or not report.get("test"):
        raise ValueError("Research checkpoint final-test evaluation is still sealed or missing.")
    if (source / "manifest.json").exists():
        names = set(verify_model(source)["files"]) | {"manifest.json"}
    else:
        names = required | {
            str(p.relative_to(source))
            for folder in ("encoder", "tokenizer")
            for p in (source / folder).rglob("*")
            if p.is_file()
        }
    return {
        name: {"sha256": digest(source / name), "bytes": (source / name).stat().st_size}
        for name in sorted(names)
    }


def freeze_checkpoint(source: Path, destination: Path, expected):
    if destination.exists():
        if checkpoint_identity(destination) != expected:
            raise ValueError("Frozen checkpoint changed; refusing to mix policies.")
        return destination
    staging = destination.with_name(destination.name + ".staging-" + uuid.uuid4().hex[:8])
    for name in expected:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    if checkpoint_identity(staging) != expected:
        raise ValueError("Checkpoint changed while freezing its inference files.")
    staging.rename(destination)
    return destination


def freeze_run(output, config):
    """A resume cannot gain time, change game seeds, or change an input identity."""
    path = output / "run.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("config") != config:
            raise ValueError("Research configuration changed; resume from the original source snapshot.")
    else:
        if any(output.glob("*/fresh/config.json")) or (output / "status.json").exists():
            raise ValueError("Existing evaluation lacks this supervisor's frozen configuration.")
        now = time.time()
        saved = {"config": config, "started_unix": now, "deadline_unix": now + config["hours"] * 3600}
        atomic_json(path, saved)
    if time.time() >= saved["deadline_unix"]:
        raise TimeoutError("Original research deadline expired; it cannot be reset by resuming.")
    return saved


def comparison_summary(output):
    """Report the four planned comparisons with simultaneous family coverage."""
    comparisons = {}
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * 4))
    for mode in ("fresh", "continuous"):
        report = json.loads((output / f"{mode}-comparison.json").read_text())
        for baseline in ("basic", "baseline"):
            pair = report["paired_candidate_minus"][baseline]
            se = (pair["ci95"][1] - pair["ci95"][0]) / (2 * 1.96)
            mean = pair["mean_difference"]
            comparisons[f"{mode}:candidate-minus-{baseline}"] = {
                "units_per_100_rounds": mean * 100,
                "nominal_ci95": [v * 100 for v in pair["ci95"]],
                "familywise_ci95": [(mean - z * se) * 100, (mean + z * se) * 100],
            }
    return {
        "comparisons": comparisons,
        "uncertainty": "Normal intervals over independent rounds or 100-round blocks. Bonferroni correction over the four predeclared aggregate comparisons gives at least 95% family coverage under the normal approximation; individual corrected intervals have 98.75% coverage. Scenario intervals remain exploratory.",
        "activated": False,
        "published": False,
    }


def run_research(
    output: Path, candidate: str, baseline: str, fresh_units=100000, blocks=1000, hours=3.0, seed=20260922
):
    if not math.isfinite(hours) or not 0 < hours <= 12:
        raise ValueError("Use a budget of at most 12 hours.")
    if any(n < len(scenarios()) * 2 or n % len(scenarios()) for n in (fresh_units, blocks)):
        raise ValueError("Both modes require equal strata with at least two independent units each.")
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "supervisor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(output, candidate, baseline, fresh_units, blocks, hours, seed)
    finally:
        lock.close()


def _run_locked(output, candidate, baseline, fresh_units, blocks, hours, seed):
    sources = {name: Path(p).absolute() for name, p in [("candidate", candidate), ("baseline", baseline)]}
    runtime = {"python": sys.version}
    for name in ("laya", "torch", "transformers", "numpy"):
        try:
            runtime[name] = version(name)
        except PackageNotFoundError:
            runtime[name] = None
    config = {
        "version": 2,
        "sources": {name: str(p) for name, p in sources.items()},
        "checkpoints": {name: checkpoint_identity(p) for name, p in sources.items()},
        "fresh_units": fresh_units,
        "blocks": blocks,
        "hours": hours,
        "seed": seed,
        "scenarios": scenarios(),
        "batch_size": 32,
        "continuous_rounds_per_unit": 100,
        "code": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "runtime": runtime,
    }
    saved = freeze_run(output, config)
    started, deadline = saved["started_unix"], saved["deadline_unix"]
    processes, logs = {}, []

    def status(state, **extra):
        progress = {}
        for path in output.glob("*/*/progress.json"):
            progress[f"{path.parent.parent.name} / {path.parent.name}"] = json.loads(path.read_text())
        atomic_json(
            output / "status.json",
            {
                "status": state,
                "stage": "policy_evaluation",
                "experiment": "paired-returns",
                "started_unix": started,
                "deadline_unix": deadline,
                "updated_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "progress": progress,
                **extra,
            },
        )

    def interrupted(signum, frame):
        raise InterruptedError(f"Evaluation received signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        status("running", stage="freezing_checkpoints")
        frozen = {
            name: freeze_checkpoint(p, output / "checkpoints" / name, config["checkpoints"][name])
            for name, p in sources.items()
        }
        if time.time() >= deadline:
            raise TimeoutError("Original research deadline expired while freezing checkpoints.")
        for name, source, device in [
            ("basic", None, "cpu"),
            ("baseline", str(frozen["baseline"]), "cuda:0"),
            ("candidate", str(frozen["candidate"]), "cuda:1"),
        ]:
            command = [
                sys.executable,
                "-u",
                "-m",
                "blackjack.cli",
                "evaluate-suite",
                "--output",
                str(output / name),
                "--policy",
                "basic" if source is None else "laya",
                "--device",
                device,
                "--fresh-units",
                str(fresh_units),
                "--blocks",
                str(blocks),
                "--seed",
                str(seed),
            ]
            if source:
                command += ["--source", source]
            log = (output / f"{name}.log").open("a")
            logs.append(log)
            processes[name] = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "4",
                    "LAYA_CPU_THREADS": "4",
                },
            )
        while any(p.poll() is None for p in processes.values()):
            failed = {name: p.returncode for name, p in processes.items() if p.poll() not in (None, 0)}
            if failed:
                raise RuntimeError(f"Evaluation failed: {failed}; completed shards are retained.")
            if time.time() >= deadline:
                raise TimeoutError("Evaluation reached its wall-clock budget.")
            status("running", pids={name: p.pid for name, p in processes.items()})
            time.sleep(3)
        if any(p.returncode != 0 for p in processes.values()):
            raise RuntimeError("Evaluation process failed; inspect policy logs.")
        for name, path in frozen.items():
            if checkpoint_identity(path) != config["checkpoints"][name]:
                raise ValueError("Frozen checkpoint changed during evaluation.")
        status("running", stage="comparing_complete_units")
        for mode in ("fresh", "continuous"):
            if time.time() >= deadline:
                raise TimeoutError("Original research deadline expired before comparison completed.")
            for name in processes:
                manifest = json.loads((output / name / mode / "manifest.json").read_text())
                if name != "basic":
                    expected = config["checkpoints"][name]
                    if (
                        manifest["config"]["source_hash"] != expected["model.safetensors"]["sha256"]
                        or manifest["config"]["config_hash"] != expected["rl_agent_config.json"]["sha256"]
                    ):
                        raise ValueError("Evaluation used a different model or calibration.")
            compare_evaluations(
                {name: output / name / mode for name in processes}, output / f"{mode}-comparison.json"
            )
        summary = comparison_summary(output)
        summary["run_config_sha256"] = digest(output / "run.json")
        summary["manifests"] = {
            f"{name}/{mode}": digest(output / name / mode / "manifest.json")
            for name in processes
            for mode in ("fresh", "continuous")
        }
        atomic_json(output / "summary.json", summary)
        status("complete", reports=["fresh-comparison.json", "continuous-comparison.json", "summary.json"])
    except BaseException as exc:
        status(
            "interrupted" if isinstance(exc, (TimeoutError, InterruptedError)) else "failed", error=str(exc)
        )
        raise
    finally:
        for p in processes.values():
            terminate_group(p)
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
