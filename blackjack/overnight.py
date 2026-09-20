"""Wall-clock-budgeted supervisor for local parallel simulation and GPU experiments."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .experiment_data import atomic_json


def terminate_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run_overnight(
    output: Path,
    source: str,
    hours=12.0,
    states=100000,
    workers=24,
    epochs=3,
    batch_size=8,
    gpus="0,1",
    seed=20260921,
    selection=2000,
    calibration=2000,
    test=5000,
    benchmark_rounds=2000,
):
    if not 0 < hours <= 12 or not 1 <= workers <= 32:
        raise ValueError("Use up to 12 hours and 1–32 workers.")
    gpu_ids = [int(x) for x in gpus.split(",")]
    if len(gpu_ids) not in (1, 2) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("Select one or two distinct local GPUs.")
    if not (Path(source) / "model.safetensors").exists():
        raise ValueError("Overnight training requires an existing local source checkpoint.")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {
        "hours": hours,
        "states": states,
        "workers": workers,
        "epochs": epochs,
        "batch_size": batch_size,
        "gpus": gpu_ids,
        "seed": seed,
        "source": source,
        "selection": selection,
        "calibration": calibration,
        "test": test,
        "benchmark_rounds": benchmark_rounds,
    }
    config_file = output / "run.json"
    if config_file.exists():
        saved = json.loads(config_file.read_text())
        if saved["config"] != config:
            raise ValueError("Run configuration changed; start a new experiment directory.")
        started = saved["started_unix"]
    else:
        started = time.time()
        atomic_json(
            config_file,
            {
                "config": config,
                "started_unix": started,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "deadline_unix": started + hours * 3600,
            },
        )
    deadline = started + hours * 3600
    status_path = output / "status.json"
    active = []
    current_stage = "starting"

    def status(state="running", **extra):
        detail = {}
        for path in [output / "dataset" / "progress.json"] + list(output.glob("gpu-*/progress.json")):
            if path.exists():
                detail[path.parent.name] = json.loads(path.read_text())
        atomic_json(
            status_path,
            {
                "status": state,
                "stage": current_stage,
                "started_unix": started,
                "deadline_unix": deadline,
                "updated_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "progress": detail,
                **extra,
            },
        )

    def interrupted(signum, frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}

    def stage(commands, until):
        processes, logs = {}, []
        try:
            for name, command in commands.items():
                log = (output / f"{name}.log").open("a")
                logs.append(log)
                env = {
                    **os.environ,
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "TOKENIZERS_PARALLELISM": "false",
                }
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                processes[name] = process
                active.append(process)
            while any(p.poll() is None for p in processes.values()):
                if time.time() >= until:
                    for process in processes.values():
                        terminate_group(process)
                    break
                status(pids={name: p.pid for name, p in processes.items()})
                time.sleep(3)
            return {name: p.poll() for name, p in processes.items()}
        finally:
            for process in processes.values():
                terminate_group(process)
                if process in active:
                    active.remove(process)
            for log in logs:
                log.close()

    # Keep the venv entrypoint string: resolving its symlink loses the venv site-packages.
    base = [sys.executable, "-u", "-m", "blackjack.cli"]
    try:
        if time.time() >= deadline:
            raise TimeoutError("The original 12-hour run budget has expired.")
        current_stage = "generation"
        status()
        dataset = output / "dataset"
        if not (dataset / "manifest.json").exists():
            generation_deadline = started + hours * 3600 * 0.28
            command = base + [
                "generate-large",
                "--output",
                str(dataset),
                "--states",
                str(states),
                "--workers",
                str(workers),
                "--shard-size",
                "64",
                "--selection",
                str(selection),
                "--calibration",
                str(calibration),
                "--test",
                str(test),
                "--seed",
                str(seed),
                "--samples",
                "1024",
                "--max-samples",
                "4096",
                "--evaluation-samples",
                "4096",
                "--evaluation-max-samples",
                "8192",
                "--deadline",
                str(generation_deadline),
            ]
            result = stage(
                {"generation": command},
                min(deadline - min(300, hours * 3600 * 0.1), generation_deadline + 600),
            )
            if result["generation"] != 0:
                raise RuntimeError(f"Generation did not complete ({result}); shard receipts are preserved.")
        current_stage = "training"
        train_until = deadline - min(1200, hours * 3600 * 0.08)
        commands = {}
        for index, gpu in enumerate(gpu_ids):
            commands[f"training-gpu-{gpu}"] = base + [
                "train-large",
                "--dataset",
                str(dataset),
                "--output",
                str(output / f"gpu-{gpu}"),
                "--source",
                source,
                "--device",
                f"cuda:{gpu}",
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--learning-rate",
                str(0.00001 if index == 0 else 0.000005),
                "--seed",
                str(seed),
                "--deadline",
                str(train_until),
            ]
        training_results = stage(commands, deadline - min(180, hours * 3600 * 0.05))
        candidates = []
        for gpu in gpu_ids:
            path = output / f"gpu-{gpu}" / "candidate" / "training_report.json"
            if path.exists():
                report = json.loads(path.read_text())
                candidates.append(
                    {
                        "path": str(path.parent),
                        "selection_regret": report["selected"]["selection"]["teacher_ev_regret"],
                        "test": report["test"],
                        "updates": report["updates"],
                    }
                )
        if not candidates:
            raise RuntimeError(f"No complete candidate; resumable checkpoints may exist. {training_results}")
        winner = min(candidates, key=lambda c: c["selection_regret"])
        atomic_json(
            output / "comparison.json",
            {
                "candidates": candidates,
                "selected": winner,
                "criterion": "selection-split EV regret; final test not used to select",
                "activated": False,
                "training_exit_codes": training_results,
            },
        )
        current_stage = "return_evaluation"
        evaluation = None
        if deadline - time.time() > 180:
            commands = {
                "return-evaluation": base
                + [
                    "benchmark",
                    "--output",
                    str(output / "returns.json"),
                    "--rounds",
                    str(benchmark_rounds),
                    "--samples",
                    "256",
                    "--players",
                    "3",
                    "--seed",
                    str(seed + 99999),
                    "--model-path",
                    winner["path"],
                ]
            }
            evaluation = stage(commands, deadline - 30)
        current_stage = "finished"
        status(
            "complete",
            selected=winner,
            candidates=candidates,
            activated=False,
            return_evaluation=evaluation,
            return_report=str(output / "returns.json") if (output / "returns.json").exists() else None,
        )
        return json.loads(status_path.read_text())
    except BaseException as exc:
        status(
            "interrupted" if isinstance(exc, (InterruptedError, TimeoutError)) else "failed", error=str(exc)
        )
        raise
    finally:
        for process in active:
            terminate_group(process)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        lock.close()
