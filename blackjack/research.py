"""Bounded local evaluation suite; keeps all raw paired units for reproduction."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .evaluation import compare_evaluations, evaluate_policy
from .experiment_data import atomic_json
from .overnight import terminate_group


def evaluate_suite(
    output, policy, source=None, device="cuda:0", fresh_units=100000, blocks=1000, seed=20260922
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
        )


def run_research(output: Path, candidate: str, baseline: str, fresh_units=100000, blocks=1000, hours=3.0):
    if output.exists() or not 0 < hours <= 12:
        raise ValueError("Use a new evaluation directory and a budget of at most 12 hours.")
    output.mkdir(parents=True)
    started, processes, logs = time.time(), {}, []
    deadline = started + hours * 3600
    atomic_json(
        output / "run.json",
        {
            "candidate": candidate,
            "baseline": baseline,
            "fresh_units": fresh_units,
            "blocks": blocks,
            "started_unix": started,
            "deadline_unix": deadline,
        },
    )

    def status(state, **extra):
        progress = {}
        for path in output.glob("*/*/progress.json"):
            progress[f"{path.parent.parent.name} / {path.parent.name}"] = json.loads(path.read_text())
        atomic_json(
            output / "status.json",
            {
                "status": state,
                "stage": "policy_evaluation",
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
        for name, source, device in [
            ("basic", None, "cpu"),
            ("baseline", baseline, "cuda:0"),
            ("candidate", candidate, "cuda:1"),
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
            ]
            if source:
                command += ["--source", source]
            log = (output / f"{name}.log").open("w")
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
        for mode in ("fresh", "continuous"):
            compare_evaluations(
                {name: output / name / mode for name in processes}, output / f"{mode}-comparison.json"
            )
        status("complete", reports=["fresh-comparison.json", "continuous-comparison.json"])
    except BaseException as exc:
        status("failed", error=str(exc))
        raise
    finally:
        for p in processes.values():
            terminate_group(p)
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
