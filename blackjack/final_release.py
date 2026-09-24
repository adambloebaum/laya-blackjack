"""Frozen candidate selection, untouched release evaluation, and one private package."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .evaluation import compare_evaluations, scenarios
from .experiment_data import atomic_json, digest, verify_dataset
from .model_audit import audit_model
from .overnight import terminate_group
from .releases import package_model, public_metadata, verify_model
from .research import checkpoint_identity, comparison_summary, freeze_checkpoint, freeze_run
from .visitation_study import inference_files

VERSION = "final-release-selection-v1"


def release_seed(seed, purpose):
    return int.from_bytes(hashlib.sha256(f"{VERSION}/{seed}/{purpose}".encode()).digest()[:16], "big")


def paired_selection(candidate, baseline, *, seed, comparisons, repeats=20000):
    """Paired game bootstrap within each fixed, equally weighted selection stratum."""
    if len(candidate) != len(baseline) or not candidate or comparisons < 1 or repeats < 100:
        raise ValueError("Selection requires matched records and a fixed comparison family.")
    groups = defaultdict(lambda: defaultdict(list))
    for c, b in zip(candidate, baseline, strict=True):
        if (c["index"], c["group_seed"], c["stratum"]) != (b["index"], b["group_seed"], b["stratum"]):
            raise ValueError("Selection decision identities differ.")
        if not all(math.isfinite(x["regret"]) for x in (c, b)):
            raise ValueError("Nonfinite selection regret.")
        groups[c["stratum"]][c["group_seed"]].append(c["regret"] - b["regret"])
    if set(groups) != {"general", "depleted"}:
        raise ValueError("Selection requires both declared strata.")
    rng = np.random.default_rng(seed)
    draws = np.zeros(repeats)
    by_stratum = {}
    for stratum in sorted(groups):
        values = groups[stratum]
        if len(values) < 2:
            raise ValueError("Selection needs at least two games per stratum.")
        totals = np.array([[len(v), sum(v)] for v in values.values()])
        by_stratum[stratum] = float(totals[:, 1].sum() / totals[:, 0].sum())
        for i in range(repeats):
            sample = totals[rng.integers(0, len(totals), len(totals))].sum(axis=0)
            draws[i] += sample[1] / sample[0] / 2
    alpha = 0.05 / comparisons
    interval = np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist()
    mean = sum(by_stratum.values()) / 2
    return {
        "regret_difference": mean,
        "familywise_ci95": interval,
        "by_stratum": by_stratum,
        "eligible": mean < 0
        and interval[1] < 0
        and by_stratum["depleted"] <= 0
        and by_stratum["general"] <= 0.0005,
        "bootstrap_replicates": repeats,
        "comparison_family": comparisons,
    }


def run_config(root):
    return json.loads((root / "run.json").read_text())["config"]


def verify_sources(root):
    config = run_config(root)
    for name, identity in config["checkpoints"].items():
        if checkpoint_identity(root / "checkpoints" / name) != identity:
            raise ValueError("Frozen candidate package changed.")


def audit_receipt(root, phase, name):
    config = run_config(root)
    if name not in config["checkpoints"] or phase not in ("selection", "final"):
        raise ValueError("Unknown release audit.")
    identity = {
        "run_sha256": digest(root / "run.json"),
        "dataset_sha256": digest(root / "data/manifest.json"),
        "files": {
            n: config["checkpoints"][name][n]["sha256"] for n in inference_files(root / "checkpoints" / name)
        },
        "split": "selection" if phase == "selection" else "test",
        "inference_mode": "sdk",
    }
    if phase == "final":
        freeze = json.loads((root / "selection-frozen.json").read_text())
        if name not in (freeze["selected_name"], "packaged"):
            raise ValueError("Only frozen finalist and packaged baseline may see final predictions.")
        identity["selection_sha256"] = digest(root / "selection-frozen.json")
    path = root / phase / name
    receipt = path / "completion.json"
    if not receipt.exists():
        return identity, None
    saved = json.loads(receipt.read_text())
    if saved["identity"] != identity or not {"report.json", "decisions.json"} <= saved["outputs"].keys():
        raise ValueError("Release audit identity changed.")
    if any(digest(path / n) != sha for n, sha in saved["outputs"].items()):
        raise ValueError("Release audit outputs changed.")
    report = json.loads((path / "report.json").read_text())
    if (
        report["inference_mode"] != "sdk"
        or report["policy_action_mismatches"] != []
        or report["batched_action_mismatches"] is not None
        or report["split"] != identity["split"]
        or report["dataset_sha256"] != identity["dataset_sha256"]
        or report["model_sha256"] != identity["files"]["model.safetensors"]
        or report["metrics"]["states"]
        != config["selection_states" if phase == "selection" else "test_states"]
    ):
        raise ValueError("Release audit violates its SDK/split/count contract.")
    return identity, report


def release_audit(root: Path, phase: str, name: str, device="cuda:0"):
    saved = json.loads((root / "run.json").read_text())
    if time.time() >= saved["deadline_unix"]:
        raise TimeoutError("Original release deadline expired.")
    if phase == "final":
        if not (root / "selection-frozen.json").exists():
            raise ValueError("Final predictions require frozen selection.")
        freeze_selection(root)
    identity, existing = audit_receipt(root, phase, name)
    if existing is not None:
        return existing
    if phase == "selection" and (root / "selection-frozen.json").exists():
        raise ValueError("Cannot add selection predictions after the freeze.")
    source, output = root / "checkpoints" / name, root / phase / name
    if inference_files(source) != identity["files"]:
        raise ValueError("Candidate inference files changed.")
    audit_model(
        root / "data",
        str(source),
        output,
        device=device,
        allow_new_dataset=True,
        inference_mode="sdk",
        split=identity["split"],
    )
    if inference_files(source) != identity["files"]:
        raise ValueError("Candidate changed during inference.")
    outputs = {
        p.name: digest(p) for p in output.glob("*.json") if p.name not in ("progress.json", "completion.json")
    }
    atomic_json(output / "completion.json", {"identity": identity, "outputs": outputs})
    return audit_receipt(root, phase, name)[1]


def freeze_selection(root):
    config = run_config(root)
    verify_sources(root)
    manifest = verify_dataset(root / "data")
    if not manifest["complete"] or any(
        manifest["actual_states"][s] != config[k]
        for s, k in (("selection", "selection_states"), ("test", "test_states"))
    ):
        raise ValueError("Release dataset is incomplete.")
    reports, records, receipts = {}, {}, {}
    for name in config["checkpoints"]:
        _, report = audit_receipt(root, "selection", name)
        if report is None:
            raise ValueError("All nominated candidates need complete selection audits.")
        reports[name] = report
        records[name] = json.loads((root / "selection" / name / "decisions.json").read_text())
        receipts[name] = digest(root / "selection" / name / "completion.json")
    identity = {
        "run_sha256": digest(root / "run.json"),
        "dataset_sha256": digest(root / "data/manifest.json"),
        "selection_receipts": receipts,
    }
    path = root / "selection-frozen.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError("Frozen selection inputs changed.")
        # Recompute the decision cheaply from the immutable selection evidence below.
    elif list((root / "final").glob("*/decisions.json")) or list((root / "returns").glob("*/*/config.json")):
        raise ValueError("Final predictions exist before selection was frozen.")
    contrasts = {
        name: paired_selection(
            rows,
            records["incumbent"],
            seed=release_seed(config["seed"], "selection-bootstrap-" + name),
            comparisons=len(records) - 1,
            repeats=config["selection_bootstrap_repeats"],
        )
        for name, rows in records.items()
        if name != "incumbent"
    }
    eligible = ["incumbent", *[name for name in sorted(contrasts) if contrasts[name]["eligible"]]]
    regrets = {name: reports[name]["metrics"]["teacher_ev_regret"] for name in reports}
    winner = min(eligible, key=lambda name: regrets[name])
    result = {
        "identity": identity,
        "selected_name": winner,
        "selection_regrets": regrets,
        "contrasts": contrasts,
        "criterion": config["selection_criterion"],
    }
    if path.exists():
        if {k: v for k, v in saved.items() if k != "frozen_unix"} != result:
            raise ValueError("Frozen selection decision changed.")
        return saved
    result["frozen_unix"] = time.time()
    atomic_json(path, result)
    return result


def prepare_package(root, freeze, summary):
    """Build one local review package; no network publication or active pointer writes."""
    name = freeze["selected_name"]
    source = root / "checkpoints" / name
    _, audit = audit_receipt(root, "final", name)
    if audit is None:
        raise ValueError("Cannot package an unaudited finalist.")
    report = json.loads((source / "training_report.json").read_text())
    report.update(
        test=audit["metrics"],
        test_deferred=False,
        prior_test=report["test"],
        audit_dataset_hash=digest(root / "data/manifest.json"),
        release_selection_sha256=digest(root / "selection-frozen.json"),
        test_inference="Canonical three-question SDK on fresh release test games",
    )
    target = root / "release-candidate"
    if not target.exists():
        staging = root / (".release-candidate-" + uuid.uuid4().hex[:8])
        # Copy only inference files, avoiding a stale inherited release inventory/card.
        for file in inference_files(source):
            dest = staging / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / file, dest)
        atomic_json(staging / "training_report.json", report)
        staging.rename(target)
    if (
        inference_files(target) != inference_files(source)
        or json.loads((target / "training_report.json").read_text()) != report
    ):
        raise ValueError("Final release-candidate copy changed.")
    evidence = root / "release-evidence"
    evidence.mkdir(exist_ok=True)
    for item in (
        "run.json",
        "selection-frozen.json",
        "summary.json",
        "fresh-comparison.json",
        "continuous-comparison.json",
    ):
        atomic_json(evidence / item, public_metadata(json.loads((root / item).read_text())))
    atomic_json(evidence / "sdk-audit.json", public_metadata(audit))
    # No checkpoints or optimizer files may enter the evidence archive.
    archive = evidence / "evaluation-records.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        with tarfile.open(temporary, "w:gz") as tar:
            for folder in ("data", "selection", "final", "returns"):
                for path in sorted((root / folder).rglob("*")):
                    if path.is_file() and path.suffix in (".json", ".jsonl") and path.name != "progress.json":
                        tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
            if (root / "source.tar").exists():
                tar.add(root / "source.tar", arcname="evaluator-source.tar")
        temporary.rename(archive)
    metrics = audit["metrics"]
    card = root / "RELEASE_MODEL_CARD.md"
    comparisons = "\n".join(
        f"| {key} | {v['units_per_100_rounds']:+.5f} | [{v['familywise_ci95'][0]:+.5f}, {v['familywise_ci95'][1]:+.5f}] |"
        for key, v in summary["comparisons"].items()
    )
    card.write_text(f"""---
language: en
license: apache-2.0
base_model: convaiinnovations/laya
library_name: laya
tags: [blackjack, simulation, decision-making]
---
# Laya Blackjack

Private release candidate selected from five completed research models. This package contains one checkpoint. Publication and dashboard activation require release review.

Selected candidate: `{name}`. No training or calibration fitting occurred during this release study. The complete prior training record is in `training_report.json`; frozen selection and fresh final evidence are in `evaluation/`. The selection criterion and uncertainty are preserved, including inconclusive or negative results.

## Fresh serving evaluation

Canonical SDK, {metrics["states"]:,} untouched test states with equal general/depleted strata: reference agreement **{100 * metrics["teacher_agreement"]:.2f}%**, reference EV regret **{metrics["teacher_ev_regret"]:.9f}** units/decision. Hit-bust Brier distance {metrics["hit_bust_brier"]:.9f}; dealer Brier distance {metrics["dealer_brier"]:.9f}. These measure agreement with a mixed exact/Monte Carlo reference, not empirical win calibration or global optimality.

| Paired contrast | Units / 100 rounds | Familywise 95% interval |
| --- | ---: | ---: |
{comparisons}

`baseline` means the previously packaged v0.2 model; `basic` is the simulator's basic-strategy heuristic. These are fixed-wager simulated returns with paired initial seeds. Fresh rounds and independent 100-round blocks are the sampling units; four aggregate intervals use Bonferroni correction under a normal approximation. Selection games cannot serve as final-test games.

## Intended use and limits

Research and simulated play with one learned player and up to six tablemates. Use only public observations from the matching project state/question contract. Action preferences are not win probabilities. Unsupported exact-reference states use Monte Carlo with basic continuation; reference agreement does not prove optimal play or real-world profitability. The 50:50 enriched test average is not ordinary gameplay visitation frequency.

See the pinned project source and reproduction artifacts in `evaluation/evaluation-records.tar.gz`. Laya and its ModernBERT backbone are credited in NOTICE. Intermediate model weights are excluded.
""")
    package = root / "release-package"
    if not package.exists():
        package_model(target, package, card, Path(__file__).parents[1], evidence)
    verified = verify_model(package)
    if inference_files(package) != inference_files(source):
        raise ValueError("Packaged inference files differ from the frozen winner.")
    if json.loads((package / "training_report.json").read_text()) != public_metadata(report):
        raise ValueError("Packaged final evidence changed.")
    if (package / "README.md").read_bytes() != card.read_bytes() or any(
        digest(package / "evaluation" / str(path.relative_to(evidence))) != digest(path)
        for path in evidence.rglob("*")
        if path.is_file()
    ):
        raise ValueError("Packaged card or evaluation evidence changed.")
    atomic_json(
        root / "release-review.json",
        {
            "selected_name": name,
            "package": "release-package",
            "manifest_sha256": digest(package / "manifest.json"),
            "inference_files": inference_files(target),
            "inventory_files": len(verified["files"]),
            "public_model_count": 1,
            "activated": False,
            "published": False,
            "next": "Review the fresh results, complete release presentation, then upload to a clean private Hub history and verify the pinned download.",
        },
    )


def run_final_release(output: Path, candidates: Path, workers=24, hours=12.0, seed=20261008, smoke=False):
    if not math.isfinite(hours) or not 0 < hours <= 12 or not 1 <= workers <= 32:
        raise ValueError("Use up to 12 hours and 1–32 workers.")
    sources = json.loads(candidates.read_text())
    if (
        len(sources) != 5
        or not {"incumbent", "packaged"} <= sources.keys()
        or any(not re.fullmatch(r"[a-z_]+", n) for n in sources)
    ):
        raise ValueError("Nominate five named completed candidates, including incumbent and packaged.")
    sources = {name: str((candidates.parent / Path(path)).absolute()) for name, path in sources.items()}
    checkpoints = {name: checkpoint_identity(Path(path)) for name, path in sources.items()}
    if len({v["model.safetensors"]["sha256"] for v in checkpoints.values()}) != 5:
        raise ValueError("Nominees must have five distinct weight identities.")
    config = {
        "version": VERSION,
        "sources": sources,
        "checkpoints": checkpoints,
        "nomination_sha256": digest(candidates),
        "workers": workers,
        "hours": hours,
        "seed": seed + (10000000 if smoke else 0),
        "smoke": smoke,
        "selection_states": 32 if smoke else 8192,
        "test_states": 32 if smoke else 16384,
        "selection_bootstrap_repeats": 400 if smoke else 20000,
        "inference_mode": "sdk",
        "fresh_units": 20 if smoke else 500000,
        "blocks": 20 if smoke else 5000,
        "scenarios": scenarios(),
        "selection_criterion": "Retain incumbent unless a challenger has lower overall SDK selection regret, an upper Bonferroni 95% paired-game bootstrap bound below zero across four challengers, depleted difference <= 0 and general difference <= 0.0005. Among eligible models choose lowest regret, with incumbent first on ties. Calibration stays frozen. Final results cannot change this selection.",
        "runtime": {
            "python": sys.version,
            **{n: version(n) for n in ("laya", "torch", "transformers", "numpy")},
        },
        "code": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = freeze_run(output, config)
        _supervise(output, saved)


def _supervise(root, saved):
    config = saved["config"]
    started = saved["started_unix"]
    deadline = saved["deadline_unix"]
    current_stage = "freezing_candidates"
    active = []

    def status(state="running", **extra):
        progress = {
            str(p.relative_to(root)): json.loads(p.read_text())
            for folder in ("data", "selection", "final", "returns")
            for p in (root / folder).rglob("progress.json")
        }
        atomic_json(
            root / "status.json",
            {
                "status": state,
                "stage": current_stage,
                "experiment": "final-release",
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
        raise InterruptedError(f"Release study received signal {signum}")

    def stage(commands, until):
        logs = []
        try:
            if time.time() >= until:
                raise TimeoutError("Original release stage deadline expired.")
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
                    raise TimeoutError("Original release stage deadline expired.")
                if any(p.poll() not in (None, 0) for p in active):
                    raise RuntimeError(f"{current_stage} worker failed; inspect its log.")
                status()
                time.sleep(3)
            if any(p.returncode != 0 for p in active):
                raise RuntimeError(f"{current_stage} worker failed.")
        finally:
            for p in active:
                terminate_group(p)
            active.clear()
            for log in logs:
                log.close()

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        status()
        for name, source in config["sources"].items():
            freeze_checkpoint(Path(source), root / "checkpoints" / name, config["checkpoints"][name])
        current_stage = "generating_release_games"
        manifest = root / "data/manifest.json"
        if not manifest.exists() or not json.loads(manifest.read_text())["complete"]:
            shard = 8 if config["smoke"] else 64
            until = started + config["hours"] * 3600 / 6
            stage(
                {
                    "data": [
                        "generate-large",
                        "--output",
                        str(root / "data"),
                        "--states",
                        str(2 * shard),
                        "--selection",
                        str(config["selection_states"]),
                        "--calibration",
                        str(shard),
                        "--test",
                        str(config["test_states"]),
                        "--workers",
                        str(config["workers"]),
                        "--shard-size",
                        str(shard),
                        "--samples",
                        "32",
                        "--max-samples",
                        "64",
                        "--evaluation-samples",
                        "32" if config["smoke"] else "4096",
                        "--evaluation-max-samples",
                        "64" if config["smoke"] else "16384",
                        "--profile",
                        "composition-v1",
                        "--teacher",
                        "hybrid-exact-v1",
                        "--seed",
                        str(release_seed(config["seed"], "data")),
                        "--deadline",
                        str(until),
                    ]
                },
                until,
            )
        current_stage = "candidate_selection"
        names = list(config["checkpoints"])
        for offset in range(0, len(names), 2):
            commands = {
                name: [
                    "release-audit",
                    "--root",
                    str(root),
                    "--phase",
                    "selection",
                    "--name",
                    name,
                    "--device",
                    f"cuda:{i}",
                ]
                for i, name in enumerate(names[offset : offset + 2])
                if audit_receipt(root, "selection", name)[1] is None
            }
            if commands:
                stage(commands, started + config["hours"] * 3600 / 4)
        freeze = freeze_selection(root)
        current_stage = "sealed_test_evaluation"
        finalists = list(dict.fromkeys((freeze["selected_name"], "packaged")))
        commands = {
            "final-" + name: [
                "release-audit",
                "--root",
                str(root),
                "--phase",
                "final",
                "--name",
                name,
                "--device",
                f"cuda:{i}",
            ]
            for i, name in enumerate(finalists)
            if audit_receipt(root, "final", name)[1] is None
        }
        if commands:
            stage(commands, started + config["hours"] * 3600 / 3)
        current_stage = "paired_returns"
        policies = {"candidate": freeze["selected_name"], "baseline": "packaged", "basic": None}
        commands = {}
        for policy, name in policies.items():
            commands["returns-" + policy] = [
                "evaluate-suite",
                "--output",
                str(root / "returns" / policy),
                "--policy",
                "laya" if name else "basic",
                "--device",
                "cuda:0" if policy == "candidate" else "cuda:1" if name else "cpu",
                "--inference-mode",
                "sdk",
                "--fresh-units",
                str(config["fresh_units"]),
                "--blocks",
                str(config["blocks"]),
                "--seed",
                str(release_seed(config["seed"], "returns")),
            ] + (["--source", str(root / "checkpoints" / name)] if name else [])
        stage(commands, deadline - 120)
        verify_sources(root)
        for mode in ("fresh", "continuous"):
            for policy, name in policies.items():
                cfg = json.loads((root / "returns" / policy / mode / "manifest.json").read_text())["config"]
                expected = config["fresh_units"] if mode == "fresh" else config["blocks"]
                if (
                    cfg["identity"]["units"] != expected
                    or cfg["identity"]["seed"] != release_seed(config["seed"], "returns")
                    or cfg["inference_mode"] != "sdk"
                ):
                    raise ValueError("Final return plan changed.")
                if name and (
                    cfg["source_hash"] != config["checkpoints"][name]["model.safetensors"]["sha256"]
                    or cfg["config_hash"] != config["checkpoints"][name]["rl_agent_config.json"]["sha256"]
                ):
                    raise ValueError("Final return model changed.")
            compare_evaluations(
                {p: root / "returns" / p / mode for p in policies}, root / f"{mode}-comparison.json"
            )
        freeze = freeze_selection(root)
        summary = {
            **comparison_summary(root),
            "selected_name": freeze["selected_name"],
            "selection_sha256": digest(root / "selection-frozen.json"),
            "smoke": config["smoke"],
            "research_evidence": not config["smoke"],
            "rounds": 3 * (config["fresh_units"] + 100 * config["blocks"]),
            "baseline": "Previously packaged v0.2 model; selection incumbent is a separate research candidate.",
            "inference_mode": "sdk",
        }
        atomic_json(root / "summary.json", summary)
        current_stage = "preparing_private_package"
        status()
        prepare_package(root, freeze, summary)
        if time.time() >= deadline:
            raise TimeoutError("Original release deadline expired during packaging.")
        current_stage = "finished"
        status(
            "complete", selected_name=freeze["selected_name"], reports=["summary.json", "release-review.json"]
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
