"""Development-only model-visited data qualification with replayable public states."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .engine import Game, Rules, basic_action, tablemate_action
from .evaluation import BatchedLaya, scenarios
from .exact_reference import hybrid_analyze
from .experiment_data import atomic_json, digest
from .model import model_state, questions
from .overnight import terminate_group
from .research import checkpoint_identity, freeze_checkpoint, freeze_run

VERSION = "model-visitation-pilot-v1"
POLICIES = ("laya", "mixture")


def seeded(seed, *parts):
    key = "/".join(map(str, (VERSION, seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:16], "big")


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def group_path(root, phase, policy, scenario, index):
    return root / phase / policy / f"{scenario}-{index:04d}.json"


def read_record(path, identity):
    receipt = path.with_suffix(".receipt.json")
    if not receipt.exists():
        return None
    saved = json.loads(receipt.read_text())
    if saved["identity"] != identity or not path.is_file() or saved["sha256"] != digest(path):
        raise ValueError(f"Pilot artifact or input identity changed: {path.name}")
    return json.loads(path.read_text())


def write_record(path, value, identity):
    atomic_json(path, value)
    atomic_json(path.with_suffix(".receipt.json"), {"identity": identity, "sha256": digest(path)})


def replay_group(group):
    """Replay explicit actions without calling a policy or consuming its RNG."""
    game = Game(Rules(**group["rules"]), group["group_seed"])
    selected = {r["event_index"]: r for r in group["rows"]}
    if len(selected) != len(group["rows"]):
        raise ValueError("Duplicate sampled decision.")
    seen = set()
    for index, action in enumerate(group["actions"]):
        if index in selected:
            row = selected[index]
            obs = game.observation()
            if (
                obs != row["observation"]
                or content_hash(obs) != row["observation_sha256"]
                or action != row["behavior_action"]
                or game.active[0] != 0
            ):
                raise ValueError("Sampled public observation/action does not replay.")
            seen.add(index)
        game.deal() if action == "deal" else game.step(action)
    if (
        seen != set(selected)
        or game.observation() != group["final_observation"]
        or game.history != group["history"]
        or len(game.history) != group["rounds"]
    ):
        raise ValueError("Pilot trajectory replay diverged or missed sampled states.")
    return len(seen)


def collect_groups(tasks, actor, config, policy, completed):
    """Uniform per-trajectory reservoir; actor receives observations only."""
    for start in range(0, len(tasks), config["batch_size"]):
        slots = []
        for scenario, index in tasks[start : start + config["batch_size"]]:
            seed = seeded(config["seed"], "game", scenario, index)
            slots.append(
                {
                    "scenario": scenario,
                    "index": index,
                    "group_seed": seed,
                    "game": Game(Rules(**config["scenarios"][scenario]), seed),
                    "sampling_rng": random.Random(seeded(config["seed"], "reservoir", scenario, index)),
                    "behavior_rng": random.Random(seeded(config["seed"], "behavior", scenario, index)),
                    "actions": [],
                    "rows": [],
                    "decisions": 0,
                }
            )
        while slots:
            if time.time() >= config["deadline_unix"]:
                raise TimeoutError("Pilot collection reached its original deadline.")
            waiting = []
            for slot in slots:
                game = slot["game"]
                while True:
                    if len(slot["actions"]) > config["rounds_per_group"] * 512:
                        raise RuntimeError("Trajectory exceeded its bounded step count.")
                    if game.phase != "playing":
                        if len(game.history) == config["rounds_per_group"]:
                            result = {
                                k: slot[k]
                                for k in ("scenario", "index", "group_seed", "actions", "decisions")
                            }
                            result.update(
                                policy=policy,
                                rules=config["scenarios"][slot["scenario"]],
                                rounds=len(game.history),
                                rows=sorted(slot["rows"], key=lambda r: r["event_index"]),
                                final_observation=game.observation(),
                                history=game.history,
                            )
                            replay_group(result)
                            completed(result)
                            break
                        slot["actions"].append("deal")
                        game.deal()
                    elif game.active[0] != 0:
                        action = tablemate_action(game)
                        slot["actions"].append(action)
                        game.step(action)
                    else:
                        waiting.append(slot)
                        break
            slots = waiting
            observations = [slot["game"].observation() for slot in slots]
            if not observations:
                continue
            if policy == "laya":
                actions = actor.actions(observations)
            else:
                actions = [
                    s["behavior_rng"].choice(o["legal_actions"])
                    if s["behavior_rng"].random() < 0.25
                    else basic_action(o)
                    for s, o in zip(slots, observations, strict=True)
                ]
            if len(actions) != len(slots):
                raise ValueError("Collector received the wrong number of actions.")
            for slot, obs, action in zip(slots, observations, actions, strict=True):
                if action not in obs["legal_actions"]:
                    raise ValueError("Collector received an illegal action.")
                row = {
                    "event_index": len(slot["actions"]),
                    "observation": obs,
                    "observation_sha256": content_hash(obs),
                    "behavior_action": action,
                }
                slot["decisions"] += 1
                k = config["states_per_group"]
                if len(slot["rows"]) < k:
                    slot["rows"].append(row)
                else:
                    index = slot["sampling_rng"].randrange(slot["decisions"])
                    if index < k:
                        slot["rows"][index] = row
                slot["actions"].append(action)
                slot["game"].step(action)


def identities(root, phase, policy, scenario, index):
    identity = {
        "run_sha256": digest(root / "run.json"),
        "phase": phase,
        "policy": policy,
        "scenario": scenario,
        "index": index,
    }
    if phase != "collect":
        identity["observations_sha256"] = digest(group_path(root, "collect", policy, scenario, index))
    return identity


def read_group(root, phase, policy, scenario, index):
    return read_record(
        group_path(root, phase, policy, scenario, index), identities(root, phase, policy, scenario, index)
    )


def label_group(task):
    group, seed, samples, max_samples = task
    records = []
    for row in group["rows"]:
        references = [
            hybrid_analyze(
                row["observation"],
                samples,
                seed=seeded(
                    seed,
                    "labels",
                    group["policy"],
                    group["scenario"],
                    group["index"],
                    row["event_index"],
                    replica,
                ),
                max_samples=max_samples,
            )
            for replica in range(2)
        ]
        records.append(
            {
                "event_index": row["event_index"],
                "observation_sha256": row["observation_sha256"],
                "references": references,
            }
        )
    return {
        "scenario": group["scenario"],
        "index": group["index"],
        "policy": group["policy"],
        "rows": records,
    }


def audit_group(group, actor):
    fast = actor.actions([r["observation"] for r in group["rows"]])
    records = []
    for row, fast_action in zip(group["rows"], fast, strict=True):
        obs = row["observation"]
        answers = actor.agent.predict(model_state(obs), questions(obs))["answers"]
        action = answers["action"]["choice"]
        if action not in obs["legal_actions"]:
            raise ValueError("SDK returned an illegal action.")
        records.append(
            {
                "event_index": row["event_index"],
                "observation_sha256": row["observation_sha256"],
                "sdk_action": action,
                "batch_action": fast_action,
                "action_probabilities": answers["action"]["probabilities"],
                "hit_bust": answers["hit_bust"]["noul"],
                "dealer": answers["dealer"]["probabilities"],
            }
        )
    return {
        "scenario": group["scenario"],
        "index": group["index"],
        "policy": group["policy"],
        "rows": records,
    }


def pilot_worker(root: Path, phase: str, policy="laya", device="cuda:1"):
    saved = json.loads((root / "run.json").read_text())
    config = saved["config"] | {"deadline_unix": saved["deadline_unix"]}
    if time.time() >= config["deadline_unix"]:
        raise TimeoutError("Original pilot deadline expired.")
    policies = POLICIES if phase == "label" else (policy,)
    tasks = [
        (p, scenario, i)
        for p in policies
        for scenario in config["scenarios"]
        for i in range(config["groups_per_scenario"])
    ]
    progress = root / "progress" / f"{phase}-{'all' if phase == 'label' else policy}.json"
    done = states = 0

    def finish(group, reuse=False):
        nonlocal done, states
        if not reuse:
            write_record(
                group_path(root, phase, group["policy"], group["scenario"], group["index"]),
                group,
                identities(root, phase, group["policy"], group["scenario"], group["index"]),
            )
        done += 1
        states += len(group["rows"])
        atomic_json(
            progress,
            {
                "stage": phase,
                "completed": done,
                "total": len(tasks),
                "unit": "trajectory groups",
                "states": states,
                "scope": "development pilot",
            },
        )

    pending = []
    for p, scenario, i in tasks:
        result = read_group(root, phase, p, scenario, i)
        if result is not None:
            if phase == "collect":
                replay_group(result)
            finish(result, reuse=True)
        else:
            pending.append((p, scenario, i))
    if not pending:
        return
    actor = (
        BatchedLaya(str(root / "checkpoints/source"), device)
        if phase == "audit" or phase == "collect" and policy == "laya"
        else None
    )
    if phase == "collect":
        # Restart the original batches: skip publishing existing groups, not their inference context.
        completed_keys = {(p, s, i) for p, s, i in tasks} - set(pending)

        def collected(group):
            key = (group["policy"], group["scenario"], group["index"])
            if key in completed_keys:
                if content_hash(group) != content_hash(read_group(root, "collect", *key)):
                    raise ValueError("Replayed collection batch changed a completed trajectory.")
            else:
                finish(group)

        for start in range(0, len(tasks), config["batch_size"]):
            chunk = tasks[start : start + config["batch_size"]]
            if any(t not in completed_keys for t in chunk):
                collect_groups([(s, i) for _, s, i in chunk], actor, config, policy, collected)
    elif phase == "label":
        with ProcessPoolExecutor(
            max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    label_group,
                    (
                        read_group(root, "collect", p, s, i),
                        config["seed"],
                        config["samples"],
                        config["max_samples"],
                    ),
                )
                for p, s, i in pending
            ]
            for future in as_completed(futures):
                if time.time() >= config["deadline_unix"]:
                    raise TimeoutError("Pilot labeling reached its original deadline.")
                finish(future.result())
    elif phase == "audit":
        for p, s, i in pending:
            if time.time() >= config["deadline_unix"]:
                raise TimeoutError("Pilot audit reached its original deadline.")
            finish(audit_group(read_group(root, "collect", p, s, i), actor))
    else:
        raise ValueError("Unknown pilot phase.")


def summarize_pilot(root):
    saved = json.loads((root / "run.json").read_text())
    config = saved["config"]
    summaries, checks, evidence = {}, {}, {}
    for policy in POLICIES:
        counts = Counter()
        groups = defaultdict(Counter)
        regret = hit_error = dealer_error = 0.0
        for scenario in config["scenarios"]:
            for index in range(config["groups_per_scenario"]):
                records = {
                    phase: read_group(root, phase, policy, scenario, index)
                    for phase in ("collect", "label", "audit")
                }
                if any(r is None for r in records.values()):
                    raise ValueError("Incomplete pilot artifacts.")
                trajectory = records["collect"]
                replay_group(trajectory)
                counts["replayed_groups"] += 1
                counts["rounds"] += trajectory["rounds"]
                counts["decisions"] += trajectory["decisions"]
                if len(trajectory["rows"]) != config["states_per_group"]:
                    counts["short_reservoirs"] += 1
                for phase in records:
                    path = group_path(root, phase, policy, scenario, index)
                    evidence[str(path.relative_to(root))] = digest(path)
                for row, label, pred in zip(
                    trajectory["rows"], records["label"]["rows"], records["audit"]["rows"], strict=True
                ):
                    if any(
                        (x["event_index"], x["observation_sha256"])
                        != (row["event_index"], row["observation_sha256"])
                        for x in (label, pred)
                    ):
                        raise ValueError("Pilot observations and predictions do not match.")
                    obs = row["observation"]
                    primary, repeat = label["references"]
                    kind = primary["teacher_kind"]
                    if kind != repeat["teacher_kind"]:
                        raise ValueError("Deterministic reference scope changed across repeats.")
                    n = counts["states"] = counts["states"] + 1
                    action = pred["sdk_action"]
                    best = primary["recommendation"]
                    cost = primary["actions"][best]["ev"] - primary["actions"][action]["ev"]
                    regret += cost
                    counts["teacher_agreement"] += action == best
                    counts["audit_action_mismatches"] += action != pred["batch_action"]
                    counts["collection_action_mismatches"] += (
                        policy == "laya" and action != row["behavior_action"]
                    )
                    counts["dealer_unresolved"] += (
                        primary["dealer_unresolved"] > 1e-12 or repeat["dealer_unresolved"] > 1e-12
                    )
                    counts[kind] += 1
                    depth = 1 - obs["cards_remaining"] / (52 * obs["rules"]["decks"])
                    counts["depleted"] += depth >= 0.5
                    hand = obs["players"][0][obs["active"][1]]
                    category = (
                        "pair" if "split" in obs["legal_actions"] else "soft" if hand["soft"] else "hard"
                    )
                    groups["category"][category] += 1
                    groups["scenario"][scenario] += 1
                    groups["depth_decile"][str(int(depth * 10))] += 1
                    groups["reference_action"][best] += 1
                    counts["repeat_agreement"] += best == repeat["recommendation"]
                    both = primary["label_quality"]["resolved"] and repeat["label_quality"]["resolved"]
                    counts["both_resolved"] += both
                    counts["resolved_repeat_agreement"] += both and best == repeat["recommendation"]
                    if kind == "exact" and primary != repeat:
                        raise ValueError("Exact reference is not deterministic.")
                    hit_error += 2 * (pred["hit_bust"] - primary["hit_bust"]) ** 2
                    dealer_error += sum((pred["dealer"][k] - v) ** 2 for k, v in primary["dealer"].items())
        n = counts["states"]
        stable = counts["resolved_repeat_agreement"] / max(1, counts["both_resolved"])
        summaries[policy] = {
            "counts": dict(counts),
            "coverage": {k: dict(v) for k, v in groups.items()},
            "reference_agreement": counts["teacher_agreement"] / n,
            "reference_regret": regret / n,
            "hit_bust_brier_distance": hit_error / n,
            "dealer_brier_distance": dealer_error / n,
            "repeat_agreement": counts["repeat_agreement"] / n,
            "both_resolved_fraction": counts["both_resolved"] / n,
            "resolved_repeat_agreement": stable,
            "depleted_fraction": counts["depleted"] / n,
        }
        thresholds = config["qualification_thresholds"]
        checks[policy] = {
            "complete_reservoirs": counts["short_reservoirs"] == 0,
            "sdk_parity": counts["audit_action_mismatches"] == counts["collection_action_mismatches"] == 0,
            "complete_dealer_targets": counts["dealer_unresolved"] == 0,
            "resolved_coverage": counts["both_resolved"] / n >= thresholds["both_resolved_fraction"],
            "resolved_label_stability": stable >= thresholds["resolved_repeat_agreement"],
            "depleted_coverage": counts["depleted"] / n >= thresholds["depleted_fraction"],
            "hand_category_coverage": all(
                groups["category"][k] >= thresholds["states_per_category"] for k in ("hard", "soft", "pair")
            ),
        }
    passed = all(all(c.values()) for c in checks.values()) and not config["smoke"]
    result = {
        "format": VERSION,
        "development_only": True,
        "smoke": config["smoke"],
        "source_model_sha256": config["checkpoint"]["model.safetensors"]["sha256"],
        "run_sha256": digest(root / "run.json"),
        "thresholds": config["qualification_thresholds"],
        "policies": summaries,
        "checks": checks,
        "qualified_for_training_design": passed,
        "decision": "prepare_matched_training"
        if passed
        else "execution_only"
        if config["smoke"]
        else "revise_pilot",
        "evidence": evidence,
        "scope": "Development-only collection/label qualification, not a final test or policy-return comparison. "
        "Each trajectory contributes a uniform reservoir with equal group weight. Policy arms share initial game "
        "seeds but visit different states. Resolved labels use a sampling heuristic, not an optimality guarantee. "
        "No training, calibration fitting, model selection, activation, or publication occurs in this command.",
        "activated": False,
        "published": False,
    }
    atomic_json(root / "report.json", result)
    return result


def run_visitation(
    output: Path, source: str, workers=24, hours=1.0, seed=20261002, smoke=False, label_budget="standard"
):
    if not math.isfinite(hours) or not 0 < hours <= 2 or not 1 <= workers <= 32:
        raise ValueError("Pilot requires 1–32 workers and at most two hours.")
    if label_budget not in ("standard", "strong"):
        raise ValueError("Unknown pilot label budget.")
    samples, max_samples = (2048, 16384) if label_budget == "strong" else (512, 4096)
    source = Path(source).absolute()
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    runtime = {"python": sys.version}
    for name in ("laya", "torch", "transformers", "numpy"):
        try:
            runtime[name] = version(name)
        except PackageNotFoundError:
            runtime[name] = None
    config = {
        "version": VERSION,
        "development_only": True,
        "source": str(source),
        "checkpoint": checkpoint_identity(source),
        "workers": workers,
        "hours": hours,
        "seed": seed + (10000000 if smoke else 0),
        "smoke": smoke,
        "scenarios": scenarios(),
        "groups_per_scenario": 1 if smoke else 10,
        "rounds_per_group": 10 if smoke else 100,
        "states_per_group": 2 if smoke else 20,
        "batch_size": 32,
        "label_budget": label_budget,
        "samples": 32 if smoke else samples,
        "max_samples": 64 if smoke else max_samples,
        "qualification_thresholds": {
            "both_resolved_fraction": 0.8,
            "resolved_repeat_agreement": 0.99,
            "depleted_fraction": 0.1,
            "states_per_category": 20,
        },
        "runtime": runtime,
        "code": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    with (output / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = freeze_run(output, config)
        _supervise(output, source, saved)


def _supervise(output, source, saved):
    config, deadline = saved["config"], saved["deadline_unix"]
    current_stage = "freezing_pilot_checkpoint"
    active = []

    def status(state="running", **extra):
        progress = {p.stem: json.loads(p.read_text()) for p in (output / "progress").glob("*.json")}
        atomic_json(
            output / "status.json",
            {
                "status": state,
                "stage": current_stage,
                "experiment": "visitation-pilot",
                "started_unix": saved["started_unix"],
                "deadline_unix": deadline,
                "updated_unix": time.time(),
                "elapsed_seconds": time.time() - saved["started_unix"],
                "progress": progress,
                **extra,
            },
        )

    def interrupted(signum, frame):
        raise InterruptedError(f"Pilot received signal {signum}")

    def stage(commands):
        logs = []
        try:
            for name, args in commands.items():
                log = (output / f"{name}.log").open("a")
                logs.append(log)
                active.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "blackjack.cli",
                            "visitation-worker",
                            "--root",
                            str(output),
                            *args,
                        ],
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
                if time.time() >= deadline:
                    raise TimeoutError("Pilot reached its original wall-clock deadline.")
                if any(p.poll() not in (None, 0) for p in active):
                    raise RuntimeError("Pilot worker failed; verified groups are retained.")
                status()
                time.sleep(3)
            if any(p.returncode != 0 for p in active):
                raise RuntimeError("Pilot worker failed; inspect its log.")
        finally:
            for p in active:
                terminate_group(p)
            active.clear()
            for log in logs:
                log.close()

    old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        status()
        frozen = freeze_checkpoint(source, output / "checkpoints/source", config["checkpoint"])
        current_stage = "collecting_pilot_states"
        stage({p: ["--phase", "collect", "--policy", p, "--device", "cuda:1"] for p in POLICIES})
        current_stage = "qualifying_pilot_labels"
        stage(
            {
                "label": ["--phase", "label"],
                "audit-laya": ["--phase", "audit", "--policy", "laya", "--device", "cuda:1"],
                "audit-mixture": ["--phase", "audit", "--policy", "mixture", "--device", "cuda:0"],
            }
        )
        current_stage = "verifying_pilot"
        status()
        if time.time() >= deadline:
            raise TimeoutError("Original deadline expired before pilot verification.")
        if checkpoint_identity(frozen) != config["checkpoint"]:
            raise ValueError("Frozen pilot checkpoint changed.")
        result = summarize_pilot(output)
        if time.time() >= deadline:
            raise TimeoutError("Original deadline expired during pilot verification.")
        status(
            "complete",
            qualified_for_training_design=result["qualified_for_training_design"],
            decision=result["decision"],
            reports=["report.json"],
        )
    except BaseException as exc:
        status(
            "interrupted" if isinstance(exc, (TimeoutError, InterruptedError)) else "failed", error=str(exc)
        )
        raise
    finally:
        for p in active:
            terminate_group(p)
        for sig, handler in old.items():
            signal.signal(sig, handler)
