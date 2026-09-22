"""Deterministic, restartable parallel generation for larger simulation experiments."""

from __future__ import annotations

import hashlib
import heapq
import json
import multiprocessing
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from .engine import Game, Rules, basic_action, tablemate_action
from .exact_reference import hybrid_analyze
from .model import model_state, questions
from .reference import MAX_REFERENCE_SAMPLES, analyze

DATA_VERSION = "shoe-reservoir-v1"
SPLITS = ("train", "selection", "calibration", "test")


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def game_seed(seed: int, split: str, shard: int, group: int, profile="standard") -> int:
    namespace = DATA_VERSION if profile == "standard" else f"{DATA_VERSION}/{profile}"
    key = f"{namespace}/{seed}/{split}/{shard}/{group}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:16], "big")


def composition_focus(obs):
    """Public-state definition fixed before generating any new experiment labels."""
    hand = obs["players"][0][obs["active"][1]]
    depth = 1 - obs["cards_remaining"] / (52 * obs["rules"]["decks"])
    return depth >= 0.5 and (
        abs(obs["true_count"]) >= 3
        or (not hand["soft"] and 9 <= hand["total"] <= 16)
        or "split" in obs["legal_actions"]
    )


def sample_observations(seed: int, training: bool, focus="general") -> list[dict]:
    """Reservoir across a complete shoe, with extra training emphasis on rare decisions."""
    rng = random.Random(seed)
    rules = Rules(
        players=rng.randint(1, 7),
        decks=rng.choice([1, 2, 4, 6, 8]),
        hit_soft_17=rng.choice([False, True]),
        double_after_split=rng.choice([False, True]),
        surrender=rng.choice([False, True]),
        blackjack_payout=rng.choice([1.2, 1.5]),
        penetration=rng.choice([0.5, 0.65, 0.75, 0.85]),
        max_hands=rng.choice([2, 3, 4]),
        tablemate_policy=rng.choice(["basic", "random", "conservative"]),
    )
    if focus not in ("general", "depleted"):
        raise ValueError("Unknown sampling focus.")
    if focus == "depleted":
        rules = replace(rules, penetration=0.85)
    game = Game(rules, seed)
    reservoir = []
    for step in range(1600):
        if game.phase != "playing":
            game.deal()
            if game.shoe_number > 1:
                break
            continue
        if game.active[0] != 0:
            game.step(tablemate_action(game))
            continue
        obs = game.observation()
        hand = obs["players"][0][obs["active"][1]]
        weight = 1.0
        if training:
            weight += 3 * ("split" in obs["legal_actions"]) + 2 * hand["soft"]
            weight += 2 * (abs(obs["true_count"]) >= 3) + (len(hand["cards"]) > 2)
        if focus == "general" or composition_focus(obs):
            if focus == "depleted" and training:
                depth = 1 - obs["cards_remaining"] / (52 * rules.decks)
                weight += 4 * (depth >= 0.65)
            priority = rng.random() ** (1 / weight)
            heapq.heappush(reservoir, (priority, step, obs))
            if len(reservoir) > 4:
                heapq.heappop(reservoir)
        action = rng.choice(obs["legal_actions"]) if rng.random() < 0.25 else basic_action(obs)
        game.step(action)
    return [obs for _, _, obs in sorted(reservoir, key=lambda item: item[1])]


def generate_shard(task: dict) -> dict:
    root = Path(task["root"])
    name = f"{task['split']}-{task['index']:05d}.jsonl"
    path = root / "shards" / name
    receipt = path.with_suffix(".receipt.json")
    identity = {k: v for k, v in task.items() if k != "root"}
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result["identity"] != identity or not path.exists() or digest(path) != result["sha256"]:
            raise ValueError(f"Existing shard or receipt is inconsistent: {name}")
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    count = resolved = total_rollouts = 0
    group = 0
    players, categories, depths, teachers = {}, {}, {}, {}
    started = time.monotonic()
    with temporary.open("w") as stream:
        while count < task["states"]:
            profile = task.get("profile", "standard")
            focus = (
                "depleted"
                if profile == "composition-v1" and task["split"] != "calibration" and task["index"] % 2
                else "general"
            )
            teacher = task.get("teacher", "monte-carlo")
            namespace = profile if teacher == "monte-carlo" else f"{profile}/{teacher}"
            seed = game_seed(task["seed"], task["split"], task["index"], group, namespace)
            group += 1
            if group > max(10000, task["states"] * 1000):
                raise RuntimeError(
                    "Sampling focus produced too few reachable states; refusing an unbounded loop."
                )
            for position, obs in enumerate(sample_observations(seed, task["split"] == "train", focus)):
                if count >= task["states"]:
                    break
                reference = analyze if teacher == "monte-carlo" else hybrid_analyze
                ref = reference(obs, task["samples"], seed=seed + position, max_samples=task["max_samples"])
                if ref["dealer_unresolved"]:
                    continue
                targets = {
                    "action": {a: float(a == ref["recommendation"]) for a in obs["legal_actions"]},
                    "hit_bust": {"false": 1 - ref["hit_bust"], "true": ref["hit_bust"]},
                    "dealer": ref["dealer"],
                }
                row = {
                    "group_seed": seed,
                    "observation": obs,
                    "state": model_state(obs),
                    "questions": questions(obs),
                    "targets": targets,
                    "reference": ref,
                }
                if profile != "standard":
                    row["stratum"] = focus
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                count += 1
                kind = ref.get("teacher_kind", "monte_carlo")
                teachers[kind] = teachers.get(kind, 0) + 1
                resolved += ref["label_quality"]["resolved"]
                total_rollouts += ref["samples"]
                p = str(obs["rules"]["players"])
                players[p] = players.get(p, 0) + 1
                h = obs["players"][0][obs["active"][1]]
                category = "pair" if "split" in obs["legal_actions"] else "soft" if h["soft"] else "hard"
                categories[category] = categories.get(category, 0) + 1
                depth = str(int((1 - obs["cards_remaining"] / (52 * obs["rules"]["decks"])) * 10) * 10)
                depths[depth] = depths.get(depth, 0) + 1
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    result = {
        "path": f"shards/{name}",
        "sha256": digest(path),
        "states": count,
        "groups": group,
        "resolved": resolved,
        "rollouts_per_action_total": total_rollouts,
        "players": players,
        "categories": categories,
        "depth_percent": depths,
        "teachers": teachers,
        "elapsed_seconds": time.monotonic() - started,
        "identity": identity,
    }
    atomic_json(receipt, result)
    return result


def generate_large_dataset(
    output: Path,
    states=100000,
    selection=2000,
    calibration=2000,
    test=5000,
    workers=8,
    shard_size=128,
    samples=512,
    max_samples=2048,
    evaluation_samples=2048,
    evaluation_max_samples=8192,
    seed=20260921,
    deadline: float | None = None,
    profile="standard",
    teacher="monte-carlo",
):
    counts = dict(zip(SPLITS, [states, selection, calibration, test], strict=True))
    if min(counts.values()) < 1 or workers < 1 or shard_size < 1:
        raise ValueError("Dataset sizes, worker count, and shard size must be positive.")
    if profile not in ("standard", "composition-v1"):
        raise ValueError("Unknown dataset profile.")
    if teacher not in ("monte-carlo", "hybrid-exact-v1"):
        raise ValueError("Unknown reference teacher.")
    if profile == "composition-v1" and any(
        count % (2 * shard_size) for split, count in counts.items() if split != "calibration"
    ):
        raise ValueError("Composition splits need equal complete general/depleted shard pairs.")
    if (
        not 16 <= samples <= max_samples <= MAX_REFERENCE_SAMPLES
        or not 16 <= evaluation_samples <= evaluation_max_samples <= MAX_REFERENCE_SAMPLES
    ):
        raise ValueError("Invalid rollout budgets.")
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": DATA_VERSION,
        "profile": profile,
        "teacher": teacher,
        "counts": counts,
        "shard_size": shard_size,
        "seed": seed,
        "samples": samples,
        "max_samples": max_samples,
        "evaluation_samples": evaluation_samples,
        "evaluation_max_samples": evaluation_max_samples,
        "code": {
            name: digest(Path(__file__).with_name(name))
            for name in ("engine.py", "reference.py", "exact_reference.py", "model.py", "experiment_data.py")
        },
    }
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Resume configuration or source differs from the existing dataset plan.")
    atomic_json(plan_path, plan)
    tasks = []
    # Generate held-out sets first so partial training budgets cannot remove evaluation coverage.
    for split in ("test", "selection", "calibration", "train"):
        for index, offset in enumerate(range(0, counts[split], shard_size)):
            tasks.append(
                {
                    "root": str(output),
                    "split": split,
                    "index": index,
                    "states": min(shard_size, counts[split] - offset),
                    "seed": seed,
                    "profile": profile,
                    "teacher": teacher,
                    "samples": samples if split == "train" else evaluation_samples,
                    "max_samples": max_samples if split == "train" else evaluation_max_samples,
                }
            )
    completed = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {}
        iterator = iter(tasks)
        for _ in range(min(workers * 2, len(tasks))):
            task = next(iterator)
            pending[pool.submit(generate_shard, task)] = task
        while pending:
            exhausted = deadline is not None and time.time() >= deadline
            ready, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                pending.pop(future)
                completed.append(future.result())
                next_task = None if exhausted else next(iterator, None)
                if next_task is not None:
                    pending[pool.submit(generate_shard, next_task)] = next_task
                progress = {
                    "stage": "generation",
                    "completed_shards": len(completed),
                    "total_shards": len(tasks),
                    "states": {
                        s: sum(r["states"] for r in completed if r["identity"]["split"] == s) for s in SPLITS
                    },
                    "elapsed_seconds": time.monotonic() - started,
                }
                atomic_json(output / "progress.json", progress)
                print(f"generation {len(completed)}/{len(tasks)} shards · {progress['states']}", flush=True)
    splits = {
        s: sorted([r for r in completed if r["identity"]["split"] == s], key=lambda r: r["path"])
        for s in SPLITS
    }
    manifest = {
        "plan": plan,
        "splits": splits,
        "elapsed_seconds": time.monotonic() - started,
        "teacher": "Public-state finite-shoe Monte Carlo, basic continuation, paired adaptive sampling",
        "selection": "Complete-shoe reservoir; rare-state weighting only in training split",
    }
    if teacher == "hybrid-exact-v1":
        manifest["teacher"] = (
            "Exact optimal finite-shoe single-player unsplit decisions within 50000 cache misses; "
            "explicit basic-continuation Monte Carlo fallback otherwise. Per-row method/coverage recorded."
        )
    manifest["teacher_counts"] = {
        s: {k: sum(r.get("teachers", {}).get(k, 0) for r in rows) for k in ("exact", "monte_carlo")}
        for s, rows in splits.items()
    }
    if profile == "composition-v1":
        manifest["selection"] = (
            "Equal general/depleted shard pairs for train, selection and final test; "
            "calibration uses general games only. Depleted games use 85% penetration, "
            "public composition_focus predicate, and training-only emphasis beyond 65% depth."
        )
    manifest["actual_states"] = {s: sum(r["states"] for r in rows) for s, rows in splits.items()}
    manifest["complete"] = manifest["actual_states"] == counts
    if any(manifest["actual_states"][s] != counts[s] for s in SPLITS if s != "train"):
        raise TimeoutError("Budget reached before evaluation sets completed; resume shard generation.")
    if not manifest["actual_states"]["train"]:
        raise TimeoutError("No complete training shards within generation budget; resume generation.")
    atomic_json(output / "manifest.json", manifest)
    return manifest


def verify_dataset(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    groups = {s: set() for s in SPLITS}
    for split, shards in manifest["splits"].items():
        for shard in shards:
            path = root / shard["path"]
            if digest(path) != shard["sha256"]:
                raise ValueError(f"Dataset hash mismatch: {path}")
            count = 0
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    groups[split].add(row["group_seed"])
                    count += 1
            if count != shard["states"]:
                raise ValueError(f"Row count mismatch: {path}")
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            if groups[a] & groups[b]:
                raise ValueError(f"Group leakage between {a} and {b}")
    return manifest
