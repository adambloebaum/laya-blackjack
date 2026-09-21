"""Serving-SDK validation and descriptive error analysis on the frozen test set."""

from __future__ import annotations

import json
import math
import multiprocessing
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .engine import value
from .evaluation import BatchedLaya
from .experiment_data import atomic_json, digest, verify_dataset
from .model import model_state, questions
from .reference import clone_world, finish_world, sample_world


def audit_model(dataset: Path, source: str, output: Path, device="cuda:1", batch_size=32):
    output.mkdir(parents=True, exist_ok=True)
    manifest = verify_dataset(dataset)
    expected = json.loads((Path(source) / "training_report.json").read_text())
    if expected["dataset_hash"] != digest(dataset / "manifest.json"):
        raise ValueError("Checkpoint and audit dataset do not match.")
    actor = BatchedLaya(source, device)
    rows = []
    for shard in manifest["splits"]["test"]:
        with (dataset / shard["path"]).open() as stream:
            rows.extend(json.loads(line) for line in stream)
    records, mismatches, latencies = [], [], []
    started = time.monotonic()
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset : offset + batch_size]
        fast_actions = actor.actions([row["observation"] for row in chunk])
        for index, (row, fast_action) in enumerate(zip(chunk, fast_actions, strict=True), offset):
            obs = row["observation"]
            before = time.monotonic()
            answers = actor.agent.predict(model_state(obs), questions(obs))["answers"]
            latencies.append(time.monotonic() - before)
            action = answers["action"]["choice"]
            if action != fast_action:
                mismatches.append(index)
            ref = row["reference"]
            hand = obs["players"][0][obs["active"][1]]
            gap = max(v["ev"] for v in ref["actions"].values()) - ref["actions"][action]["ev"]
            category = "pair" if "split" in obs["legal_actions"] else "soft" if hand["soft"] else "hard"
            dealer_brier = sum(
                (answers["dealer"]["probabilities"][k] - p) ** 2 for k, p in ref["dealer"].items()
            )
            hit_brier = 2 * (answers["hit_bust"]["noul"] - ref["hit_bust"]) ** 2
            record = {
                "index": index,
                "group_seed": row["group_seed"],
                "players": obs["rules"]["players"],
                "decks": obs["rules"]["decks"],
                "hit_soft_17": obs["rules"]["hit_soft_17"],
                "category": category,
                "hand": [value(c) for c in hand["cards"]],
                "total": hand["total"],
                "dealer_up": value(obs["dealer"][0]),
                "true_count": obs["true_count"],
                "depth": min(9, int((1 - obs["cards_remaining"] / (52 * obs["rules"]["decks"])) * 10)),
                "action": action,
                "reference_action": ref["recommendation"],
                "regret": gap,
                "confidence": answers["action"]["probabilities"][action],
                "resolved": ref["label_quality"]["resolved"],
                "correct": action == ref["recommendation"],
                "action_values": {k: v["ev"] for k, v in ref["actions"].items()},
                "hit_bust_brier": hit_brier,
                "dealer_brier": dealer_brier,
            }
            records.append(record)
        atomic_json(
            output / "progress.json",
            {
                "stage": "sdk_audit",
                "completed": len(records),
                "total": len(rows),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        print(f"SDK audit: {len(records)}/{len(rows)}", flush=True)

    def summarize(items):
        return {
            "states": len(items),
            "teacher_agreement": float(np.mean([r["correct"] for r in items])),
            "teacher_ev_regret": float(np.mean([r["regret"] for r in items])),
            "hit_bust_brier": float(np.mean([r["hit_bust_brier"] for r in items])),
            "dealer_brier": float(np.mean([r["dealer_brier"] for r in items])),
        }

    grouped = {}
    for field in ("players", "decks", "hit_soft_17", "category", "depth", "resolved"):
        groups = defaultdict(list)
        for record in records:
            groups[str(record[field])].append(record)
        grouped[field] = {key: summarize(items) for key, items in sorted(groups.items())}
    # Bootstrap whole-game groups, never individual correlated decision rows.
    group_rows = defaultdict(list)
    for r in records:
        group_rows[r["group_seed"]].append(r)
    group_stats = np.array(
        [
            [len(items), sum(r["correct"] for r in items), sum(r["regret"] for r in items)]
            for items in group_rows.values()
        ]
    )
    rng = np.random.default_rng(20260923)
    replicates = []
    for _ in range(2000):
        total = group_stats[rng.integers(0, len(group_stats), len(group_stats))].sum(axis=0)
        replicates.append(total[1:] / total[0])
    intervals = np.quantile(replicates, [0.025, 0.975], axis=0)
    metrics = summarize(records)
    worst = sorted(records, key=lambda r: r["regret"], reverse=True)[:30]
    summary = {
        "model_sha256": digest(Path(source) / "model.safetensors"),
        "dataset_sha256": digest(dataset / "manifest.json"),
        "metrics": metrics,
        "game_groups": len(group_rows),
        "cluster_bootstrap_replicates": 2000,
        "teacher_agreement_ci95": intervals[:, 0].tolist(),
        "teacher_ev_regret_ci95": intervals[:, 1].tolist(),
        "trainer_report_metrics": expected["test"],
        "batched_action_mismatches": mismatches,
        "batch_size": batch_size,
        "serving_fallbacks": actor.fallbacks,
        "median_sdk_latency_ms": float(np.median(latencies) * 1000),
        "by_group": grouped,
        "largest_regrets": worst,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "Descriptive analysis of the already frozen final test, not a new independent validation or a basis for selecting this release. Reusing these examples for tuning requires a new final test set.",
        "uncertainty": "Game-cluster bootstrap; conditions on the approximate Monte Carlo labels, excluding teacher sampling uncertainty.",
    }
    atomic_json(output / "report.json", summary)
    atomic_json(output / "decisions.json", records)
    # Preserve exact public states for reproducible investigation, never hidden simulator state.
    atomic_json(
        output / "worst-states.json",
        [{"audit": r, "observation": rows[r["index"]]["observation"]} for r in worst],
    )
    atomic_json(
        output / "progress.json", {"stage": "complete", "completed": len(records), "total": len(rows)}
    )
    return summary


def _recheck_contrast(task):
    obs, model_action, teacher_action, seed, samples = task
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        world = sample_world(obs, rng)
        teacher = finish_world(clone_world(world), teacher_action, obs["active"][0])
        model = finish_world(clone_world(world), model_action, obs["active"][0])
        differences.append(teacher - model)
    return {
        "samples": samples,
        "sum": sum(differences),
        "sum_squares": sum(x * x for x in differences),
        "seed": seed,
    }


def recheck_errors(audit: Path, output: Path, workers=24, samples=10000, repeats=4):
    worst = json.loads((audit / "worst-states.json").read_text())
    tasks = [
        (
            item["observation"],
            item["audit"]["action"],
            item["audit"]["reference_action"],
            202609230000 + i * repeats + repeat,
            samples,
        )
        for i, item in enumerate(worst)
        for repeat in range(repeats)
    ]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        contrasts = list(pool.map(_recheck_contrast, tasks))
    results = []
    for i, item in enumerate(worst):
        chunks = contrasts[i * repeats : (i + 1) * repeats]
        n = sum(r["samples"] for r in chunks)
        total = sum(r["sum"] for r in chunks)
        ss = sum(r["sum_squares"] for r in chunks)
        mean = total / n
        se = math.sqrt(max(0, (ss - total * total / n) / (n - 1) / n))
        results.append(
            {
                "decision": item["audit"],
                "fresh_worlds": n,
                "teacher_minus_model_ev": mean,
                "standard_error": se,
                "ci95": [mean - 1.96 * se, mean + 1.96 * se],
                "replicates": chunks,
            }
        )
    report = {
        "cases": results,
        "elapsed_seconds": time.monotonic() - started,
        "method": "Fixed original teacher action versus fixed model action, paired fresh hidden worlds, basic continuation. No reselection of actions from these samples.",
        "scope": "Exploratory recheck of the 30 largest frozen-test regrets; per-case intervals are not simultaneous confidence guarantees.",
    }
    atomic_json(output, report)
    return report
