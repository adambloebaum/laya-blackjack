"""Matched training data: shared broad coverage and paired behavior reservoirs."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .exact_reference import hybrid_analyze
from .experiment_data import SPLITS, atomic_json, digest, verify_dataset
from .model import model_state, questions
from .visitation import content_hash, group_path, read_group, read_record, replay_group, write_record

VERSION = "model-visitation-training-v1"
ARMS = {"control": "mixture", "visited": "laya"}


def study_seed(seed, *parts):
    value = "/".join(map(str, (VERSION, seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:16], "big")


def collection_config(config):
    return {
        "version": VERSION,
        "seed": study_seed(config["seed"], "replacement", "train"),
        "scenarios": config["scenarios"],
        "groups_per_scenario": config["groups_per_scenario"],
        "rounds_per_group": config["rounds_per_group"],
        "states_per_group": config["states_per_group"],
        "batch_size": 32,
        "checkpoint": config["checkpoint"],
        "parent_config_sha256": content_hash(config),
    }


def pair_records(root, scenario, index):
    collection = root / "collection"
    records = {
        arm: read_group(collection, "collect", policy, scenario, index) for arm, policy in ARMS.items()
    }
    if any(value is None for value in records.values()):
        raise ValueError("A paired collection group is missing.")
    if records["control"]["group_seed"] != records["visited"]["group_seed"]:
        raise ValueError("Replacement arms do not share initial game seeds.")
    for group in records.values():
        replay_group(group)
    identity = {
        "run_sha256": digest(root / "run.json"),
        "parents": {
            arm: digest(group_path(collection, "collect", policy, scenario, index))
            for arm, policy in ARMS.items()
        },
    }
    return records, identity


def label_pair(task):
    root, scenario, index = task
    saved = json.loads((root / "run.json").read_text())
    config = saved["config"]
    if time.time() >= saved["deadline_unix"]:
        raise TimeoutError("Original study deadline expired during labeling.")
    records, identity = pair_records(root, scenario, index)
    path = root / "assembly/labels" / f"{scenario}-{index:04d}.json"
    cached = read_record(path, identity)
    if cached is not None:
        return cached
    references, observed = {}, {}
    for arm, group in records.items():
        observed[arm] = set()
        for row in group["rows"]:
            key = row["observation_sha256"]
            observed[arm].add(key)
            if key in references:
                continue
            if time.time() >= saved["deadline_unix"]:
                raise TimeoutError("Original study deadline expired during labeling.")
            ref = hybrid_analyze(
                row["observation"],
                config["samples"],
                seed=study_seed(config["seed"], "replacement-label", "train", group["group_seed"], key),
                max_samples=config["max_samples"],
            )
            if abs(ref["dealer_unresolved"]) > 1e-12:
                raise ValueError("Replacement label has incomplete dealer targets.")
            references[key] = ref
    result = {
        "scenario": scenario,
        "index": index,
        "references": references,
        "shared_observations": len(observed["control"] & observed["visited"]),
        "states": sum(len(g["rows"]) for g in records.values()),
    }
    write_record(path, result, identity)
    return result


def row_from_reference(group, row, reference):
    obs = row["observation"]
    if reference["recommendation"] not in obs["legal_actions"]:
        raise ValueError("Reference selected an illegal action.")
    return {
        "group_seed": group["group_seed"],
        "observation": obs,
        "state": model_state(obs),
        "questions": questions(obs),
        "targets": {
            "action": {a: float(a == reference["recommendation"]) for a in obs["legal_actions"]},
            "hit_bust": {"false": 1 - reference["hit_bust"], "true": reference["hit_bust"]},
            "dealer": reference["dealer"],
        },
        "reference": reference,
        "stratum": "replacement",
        "source_family": group["policy"],
    }


def publish_bytes(path, body):
    """Idempotent publication; never silently replace an inconsistent completed file."""
    expected = hashlib.sha256(body).hexdigest()
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f"Existing assembled artifact changed: {path.name}")
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return expected


def replacement_shard(root, index, rows, inputs):
    name = f"shards/train-replacement-{index:05d}.jsonl"
    body = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    sha256 = publish_bytes(root / name, body)
    result = {
        "path": name,
        "sha256": sha256,
        "states": len(rows),
        "groups": len({r["group_seed"] for r in rows}),
        "teachers": dict(Counter(r["reference"]["teacher_kind"] for r in rows)),
        "resolved": sum(r["reference"]["label_quality"]["resolved"] for r in rows),
        "rollouts_per_action_total": sum(r["reference"]["samples"] for r in rows),
        "source_family": "replacement",
        "input_sha256": inputs,
    }
    publish_bytes((root / name).with_suffix(".receipt.json"), (json.dumps(result, indent=2) + "\n").encode())
    return result


def copy_shared(root, broad, manifest):
    result = {split: [] for split in SPLITS}
    for split, shards in manifest["splits"].items():
        for shard in shards:
            name = shard["path"].replace("train-", "train-broad-") if split == "train" else shard["path"]
            target = root / name
            if target.exists():
                if digest(target) != shard["sha256"]:
                    raise ValueError("Shared dataset copy changed.")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".partial")
                shutil.copyfile(broad / shard["path"], temporary)
                if digest(temporary) != shard["sha256"]:
                    raise ValueError("Shared dataset changed while copying.")
                temporary.replace(target)
            result[split].append({**shard, "path": name, "source_family": "broad"})
    return result


def verify_matched_data(root):
    config = json.loads((root / "run.json").read_text())["config"]
    broad = verify_dataset(root / "broad")
    manifests = {arm: verify_dataset(root / "datasets" / arm) for arm in ARMS}
    expected = {"train": 2 * config["shared_states"], **config["heldout_states"]}
    signatures = []
    for manifest in manifests.values():
        if not manifest["complete"] or manifest["actual_states"] != expected:
            raise ValueError("Matched datasets require every planned state.")
        for split in ("selection", "calibration", "test"):
            a = [(s["sha256"], s["states"]) for s in manifest["splits"][split]]
            b = [(s["sha256"], s["states"]) for s in broad["splits"][split]]
            if a != b:
                raise ValueError("Held-out rows differ between training arms and the common audit dataset.")
        shared = [s for s in manifest["splits"]["train"] if s["source_family"] == "broad"]
        if [(s["sha256"], s["states"]) for s in shared] != [
            (s["sha256"], s["states"]) for s in broad["splits"]["train"]
        ]:
            raise ValueError("Broad training rows are not shared exactly.")
        signatures.append([s["states"] for s in manifest["splits"]["train"]])
        if sum(s["states"] for s in shared) != config["shared_states"]:
            raise ValueError("Wrong shared training state count.")
    if signatures[0] != signatures[1]:
        raise ValueError("Training arms have different shard/batch schedules.")
    updates = config["epochs"] * sum(math.ceil(3 * n / config["batch_size"]) for n in signatures[0])
    return {
        "datasets": {arm: digest(root / "datasets" / arm / "manifest.json") for arm in ARMS},
        "common_audit_dataset": digest(root / "broad/manifest.json"),
        "planned_updates_per_arm": updates,
        "states_per_arm": expected,
    }


def assemble_study(root: Path):
    saved = json.loads((root / "run.json").read_text())
    config = saved["config"]
    broad = verify_dataset(root / "broad")
    if not broad["complete"] or broad["actual_states"] != {
        "train": config["shared_states"],
        **config["heldout_states"],
    }:
        raise ValueError("Broad source data must be complete before assembly.")
    tasks = [
        (root, scenario, i) for scenario in config["scenarios"] for i in range(config["groups_per_scenario"])
    ]
    done = states = 0
    with ProcessPoolExecutor(
        max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for future in as_completed([pool.submit(label_pair, t) for t in tasks]):
            result = future.result()
            done += 1
            states += result["states"]
            atomic_json(
                root / "assembly/progress.json",
                {
                    "stage": "replacement_labeling",
                    "completed": done,
                    "total": len(tasks),
                    "unit": "paired trajectory groups",
                    "states": states,
                },
            )
    manifests = {arm: copy_shared(root / "datasets" / arm, root / "broad", broad) for arm in ARMS}
    buffers = {arm: [] for arm in ARMS}
    source_stats = {arm: Counter() for arm in ARMS}
    indices = dict.fromkeys(ARMS, 0)
    evidence = {}
    shared_labels = 0
    unique_worlds = unique_labels = 0
    inputs = content_hash({"run": digest(root / "run.json"), "broad": digest(root / "broad/manifest.json")})
    for _, scenario, index in tasks:
        if time.time() >= saved["deadline_unix"]:
            raise TimeoutError("Original study deadline expired during assembly.")
        groups, identity = pair_records(root, scenario, index)
        path = root / "assembly/labels" / f"{scenario}-{index:04d}.json"
        labels = read_record(path, identity)
        if labels is None:
            raise ValueError("Missing replacement labels.")
        evidence[str(path.relative_to(root))] = digest(path)
        evidence[str(path.with_suffix(".receipt.json").relative_to(root))] = digest(
            path.with_suffix(".receipt.json")
        )
        for policy in ARMS.values():
            collection_path = group_path(root / "collection", "collect", policy, scenario, index)
            for artifact in (collection_path, collection_path.with_suffix(".receipt.json")):
                evidence[str(artifact.relative_to(root))] = digest(artifact)
        shared_labels += labels["shared_observations"]
        unique_labels += len(labels["references"])
        unique_worlds += sum(r["samples"] for r in labels["references"].values())
        for arm, group in groups.items():
            if (
                len(group["rows"]) != config["states_per_group"]
                or group["rounds"] != config["rounds_per_group"]
            ):
                raise ValueError("Replacement reservoir or trajectory is incomplete.")
            for row in group["rows"]:
                reference = labels["references"][row["observation_sha256"]]
                buffers[arm].append(row_from_reference(group, row, reference))
                source_stats[arm]["states"] += 1
                source_stats[arm][reference["teacher_kind"]] += 1
                source_stats[arm]["resolved"] += reference["label_quality"]["resolved"]
                source_stats[arm]["worlds"] += reference["samples"]
                if len(buffers[arm]) == config["shard_size"]:
                    manifests[arm]["train"].append(
                        replacement_shard(root / "datasets" / arm, indices[arm], buffers[arm], inputs)
                    )
                    indices[arm] += 1
                    buffers[arm] = []
    if any(buffers.values()) or any(s["states"] != config["shared_states"] for s in source_stats.values()):
        raise ValueError("Replacement states do not form the planned complete shards.")
    for arm, splits in manifests.items():
        manifest = {
            "plan": {"version": VERSION, "arm": arm, "input_sha256": inputs, "seed": config["seed"]},
            "splits": splits,
            "actual_states": {s: sum(r["states"] for r in rows) for s, rows in splits.items()},
            "complete": True,
            "teacher": broad["teacher"],
            "teacher_counts": {
                s: dict(
                    Counter(
                        {
                            k: sum(r.get("teachers", {}).get(k, 0) for r in rows)
                            for k in ("exact", "monte_carlo")
                        }
                    )
                )
                for s, rows in splits.items()
            },
            "selection": "Common group-disjoint general/depleted selection and test; general calibration. "
            "Training is half identical broad rows, half paired behavior-specific uniform reservoirs.",
        }
        publish_bytes(
            root / "datasets" / arm / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode()
        )
    matched = verify_matched_data(root)
    result = {
        **matched,
        "source_stats": source_stats,
        "shared_replacement_labels": shared_labels,
        "unique_reference_calls": unique_labels,
        "unique_reference_worlds": unique_worlds,
        "label_evidence": evidence,
        "input_sha256": inputs,
    }
    publish_bytes(root / "assembly/report.json", (json.dumps(result, indent=2) + "\n").encode())
    return result


def verify_collection_serving(root):
    config = json.loads((root / "run.json").read_text())["config"]
    count = 0
    for scenario in config["scenarios"]:
        for index in range(config["groups_per_scenario"]):
            group = read_group(root / "collection", "collect", "laya", scenario, index)
            audit = read_group(root / "collection", "audit", "laya", scenario, index)
            if group is None or audit is None:
                raise ValueError("Collection serving audit is incomplete.")
            for row, pred in zip(group["rows"], audit["rows"], strict=True):
                if (row["event_index"], row["observation_sha256"]) != (
                    pred["event_index"],
                    pred["observation_sha256"],
                ):
                    raise ValueError("Collection serving audit identities differ.")
                if row["behavior_action"] != pred["sdk_action"] or pred["batch_action"] != pred["sdk_action"]:
                    raise ValueError("Frozen collector and serving actions differ.")
                count += 1
    if count != config["shared_states"]:
        raise ValueError("Collection serving audit has the wrong state count.")
    return {"states": count, "action_mismatches": 0}
