"""Paired policy evaluation with fixed strata and independent round/block units."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .engine import Game, Rules, basic_action
from .experiment_data import atomic_json, digest
from .model import ensure_fits, load_agent, model_state, questions
from .reference import analyze, rollout_action

EVALUATION_VERSION = "paired-policy-v1"


def scenarios():
    result = {f"s17-6d-{p}p": asdict(Rules(players=p)) for p in range(1, 8)}
    result["h17-no-das-6to5"] = asdict(
        Rules(hit_soft_17=True, double_after_split=False, blackjack_payout=1.2)
    )
    result["s17-2d-3p"] = asdict(Rules(decks=2))
    result["random-tablemates-7p"] = asdict(Rules(players=7, tablemate_policy="random"))
    return result


def evaluation_seed(seed, mode, scenario, index):
    key = f"{EVALUATION_VERSION}/{seed}/{mode}/{scenario}/{index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:16], "big")


class BasicPolicy:
    def actions(self, observations):
        return [basic_action(obs) for obs in observations]


class ReferencePolicy:
    def __init__(self, samples):
        self.samples = samples

    def actions(self, observations):
        return [analyze(obs, self.samples)["recommendation"] for obs in observations]


class BatchedLaya:
    """Action-only batches; near ties use the exact three-question serving call."""

    def __init__(self, source, device):
        self.agent = load_agent(source, device)
        if device.startswith("cuda") and self.agent.device.type != "cuda":
            raise RuntimeError("Requested GPU is unavailable; refusing a silent CPU evaluation.")
        self.fallbacks = 0

    def actions(self, observations):
        import torch
        from laya.common import QTYPES, build_sequence, collate_items

        if not observations:
            return []
        items, states, keys = [], [], []
        for obs in observations:
            state, qs = model_state(obs), questions(obs)
            ensure_fits(self.agent, state, qs)
            q = qs["action"]
            ids, markers = build_sequence(
                self.agent.tok,
                state,
                self.agent._to_internal(q),
                self.agent.cfg["max_len"],
                self.agent.cfg["head_max_len"],
            )
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES["choice"]})
            states.append(state)
            keys.append(list(q["criteria"]))
        batch = collate_items([items], self.agent.tok.pad_token_id)
        kwargs = {
            k: batch[k].to(self.agent.device)
            for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
        }
        with (
            torch.inference_mode(),
            torch.autocast(
                self.agent.device.type, dtype=self.agent.dtype, enabled=self.agent.device.type == "cuda"
            ),
        ):
            logits, _ = self.agent.model(**kwargs)
        logits = logits.float().cpu().numpy()
        actions = []
        for i, choices in enumerate(keys):
            z = logits[i, : len(choices)]
            # BF16 batching changes last-bit rounding. Check uncertain actions through serving.
            if len(z) > 1 and np.sort(z)[-1] - np.sort(z)[-2] < 0.15:
                result = self.agent.predict(states[i], questions(observations[i]))
                action = result["answers"]["action"]["choice"]
                self.fallbacks += 1
            else:
                action = choices[int(z.argmax())]
            actions.append(action)
        return actions


class ServingLaya(BatchedLaya):
    """Use the dashboard's complete three-question SDK call for every decision."""

    def actions(self, observations):
        actions = []
        device = self.agent.device.type
        for obs in observations:
            state, qs = model_state(obs), questions(obs)
            ensure_fits(self.agent, state, qs)
            result = self.agent.predict(state, qs)
            if self.agent.device.type != device:
                raise RuntimeError("SDK changed inference device during evaluation.")
            action = result["answers"]["action"]["choice"]
            if action not in obs["legal_actions"]:
                raise ValueError("Serving SDK returned an illegal action.")
            actions.append(action)
        return actions


def simulate_units(policy, scenario, rules, indices, *, seed, mode, rounds_per_unit, batch_size):
    records = []
    indices = list(indices)
    for start in range(0, len(indices), batch_size):
        slots = []
        for index in indices[start : start + batch_size]:
            game = Game(Rules(**rules), evaluation_seed(seed, mode, scenario, index))
            game.deal()
            slots.append(
                {
                    "game": game,
                    "index": index,
                    "rounds": 0,
                    "profit": 0.0,
                    "positive": 0,
                    "voids": 0,
                    "decisions": 0,
                }
            )
        while slots:
            waiting = []
            for slot in slots:
                game = slot["game"]
                while True:
                    if game.phase != "playing":
                        slot["profit"] += game.history[-1]["profit"]
                        slot["positive"] += game.history[-1]["profit"] > 0
                        slot["voids"] += game.phase == "void"
                        slot["rounds"] += 1
                        if slot["rounds"] == rounds_per_unit:
                            records.append(
                                {
                                    "scenario": scenario,
                                    "index": slot["index"],
                                    **{
                                        k: slot[k]
                                        for k in ("rounds", "profit", "positive", "voids", "decisions")
                                    },
                                    "shoes": game.shoe_number,
                                }
                            )
                            break
                        game.deal()
                    elif game.active[0] != 0:
                        game.step(rollout_action(game, 0))
                    else:
                        waiting.append(slot)
                        break
            slots = waiting
            if slots:
                observations = [slot["game"].observation() for slot in slots]
                actions = policy.actions(observations)
                if len(actions) != len(slots):
                    raise ValueError("Policy returned the wrong number of actions.")
                for slot, action in zip(slots, actions, strict=True):
                    slot["game"].step(action)
                    slot["decisions"] += 1
    return sorted(records, key=lambda row: row["index"])


def evaluate_policy(
    output: Path,
    policy="laya",
    source=None,
    device="cuda:0",
    mode="fresh",
    units=100000,
    seed=20260922,
    batch_size=32,
    shard_size=256,
    samples=256,
    inference_mode="batched",
):
    strata = scenarios()
    if units < len(strata) * 2 or units % len(strata) or min(batch_size, shard_size) < 1:
        raise ValueError("Use a positive batch/shard size and equal strata with at least two units each.")
    if mode not in ("fresh", "continuous") or policy not in ("laya", "basic", "reference"):
        raise ValueError("Invalid evaluation mode or policy.")
    if policy == "laya" and not source:
        raise ValueError("Laya evaluation requires a checkpoint.")
    if inference_mode not in ("batched", "sdk"):
        raise ValueError("Unknown inference mode.")
    rounds_per_unit = 1 if mode == "fresh" else 100
    identity = {
        "version": EVALUATION_VERSION,
        "scenarios": strata,
        "mode": mode,
        "units": units,
        "rounds_per_unit": rounds_per_unit,
        "seed": seed,
    }
    config = {
        "identity": identity,
        "policy": policy,
        "batch_size": batch_size,
        "shard_size": shard_size,
        "samples": samples,
        "receipt_version": 2,
        "inference_mode": inference_mode,
        "source_hash": digest(Path(source) / "model.safetensors") if source else None,
        "config_hash": digest(Path(source) / "rl_agent_config.json") if source else None,
        "code": {
            name: digest(Path(__file__).with_name(name))
            for name in ("engine.py", "reference.py", "model.py", "evaluation.py")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Evaluation resume configuration changed.")
    atomic_json(config_path, config)
    actor = (
        (ServingLaya if inference_mode == "sdk" else BatchedLaya)(source, device)
        if policy == "laya"
        else ReferencePolicy(samples)
        if policy == "reference"
        else BasicPolicy()
    )
    started, receipts, completed, fallbacks = time.monotonic(), [], 0, 0
    per_stratum = units // len(strata)
    for scenario, rules in strata.items():
        for offset in range(0, per_stratum, shard_size):
            path = output / "shards" / f"{scenario}-{offset:06d}.json"
            receipt_path = path.with_suffix(".receipt.json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if digest(path) != receipt["sha256"]:
                    raise ValueError(f"Evaluation shard hash mismatch: {path}")
            else:
                before_fallbacks = getattr(actor, "fallbacks", 0)
                rows = simulate_units(
                    actor,
                    scenario,
                    rules,
                    range(offset, min(offset + shard_size, per_stratum)),
                    seed=seed,
                    mode=mode,
                    rounds_per_unit=rounds_per_unit,
                    batch_size=batch_size,
                )
                atomic_json(path, rows)
                receipt = {
                    "path": str(path.relative_to(output)),
                    "sha256": digest(path),
                    "units": len(rows),
                    "serving_fallbacks": getattr(actor, "fallbacks", 0) - before_fallbacks,
                }
                atomic_json(receipt_path, receipt)
            receipts.append(receipt)
            completed += receipt["units"]
            fallbacks += receipt["serving_fallbacks"]
            progress = {
                "stage": "evaluation",
                "policy": policy,
                "mode": mode,
                "completed_units": completed,
                "total_units": units,
                "rounds": completed * rounds_per_unit,
                "elapsed_seconds": time.monotonic() - started,
                "updated_unix": time.time(),
                "serving_fallbacks": fallbacks,
            }
            atomic_json(output / "progress.json", progress)
            print(f"{policy} {mode}: {completed}/{units} independent units", flush=True)
    manifest = {"config": config, "shards": receipts, "elapsed_seconds": time.monotonic() - started}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "progress.json", {**progress, "stage": "complete"})
    return manifest


def read_evaluation(root):
    manifest = json.loads((root / "manifest.json").read_text())
    rows = {}
    for shard in manifest["shards"]:
        path = root / shard["path"]
        if digest(path) != shard["sha256"]:
            raise ValueError(f"Evaluation shard hash mismatch: {path}")
        values = json.loads(path.read_text())
        if len(values) != shard["units"]:
            raise ValueError("Evaluation shard length mismatch.")
        for row in values:
            key = (row["scenario"], row["index"])
            if key in rows:
                raise ValueError("Duplicate independent evaluation unit.")
            rows[key] = row
    expected = manifest["config"]["identity"]["units"]
    identity = manifest["config"]["identity"]
    expected_keys = {
        (name, index)
        for name in identity["scenarios"]
        for index in range(expected // len(identity["scenarios"]))
    }
    if set(rows) != expected_keys or any(
        row["rounds"] != identity["rounds_per_unit"] or not math.isfinite(row["profit"])
        for row in rows.values()
    ):
        raise ValueError("Incomplete evaluation.")
    return manifest, rows


def estimate(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "mean_units_per_round": mean,
        "standard_error": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "independent_units": len(values),
    }


def compare_evaluations(inputs: dict[str, Path], output: Path, candidate="candidate"):
    datasets = {name: read_evaluation(path) for name, path in inputs.items()}
    identities = [data[0]["config"]["identity"] for data in datasets.values()]
    if (
        candidate not in inputs
        or len(inputs) < 2
        or any(identity != identities[0] for identity in identities)
    ):
        raise ValueError("Compare the candidate with at least one policy on identical evaluation units.")
    code_versions = [data[0]["config"]["code"] for data in datasets.values()]
    if any(code != code_versions[0] for code in code_versions):
        raise ValueError("Policies must use identical simulator and evaluator source code.")
    identity = identities[0]
    keys = sorted(datasets[candidate][1])
    if any(sorted(data[1]) != keys for data in datasets.values()):
        raise ValueError("Paired evaluation keys do not match.")
    strata = list(identity["scenarios"])
    results = {}
    for name, (manifest, rows) in datasets.items():
        values = {s: [rows[k]["profit"] / rows[k]["rounds"] for k in keys if k[0] == s] for s in strata}
        by_scenario = {s: estimate(v) for s, v in values.items()}
        mean = float(np.mean([r["mean_units_per_round"] for r in by_scenario.values()]))
        # Equal fixed stratum weights; uncertainty respects round/block independence.
        se = math.sqrt(sum(r["standard_error"] ** 2 for r in by_scenario.values())) / len(strata)
        results[name] = {
            "mean_units_per_round": mean,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "rounds": sum(r["rounds"] for r in rows.values()),
            "voids": sum(r["voids"] for r in rows.values()),
            "by_scenario": by_scenario,
            "model_hash": manifest["config"]["source_hash"],
            **(
                {
                    "inference_mode": manifest["config"]["inference_mode"]
                    if manifest["config"]["policy"] == "laya"
                    else "not_applicable"
                }
                if "inference_mode" in manifest["config"]
                else {}
            ),
        }
    comparisons = {}
    candidate_rows = datasets[candidate][1]
    for name, (_, rows) in datasets.items():
        if name == candidate:
            continue
        values = {
            s: [
                (candidate_rows[k]["profit"] - rows[k]["profit"]) / rows[k]["rounds"]
                for k in keys
                if k[0] == s
            ]
            for s in strata
        }
        by_scenario = {s: estimate(v) for s, v in values.items()}
        mean = float(np.mean([r["mean_units_per_round"] for r in by_scenario.values()]))
        se = math.sqrt(sum(r["standard_error"] ** 2 for r in by_scenario.values())) / len(strata)
        comparisons[name] = {
            "mean_difference": mean,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "by_scenario": by_scenario,
        }
    report = {
        "identity": identity,
        "results": results,
        "candidate": candidate,
        "paired_candidate_minus": comparisons,
        "inference": "Fixed equal-weight strata; normal 95% intervals over independent rounds or 100-round blocks. Scenario intervals are exploratory, without multiplicity correction.",
        "pairing": "Identical initial RNG seeds per unit; different actions can diverge subsequent trajectories.",
        "limitations": "The basic policy is a multi-deck heuristic. Fixed wagers, no insurance; these simulator results do not establish optimality or real-world profitability.",
    }
    atomic_json(output, report)
    return report
