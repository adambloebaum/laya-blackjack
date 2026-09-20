"""Simulation distillation, separate-seed calibration, and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .engine import Game, Rules, basic_action, tablemate_action
from .model import BASE_MODEL, BASE_REVISION, ensure_fits, load_agent, model_state, questions
from .reference import analyze


def generate_dataset(output: Path, states: int = 500, samples: int = 256, seed: int = 42):
    if states < 10:
        raise ValueError("At least 10 training states are required.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "seed": seed,
        "states": states,
        "samples": samples,
        "splits": {},
        "teacher": "finite-shoe MC, basic continuation; approximate policy improvement",
    }
    for index, (split, count) in enumerate(
        (("train", states), ("validation", max(10, states // 5)), ("test", max(10, states // 5)))
    ):
        rows = []
        rng = random.Random(seed + index * 1000000)
        group = 0
        while len(rows) < count:
            group_seed = seed * 100000000 + index * 1000000 + group
            rules = Rules(
                players=rng.randint(1, 7),
                decks=rng.choice([1, 2, 6, 8]),
                hit_soft_17=rng.choice([False, True]),
                double_after_split=rng.choice([False, True]),
                surrender=rng.choice([False, True]),
                blackjack_payout=rng.choice([1.2, 1.5]),
                penetration=rng.choice([0.5, 0.65, 0.75, 0.85]),
                tablemate_policy=rng.choice(["basic", "random", "conservative"]),
            )
            game = Game(rules, group_seed)
            # Collect several positions across a shoe; the complete game stays in one split.
            collected = 0
            for _ in range(200):
                if len(rows) >= count or collected >= 4:
                    break
                if game.phase != "playing":
                    game.deal()
                    continue
                obs = game.observation()
                if game.active[0] != 0:
                    game.step(tablemate_action(game))
                    continue
                if rng.random() < 0.4:
                    ref = analyze(obs, samples, seed=group_seed + game.round * 997 + collected)
                    # If a pathological tiny shoe cannot resolve dealer totals, omit that state.
                    if ref["dealer_unresolved"] == 0:
                        targets = {
                            "action": {a: float(a == ref["recommendation"]) for a in obs["legal_actions"]},
                            "hit_bust": {"false": 1 - ref["hit_bust"], "true": ref["hit_bust"]},
                            "dealer": ref["dealer"],
                        }
                        rows.append(
                            {
                                "group_seed": group_seed,
                                "observation": obs,
                                "state": model_state(obs),
                                "questions": questions(obs),
                                "targets": targets,
                                "reference": ref,
                            }
                        )
                        collected += 1
                game.step(rng.choice(obs["legal_actions"]) if rng.random() < 0.25 else basic_action(obs))
            group += 1
            if group % 10 == 0:
                print(f"dataset {split}: {len(rows)}/{count} states", flush=True)
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["splits"][split] = {
            "states": len(rows),
            "groups": group,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"dataset {split}: complete ({len(rows)} states)", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def train(
    dataset: Path,
    output: Path,
    source=BASE_MODEL,
    device=None,
    epochs=3,
    batch_size=4,
    learning_rate=0.0001,
    full_model=False,
    seed=42,
):
    import torch
    from laya.common import QTYPES, build_sequence, collate_items, temp_bucket
    from safetensors.torch import save_file

    if epochs < 1 or batch_size < 1 or output.exists():
        raise ValueError("Positive epochs/batch size and a new output directory are required.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    started = time.perf_counter()
    manifest = json.loads((dataset / "manifest.json").read_text())
    for name, info in manifest["splits"].items():
        if hashlib.sha256((dataset / f"{name}.jsonl").read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Dataset {name} hash does not match its manifest.")
    rows = {name: read_rows(dataset / f"{name}.jsonl") for name in ("train", "validation", "test")}
    groups = {name: {row["group_seed"] for row in data} for name, data in rows.items()}
    if any(
        groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Train, calibration, and test must use disjoint game seeds.")
    agent = load_agent(source, device)
    print(f"Loaded {source} on {agent.device}", flush=True)
    model = agent.model
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith("act_head.") and (full_model or not name.startswith("encoder.")))
    if full_model:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def prepare(data):
        items = []
        for row in data:
            ensure_fits(agent, row["state"], row["questions"])
            for qid, q in row["questions"].items():
                qtype = QTYPES[q["type"]]
                ids, markers = build_sequence(
                    agent.tok,
                    row["state"],
                    agent._to_internal(q),
                    agent.cfg["max_len"],
                    agent.cfg["head_max_len"],
                )
                keys = ["false", "true"] if q["type"] == "noul" else list(q["criteria"])
                target = [row["targets"][qid][k] for k in keys]
                items.append(
                    {
                        "ids": ids,
                        "markers": markers,
                        "qtype": qtype,
                        "target": target,
                        "qid": qid,
                        "keys": keys,
                        "row": row,
                    }
                )
        return items

    items = {name: prepare(data) for name, data in rows.items()}

    def forward(chunk):
        batch = collate_items([chunk], agent.tok.pad_token_id)
        kwargs = {
            k: batch[k].to(agent.device)
            for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
        }
        with torch.autocast(agent.device.type, dtype=torch.bfloat16, enabled=agent.device.type == "cuda"):
            logits, _ = model(**kwargs)
        return logits.float(), batch["target"].to(agent.device)

    def predictions(data):
        model.eval()
        output = []
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                chunk = data[start : start + batch_size]
                logits, _ = forward(chunk)
                for it, z in zip(chunk, logits.cpu().numpy(), strict=True):
                    output.append((it, z[: len(it["target"])]))
        return output

    def metrics(preds, temps):
        scores = defaultdict(list)
        action_confs, action_correct = [], []
        for it, z in preds:
            temp = temps.get(temp_bucket(it["qtype"], len(z)), 1.0)
            z = (z - z.max()) / temp
            p = np.exp(z)
            p /= p.sum()
            target = np.array(it["target"])
            scores[it["qid"] + "_brier"].append(float(((p - target) ** 2).sum()))
            if it["qid"] == "action":
                action = it["keys"][int(p.argmax())]
                ref = it["row"]["reference"]
                scores["teacher_agreement"].append(float(action == ref["recommendation"]))
                scores["teacher_ev_regret"].append(
                    ref["actions"][ref["recommendation"]]["ev"] - ref["actions"][action]["ev"]
                )
                action_confs.append(float(p.max()))
                action_correct.append(float(action == ref["recommendation"]))
        ece = 0.0
        for lo in np.linspace(0, 0.9, 10):
            selected = [i for i, p in enumerate(action_confs) if lo < p <= lo + 0.1000001]
            if selected:
                ece += (
                    len(selected)
                    / len(action_confs)
                    * abs(
                        np.mean([action_confs[i] for i in selected])
                        - np.mean([action_correct[i] for i in selected])
                    )
                )
        return {**{k: float(np.mean(v)) for k, v in scores.items()}, "teacher_action_ece": float(ece)}

    print("Evaluating base model on held-out test split", flush=True)
    baseline = metrics(predictions(items["test"]), {})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.01
    )
    losses = []
    for epoch in range(epochs):
        model.train()
        if not full_model:
            model.encoder.eval()  # Frozen encoder must not add dropout noise.
        random.shuffle(items["train"])
        epoch_losses = []
        for start in range(0, len(items["train"]), batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits, target = forward(items["train"][start : start + batch_size])
            loss = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; checkpoint not published.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
            if start // batch_size % 50 == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} · batch {start // batch_size} · loss {loss.item():.4f}",
                    flush=True,
                )
        losses.append(float(np.mean(epoch_losses)))
    del optimizer
    print("Fitting temperatures on separate validation games", flush=True)
    buckets = defaultdict(list)
    for it, z in predictions(items["validation"]):
        buckets[temp_bucket(it["qtype"], len(z))].append((z, np.array(it["target"])))
    temps = {}
    for bucket, pairs in buckets.items():

        def objective(temp, pairs=pairs):
            terms = []
            for z, target in pairs:
                z = (z - z.max()) / temp
                logp = z - np.log(np.exp(z).sum())
                terms.append(-float((target * logp).sum()))
            return float(np.mean(terms))

        temps[bucket] = float(min(np.geomspace(0.1, 10, 81), key=objective))
    report = {
        "source": source,
        "base_revision": BASE_REVISION,
        "method": "supervised simulator distillation / soft cross entropy",
        "full_model": full_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(agent.device),
        "loss": losses,
        "baseline": baseline,
        "test": metrics(predictions(items["test"]), temps),
        "temperature_by_options": temps,
        "states": {k: len(v) for k, v in rows.items()},
        "dataset": json.loads((dataset / "manifest.json").read_text()),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": "Teacher is approximate. Metrics measure held-out teacher agreement, not casino profitability or proven optimality.",
    }
    # Publish the checkpoint only after all its components and evaluation are written.
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(parents=True, exist_ok=False)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(staging / "model.safetensors"),
    )
    model.encoder.config.save_pretrained(staging / "encoder")
    agent.tok.save_pretrained(staging / "tokenizer")
    cfg = {**agent.cfg, "temperature": [1.0, 1.0, 1.0], "temperature_by_options": temps, "fine_tuned": True}
    (staging / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    (staging / "training_report.json").write_text(json.dumps(report, indent=2))
    staging.rename(output)
    print(json.dumps(report, indent=2), flush=True)
    return report


def benchmark(output: Path, rounds=200, seed=12345, players=3, samples=128, model_path=None):
    from .model import LayaPolicy

    model = LayaPolicy()
    policies = ["basic", "reference"]
    if model_path:
        model.load(model_path)
        policies.append("laya")
    results = {}
    for policy in policies:
        returns = []
        voids = 0
        # Independent fresh-shoe rounds permit meaningful round-level standard errors.
        for i in range(rounds):
            game = Game(Rules(players=players), seed + i)
            game.deal()
            while game.phase == "playing":
                if game.active[0] != 0:
                    action = tablemate_action(game)
                elif policy == "basic":
                    action = basic_action(game.observation())
                elif policy == "reference":
                    action = analyze(game.observation(), samples)["recommendation"]
                else:
                    action = model.predict(game.observation())["answers"]["action"]["choice"]
                game.step(action)
            returns.append(game.history[-1]["profit"])
            voids += game.phase == "void"
            if (i + 1) % 25 == 0:
                print(f"benchmark {policy}: {i + 1}/{rounds} rounds", flush=True)
        mean = float(np.mean(returns))
        se = float(np.std(returns, ddof=1) / math.sqrt(rounds)) if rounds > 1 else 0
        results[policy] = {
            "rounds": rounds,
            "mean_units": mean,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "positive_rounds": sum(x > 0 for x in returns),
            "voids": voids,
        }
    report = {
        "results": results,
        "seed": seed,
        "rules": asdict(Rules(players=players)),
        "sampling": "Independent fresh shoes per round, paired initial seeds across policies",
        "samples": samples,
        "model": model_path,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report
