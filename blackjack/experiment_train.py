"""Bounded-memory fine-tuning with recoverable optimizer state and untouched final tests."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

from .experiment_data import atomic_json, digest, verify_dataset
from .model import ensure_fits, load_agent


def scores(predictions, temperatures):
    from laya.common import temp_bucket

    values = {}
    per_players = {}
    per_stratum = {}
    per_teacher = {}
    action_p, action_y = [], []
    resolved_correct, unresolved_correct = [], []
    for item, logits in predictions:
        z = np.asarray(logits, dtype=float)
        z = (z - z.max()) / temperatures.get(temp_bucket(item["qtype"], len(z)), 1.0)
        p = np.exp(z)
        p /= p.sum()
        target = np.asarray(item["target"])
        values.setdefault(item["qid"] + "_brier", []).append(float(((p - target) ** 2).sum()))
        if item["qid"] != "action":
            continue
        choice = int(p.argmax())
        correct = float(choice == item["label"])
        regret = max(item["ev"]) - item["ev"][choice]
        values.setdefault("teacher_agreement", []).append(correct)
        values.setdefault("teacher_ev_regret", []).append(regret)
        per_players.setdefault(str(item["players"]), []).append((correct, regret))
        per_stratum.setdefault(item.get("stratum", "general"), []).append((correct, regret))
        per_teacher.setdefault(item.get("teacher_kind", "monte_carlo"), []).append((correct, regret))
        action_p.append(float(p.max()))
        action_y.append(correct)
        (resolved_correct if item["resolved"] else unresolved_correct).append(correct)
    ece = 0.0
    # Half-open disjoint bins; a boundary must never contribute twice.
    assignments = np.minimum((np.array(action_p) * 10).astype(int), 9)
    for bucket in range(10):
        selected = assignments == bucket
        if selected.any():
            ece += selected.mean() * abs(
                np.array(action_p)[selected].mean() - np.array(action_y)[selected].mean()
            )
    return {
        **{k: float(np.mean(v)) for k, v in values.items()},
        "teacher_action_ece": float(ece),
        "resolved_agreement": float(np.mean(resolved_correct)) if resolved_correct else None,
        "unresolved_agreement": float(np.mean(unresolved_correct)) if unresolved_correct else None,
        "resolved_states": len(resolved_correct),
        "unresolved_states": len(unresolved_correct),
        "by_players": {
            k: {
                "states": len(v),
                "teacher_agreement": float(np.mean([x[0] for x in v])),
                "teacher_ev_regret": float(np.mean([x[1] for x in v])),
            }
            for k, v in per_players.items()
        },
        "by_stratum": {
            k: {
                "states": len(v),
                "teacher_agreement": float(np.mean([x[0] for x in v])),
                "teacher_ev_regret": float(np.mean([x[1] for x in v])),
            }
            for k, v in per_stratum.items()
        },
        "by_teacher": {
            k: {
                "states": len(v),
                "teacher_agreement": float(np.mean([x[0] for x in v])),
                "teacher_ev_regret": float(np.mean([x[1] for x in v])),
            }
            for k, v in per_teacher.items()
        },
    }


def selection_eligible(candidate, baseline, general_margin):
    """Predeclared composition criterion, using selection data only."""
    if general_margin is None:
        return True
    c, b = candidate["by_stratum"], baseline["by_stratum"]
    return (
        c["general"]["teacher_ev_regret"] <= b["general"]["teacher_ev_regret"] + general_margin
        and c["depleted"]["teacher_ev_regret"] < b["depleted"]["teacher_ev_regret"]
        and candidate["teacher_ev_regret"] < baseline["teacher_ev_regret"]
    )


def fit_temperatures(predictions):
    from laya.common import temp_bucket

    buckets = {}
    for item, logits in predictions:
        buckets.setdefault(temp_bucket(item["qtype"], len(logits)), []).append(
            (np.array(logits), np.array(item["target"]))
        )
    result = {}
    for name, pairs in buckets.items():
        best_loss = float("inf")
        for temp in np.geomspace(0.1, 10, 81):
            loss = 0.0
            for z, target in pairs:
                z = (z - z.max()) / temp
                loss -= float((target * (z - np.log(np.exp(z).sum()))).sum())
            if loss < best_loss:
                result[name], best_loss = float(temp), loss
    return result


class ShardItems:
    """At most one tokenized shard needs to reside in memory."""

    def __init__(self, agent, dataset: Path, cache: Path):
        self.agent, self.dataset, self.cache = agent, dataset, cache
        cache.mkdir(parents=True, exist_ok=True)
        contract = {
            "dataset": digest(dataset / "manifest.json"),
            "max_len": agent.cfg["max_len"],
            "head_max_len": agent.cfg["head_max_len"],
            "tokenizer": agent.tok.backend_tokenizer.to_str(),
            "version": 3,
        }
        contract_path = cache / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Token cache contract changed; use a new experiment directory.")
        atomic_json(contract_path, contract)

    def load(self, shard):
        import torch
        from laya.common import QTYPES, build_sequence

        path = self.cache / (Path(shard["path"]).stem + ".pt")
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["source_hash"] != shard["sha256"]:
                raise ValueError("Token cache source hash mismatch.")
            return saved["items"]
        items = []
        with (self.dataset / shard["path"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                ensure_fits(self.agent, row["state"], row["questions"])
                for qid, q in row["questions"].items():
                    ids, markers = build_sequence(
                        self.agent.tok,
                        row["state"],
                        self.agent._to_internal(q),
                        self.agent.cfg["max_len"],
                        self.agent.cfg["head_max_len"],
                    )
                    keys = ["false", "true"] if q["type"] == "noul" else list(q["criteria"])
                    target = [row["targets"][qid][k] for k in keys]
                    resolved = row["reference"]["label_quality"]["resolved"]
                    item = {
                        "ids": ids,
                        "markers": markers,
                        "target": target,
                        "qtype": QTYPES[q["type"]],
                        "qid": qid,
                        "label": int(np.argmax(target)),
                        "weight": (2.0 if resolved else 0.5) if qid == "action" else 1.0,
                    }
                    if qid == "action":
                        item.update(
                            ev=[row["reference"]["actions"][k]["ev"] for k in keys],
                            players=row["observation"]["rules"]["players"],
                            resolved=resolved,
                            stratum=row.get("stratum", "general"),
                            teacher_kind=row["reference"].get("teacher_kind", "monte_carlo"),
                        )
                    items.append(item)
        temporary = path.with_suffix(".tmp")
        torch.save({"items": items, "source_hash": shard["sha256"]}, temporary)
        temporary.replace(path)
        return items


def decision_loss(logits, target, items, objective="imitation"):
    """Retain probability losses; optionally penalize estimated action regret.

    The additional term is 0.25 * E[min(reference regret, 1 unit)] / 0.1 unit.
    It uses the existing resolution weights and never interprets EV as probability.
    """
    import torch

    if objective not in ("imitation", "cost-sensitive"):
        raise ValueError("Unknown training objective.")
    weights = torch.tensor([it["weight"] for it in items], device=logits.device)
    losses = -(target * torch.log_softmax(logits, -1)).sum(-1)
    if objective == "cost-sensitive":
        costs = torch.zeros_like(logits)
        for i, it in enumerate(items):
            if it["qid"] == "action":
                values = torch.tensor(it["ev"], device=logits.device)
                costs[i, : len(values)] = (values.max() - values).clamp(0, 1) / 0.1
        losses = losses + 0.25 * (torch.softmax(logits, -1) * costs).sum(-1)
    return (losses * weights).sum() / weights.sum()


def train_large(
    dataset: Path,
    output: Path,
    source: str,
    device="cuda:0",
    epochs=3,
    batch_size=8,
    learning_rate=0.00001,
    seed=20260921,
    checkpoint_steps=1000,
    deadline: float | None = None,
    defer_test=False,
    general_margin: float | None = None,
    objective="imitation",
):
    import torch
    from laya.common import collate_items
    from safetensors.torch import load_file, save_file

    if (
        min(epochs, batch_size, checkpoint_steps) < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Epochs, batch size, learning rate, and checkpoint interval must be positive.")
    if general_margin is not None and (not math.isfinite(general_margin) or general_margin < 0):
        raise ValueError("General-play selection margin must be finite and nonnegative.")
    if objective not in ("imitation", "cost-sensitive"):
        raise ValueError("Unknown training objective.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = verify_dataset(dataset)
    config = {
        "dataset_hash": digest(dataset / "manifest.json"),
        "source": source,
        "source_hash": digest(Path(source) / "model.safetensors") if Path(source).is_dir() else source,
        "source_config_hash": digest(Path(source) / "rl_agent_config.json")
        if Path(source).is_dir()
        else source,
        "trainer_hash": digest(Path(__file__)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "version": 1,
        "defer_test": defer_test,
        "general_margin": general_margin,
        "objective": objective,
    }
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != config:
        raise ValueError("Resume training configuration changed.")
    if (output / "candidate" / "training_report.json").exists():
        return json.loads((output / "candidate" / "training_report.json").read_text())
    atomic_json(output / "config.json", config)
    random.seed(seed)
    torch.manual_seed(seed)
    agent = load_agent(source, device)
    if agent.device.type != "cuda":
        raise RuntimeError("Large training requires the requested CUDA device; refusing silent CPU fallback.")
    model = agent.model
    for name, param in model.named_parameters():
        param.requires_grad_(not name.startswith("act_head."))
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.01
    )
    store = ShardItems(agent, dataset, output / "tokens")
    training_shards = manifest["splits"]["train"]
    steps_per_epoch = sum(math.ceil(3 * shard["states"] / batch_size) for shard in training_shards)
    total_steps = steps_per_epoch * epochs
    start_epoch = start_shard = start_batch = updates = 0
    best_regret = float("inf")
    history = []
    resume = output / "resume.pt"
    started = time.monotonic()

    def checkpoint(next_epoch, next_shard, next_batch):
        temporary = output / "resume.tmp"
        optimizer.zero_grad(set_to_none=True)
        state = {
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": next_epoch,
            "shard": next_shard,
            "batch": next_batch,
            "updates": updates,
            "history": history,
            "best_regret": best_regret,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(agent.device),
        }
        torch.save(state, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(resume)

    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError("Checkpoint does not match this experiment.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, start_shard, start_batch, updates = (
            state[k] for k in ("epoch", "shard", "batch", "updates")
        )
        history, best_regret = state["history"], state["best_regret"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], agent.device)
        del state
        gc.collect()
        print(f"Resumed at epoch={start_epoch}, shard={start_shard}, batch={start_batch}", flush=True)
    best_pointer = output / "best.json"
    if best_pointer.exists():
        best_regret = min(
            best_regret, json.loads(best_pointer.read_text())["record"]["selection"]["teacher_ev_regret"]
        )

    def forward(items):
        batch = collate_items([items], agent.tok.pad_token_id)
        kwargs = {
            k: batch[k].to(agent.device)
            for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(**kwargs)
        return logits.float(), batch["target"].to(agent.device)

    def predict(split):
        model.eval()
        predictions = []
        with torch.no_grad():
            for shard in manifest["splits"][split]:
                items = store.load(shard)
                for start in range(0, len(items), batch_size):
                    chunk = items[start : start + batch_size]
                    z, _ = forward(chunk)
                    for item, logits in zip(chunk, z.cpu().numpy(), strict=True):
                        meta = {k: v for k, v in item.items() if k not in ("ids", "markers")}
                        predictions.append((meta, logits[: len(item["target"])].copy()))
        return predictions

    # Baseline test values are fixed before training, never used for selection or stopping.
    baseline_file = output / "baseline-test.json"
    if not defer_test and not baseline_file.exists():
        if updates:
            raise ValueError("Resumed run is missing its pre-training baseline.")
        print("Recording fixed source baseline on final test (not used for training decisions)", flush=True)
        atomic_json(baseline_file, scores(predict("test"), agent.temperature_by_options))
    baseline_selection = None
    if general_margin is not None:
        selection_file = output / "baseline-selection.json"
        if not selection_file.exists():
            if updates:
                raise ValueError("Resumed run is missing its frozen selection baseline.")
            atomic_json(selection_file, scores(predict("selection"), agent.temperature_by_options))
        baseline_selection = json.loads(selection_file.read_text())
        if set(baseline_selection["by_stratum"]) != {"general", "depleted"}:
            raise ValueError("Guarded selection requires general and depleted selection strata.")
        if not best_pointer.exists():
            # The unchanged incumbent is an explicit option if every epoch fails the guard.
            best_dir = output / "best-incumbent"
            best_staging = output / "best-incumbent.staging"
            best_staging.mkdir(exist_ok=True)
            record = {
                "epoch": 0,
                "complete_epoch": True,
                "updates": 0,
                "mean_loss": None,
                "selection": baseline_selection,
                "incumbent": True,
            }
            shutil.copyfile(Path(source) / "model.safetensors", best_staging / "model.safetensors")
            atomic_json(best_staging / "selection.json", record)
            if not best_dir.exists():
                best_staging.rename(best_dir)
            atomic_json(best_pointer, {"path": best_dir.name, "record": record})
            best_regret = baseline_selection["teacher_ev_regret"]
    stop = False
    for epoch in range(start_epoch, epochs):
        order = list(range(len(training_shards)))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        losses = []
        for shard_position in range(start_shard if epoch == start_epoch else 0, len(order)):
            items = store.load(training_shards[order[shard_position]])
            random.Random(seed + epoch * 1000000 + shard_position).shuffle(items)
            offset = start_batch if epoch == start_epoch and shard_position == start_shard else 0
            for batch_index in range(offset, math.ceil(len(items) / batch_size)):
                if deadline is not None and time.time() >= deadline:
                    checkpoint(epoch, shard_position, batch_index)
                    stop = True
                    break
                chunk = items[batch_index * batch_size : (batch_index + 1) * batch_size]
                warmup = max(1, int(total_steps * 0.03))
                scale = (
                    (updates + 1) / warmup
                    if updates < warmup
                    else 0.1
                    + 0.9 * (1 + math.cos(math.pi * (updates - warmup) / max(1, total_steps - warmup))) / 2
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * scale
                optimizer.zero_grad(set_to_none=True)
                logits, target = forward(chunk)
                loss = decision_loss(logits, target, chunk, objective)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; refusing to publish candidate.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                updates += 1
                losses.append(loss.item())
                if updates % checkpoint_steps == 0:
                    checkpoint(epoch, shard_position, batch_index + 1)
                if updates % 50 == 0:
                    progress = {
                        "stage": "training",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "updates": updates,
                        "planned_updates": total_steps,
                        "shard": shard_position + 1,
                        "shards": len(order),
                        "loss": float(np.mean(losses[-50:])),
                        "device": str(agent.device),
                        "elapsed_seconds": time.monotonic() - started,
                        "gpu_peak_gb": torch.cuda.max_memory_allocated(agent.device) / 1e9,
                        "best_selection_regret": None if math.isinf(best_regret) else best_regret,
                    }
                    atomic_json(output / "progress.json", progress)
                    print(
                        f"epoch {epoch + 1}/{epochs} · update {updates}/{total_steps} · loss {progress['loss']:.4f}",
                        flush=True,
                    )
            if stop:
                break
        if not updates:
            raise TimeoutError("Training budget expired before the first update; no candidate published.")
        # A time-limited partial epoch can still be evaluated against the selection set.
        print(f"Evaluating selection split after epoch {epoch + 1}", flush=True)
        selection_scores = scores(predict("selection"), {})
        history.append(
            {
                "epoch": epoch + 1,
                "complete_epoch": not stop,
                "updates": updates,
                "mean_loss": float(np.mean(losses)) if losses else None,
                "selection": selection_scores,
            }
        )
        eligible = selection_eligible(selection_scores, baseline_selection, general_margin)
        history[-1]["selection_eligible"] = eligible
        if eligible and selection_scores["teacher_ev_regret"] < best_regret:
            best_regret = selection_scores["teacher_ev_regret"]
            # The pointer activates weights and their selection record together.
            best_dir = output / f"best-{epoch + 1}-{updates}-{time.time_ns()}"
            best_staging = best_dir.with_suffix(".staging")
            best_staging.mkdir()
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                str(best_staging / "model.safetensors"),
            )
            atomic_json(best_staging / "selection.json", history[-1])
            best_staging.rename(best_dir)
            atomic_json(best_pointer, {"path": best_dir.name, "record": history[-1]})
        if stop:
            break
        checkpoint(epoch + 1, 0, 0)
        start_shard = start_batch = 0
    optimizer = None
    gc.collect()
    torch.cuda.empty_cache()
    best = json.loads(best_pointer.read_text())
    best_weights = output / best["path"] / "model.safetensors"
    model.load_state_dict(load_file(str(best_weights)), strict=True)
    print(
        "Calibrating on separate games; "
        + ("final test remains sealed" if defer_test else "evaluating final test"),
        flush=True,
    )
    temperatures = fit_temperatures(predict("calibration"))
    test_scores = None if defer_test else scores(predict("test"), temperatures)
    report = {
        "source": source,
        "method": "Full-model simulation distillation, uncertainty-weighted action loss",
        "objective": objective,
        "reference_method": manifest["teacher"],
        "teacher_counts": manifest.get("teacher_counts"),
        "full_model": True,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(agent.device),
        "history": history,
        "stopped_for_budget": stop,
        "updates": updates,
        "baseline": None if defer_test else json.loads(baseline_file.read_text()),
        "baseline_selection": baseline_selection,
        "test_deferred": defer_test,
        "general_margin": general_margin,
        "test": test_scores,
        "temperature_by_options": temperatures,
        "states": {**manifest["actual_states"], "validation": manifest["actual_states"]["calibration"]},
        "dataset_hash": config["dataset_hash"],
        "selected": best["record"],
        "elapsed_seconds": time.monotonic() - started,
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated(agent.device) / 1e9,
        "limitations": "Reference scope is recorded in reference_method. Monte Carlo fallbacks use basic continuation and heuristic stopping. Teacher agreement does not establish global optimality or profit. Candidate is not auto-activated.",
    }
    staging = output / "candidate.staging"
    staging.mkdir(exist_ok=True)
    # Best weights were already serialized atomically; copy avoids another full CPU allocation.
    shutil.copyfile(best_weights, staging / "model.safetensors")
    model.encoder.config.save_pretrained(staging / "encoder")
    agent.tok.save_pretrained(staging / "tokenizer")
    atomic_json(
        staging / "rl_agent_config.json",
        {
            **agent.cfg,
            "temperature": [1.0, 1.0, 1.0],
            "temperature_by_options": temperatures,
            "fine_tuned": True,
        },
    )
    atomic_json(staging / "training_report.json", report)
    staging.rename(output / "candidate")
    atomic_json(
        output / "progress.json",
        {
            "stage": "complete",
            "updates": updates,
            "candidate": str(output / "candidate"),
            "test": test_scores,
        },
    )
    print(json.dumps(report, indent=2), flush=True)
    return report
