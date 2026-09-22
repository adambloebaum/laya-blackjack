"""Verify raw paired units and export portable return/error diagnostics.

This reads completed experiments. It never trains, selects, or activates a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

from blackjack.evaluation import compare_evaluations
from blackjack.experiment_data import atomic_json, digest
from blackjack.research import comparison_summary


def interval(mean, se, comparisons=1):
    z = 1.96 if comparisons == 1 else NormalDist().inv_cdf(1 - 0.05 / (2 * comparisons))
    return [mean - z * se, mean + z * se]


def scenario_diagnostics(reports):
    """Keep equal scenario weights and distinguish post-hoc from planned families."""
    count = sum(len(r["identity"]["scenarios"]) * len(r["paired_candidate_minus"]) for r in reports.values())
    rows = []
    for mode, report in reports.items():
        scenarios = report["identity"]["scenarios"]
        for comparator, pair in report["paired_candidate_minus"].items():
            for scenario in scenarios:
                item = pair["by_scenario"][scenario]
                mean, se = item["mean_units_per_round"] * 100, item["standard_error"] * 100
                rows.append(
                    {
                        "mode": mode,
                        "scenario": scenario,
                        "comparator": comparator,
                        "independent_units": item["independent_units"],
                        "rounds_per_unit": report["identity"]["rounds_per_unit"],
                        "difference_per_100": mean,
                        "standard_error_per_100": se,
                        "nominal_ci95": interval(mean, se),
                        "exploratory_familywise_ci95": interval(mean, se, count),
                        "contribution_to_aggregate_per_100": mean / len(scenarios),
                    }
                )
    return {"comparisons": count, "rows": rows}


def audit_diagnostics(candidate, baseline):
    if not candidate or len(candidate) != len(baseline):
        raise ValueError("Audit decision populations differ or are empty.")
    identity = ("index", "group_seed", "hand", "reference_action", "action_values", "stratum", "teacher_kind")
    for new, old in zip(candidate, baseline, strict=True):
        if any(new[key] != old[key] for key in identity):
            raise ValueError("Audit decision identities or reference targets differ.")
    if len({(r["group_seed"], r["index"]) for r in candidate}) != len(candidate):
        raise ValueError("Duplicate audit decision identity.")
    total_regret = sum(r["regret"] for r in candidate)
    fields = ("stratum", "category", "depth", "players", "resolved", "teacher_kind", "reference_action")
    groups = {}
    for field in fields:
        items = defaultdict(list)
        for new, old in zip(candidate, baseline, strict=True):
            items[str(new[field])].append((new, old))
        groups[field] = {}
        for key, pairs in sorted(items.items()):
            new_regret = sum(n["regret"] for n, _ in pairs)
            groups[field][key] = {
                "states": len(pairs),
                "candidate_errors": sum(not n["correct"] for n, _ in pairs),
                "baseline_errors": sum(not o["correct"] for _, o in pairs),
                "candidate_regret_sum": new_regret,
                "baseline_regret_sum": sum(o["regret"] for _, o in pairs),
                "candidate_regret_share": new_regret / total_regret if total_regret else 0,
            }
    transitions = defaultdict(list)
    for row in candidate:
        if not row["correct"]:
            transitions[(row["reference_action"], row["action"])].append(row)
    confusions = [
        {
            "reference_action": ref,
            "candidate_action": action,
            "states": len(rows),
            "resolved_states": sum(r["resolved"] for r in rows),
            "regret_sum": sum(r["regret"] for r in rows),
        }
        for (ref, action), rows in transitions.items()
    ]
    return {
        "states": len(candidate),
        "candidate_errors": sum(not r["correct"] for r in candidate),
        "baseline_errors": sum(not r["correct"] for r in baseline),
        "candidate_regret_sum": total_regret,
        "changed_actions": sum(n["action"] != o["action"] for n, o in zip(candidate, baseline, strict=True)),
        "groups": groups,
        "confusions": sorted(confusions, key=lambda r: r["regret_sum"], reverse=True),
        "top30_regret_share": (
            sum(r["regret"] for r in sorted(candidate, key=lambda r: r["regret"], reverse=True)[:30])
            / total_regret
            if total_regret
            else 0
        ),
        "scope": f"Descriptive inspected {len(candidate):,}-state audit; reference cost is not realized return. "
        "The enriched test population differs from ordinary play. No inference on new test games.",
    }


def recheck_diagnostics(root, audit):
    plan = json.loads((root / "recheck-plan.json").read_text())
    if plan["source_audit_sha256"] != digest(audit / "candidate/report.json"):
        raise ValueError("Recheck does not belong to this candidate audit.")
    if plan["source_worst_states_sha256"] != digest(audit / "candidate/worst-states.json"):
        raise ValueError("Original error-state inventory changed.")
    report = json.loads((root / "mc-recheck.json").read_text())
    cases = report["cases"]
    if [c["decision"]["index"] for c in cases] != plan["indices"]:
        raise ValueError("Rechecked case identities changed.")
    expected_worlds = plan["samples_per_replicate"] * plan["replicates"]
    original = {
        item["audit"]["index"]: item["audit"]
        for item in json.loads((audit / "candidate/worst-states.json").read_text())
    }
    rows = []
    for case in cases:
        decision = case["decision"]
        if (
            decision != original[decision["index"]]
            or decision["teacher_kind"] != "monte_carlo"
            or case["fresh_worlds"] != expected_worlds
        ):
            raise ValueError("Recheck changed the target, action, teacher kind, or sample count.")
        rows.append(
            {
                "index": decision["index"],
                "reference_action": decision["reference_action"],
                "candidate_action": decision["action"],
                "original_resolved": decision["resolved"],
                "original_regret": decision["regret"],
                "fresh_worlds": case["fresh_worlds"],
                "reference_minus_candidate_ev": case["teacher_minus_model_ev"],
                "nominal_ci95": case["ci95"],
                "exploratory_familywise_ci95": interval(
                    case["teacher_minus_model_ev"], case["standard_error"], len(cases)
                ),
            }
        )
    return {
        "plan": plan,
        "elapsed_seconds": report["elapsed_seconds"],
        "cases": rows,
        "positive_nominal_intervals": sum(r["nominal_ci95"][0] > 0 for r in rows),
        "positive_adjusted_intervals": sum(r["exploratory_familywise_ci95"][0] > 0 for r in rows),
        "negative_adjusted_intervals": sum(r["exploratory_familywise_ci95"][1] < 0 for r in rows),
        "scope": "Exploratory fixed-action recheck of selected Monte Carlo errors with basic continuation. "
        "Intervals are conditional on these public states; selection bias and continuation-policy bias remain. "
        "Exact-teacher states are excluded. No model was retrained or selected using these worlds.",
    }


def analyze(run, audit, recheck, output):
    status = json.loads((run / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Return run is incomplete.")
    summary = json.loads((run / "summary.json").read_text())
    if summary["run_config_sha256"] != digest(run / "run.json"):
        raise ValueError("Frozen run configuration changed.")
    config = json.loads((run / "run.json").read_text())["config"]
    reports, identities = {}, {}
    with tempfile.TemporaryDirectory(prefix="laya-return-audit-") as temp:
        for mode in ("fresh", "continuous"):
            inputs = {name: run / name / mode for name in ("basic", "baseline", "candidate")}
            for name, path in inputs.items():
                key = f"{name}/{mode}"
                if digest(path / "manifest.json") != summary["manifests"][key]:
                    raise ValueError("Evaluation manifest changed.")
                manifest = json.loads((path / "manifest.json").read_text())
                expected_units = config["fresh_units"] if mode == "fresh" else config["blocks"]
                if manifest["config"]["identity"]["units"] != expected_units:
                    raise ValueError("Evaluation sample count differs from the frozen plan.")
                if name != "basic":
                    for field, filename in (
                        ("source_hash", "model.safetensors"),
                        ("config_hash", "rl_agent_config.json"),
                    ):
                        if manifest["config"][field] != config["checkpoints"][name][filename]["sha256"]:
                            raise ValueError("Evaluation checkpoint differs from the frozen plan.")
            report = compare_evaluations(inputs, Path(temp) / f"{mode}-comparison.json")
            if report != json.loads((run / f"{mode}-comparison.json").read_text()):
                raise ValueError("Saved comparison does not reproduce from verified raw units.")
            reports[mode] = report
            identities[f"{mode}-comparison.json"] = digest(run / f"{mode}-comparison.json")
        if comparison_summary(Path(temp))["comparisons"] != summary["comparisons"]:
            raise ValueError("Four-comparison summary does not reproduce.")
    rows, datasets = {}, set()
    for name in ("candidate", "baseline"):
        p = audit / name
        report = json.loads((p / "report.json").read_text())
        expected = config["checkpoints"][name]["model.safetensors"]["sha256"]
        if report["model_sha256"] != expected:
            raise ValueError("Audit and return checkpoint identities differ.")
        datasets.add(report["dataset_sha256"])
        rows[name] = json.loads((p / "decisions.json").read_text())
        if len(rows[name]) != report["metrics"]["states"] or not math.isclose(
            sum(r["regret"] for r in rows[name]) / len(rows[name]),
            report["metrics"]["teacher_ev_regret"],
        ):
            raise ValueError("Audit decisions do not reconcile to reported metrics.")
        identities[f"audit/{name}/report.json"] = digest(p / "report.json")
        identities[f"audit/{name}/decisions.json"] = digest(p / "decisions.json")
    if len(datasets) != 1:
        raise ValueError("Audits use different datasets.")
    raw_rounds = sum(item["rounds"] for report in reports.values() for item in report["results"].values())
    if raw_rounds != sum(p["rounds"] for p in status["progress"].values()):
        raise ValueError("Status round counts disagree with verified raw records.")
    scenarios = scenario_diagnostics(reports)
    result = {
        "run": run.name,
        "source_commit": json.loads((run / "launch.json").read_text())["source_commit"],
        "completed_unix": status["updated_unix"],
        "elapsed_seconds": status["elapsed_seconds"],
        "raw_units_reverified": True,
        "raw_rounds": raw_rounds,
        "analysis_source_sha256": digest(Path(__file__)),
        "source_hashes": identities,
        "summary": summary,
        "reports": reports,
        "scenarios": scenarios,
        "audit": audit_diagnostics(rows["candidate"], rows["baseline"]),
        "recheck": recheck_diagnostics(recheck, audit),
        "uncertainty": "The four aggregate comparisons were predeclared. The 40 scenario contrasts and "
        "25 selected-error contrasts are separate exploratory families, each using Bonferroni-adjusted "
        "normal intervals alongside nominal 95% intervals. They do not establish causal mechanisms.",
        "activated": False,
        "published": False,
    }
    for name in ("recheck-plan.json", "mc-recheck.json"):
        identities[f"recheck/{name}"] = digest(recheck / name)
    atomic_json(output, result)
    csv_path = output.with_suffix(".csv")
    columns = (
        "mode",
        "scenario",
        "comparator",
        "independent_units",
        "rounds_per_unit",
        "difference_per_100",
        "standard_error_per_100",
        "contribution_to_aggregate_per_100",
        "nominal_low",
        "nominal_high",
        "adjusted_low",
        "adjusted_high",
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in scenarios["rows"]:
            writer.writerow(
                {key: row[key] for key in columns if key in row}
                | dict(
                    zip(columns[-4:], row["nominal_ci95"] + row["exploratory_familywise_ci95"], strict=True)
                )
            )
    print(f"Verified and archived {result['raw_rounds']:,} rounds; wrote {output} and {csv_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--recheck", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/results/large-return-analysis.json"))
    analyze(**vars(parser.parse_args()))
