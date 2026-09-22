"""Reproduce a completed pilot report and export evidence without private paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blackjack.experiment_data import atomic_json, digest
from blackjack.research import checkpoint_identity
from blackjack.visitation import summarize_pilot


def export_pilot(root: Path, output: Path):
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Export outside the immutable run directory.")
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Only completed pilot reports can be exported.")
    run = json.loads((root / "run.json").read_text())
    launch_path = root / "launch.json"
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
    config = run["config"]
    if checkpoint_identity(root / "checkpoints/source") != config["checkpoint"]:
        raise ValueError("Frozen pilot checkpoint changed.")
    if launch:
        if digest(root / "source.tar") != launch["source_archive_sha256"]:
            raise ValueError("Pilot source archive changed.")
        for name, sha256 in config["code"].items():
            if digest(root / "source/blackjack" / name) != sha256:
                raise ValueError("Pilot source snapshot changed.")
    elif (root / "source.tar").exists() or (root / "source").exists():
        raise ValueError("Archived source has no launch receipt.")
    report = json.loads((root / "report.json").read_text())
    if summarize_pilot(root, write_report=False) != report:
        raise ValueError("Reproduced pilot report differs; original report was preserved.")
    if (
        status["qualified_for_training_design"] != report["qualified_for_training_design"]
        or status["decision"] != report["decision"]
    ):
        raise ValueError("Pilot status and qualification report disagree.")
    exported = {
        "run_id": root.name,
        "source_commit": launch.get("source_commit"),
        "source_archive_sha256": launch.get("source_archive_sha256"),
        "source_snapshot_verified": bool(launch),
        "report_sha256": digest(root / "report.json"),
        "started_unix": run["started_unix"],
        "deadline_unix": run["deadline_unix"],
        "elapsed_seconds": status["elapsed_seconds"],
        "config": {k: v for k, v in config.items() if k != "source"},
        "report": report,
    }
    for key in ("previous_failed_attempt", "original_followup_deadline_unix"):
        if key in launch:
            exported[key] = launch[key]
    atomic_json(output, exported)
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_pilot(args.run, args.output)
    print(f"Verified {result['run_id']}: {result['report']['decision']}; exported {args.output}")
