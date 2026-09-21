"""Render publication figures from the committed, measured result reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "figures"
COLORS = {"baseline": "#87929c", "candidate": "#087f6c", "basic": "#364e75"}


def read(name):
    return json.loads((DATA / name).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        path = OUT / f"{name}.{extension}"
        metadata = {"Date": None} if extension == "svg" else None
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", metadata=metadata)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": "#cbd1d6",
        "axes.labelcolor": "#334155", "text.color": "#1e293b",
        "xtick.color": "#64748b", "ytick.color": "#334155",
        "svg.hashsalt": "laya-blackjack-v0.2",
    })
    report = read("overnight-100k.json")
    audit = read("sdk-audit.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), layout="constrained")
    for ax, metric, title, scale in [
        (axes[0], "teacher_agreement", "Reference action agreement (%) ↑", 100),
        (axes[1], "teacher_ev_regret", "Reference EV regret (units / decision) ↓", 1),
    ]:
        values = [report["baseline"][metric] * scale, report["test"][metric] * scale]
        ax.barh([1, 0], values, color=[COLORS["baseline"], COLORS["candidate"]], height=0.5)
        ax.set_yticks([1, 0], ["Warm start", "Selected model"])
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=11, pad=14)
        ax.set_xlim(0, max(values) * 1.25)
        for y, value in zip([1, 0], values, strict=True):
            ax.text(value + max(values) * 0.025, y, f"{value:.2f}" if scale == 100 else f"{value:.5f}",
                    va="center", fontsize=10)
    fig.supxlabel("5,000 frozen test states · trainer batches · approximate teacher, not an optimal solver",
                  fontsize=9, color="#64748b")
    save(fig, "model-quality")

    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    groups = audit["by_group"]["depth"]
    x = np.array([int(k) for k in groups])
    y = [groups[str(k)]["teacher_agreement"] * 100 for k in x]
    ax.plot(x, y, color=COLORS["candidate"], marker="o", linewidth=2)
    for k, value in zip(x, y, strict=True):
        ax.annotate(f'n={groups[str(k)]["states"]}', (k, value), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8, color="#64748b")
    ax.set_xticks(x, [f"{k * 10}–{(k + 1) * 10}%" for k in x], fontsize=8)
    ax.set_ylim(70, 105)
    ax.set_ylabel("Reference agreement (%)")
    ax.set_xlabel("Fraction of the shoe already consumed")
    ax.set_title("Depleted shoes remain a weakness", loc="left", fontsize=13, pad=20)
    fig.supxlabel("Serving SDK · descriptive test subgroups · small late-shoe samples", fontsize=9,
                  color="#64748b")
    save(fig, "shoe-depth")

    paths = [DATA / f"{mode}-comparison.json" for mode in ("fresh", "continuous")]
    if not all(p.exists() for p in paths):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True, layout="constrained")
    for ax, path, label in zip(axes, paths, ["Fresh shoes", "Continuous shoes"], strict=True):
        data = json.loads(path.read_text())
        ax.axvline(0, color="#aab4bd", linewidth=1)
        for y, name in enumerate(["basic", "baseline"]):
            row = data["paired_candidate_minus"][name]
            mean = row["mean_difference"] * 100
            low, high = [v * 100 for v in row["ci95"]]
            ax.errorbar(mean, 1-y, xerr=[[mean-low], [high-mean]], fmt="o", capsize=4,
                        color=COLORS["candidate"], markersize=7)
            ax.annotate(f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]", (mean, 1-y),
                        xytext=(0, 15), textcoords="offset points", ha="center", fontsize=8)
        ax.set_yticks([1, 0], ["vs basic heuristic", "vs warm start"])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, 1.8)
        ax.set_title(label, loc="left", fontsize=12)
        ax.set_xlabel("Selected-model advantage (units / 100 rounds)")
    fig.supxlabel("Paired 95% intervals · 100,000 rounds per policy and setting · fixed equal scenario weights",
                  fontsize=9, color="#64748b")
    save(fig, "paired-returns")


if __name__ == "__main__":
    main()
