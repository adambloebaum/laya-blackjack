"""Render the selected model's figures from verified, committed release evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, TEAL, GRAY = "#203648", "#087f6c", "#87929c"


def save(fig, name):
    for ext in ("svg", "png"):
        path = ROOT / "docs/figures" / f"{name}.{ext}"
        fig.savefig(path, dpi=180, facecolor="white", metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    data = json.loads((ROOT / "docs/results/final-release-completed.json").read_text())
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#cbd1d6", "svg.hashsalt": "laya-final-release-v1",
    })
    fig, ax = plt.subplots(figsize=(11.8, 5.1))
    fig.subplots_adjust(left=0.28, right=0.70, top=0.73, bottom=0.29)
    fig.text(0.035, 0.93, "Laya Blackjack · 3 million simulated rounds", fontsize=17, weight="bold")
    fig.text(0.035, 0.865, "Continuous-play gains; fresh-shoe differences remain inconclusive", fontsize=11)
    ax.axvline(0, color=INK, linewidth=1, alpha=0.45)
    labels = []
    for y, (mode, comparator) in zip((3, 2, 1, 0), (
        ("fresh", "basic"), ("fresh", "baseline"), ("continuous", "basic"), ("continuous", "baseline"),
    ), strict=True):
        row = data["summary"]["comparisons"][f"{mode}:candidate-minus-{comparator}"]
        mean, (low, high) = row["units_per_100_rounds"], row["familywise_ci95"]
        ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt="o", color=TEAL, capsize=4, markersize=7)
        ax.text(1.04, y, f"{mean:+.3f}  [{low:+.3f}, {high:+.3f}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=10)
        labels.append(f"{'Fresh shoe' if mode == 'fresh' else 'Continuous'} · vs "
                      f"{'basic strategy' if comparator == 'basic' else 'broad training'}")
    ax.set_yticks([3, 2, 1, 0], labels)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(-0.15, 0.65)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlabel("Laya Blackjack advantage · betting units per 100 rounds", labelpad=12)
    fig.text(0.035, 0.135, "Broad-training baseline: model after initial adaptation and 100,000 training states, before the two refinement stages.", fontsize=9)
    fig.text(0.035, 0.09, "Bonferroni-adjusted 95% intervals across four planned comparisons; normal approximation.", fontsize=9)
    fig.text(0.035, 0.045, "Per policy: 500,000 fresh rounds + 5,000 independent 100-round blocks. All mean returns remained negative.", fontsize=9)
    save(fig, "final-returns")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.71, bottom=0.29, wspace=0.65)
    fig.text(0.035, 0.93, "Closer to the reference on a fresh locked test", fontsize=17, weight="bold")
    fig.text(0.035, 0.865, "16,384 states · 4,840 game groups · full three-question inference", fontsize=11)
    for ax, metric, interval, title, scale in (
        (axes[0], "teacher_agreement", "teacher_agreement_ci95", "Reference action agreement (%) ↑", 100),
        (axes[1], "teacher_ev_regret", "teacher_ev_regret_ci95", "Reference regret (units / decision) ↓", 1),
    ):
        for y, name, color in ((1, "packaged", GRAY), (0, "incumbent", TEAL)):
            audit = data["final_audits"][name]
            value = audit["metrics"][metric] * scale
            low, high = [x * scale for x in audit[interval]]
            ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt="o", color=color, capsize=4, markersize=7)
            label = f"{value:.2f}%" if scale == 100 else f"{value:.6f}"
            ax.annotate(label, (value, y), xytext=(0, 15), textcoords="offset points", ha="center")
        ax.set_yticks([1, 0], ["Broad-training\nbaseline", "Laya Blackjack"])
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=11, pad=15)
        ax.set_ylim(-0.55, 1.65)
        ax.set_xlim((91.5, 97) if scale == 100 else (0, 0.0038))
        if scale == 1:
            ax.xaxis.set_major_locator(MultipleLocator(0.001))
    fig.text(0.035, 0.135, "Broad-training baseline: model after initial adaptation and 100,000 training states, before the two refinement stages.", fontsize=9)
    fig.text(0.035, 0.09, "Individual 95% whole-game bootstrap intervals, conditional on recorded reference labels.", fontsize=9)
    fig.text(0.035, 0.045, "Equal general/depleted state mix; includes Monte Carlo reference approximations. Agreement is not win probability.", fontsize=9)
    save(fig, "final-quality")


if __name__ == "__main__":
    main()
