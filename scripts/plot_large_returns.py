"""Export research figures from the verified large-return analysis JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
INK, TEAL, GREY = "#203648", "#087f6c", "#b9c9c5"
LABELS = {
    **{f"s17-6d-{p}p": f"6 decks · S17 · {p} player{'s' if p > 1 else ''}" for p in range(1, 8)},
    "h17-no-das-6to5": "6 decks · H17 · no DAS · 6:5",
    "s17-2d-3p": "2 decks · S17 · 3 players",
    "random-tablemates-7p": "6 decks · 7 players · random",
}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=180, facecolor="white", metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    data = json.loads((ROOT / "docs/results/large-return-analysis.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#cbd1d6",
            "svg.hashsalt": "laya-large-returns-20260922",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.subplots_adjust(left=0.235, right=0.71, top=0.72, bottom=0.25)
    fig.text(0.035, 0.93, "12 million rounds · paired model comparisons", fontsize=17, weight="bold")
    fig.text(0.035, 0.865, "Latest candidate versus basic strategy and its predecessor", fontsize=11)
    ax.axvline(0, color=INK, linewidth=1, alpha=0.45)
    labels = []
    for y, (mode, comparator) in zip(
        (3, 2, 1, 0),
        (("fresh", "basic"), ("fresh", "baseline"), ("continuous", "basic"), ("continuous", "baseline")),
        strict=True,
    ):
        row = data["summary"]["comparisons"][f"{mode}:candidate-minus-{comparator}"]
        mean = row["units_per_100_rounds"]
        low, high = row["familywise_ci95"]
        ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", color=TEAL, capsize=4, markersize=7)
        ax.text(
            1.04,
            y,
            f"{mean:+.3f}  [{low:+.3f}, {high:+.3f}]",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=10,
        )
        labels.append(
            f"{'Fresh shoe' if mode == 'fresh' else 'Continuous'} · vs {'basic' if comparator == 'basic' else 'predecessor'}"
        )
    ax.set_yticks([3, 2, 1, 0], labels)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlim(-0.12, 0.36)
    ax.set_ylim(-0.5, 3.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("Candidate advantage · betting units per 100 rounds", labelpad=12)
    fig.text(
        0.035,
        0.095,
        "Intervals: 95% simultaneous family coverage across four predeclared comparisons (normal approximation).",
        fontsize=9,
    )
    fig.text(
        0.035,
        0.05,
        "Per policy: 1 million fresh rounds + 30,000 independent 100-round blocks. Equal weights across ten scenarios.",
        fontsize=9,
    )
    save(fig, "large-return-overview")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.83, bottom=0.12, hspace=0.23, wspace=0.12)
    fig.text(0.035, 0.95, "Returns by rules and table size", fontsize=18, weight="bold")
    fig.text(
        0.035, 0.91, "Candidate-minus-comparator differences · betting units per 100 rounds", fontsize=11
    )
    scenarios = list(data["reports"]["fresh"]["identity"]["scenarios"])
    for row_index, mode in enumerate(("fresh", "continuous")):
        for col_index, comparator in enumerate(("basic", "baseline")):
            ax = axes[row_index, col_index]
            ax.axvline(0, color=INK, linewidth=1, alpha=0.5)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#e8edeb", linewidth=0.8)
            for y, scenario in enumerate(scenarios):
                item = next(
                    r
                    for r in data["scenarios"]["rows"]
                    if r["mode"] == mode and r["comparator"] == comparator and r["scenario"] == scenario
                )
                low, high = item["exploratory_familywise_ci95"]
                ax.plot([low, high], [y, y], color=GREY, linewidth=2)
                low, high = item["nominal_ci95"]
                ax.plot([low, high], [y, y], color=TEAL, linewidth=2)
                ax.plot(item["difference_per_100"], y, "o", color=TEAL, markersize=4.5)
            ax.set_yticks(range(len(scenarios)), [LABELS[s] for s in scenarios])
            ax.tick_params(axis="y", length=0, pad=10)
            ax.set_ylim(9.65, -0.65)
            ax.set_xlim(-0.92, 0.98)
            ax.xaxis.set_major_locator(MultipleLocator(0.4))
            title = f"{'Fresh shoes' if mode == 'fresh' else 'Continuous play'} · vs {'basic heuristic' if comparator == 'basic' else 'predecessor'}"
            ax.set_title(title, loc="left", fontsize=11, pad=12)
    fig.legend(
        [Line2D([0], [0], color=TEAL, linewidth=2), Line2D([0], [0], color=GREY, linewidth=2)],
        ["Individual 95% interval", "95% family coverage across 40 contrasts"],
        loc="upper left",
        bbox_to_anchor=(0.26, 0.895),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    fig.text(
        0.035,
        0.065,
        "Exploratory subgroups; normal intervals. Per scenario and policy: 100,000 fresh rounds or 3,000 independent blocks.",
        fontsize=9,
    )
    fig.text(
        0.035,
        0.03,
        "S17/H17: dealer stands/hits soft 17. DAS: double after split. Positive differences do not establish positive profit.",
        fontsize=9,
    )
    save(fig, "large-return-scenarios")


if __name__ == "__main__":
    main()
