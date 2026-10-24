"""Draw the four figures used by docs/analysis.md from results/.

    python scripts/make_figures.py        # writes docs/figures/*.png

Nothing is computed here that is not already in results/: medians over seeds for the curves, the weekly
counts, the monthly table, the reliability bins and the drift counts are read and plotted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SPLIT_NAMES = {"valid": "validation", "test": "test", "post": "after the test"}
PNG_COMPRESS_LEVEL = 6  # zlib level of the PNG writer; fixed so the file bytes depend only on the drawing


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2.0, "legend.frameon": False,
    })


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None},
                pil_kwargs={"compress_level": PNG_COMPRESS_LEVEL})
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def percent_axis(ax) -> None:
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def capacity_curves(results: Path, summary: dict, out: Path) -> None:
    curve = pd.read_csv(results / "capacity_curve_test.csv")
    capacity = summary["setup"]["capacity_per_week"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    for ax, metric, title in ((axes[0], "precision", "Precision"), (axes[1], "recall", "Recall")):
        for model, color, label, dashed in (("perfect_ranking", MUTED, "Best possible ranking", True),
                                            ("base_rate", MUTED, "No model (random order)", False),
                                            ("previous_result_rule", ORANGE, "Previous-result rule", False),
                                            ("logistic", AQUA, "Logistic regression", False),
                                            ("gbm", BLUE, "Boosted model", False)):
            frame = curve[curve["model"] == model].pivot(index="capacity", columns="seed", values=metric)
            x = frame.index.to_numpy()
            ax.plot(x, frame.median(axis=1), color=color, label=label, linestyle="--" if dashed else "-",
                    linewidth=1.5 if model in ("perfect_ranking", "base_rate") else 2.0)
            if model == "gbm":
                ax.fill_between(x, frame.min(axis=1), frame.max(axis=1), color=color, alpha=0.15, linewidth=0)
        ax.axvline(capacity, color=INK2, linewidth=1)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Inspections put first per week")
        ax.set_ylim(0, 1.02)
        percent_axis(ax)
    axes[0].text(capacity + 4, 0.04, f"{capacity} a week", color=INK, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
    fig.tight_layout()
    save(fig, out / "capacity_curves.png")


def weekly(results: Path, summary: dict, out: Path) -> None:
    weekly_frame = pd.read_csv(results / "weekly_test.csv", parse_dates=["week"])
    visits = weekly_frame["visits"]
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    x = weekly_frame["week"]
    ax.plot(x, weekly_frame["positives"] / weekly_frame["pool"], color=MUTED, linewidth=1.5, label="Share of the whole week's pool that found the problem")
    ax.plot(x, weekly_frame["hits_previous_result_rule"] / visits, color=ORANGE, label="Previous-result rule, top of the list")
    ax.plot(x, weekly_frame["hits_gbm"] / visits, color=BLUE, label="Boosted model, top of the list")
    ax.set_ylim(0.3, 0.9)
    percent_axis(ax)
    ax.set_title(f"Precision of the top {summary['setup']['capacity_per_week']} in each test week", fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.legend(loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.5), fontsize=8.5)
    fig.tight_layout()
    save(fig, out / "weekly_precision.png")


def calibration(results: Path, summary: dict, out: Path) -> None:
    cal = summary["calibration"]
    months = pd.read_csv(results / "monthly_calibration.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.5, label="Perfect calibration")
    for key, color, label in (("before", BLUE, "Raw scores"), ("after", ORANGE, "After " + cal["chosen_method"].capitalize() + " scaling")):
        table = pd.DataFrame(cal["reliability_test_5_seed_average"][key])
        table = table[table["n"] > 0]
        ax.plot(table["mean_predicted"], table["observed_rate"], color=color, marker="o", markersize=5, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability in the bin")
    ax.set_ylabel("Observed share")
    percent_axis(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Reliability on the test window", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5)
    ax = axes[1]
    x = np.arange(len(months))
    for split in months["split"].unique():
        index = np.where(months["split"] == split)[0]
        if split == "test":
            ax.axvspan(index.min() - 0.5, index.max() + 0.5, color="#f4f3ee", linewidth=0)
        ax.text((index.min() + index.max()) / 2, 0.592, SPLIT_NAMES[split], ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.plot(x, months["observed_rate"], color=INK, label="Observed share with a hit")
    ax.plot(x, months["mean_raw_probability"], color=BLUE, label="Mean raw score")
    ax.plot(x, months["mean_calibrated_probability"], color=ORANGE, label="Mean after " + cal["chosen_method"].capitalize() + " scaling")
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{names[int(m[5:7]) - 1]} {m[2:4]}" for m in months["month"]][::2])
    ax.set_ylim(0.40, 0.60)
    ax.set_yticks(np.arange(0.40, 0.601, 0.05))
    percent_axis(ax)
    ax.set_title("Observed share and mean prediction, by month", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=1, fontsize=8.5)
    fig.tight_layout()
    save(fig, out / "calibration.png")


def drift(results: Path, out: Path) -> None:
    matched = json.loads((results / "drift_matched_history.json").read_text(encoding="utf-8"))
    groups = list(matched["all_rows"]["by_group"])
    shares = {name: [matched[name]["by_group"][g]["major"] / matched[name]["by_group"][g]["feature_windows"] for g in groups]
              for name in ("all_rows", "matched")}
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    y = np.arange(len(groups))
    height = 0.34
    ax.barh(y - height / 2, shares["all_rows"], height=height, color=BLUE, label="All inspections")
    ax.barh(y + height / 2, shares["matched"], height=height, color=ORANGE, label="Inspections with the same visible history in every window")
    for index, group in enumerate(groups):
        for offset, name in ((-height / 2, "all_rows"), (height / 2, "matched")):
            block = matched[name]["by_group"][group]
            ax.text(shares[name][index] + 0.01, index + offset, f"{block['major']} of {block['feature_windows']}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([g.capitalize() for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(f"Share of feature-windows with a major shift in the feature distribution (PSI at or above {matched['cutoffs']['major']})")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2, fontsize=8.5)
    fig.tight_layout()
    save(fig, out / "drift_alerts.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="docs/figures")
    args = parser.parse_args()
    results, out = ROOT / args.results, ROOT / args.out
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    style()
    capacity_curves(results, summary, out)
    weekly(results, summary, out)
    calibration(results, summary, out)
    drift(results, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
