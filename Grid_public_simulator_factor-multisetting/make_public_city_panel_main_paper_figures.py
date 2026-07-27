"""Create figures for the reported public-city-panel main configuration.

Operational baselines are stored in both learning-method result directories;
they are de-duplicated here before being replicated across rounds for
like-for-like plotting.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


_mplconfig = Path.cwd() / ".matplotlib"
_mplconfig.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


MAIN_TAG = (
    "pc_main_M3T20_B10_D4500N300_q3_lp0p015_bs0p03_"
    "a1t0p08_i0p10_d3e150c128b1024_s2026"
)
LEARNING_METHODS = ("sacsa_reserve", "dfl_pto")
OPERATIONAL_METHODS = ("gap_pacing", "uniform_pacing", "random_pacing", "no_subsidy")

METHOD_LABELS = {
    "sacsa_reserve": "Proposed Beta-Dirichlet SAC-SA",
    "dfl_pto": "DFL-PTO",
    "gap_pacing": "Demand-Supply Gap",
    "uniform_pacing": "Uniform Pacing",
    "random_pacing": "Random Pacing",
    "no_subsidy": "No Subsidy",
}
METHOD_ORDER = [
    "sacsa_reserve",
    "dfl_pto",
    "gap_pacing",
    "uniform_pacing",
    "random_pacing",
    "no_subsidy",
]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "sacsa_reserve": "#5477C4",
    "dfl_pto": "#CC6F47",
    "gap_pacing": "#B8A037",
    "uniform_pacing": "#7A828F",
    "random_pacing": "#BD569B",
    "no_subsidy": "#464C55",
}
MARKERS = {
    "sacsa_reserve": "D",
    "dfl_pto": "o",
    "gap_pacing": "^",
    "uniform_pacing": "P",
    "random_pacing": "X",
    "no_subsidy": "v",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def method_order_present(keys: Iterable[str]) -> List[str]:
    present = set(str(key) for key in keys)
    return [key for key in METHOD_ORDER if key in present]


def default_result_dirs(root: Path) -> list[Path]:
    return [root / f"results_{MAIN_TAG}_{method}" for method in LEARNING_METHODS]


def load_main_results(result_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    for directory in result_dirs:
        summary_path = directory / "final_eval_summary.csv"
        raw_path = directory / "final_eval_raw.csv"
        if not summary_path.exists() or not raw_path.exists():
            raise FileNotFoundError(f"Incomplete result directory: {directory}")
        summary = pd.read_csv(summary_path)
        raw = pd.read_csv(raw_path)
        summary["source_dir"] = directory.name
        raw["source_dir"] = directory.name
        summary_frames.append(summary)
        raw_frames.append(raw)

    summary = pd.concat(summary_frames, ignore_index=True)
    raw = pd.concat(raw_frames, ignore_index=True)
    summary = summary[summary["method"].isin(METHOD_ORDER)].copy()
    raw = raw[raw["method"].isin(METHOD_ORDER)].copy()

    learned_summary = summary[summary["method"].isin(LEARNING_METHODS) & (summary["round"] > 0)].copy()
    learned_raw = raw[raw["method"].isin(LEARNING_METHODS) & (raw["round"] > 0)].copy()
    rounds = sorted(learned_summary["round"].astype(int).unique().tolist())
    if rounds != [1, 2, 3]:
        raise ValueError(f"Expected completed outer rounds [1, 2, 3], found {rounds}.")

    operational_summary = (
        summary[summary["method"].isin(OPERATIONAL_METHODS)]
        .sort_values("source_dir")
        .drop_duplicates(["method"], keep="first")
    )
    operational_raw = (
        raw[raw["method"].isin(OPERATIONAL_METHODS)]
        .sort_values("source_dir")
        .drop_duplicates(["method", "episode"], keep="first")
    )

    replicated_summary = []
    replicated_raw = []
    for q in rounds:
        summary_q = operational_summary.copy()
        raw_q = operational_raw.copy()
        summary_q["round"] = q
        raw_q["round"] = q
        replicated_summary.append(summary_q)
        replicated_raw.append(raw_q)

    summary = pd.concat([learned_summary, *replicated_summary], ignore_index=True)
    raw = pd.concat([learned_raw, *replicated_raw], ignore_index=True)
    summary["round"] = summary["round"].astype(int)
    raw["round"] = raw["round"].astype(int)
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    raw["method_label"] = raw["method"].map(METHOD_LABELS)
    return summary, raw


def round_summary(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["round", "method", "method_label"], as_index=False)
        .agg(
            net_reward_mean=("net_reward", "mean"),
            net_reward_se=("net_reward", lambda x: x.std(ddof=1) / math.sqrt(max(len(x), 1))),
            response_score_mean=("response_score", "mean"),
            response_score_se=("response_score", lambda x: x.std(ddof=1) / math.sqrt(max(len(x), 1))),
            spend_mean=("spend", "mean"),
            spend_se=("spend", lambda x: x.std(ddof=1) / math.sqrt(max(len(x), 1))),
            final_budget_mean=("final_budget", "mean"),
            eval_episodes=("net_reward", "size"),
        )
    )


def save_figure(fig: plt.Figure, base_paths: list[Path]) -> None:
    for base_path in base_paths:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf", ".svg"):
            output = base_path.with_suffix(suffix)
            kwargs = {"bbox_inches": "tight"}
            if suffix == ".png":
                kwargs["dpi"] = 360
            fig.savefig(output, **kwargs)
    plt.close(fig)


def plot_round_boxplots(raw: pd.DataFrame, output_paths: list[Path]) -> None:
    rounds = sorted(raw["round"].unique())
    methods = method_order_present(raw["method"].unique())
    positions = list(range(len(methods), 0, -1))
    fig, axes = plt.subplots(1, len(rounds), figsize=(13.8, 6.1), sharey=True)
    x_min = math.floor(float(raw["net_reward"].quantile(0.005)) - 2.0)
    x_max = math.ceil(float(raw["net_reward"].quantile(0.995)) + 2.0)

    for axis, q in zip(axes, rounds):
        q_data = raw[raw["round"] == q]
        series = [q_data.loc[q_data["method"] == method, "net_reward"].to_numpy() for method in methods]
        for y, values, method in zip(positions, series, methods):
            axis.boxplot(
                [values],
                vert=False,
                positions=[y],
                widths=0.56,
                patch_artist=True,
                showfliers=True,
                boxprops={
                    "facecolor": COLORS[method] + "4D",
                    "edgecolor": COLORS[method],
                    "linewidth": 1.05,
                },
                medianprops={"color": TOKENS["ink"], "linewidth": 1.15},
                whiskerprops={"color": TOKENS["muted"], "linewidth": 0.9},
                capprops={"color": TOKENS["muted"], "linewidth": 0.9},
                flierprops={
                    "marker": ".",
                    "markersize": 2.0,
                    "markerfacecolor": COLORS[method],
                    "markeredgecolor": COLORS[method],
                    "alpha": 0.13,
                },
            )
            mean = float(np.mean(values))
            axis.scatter(
                mean,
                y,
                s=22,
                marker="D",
                facecolor=COLORS[method],
                edgecolor=TOKENS["panel"],
                linewidth=0.45,
                zorder=4,
            )
        axis.set_title(f"Round q = {q}", fontsize=11, fontweight="semibold", color=TOKENS["ink"], pad=10)
        axis.set_xlim(x_min, x_max)
        axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        axis.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
        axis.grid(axis="y", visible=False)
        axis.spines["left"].set_color(TOKENS["axis"])
        axis.spines["bottom"].set_color(TOKENS["axis"])
        axis.tick_params(axis="x", labelsize=8.5)
        axis.tick_params(axis="y", length=0, labelsize=9.1)
        axis.set_xlabel("Net reward", fontsize=9.5, color=TOKENS["ink"])

    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(
        [METHOD_LABELS[method].replace("Proposed Beta-Dirichlet SAC-SA", "Proposed\nBeta-Dirichlet SAC-SA") for method in methods]
    )
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.27, right=0.985, bottom=0.14, top=0.91, wspace=0.085)
    save_figure(fig, output_paths)


def plot_round_curve(summary: pd.DataFrame, output_paths: list[Path]) -> None:
    methods = method_order_present(summary["method"].unique())
    fig, axis = plt.subplots(figsize=(9.8, 5.5))
    for method in methods:
        rows = summary[summary["method"] == method].sort_values("round")
        proposed = method == "sacsa_reserve"
        axis.plot(
            rows["round"],
            rows["net_reward_mean"],
            marker=MARKERS[method],
            markersize=7 if proposed else 5.5,
            linewidth=2.2 if proposed else 1.35,
            color=COLORS[method],
            markerfacecolor=COLORS[method] if proposed else TOKENS["panel"],
            markeredgecolor=COLORS[method],
            alpha=0.98 if proposed else 0.84,
            label=METHOD_LABELS[method],
        )
    axis.set_xlabel("Outer round q", fontsize=10, color=TOKENS["ink"])
    axis.set_ylabel("Mean net reward", fontsize=10, color=TOKENS["ink"])
    axis.set_xticks(sorted(summary["round"].unique()))
    axis.grid(True, color=TOKENS["grid"], linewidth=0.8)
    axis.spines["left"].set_color(TOKENS["axis"])
    axis.spines["bottom"].set_color(TOKENS["axis"])
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        ncol=3,
        fontsize=8.5,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.31, top=0.94)
    save_figure(fig, output_paths)


def plot_reward_spend(summary: pd.DataFrame, output_paths: list[Path]) -> None:
    rounds = sorted(summary["round"].unique())
    methods = method_order_present(summary["method"].unique())
    fig, axes = plt.subplots(1, len(rounds), figsize=(14.2, 5.7), sharey=True)
    y_min = float(summary["net_reward_mean"].min()) - 2.0
    y_max = float(summary["net_reward_mean"].max()) + 2.0

    for column, (axis, q) in enumerate(zip(axes, rounds)):
        q_data = summary[summary["round"] == q]
        for method in methods:
            row = q_data[q_data["method"] == method]
            if row.empty:
                continue
            value = row.iloc[0]
            proposed = method == "sacsa_reserve"
            axis.errorbar(
                value["spend_mean"],
                value["net_reward_mean"],
                xerr=value["spend_se"],
                yerr=value["net_reward_se"],
                marker=MARKERS[method],
                markersize=7.0 if proposed else 5.6,
                color=COLORS[method],
                markerfacecolor=COLORS[method] if proposed else TOKENS["panel"],
                markeredgecolor=COLORS[method],
                elinewidth=1.0,
                capsize=2.0,
                linestyle="none",
                alpha=0.98 if proposed else 0.82,
                zorder=4 if proposed else 3,
            )
        axis.set_title(f"Round q = {q}", fontsize=11.2, fontweight="semibold", color=TOKENS["ink"], pad=8)
        axis.set_xlabel("Mean spend", fontsize=9.4, color=TOKENS["ink"])
        if column == 0:
            axis.set_ylabel("Mean net reward", fontsize=9.4, color=TOKENS["ink"])
        axis.set_xlim(-0.25, 10.55)
        axis.set_ylim(y_min, y_max)
        axis.grid(True, color=TOKENS["grid"], linewidth=0.8)
        axis.spines["left"].set_color(TOKENS["axis"])
        axis.spines["bottom"].set_color(TOKENS["axis"])
        axis.tick_params(labelsize=8.5)

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[method],
            color=COLORS[method],
            label=METHOD_LABELS[method],
            markerfacecolor=COLORS[method] if method == "sacsa_reserve" else TOKENS["panel"],
            markeredgecolor=COLORS[method],
            linewidth=1.6 if method == "sacsa_reserve" else 1.0,
            markersize=6.5,
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=8.6,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.29, top=0.90, wspace=0.25)
    save_figure(fig, output_paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dirs", nargs="+", type=Path)
    parser.add_argument("--out_dir", type=Path)
    args = parser.parse_args()

    configure_matplotlib()
    root = Path(__file__).resolve().parent
    result_dirs = args.result_dirs or default_result_dirs(root)
    result_dirs = [path if path.is_absolute() else root / path for path in result_dirs]
    out_dir = args.out_dir or root / "paper_figures_public_city_main"
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    summary_input, raw = load_main_results(result_dirs)
    summary = round_summary(raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(source_dir / "public_city_main_round_eval_raw.csv", index=False)
    summary.to_csv(source_dir / "public_city_main_round_eval_summary.csv", index=False)
    summary_input.to_csv(source_dir / "public_city_main_reported_summary.csv", index=False)

    plot_round_boxplots(
        raw,
        [out_dir / "fig_public_city_main_round_boxplot"],
    )
    plot_round_curve(
        summary,
        [out_dir / "fig_public_city_main_round_curve"],
    )
    plot_reward_spend(
        summary,
        [out_dir / "fig_public_city_main_reward_spend"],
    )

    final_round = int(summary["round"].max())
    final_summary = summary[summary["round"] == final_round].sort_values("net_reward_mean", ascending=False)
    final_summary.to_csv(source_dir / "public_city_main_final_summary.csv", index=False)
    print(f"Main configuration: {MAIN_TAG}")
    print(f"Source result directories: {', '.join(str(path) for path in result_dirs)}")
    print(f"Paper figures written to: {out_dir}")
    print(final_summary[["method_label", "net_reward_mean", "net_reward_se", "spend_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
