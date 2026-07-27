"""Create paper-style figures for Synthetic T=6 one-factor ablation sweeps."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_mplconfig = Path.cwd() / ".matplotlib"
_mplconfig.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


METHOD_LABELS = {
    "sacsa_reserve": "Proposed Beta-Dirichlet SAC-SA",
    "dfl_pto": "DFL-PTO",
    "gap_heuristic": "Demand-Supply Gap",
    "uniform_pacing": "Uniform Pacing",
    "random_pacing": "Random Pacing",
    "no_subsidy": "No Subsidy",
}

METHOD_ORDER = [
    "sacsa_reserve",
    "dfl_pto",
    "gap_heuristic",
    "uniform_pacing",
    "random_pacing",
    "no_subsidy",
]

OPERATIONAL_METHODS = {"gap_heuristic", "uniform_pacing", "random_pacing", "no_subsidy"}

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
    "gap_heuristic": "#B8A037",
    "uniform_pacing": "#7A828F",
    "random_pacing": "#BD569B",
    "no_subsidy": "#464C55",
}

MARKERS = {
    "sacsa_reserve": "D",
    "dfl_pto": "o",
    "gap_heuristic": "^",
    "uniform_pacing": "P",
    "random_pacing": "X",
    "no_subsidy": "v",
}

FACET_SPECS = {
    "m": ("config_M", r"$M$", r"$M$"),
    "d0": ("config_initial_logged_days", r"$n_0$", r"$n_0$"),
    "newdays": ("config_new_logged_days", r"$n_q^{\mathrm{new}}$", r"$n_q^{\mathrm{new}}$"),
    "seed": ("source_seed", "seed", "seed"),
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )


def _sem(x: pd.Series) -> float:
    values = pd.to_numeric(x, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def method_order_present(keys: Iterable[str]) -> list[str]:
    present = {str(k) for k in keys}
    return [key for key in METHOD_ORDER if key in present]


def legend_handles(methods: list[str]) -> list[Line2D]:
    handles: list[Line2D] = []
    for method in methods:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=MARKERS[method],
                color=COLORS[method],
                label=METHOD_LABELS[method],
                markerfacecolor=COLORS[method] if method == "sacsa_reserve" else TOKENS["panel"],
                markeredgecolor=COLORS[method],
                linewidth=1.8 if method == "sacsa_reserve" else 1.1,
                markersize=6.5,
            )
        )
    return handles


def pretty_method_label(method: str) -> str:
    return METHOD_LABELS[method].replace(
        "Proposed Beta-Dirichlet SAC-SA", "Proposed\nBeta-Dirichlet SAC-SA"
    )


def save_figure(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {".png": {"dpi": 360}, ".pdf": {}, ".svg": {}}.items():
        fig.savefig(fig_dir / f"{name}{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _facet_sort_key(value: object) -> tuple[int, float | str]:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return (0, float(numeric))
    return (1, str(value))


def _format_facet_value(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        number = float(numeric)
        if number.is_integer():
            return str(int(number))
        return f"{number:g}"
    return str(value)


def load_raw(summary_dir: Path, facet: str) -> tuple[pd.DataFrame, str, str]:
    raw_path = summary_dir / "round_eval_raw_all_methods.csv"
    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}")
    raw = pd.read_csv(raw_path)
    raw = raw[raw["method_key"].isin(METHOD_ORDER)].copy()
    if raw.empty:
        raise SystemExit(f"No expected methods found in {raw_path}")

    facet_column, facet_prefix, facet_axis_label = FACET_SPECS[facet]
    if facet_column not in raw.columns:
        raise SystemExit(f"Missing facet column {facet_column} in {raw_path}")

    raw["facet_value"] = raw[facet_column].map(_format_facet_value)
    raw["facet_label"] = raw["facet_value"].map(lambda value: f"{facet_prefix}={value}")
    raw["round"] = pd.to_numeric(raw["round"], errors="coerce").astype(int)
    for column in ["net_reward", "spend", "response_score", "final_budget"]:
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    learning = raw[~raw["method_key"].isin(OPERATIONAL_METHODS)].copy()
    operational = raw[raw["method_key"].isin(OPERATIONAL_METHODS)].drop_duplicates(
        ["facet_value", "round", "method_key", "eval_seed", "eval_episode"]
    )
    out = pd.concat([learning, operational], ignore_index=True)
    out["method_label"] = out["method_key"].map(METHOD_LABELS)
    return out, facet_prefix, facet_axis_label


def facet_order(raw: pd.DataFrame) -> list[str]:
    values = raw["facet_value"].dropna().unique().tolist()
    return sorted(values, key=_facet_sort_key)


def mean_summary(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["facet_value", "facet_label", "round", "method_key", "method_label"], as_index=False)
        .agg(
            net_reward_mean=("net_reward", "mean"),
            net_reward_se=("net_reward", _sem),
            spend_mean=("spend", "mean"),
            spend_se=("spend", _sem),
            response_score_mean=("response_score", "mean"),
            response_score_se=("response_score", _sem),
            final_budget_mean=("final_budget", "mean"),
            final_budget_se=("final_budget", _sem),
            eval_episodes=("net_reward", "size"),
        )
    )


def plot_final_boxplots(raw: pd.DataFrame, fig_dir: Path, name_prefix: str) -> None:
    facets = facet_order(raw)
    methods = method_order_present(raw["method_key"].unique())
    final_round = int(raw["round"].max())
    fig, axes = plt.subplots(
        1,
        len(facets),
        figsize=(4.0 * len(facets), 3.1),
        squeeze=False,
        sharey=True,
    )
    y_positions = np.arange(len(methods))
    for idx, facet_value in enumerate(facets):
        ax = axes[0, idx]
        subset = raw[(raw["facet_value"] == facet_value) & (raw["round"] == final_round)]
        for y, method in zip(y_positions, methods):
            values = subset.loc[subset["method_key"] == method, "net_reward"].dropna().to_numpy()
            if values.size == 0:
                continue
            ax.boxplot(
                [values],
                positions=[y],
                vert=False,
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
            ax.scatter(
                [float(np.mean(values))],
                [y],
                marker="D",
                s=22,
                color=COLORS[method],
                edgecolor=TOKENS["panel"],
                linewidth=0.45,
                zorder=4,
            )
        title = subset["facet_label"].iloc[0] if not subset.empty else str(facet_value)
        ax.set_title(f"{title}, q={final_round}", fontsize=9.5, weight="bold", color=TOKENS["ink"])
        ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.7)
        ax.set_xlabel("Net reward", fontsize=8.8)
        ax.set_yticks(y_positions)
        if idx == 0:
            ax.set_yticklabels([pretty_method_label(method) for method in methods], fontsize=8.4)
        else:
            ax.tick_params(labelleft=False)
        ax.invert_yaxis()

    handles = legend_handles(methods)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        fontsize=8.2,
        columnspacing=1.2,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.20, right=0.985, top=0.86, bottom=0.30, wspace=0.12)
    save_figure(fig, fig_dir, f"{name_prefix}_final_boxplot")


def plot_round_curve(summary: pd.DataFrame, fig_dir: Path, name_prefix: str) -> None:
    facets = facet_order(summary)
    methods = method_order_present(summary["method_key"].unique())
    fig, axes = plt.subplots(
        1,
        len(facets),
        figsize=(4.0 * len(facets), 3.55),
        squeeze=False,
        sharey=False,
    )
    for idx, facet_value in enumerate(facets):
        ax = axes[0, idx]
        subset = summary[summary["facet_value"] == facet_value]
        for method in methods:
            method_df = subset[subset["method_key"] == method].sort_values("round")
            if method_df.empty:
                continue
            ax.errorbar(
                method_df["round"],
                method_df["net_reward_mean"],
                yerr=method_df["net_reward_se"],
                color=COLORS[method],
                marker=MARKERS[method],
                markerfacecolor=COLORS[method] if method == "sacsa_reserve" else TOKENS["panel"],
                markeredgecolor=COLORS[method],
                linewidth=1.8 if method == "sacsa_reserve" else 1.15,
                markersize=5.5,
                capsize=2.0,
                alpha=0.98,
            )
        title = subset["facet_label"].iloc[0] if not subset.empty else str(facet_value)
        ax.set_title(title, fontsize=9.5, weight="bold", color=TOKENS["ink"])
        ax.set_xlabel("Outer round q", fontsize=8.8)
        if idx == 0:
            ax.set_ylabel("Mean net reward", fontsize=8.8)
        ax.grid(color=TOKENS["grid"], linewidth=0.7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    handles = legend_handles(methods)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        fontsize=8.2,
        columnspacing=1.2,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.27, wspace=0.22)
    save_figure(fig, fig_dir, f"{name_prefix}_round_curve")


def plot_reward_spend(summary: pd.DataFrame, fig_dir: Path, name_prefix: str) -> None:
    facets = facet_order(summary)
    methods = method_order_present(summary["method_key"].unique())
    final_round = int(summary["round"].max())
    fig, axes = plt.subplots(
        1,
        len(facets),
        figsize=(4.0 * len(facets), 3.55),
        squeeze=False,
        sharey=False,
    )
    for idx, facet_value in enumerate(facets):
        ax = axes[0, idx]
        subset = summary[(summary["facet_value"] == facet_value) & (summary["round"] == final_round)]
        for method in methods:
            row = subset[subset["method_key"] == method]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.errorbar(
                row["spend_mean"],
                row["net_reward_mean"],
                xerr=row["spend_se"],
                yerr=row["net_reward_se"],
                color=COLORS[method],
                marker=MARKERS[method],
                markerfacecolor=COLORS[method] if method == "sacsa_reserve" else TOKENS["panel"],
                markeredgecolor=COLORS[method],
                markersize=5.8,
                linewidth=0,
                elinewidth=0.8,
                capsize=2.0,
                alpha=0.98,
            )
        title = subset["facet_label"].iloc[0] if not subset.empty else str(facet_value)
        ax.set_title(f"{title}, q={final_round}", fontsize=9.5, weight="bold", color=TOKENS["ink"])
        ax.set_xlabel("Mean spend", fontsize=8.8)
        if idx == 0:
            ax.set_ylabel("Mean net reward", fontsize=8.8)
        ax.grid(color=TOKENS["grid"], linewidth=0.7)

    handles = legend_handles(methods)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        fontsize=8.2,
        columnspacing=1.2,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.27, wspace=0.22)
    save_figure(fig, fig_dir, f"{name_prefix}_reward_spend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_dir", default=os.environ.get("SYNTHETIC_T6_ABLATION_SUMMARY_DIR"))
    parser.add_argument("--facet", choices=sorted(FACET_SPECS), required=True)
    parser.add_argument(
        "--keep_values",
        default=None,
        help="Optional comma-separated facet values to keep, after display formatting.",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--name_prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary_dir:
        raise SystemExit("Pass --summary_dir or set SYNTHETIC_T6_ABLATION_SUMMARY_DIR.")
    configure_matplotlib()
    summary_dir = Path(args.summary_dir)
    raw, _, _ = load_raw(summary_dir, args.facet)
    if args.keep_values:
        keep = {
            _format_facet_value(value.strip())
            for value in str(args.keep_values).split(",")
            if value.strip()
        }
        raw = raw[raw["facet_value"].isin(keep)].copy()
        if raw.empty:
            raise SystemExit(f"No rows left after --keep_values={args.keep_values}")
    summary = mean_summary(raw)
    fig_dir = Path(args.out_dir) if args.out_dir else summary_dir / "paper_figures"
    name_prefix = args.name_prefix or f"fig_synT6_ab{args.facet}"

    source_dir = fig_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(source_dir / f"{name_prefix}_round_eval_raw.csv", index=False)
    summary.to_csv(source_dir / f"{name_prefix}_round_eval_summary.csv", index=False)

    plot_final_boxplots(raw, fig_dir, name_prefix)
    plot_round_curve(summary, fig_dir, name_prefix)
    plot_reward_spend(summary, fig_dir, name_prefix)
    print(f"[done] wrote ablation figures to {fig_dir}")


if __name__ == "__main__":
    main()
