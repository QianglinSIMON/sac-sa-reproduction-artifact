"""Create main-paper synthetic figures from a finished SAC-SA summary.

This script draws the three figures used in the manuscript's synthetic
experiment section:

1. round-faceted episode-level net-reward boxplots;
2. round-by-round mean net reward;
3. reward-spend tradeoff by round.

Only the main paper method set is shown. Operational baselines are
de-duplicated because each learning-method run writes the same operational
policies into its raw evaluation table.
"""
from __future__ import annotations

import argparse
import math
import os
import re
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
    "td3_projection": "TD3 feasible shares",
    "dfl_pto": "DFL-PTO",
    "gap_heuristic": "Demand-Supply Gap",
    "uniform_pacing": "Uniform Pacing",
    "random_pacing": "Random Pacing",
    "no_subsidy": "No Subsidy",
}

METHOD_ORDER = [
    "sacsa_reserve",
    "td3_projection",
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
    "td3_projection": "#71B436",
    "dfl_pto": "#CC6F47",
    "gap_heuristic": "#B8A037",
    "uniform_pacing": "#7A828F",
    "random_pacing": "#BD569B",
    "no_subsidy": "#464C55",
}

MARKERS = {
    "sacsa_reserve": "D",
    "td3_projection": "s",
    "dfl_pto": "o",
    "gap_heuristic": "^",
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


def method_order_present(keys: Iterable[str]) -> list[str]:
    present = set(str(k) for k in keys)
    return [k for k in METHOD_ORDER if k in present]


def _sem(x: pd.Series) -> float:
    values = pd.to_numeric(x, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def _first_config_value(raw: pd.DataFrame, column: str) -> float | None:
    if column not in raw.columns:
        return None
    values = pd.to_numeric(raw[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _infer_horizon(summary_dir: Path, raw: pd.DataFrame) -> tuple[str, int | None]:
    t_value = _first_config_value(raw, "config_T")
    if t_value is not None:
        t_int = int(round(t_value))
        return f"T={t_int}", t_int
    match = re.search(r"(?:^|_)T(\d+)(?:_|$)", str(summary_dir))
    if match:
        t_int = int(match.group(1))
        return f"T={t_int}", t_int
    return "Synthetic", None


def summary_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    for env_name in ("SYNTHETIC_T6_SUMMARY_DIR", "SYNTHETIC_T12_SUMMARY_DIR", "SYNTHETIC_MAIN_SUMMARY_DIR"):
        value = os.environ.get(env_name)
        if value:
            dirs.append(Path(value))
    if args.summary_dir:
        dirs.extend(Path(p) for p in args.summary_dir)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise SystemExit(
            "No summary directory provided. Set SYNTHETIC_MAIN_SUMMARY_DIR or pass --summary_dir."
        )
    return unique


def load_summary_raw(summary_dir: Path) -> pd.DataFrame:
    raw_path = summary_dir / "round_eval_raw_all_methods.csv"
    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}")
    raw = pd.read_csv(raw_path)
    raw = raw[raw["method_key"].isin(METHOD_ORDER)].copy()
    if raw.empty:
        raise SystemExit(f"No main-paper methods found in {raw_path}")

    learning = raw[~raw["method_key"].isin(OPERATIONAL_METHODS)].copy()
    operational = raw[raw["method_key"].isin(OPERATIONAL_METHODS)].drop_duplicates(
        ["round", "method_key", "eval_seed", "eval_episode"]
    )
    out = pd.concat([learning, operational], ignore_index=True)
    out["round"] = pd.to_numeric(out["round"], errors="coerce").astype(int)
    for col in ["net_reward", "spend", "response_score", "final_budget"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    horizon, t_value = _infer_horizon(summary_dir, raw)
    out["horizon"] = horizon
    out["T"] = t_value if t_value is not None else np.nan
    out["method_label"] = out["method_key"].map(METHOD_LABELS)
    out["source_summary_dir"] = str(summary_dir)
    return out


def load_all_raw(summary_dirs: list[Path]) -> pd.DataFrame:
    return pd.concat([load_summary_raw(path) for path in summary_dirs], ignore_index=True)


def mean_summary(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["horizon", "T", "round", "method_key", "method_label"], dropna=False, as_index=False)
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


def save_figure(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {".png": {"dpi": 360}, ".pdf": {}, ".svg": {}}.items():
        fig.savefig(fig_dir / f"{name}{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def legend_handles(methods: list[str]) -> list[Line2D]:
    handles = []
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
    label = METHOD_LABELS[method]
    return label.replace("Proposed Beta-Dirichlet SAC-SA", "Proposed\nBeta-Dirichlet SAC-SA")


def horizon_order(raw: pd.DataFrame) -> list[str]:
    tmp = raw[["horizon", "T"]].drop_duplicates()
    tmp = tmp.assign(sort_T=pd.to_numeric(tmp["T"], errors="coerce"))
    tmp = tmp.sort_values(["sort_T", "horizon"], na_position="last")
    return tmp["horizon"].tolist()


def plot_round_boxplots(raw: pd.DataFrame, fig_dir: Path, name_prefix: str) -> None:
    horizons = horizon_order(raw)
    rounds = sorted(raw["round"].unique())
    methods = method_order_present(raw["method_key"].unique())
    n_rows = len(horizons)
    n_cols = len(rounds)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.15 * n_cols, 2.7 * n_rows + 0.85),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    y_positions = np.arange(len(methods))
    for r_idx, horizon in enumerate(horizons):
        for c_idx, round_id in enumerate(rounds):
            ax = axes[r_idx, c_idx]
            subset = raw[(raw["horizon"] == horizon) & (raw["round"] == round_id)]
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
            ax.set_title(f"{horizon}, round q = {round_id}", fontsize=9.5, weight="bold", color=TOKENS["ink"])
            ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set_xlabel("Net reward" if r_idx == n_rows - 1 else "", fontsize=8.8)
            if c_idx == 0:
                ax.set_yticks(y_positions)
                ax.set_yticklabels([pretty_method_label(m) for m in methods], fontsize=8.5)
            else:
                ax.set_yticks(y_positions)
                ax.tick_params(labelleft=False)
            ax.invert_yaxis()

    handles = legend_handles(methods)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=min(4, len(handles)),
        fontsize=8.2,
        columnspacing=1.2,
        handlelength=2.1,
    )
    bottom = 0.27 if n_rows == 1 else 0.16
    fig.subplots_adjust(left=0.18, right=0.985, top=0.95, bottom=bottom, wspace=0.12, hspace=0.42)
    save_figure(fig, fig_dir, f"{name_prefix}_round_boxplot")


def plot_round_curve(summary: pd.DataFrame, fig_dir: Path, name_prefix: str) -> None:
    horizons = horizon_order(summary)
    methods = method_order_present(summary["method_key"].unique())
    n_cols = len(horizons)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.25 * n_cols, 3.5), squeeze=False, sharey=False)

    for idx, horizon in enumerate(horizons):
        ax = axes[0, idx]
        subset = summary[summary["horizon"] == horizon]
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
        ax.set_title(horizon, fontsize=9.5, weight="bold", color=TOKENS["ink"])
        ax.set_xlabel("Outer round q", fontsize=8.8)
        if idx == 0:
            ax.set_ylabel("Mean net reward", fontsize=8.8)
        ax.grid(color=TOKENS["grid"], linewidth=0.7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    handles = legend_handles(methods)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(4, len(handles)),
        fontsize=8.2,
        columnspacing=1.2,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.22, wspace=0.22)
    save_figure(fig, fig_dir, f"{name_prefix}_round_curve")


def plot_reward_spend(summary: pd.DataFrame, fig_dir: Path, name_prefix: str) -> None:
    horizons = horizon_order(summary)
    rounds = sorted(summary["round"].unique())
    methods = method_order_present(summary["method_key"].unique())
    n_rows = len(horizons)
    n_cols = len(rounds)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.15 * n_cols, 2.75 * n_rows + 0.75),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for r_idx, horizon in enumerate(horizons):
        for c_idx, round_id in enumerate(rounds):
            ax = axes[r_idx, c_idx]
            subset = summary[(summary["horizon"] == horizon) & (summary["round"] == round_id)]
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
            ax.set_title(f"{horizon}, round q = {round_id}", fontsize=9.5, weight="bold", color=TOKENS["ink"])
            ax.set_xlabel("Mean spend" if r_idx == n_rows - 1 else "", fontsize=8.8)
            if c_idx == 0:
                ax.set_ylabel("Mean net reward", fontsize=8.8)
            ax.grid(color=TOKENS["grid"], linewidth=0.7)

    handles = legend_handles(methods)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=min(4, len(handles)),
        fontsize=8.2,
        columnspacing=1.2,
        handlelength=2.1,
    )
    bottom = 0.27 if n_rows == 1 else 0.16
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=bottom, wspace=0.20, hspace=0.40)
    save_figure(fig, fig_dir, f"{name_prefix}_reward_spend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary_dir",
        action="append",
        default=[],
        help="Summary directory containing round_eval_raw_all_methods.csv. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for figures. Defaults to <first_summary_dir>/paper_figures.",
    )
    parser.add_argument(
        "--name_prefix",
        default="fig_synthetic_main",
        help="Output filename prefix. Use fig_synthetic_t6_t12 to replace the manuscript Fig. 1-3 files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    summary_dirs = summary_dirs_from_args(args)
    raw = load_all_raw(summary_dirs)
    summary = mean_summary(raw)

    fig_dir = Path(args.out_dir) if args.out_dir else summary_dirs[0] / "paper_figures"
    source_dir = fig_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(source_dir / "synthetic_main_round_eval_raw.csv", index=False)
    summary.to_csv(source_dir / "synthetic_main_round_eval_summary.csv", index=False)

    plot_round_boxplots(raw, fig_dir, args.name_prefix)
    plot_round_curve(summary, fig_dir, args.name_prefix)
    plot_reward_spend(summary, fig_dir, args.name_prefix)
    print(f"[done] wrote main-paper figures to {fig_dir}")


if __name__ == "__main__":
    main()
