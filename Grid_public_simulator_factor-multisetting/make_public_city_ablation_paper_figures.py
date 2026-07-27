"""Aggregate and plot one-factor public-city-panel ablation sweeps."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from make_public_city_panel_main_paper_figures import (
    COLORS,
    MARKERS,
    METHOD_LABELS,
    METHOD_ORDER,
    TOKENS,
    configure_matplotlib,
    save_figure,
)
import matplotlib.pyplot as plt


LEARNING_METHODS = {"sacsa_reserve", "dfl_pto"}
OPERATIONAL_METHODS = {"gap_pacing", "uniform_pacing", "random_pacing", "no_subsidy"}
FACETS = {
    "budget": ("config_budget", r"$B$"),
    "seed": ("config_seed", "seed"),
    "d0": ("config_d0", r"$n_0$"),
    "newdays": ("config_newdays", r"$n_q^{\mathrm{new}}$"),
    "m": ("config_m", r"$M$"),
}


def _sem(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1) / math.sqrt(len(numeric)))


def _method_order(keys: Iterable[str]) -> list[str]:
    present = {str(key) for key in keys}
    return [method for method in METHOD_ORDER if method in present]


def _format_value(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(number):
        number = float(number)
        return str(int(number)) if number.is_integer() else f"{number:g}"
    return str(value)


def _config_values(config: dict) -> dict[str, object]:
    experiment = config["experiment"]
    return {
        "config_m": int(experiment["M"]),
        "config_t": int(experiment["T"]),
        "config_budget": float(experiment["daily_budget"]),
        "config_d0": int(experiment["initial_logged_days"]),
        "config_newdays": int(experiment["new_logged_days"]),
        "config_seed": int(experiment["seed"]),
        "config_q_max": int(experiment["q_max"]),
    }


def load_sweep(root: Path, pattern: str, facet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    patterns = [value.strip() for value in pattern.split(",") if value.strip()]
    result_dirs = sorted(
        {
            path
            for current_pattern in patterns
            for path in root.glob(current_pattern)
            if path.is_dir()
        }
    )
    raw_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for directory in result_dirs:
        config_path = directory / "config.json"
        raw_path = directory / "final_eval_raw.csv"
        if not config_path.exists() or not raw_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_values = _config_values(config)
        source_method = str(config["method"])
        frame = pd.read_csv(raw_path)
        frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
        for key, value in config_values.items():
            frame[key] = value
        frame["source_method"] = source_method
        frame["source_dir"] = directory.name
        frame["setting_id"] = "|".join(
            str(config_values[key])
            for key in ("config_m", "config_budget", "config_d0", "config_newdays", "config_seed")
        )
        raw_frames.append(frame)
        manifest_rows.append(
            {"source_dir": directory.name, "source_method": source_method, **config_values}
        )
    if not raw_frames:
        raise SystemExit(f"No complete result directories match patterns {patterns}")

    raw = pd.concat(raw_frames, ignore_index=True)
    learned = raw[
        raw["method"].isin(LEARNING_METHODS)
        & raw["method"].eq(raw["source_method"])
        & raw["round"].gt(0)
    ].copy()
    operational = (
        raw[raw["method"].isin(OPERATIONAL_METHODS)]
        .sort_values("source_dir")
        .drop_duplicates(["setting_id", "method", "episode"], keep="first")
    )
    q_by_setting = learned.groupby("setting_id")["round"].max().astype(int).to_dict()
    replicated: list[pd.DataFrame] = []
    for setting_id, rows in operational.groupby("setting_id", sort=False):
        for q in range(1, q_by_setting[setting_id] + 1):
            copy = rows.copy()
            copy["round"] = q
            replicated.append(copy)
    raw = pd.concat([learned, *replicated], ignore_index=True)

    facet_column, facet_symbol = FACETS[facet]
    raw["facet_value"] = raw[facet_column].map(_format_value)
    raw["facet_label"] = raw["facet_value"].map(lambda value: f"{facet_symbol}={value}")
    raw["method_label"] = raw["method"].map(METHOD_LABELS)
    raw["round"] = pd.to_numeric(raw["round"], errors="raise").astype(int)
    for column in ("net_reward", "response_score", "spend", "final_budget"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    return raw, pd.DataFrame(manifest_rows).drop_duplicates()


def _facet_order(raw: pd.DataFrame) -> list[str]:
    def key(value: str) -> tuple[int, float | str]:
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return (0, float(number)) if pd.notna(number) else (1, str(value))

    return sorted(raw["facet_value"].unique().tolist(), key=key)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["facet_value", "facet_label", "round", "method", "method_label"], as_index=False)
        .agg(
            net_reward_mean=("net_reward", "mean"),
            net_reward_se=("net_reward", _sem),
            response_score_mean=("response_score", "mean"),
            response_score_se=("response_score", _sem),
            spend_mean=("spend", "mean"),
            spend_se=("spend", _sem),
            final_budget_mean=("final_budget", "mean"),
            eval_episodes=("net_reward", "size"),
        )
    )


def plot_boxplots(raw: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    facets = _facet_order(raw)
    methods = _method_order(raw["method"].unique())
    final_round = int(raw["round"].max())
    positions = list(range(len(methods), 0, -1))
    figure, axes = plt.subplots(
        1, len(facets), figsize=(5.4 * len(facets), 5.4), sharey=True, squeeze=False
    )
    for column, facet_value in enumerate(facets):
        axis = axes[0, column]
        subset = raw[(raw["facet_value"] == facet_value) & (raw["round"] == final_round)]
        for y, method in zip(positions, methods):
            values = subset.loc[subset["method"] == method, "net_reward"].dropna().to_numpy()
            if values.size == 0:
                continue
            axis.boxplot(
                [values], vert=False, positions=[y], widths=0.56, patch_artist=True,
                boxprops={"facecolor": COLORS[method] + "4D", "edgecolor": COLORS[method], "linewidth": 1.05},
                medianprops={"color": TOKENS["ink"], "linewidth": 1.15},
                whiskerprops={"color": TOKENS["muted"], "linewidth": 0.9},
                capprops={"color": TOKENS["muted"], "linewidth": 0.9},
                flierprops={"marker": ".", "markersize": 2, "markerfacecolor": COLORS[method],
                            "markeredgecolor": COLORS[method], "alpha": 0.13},
            )
            axis.scatter(float(values.mean()), y, marker="D", s=22, color=COLORS[method],
                         edgecolor=TOKENS["panel"], linewidth=0.45, zorder=4)
        axis.set_title(f"{subset['facet_label'].iloc[0]}, q={final_round}", fontsize=10.5, fontweight="semibold")
        axis.set_xlabel("Net reward")
        axis.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
        axis.set_yticks(positions)
        if column == 0:
            axis.set_yticklabels([
                METHOD_LABELS[method].replace("Proposed Beta-Dirichlet SAC-SA", "Proposed\nBeta-Dirichlet SAC-SA")
                for method in methods
            ])
        else:
            axis.tick_params(axis="y", labelleft=False)
    figure.subplots_adjust(left=0.24, right=0.985, bottom=0.14, top=0.89, wspace=0.12)
    save_figure(figure, [out_dir / f"{prefix}_final_boxplot"])


def plot_round_curve(summary: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    facets = _facet_order(summary)
    methods = _method_order(summary["method"].unique())
    figure, axes = plt.subplots(1, len(facets), figsize=(5.3 * len(facets), 4.8), squeeze=False)
    for column, facet_value in enumerate(facets):
        axis = axes[0, column]
        subset = summary[summary["facet_value"] == facet_value]
        for method in methods:
            rows = subset[subset["method"] == method].sort_values("round")
            if rows.empty:
                continue
            proposed = method == "sacsa_reserve"
            axis.plot(
                rows["round"], rows["net_reward_mean"], marker=MARKERS[method],
                markersize=6.5 if proposed else 5.2, linewidth=2.0 if proposed else 1.25,
                color=COLORS[method], markerfacecolor=COLORS[method] if proposed else TOKENS["panel"],
                markeredgecolor=COLORS[method], label=METHOD_LABELS[method],
            )
        axis.set_title(subset["facet_label"].iloc[0], fontsize=10.5, fontweight="semibold")
        axis.set_xlabel("Outer round q")
        if column == 0:
            axis.set_ylabel("Mean net reward")
        axis.set_xticks(sorted(subset["round"].unique()))
        axis.grid(color=TOKENS["grid"], linewidth=0.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8.5)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.25, top=0.88, wspace=0.22)
    save_figure(figure, [out_dir / f"{prefix}_round_curve"])


def plot_reward_spend(summary: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    facets = _facet_order(summary)
    methods = _method_order(summary["method"].unique())
    final_round = int(summary["round"].max())
    figure, axes = plt.subplots(1, len(facets), figsize=(5.3 * len(facets), 4.8), squeeze=False)
    for column, facet_value in enumerate(facets):
        axis = axes[0, column]
        subset = summary[(summary["facet_value"] == facet_value) & (summary["round"] == final_round)]
        for method in methods:
            rows = subset[subset["method"] == method]
            if rows.empty:
                continue
            row = rows.iloc[0]
            proposed = method == "sacsa_reserve"
            axis.errorbar(
                row["spend_mean"], row["net_reward_mean"], xerr=row["spend_se"], yerr=row["net_reward_se"],
                marker=MARKERS[method], markersize=6.5 if proposed else 5.2, color=COLORS[method],
                markerfacecolor=COLORS[method] if proposed else TOKENS["panel"],
                markeredgecolor=COLORS[method], elinewidth=0.9, capsize=2, linestyle="none",
                label=METHOD_LABELS[method],
            )
        axis.set_title(f"{subset['facet_label'].iloc[0]}, q={final_round}", fontsize=10.5, fontweight="semibold")
        axis.set_xlabel("Mean spend")
        if column == 0:
            axis.set_ylabel("Mean net reward")
        axis.grid(color=TOKENS["grid"], linewidth=0.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8.5)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.25, top=0.88, wspace=0.22)
    save_figure(figure, [out_dir / f"{prefix}_reward_spend"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--facet", choices=sorted(FACETS), required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--name_prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    raw, manifest = load_sweep(args.root, args.pattern, args.facet)
    summary = summarize(raw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(source_dir / f"{args.name_prefix}_manifest.csv", index=False)
    raw.to_csv(source_dir / f"{args.name_prefix}_round_eval_raw.csv", index=False)
    summary.to_csv(source_dir / f"{args.name_prefix}_round_eval_summary.csv", index=False)
    plot_boxplots(raw, args.out_dir, args.name_prefix)
    plot_round_curve(summary, args.out_dir, args.name_prefix)
    plot_reward_spend(summary, args.out_dir, args.name_prefix)
    print(f"[done] wrote public-city ablation figures to {args.out_dir}")


if __name__ == "__main__":
    main()
