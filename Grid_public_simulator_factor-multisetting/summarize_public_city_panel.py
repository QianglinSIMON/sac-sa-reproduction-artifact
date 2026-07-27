"""Combine method runs and create compact paper-style public-city figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "sacsa_reserve": "Proposed SAC-SA",
    "dfl_pto": "DFL-PTO",
    "no_subsidy": "No Subsidy",
    "uniform_pacing": "Uniform Pacing",
    "gap_pacing": "Demand-Supply Gap",
    "random_pacing": "Random Pacing",
}
ORDER = list(LABELS)
COLORS = {
    "sacsa_reserve": "#c23b52",
    "dfl_pto": "#3b6fb6",
    "no_subsidy": "#737b87",
    "uniform_pacing": "#4c9f70",
    "gap_pacing": "#d58b2a",
    "random_pacing": "#9b6bb3",
}


def load_results(result_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    raw = []
    for directory in result_dirs:
        summary_path = directory / "final_eval_summary.csv"
        raw_path = directory / "final_eval_raw.csv"
        if summary_path.exists():
            frame = pd.read_csv(summary_path)
            frame["source_dir"] = directory.name
            summaries.append(frame)
        if raw_path.exists():
            frame = pd.read_csv(raw_path)
            frame["source_dir"] = directory.name
            raw.append(frame)
    if not summaries or not raw:
        raise FileNotFoundError("No complete final_eval_summary.csv/final_eval_raw.csv pairs found.")
    summary = pd.concat(summaries, ignore_index=True)
    raw_frame = pd.concat(raw, ignore_index=True)
    # Operational baselines are repeated in each learned-method run.
    summary = summary.sort_values("source_dir").drop_duplicates(["method", "round"], keep="first")
    raw_frame = raw_frame.sort_values("source_dir").drop_duplicates(
        ["method", "round", "episode"], keep="first"
    )
    return summary, raw_frame


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#dfe3eb", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def make_figures(summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    learned = summary[summary["round"] > 0]
    final_round = int(learned["round"].max())
    final_raw = raw[(raw["round"] == 0) | (raw["round"] == final_round)].copy()
    methods = [method for method in ORDER if method in set(final_raw["method"])]

    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    values = [final_raw.loc[final_raw.method == method, "net_reward"].to_numpy() for method in methods]
    box = axis.boxplot(values, patch_artist=True, showfliers=False, widths=0.58)
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.65)
    axis.set_xticks(range(1, len(methods) + 1), [LABELS[m] for m in methods], rotation=18, ha="right")
    axis.set_ylabel("Episode net reward")
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(out_dir / "public_city_final_boxplot.png", dpi=300)
    figure.savefig(out_dir / "public_city_final_boxplot.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for method in ("sacsa_reserve", "dfl_pto"):
        rows = summary[(summary.method == method) & (summary["round"] > 0)].sort_values("round")
        if rows.empty:
            continue
        axis.errorbar(
            rows["round"], rows["net_reward"], yerr=rows.get("net_reward_se"),
            marker="o", linewidth=2.0, capsize=3, color=COLORS[method], label=LABELS[method]
        )
    for method in ("uniform_pacing", "gap_pacing", "random_pacing", "no_subsidy"):
        rows = summary[summary.method == method]
        if rows.empty:
            continue
        axis.axhline(float(rows.iloc[0].net_reward), color=COLORS[method], linestyle="--", linewidth=1.3, label=LABELS[method])
    axis.set_xlabel("Interaction round $q$")
    axis.set_ylabel("Mean true-DGP net reward")
    axis.set_xticks(sorted(learned["round"].unique()))
    style_axis(axis)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    figure.tight_layout()
    figure.savefig(out_dir / "public_city_round_curve.png", dpi=300, bbox_inches="tight")
    figure.savefig(out_dir / "public_city_round_curve.pdf", bbox_inches="tight")
    plt.close(figure)

    final_summary = summary[(summary["round"] == 0) | (summary["round"] == final_round)]
    figure, axis = plt.subplots(figsize=(6.5, 4.6))
    for method in methods:
        rows = final_summary[final_summary.method == method]
        if rows.empty:
            continue
        row = rows.iloc[-1]
        axis.errorbar(
            row.spend, row.net_reward, yerr=row.get("net_reward_se", 0.0),
            marker="o", markersize=7, capsize=3, color=COLORS[method], label=LABELS[method]
        )
    axis.set_xlabel("Mean subsidy spend")
    axis.set_ylabel("Mean net reward")
    style_axis(axis)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(out_dir / "public_city_reward_spend.png", dpi=300)
    figure.savefig(out_dir / "public_city_reward_spend.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    summary, raw = load_results(args.result_dirs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "combined_summary.csv", index=False)
    raw.to_csv(args.out_dir / "combined_raw.csv", index=False)
    make_figures(summary, raw, args.out_dir)
    marker = args.out_dir / "summary_complete.json"
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"status": "complete", "result_dir_count": len(args.result_dirs)}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(marker)


if __name__ == "__main__":
    main()
