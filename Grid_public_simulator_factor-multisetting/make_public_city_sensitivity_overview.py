"""Create the combined public-grid single-factor sensitivity figure.

Run this script after all five sweep wrappers have completed. It reads only
result directories in this experiment folder and has no dependency on files
outside the anonymous artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from make_public_city_ablation_paper_figures import load_sweep, summarize
from make_public_city_panel_main_paper_figures import (
    COLORS,
    MARKERS,
    METHOD_LABELS,
    METHOD_ORDER,
    TOKENS,
    configure_matplotlib,
    save_figure,
)


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Sweep:
    facet: str
    pattern: str
    titles: dict[str, str]


SWEEPS = (
    Sweep(
        "d0",
        "results_pc_sensitivity_base_*,results_pc_sensitivity_d0_*",
        {"3500": r"$n_0=3500$", "4500": r"$n_0=4500$"},
    ),
    Sweep(
        "newdays",
        "results_pc_sensitivity_base_*,results_pc_sensitivity_newdays_*",
        {"250": r"$n_q^{\mathrm{new}}=250$", "300": r"$n_q^{\mathrm{new}}=300$"},
    ),
    Sweep(
        "budget",
        "results_pc_sensitivity_base_*,results_pc_sensitivity_budget_*",
        {"5": r"$B=5$", "10": r"$B=10$"},
    ),
    Sweep(
        "m",
        "results_pc_sensitivity_base_*,results_pc_sensitivity_m_*",
        {"3": r"$M=3$", "6": r"$M=6$"},
    ),
    Sweep(
        "seed",
        "results_pc_sensitivity_base_*,results_pc_sensitivity_seed_*",
        {"2026": "Run seed 2026", "2028": "Run seed 2028"},
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=HERE / "paper_figures_pc_sensitivity_overview",
    )
    parser.add_argument("--name", default="fig_pc_sensitivity_overview")
    return parser.parse_args()


def ordered_values(values: pd.Series) -> list[str]:
    unique = {str(value) for value in values}

    def sort_key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return sorted(unique, key=sort_key)


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    summaries: list[pd.DataFrame] = []
    manifests: list[pd.DataFrame] = []
    for sweep in SWEEPS:
        raw, manifest = load_sweep(args.root, sweep.pattern, sweep.facet)
        current = summarize(raw)
        current["factor"] = sweep.facet
        summaries.append(current)
        manifest["factor"] = sweep.facet
        manifests.append(manifest)

    combined = pd.concat(summaries, ignore_index=True)
    manifest = pd.concat(manifests, ignore_index=True).drop_duplicates()

    figure, axes = plt.subplots(
        len(SWEEPS),
        2,
        figsize=(7.15, 8.2),
        squeeze=False,
    )
    panel_letters = iter("abcdefghij")
    for row, sweep in enumerate(SWEEPS):
        factor_rows = combined[combined["factor"] == sweep.facet]
        values = ordered_values(factor_rows["facet_value"])
        if len(values) != 2:
            raise ValueError(
                f"Expected two values for {sweep.facet}, found {values}"
            )
        for column, value in enumerate(values):
            axis = axes[row, column]
            setting = factor_rows[factor_rows["facet_value"].astype(str) == value]
            for method in METHOD_ORDER:
                rows = setting[setting["method"] == method].sort_values("round")
                if rows.empty:
                    continue
                axis.plot(
                    rows["round"],
                    rows["net_reward_mean"],
                    color=COLORS[method],
                    marker=MARKERS[method],
                    linewidth=1.55 if method == "sacsa_reserve" else 1.15,
                    markersize=4.6,
                    markerfacecolor=COLORS[method],
                    markeredgecolor=COLORS[method],
                )
            title = sweep.titles.get(value, setting["facet_label"].iloc[0])
            axis.set_title(
                f"({next(panel_letters)}) {title}",
                fontsize=8.8,
                fontweight="bold",
                pad=3,
            )
            axis.set_xticks((1, 2, 3))
            axis.set_xticklabels(("1", "2", "3"), fontweight="bold")
            if column == 0:
                axis.set_ylabel("Mean net reward", fontsize=8.2, fontweight="bold")
            axis.grid(False)
            axis.tick_params(axis="both", labelsize=7.2, pad=1.5)
            for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
                label.set_fontweight("bold")
            axis.spines["left"].set_color(TOKENS["axis"])
            axis.spines["bottom"].set_color(TOKENS["axis"])

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.3,
            markersize=4.5,
            label=METHOD_LABELS[method]
            .replace("Proposed Beta-Dirichlet SAC-SA", "SAC-SA")
            .replace(" Pacing", ""),
        )
        for method in METHOD_ORDER
    ]
    legend = figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=6,
        columnspacing=0.62,
        handlelength=1.25,
        handletextpad=0.25,
        borderaxespad=0,
        prop={"size": 7.2, "weight": "bold"},
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    figure.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.99,
        bottom=0.06,
        wspace=0.22,
        hspace=0.48,
    )
    output_base = args.out_dir / args.name
    save_figure(figure, [output_base])

    source_dir = args.out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        source_dir / f"{args.name}_round_summary.csv",
        index=False,
    )
    manifest.to_csv(
        source_dir / f"{args.name}_manifest.csv",
        index=False,
    )
    print(f"[done] wrote {output_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
