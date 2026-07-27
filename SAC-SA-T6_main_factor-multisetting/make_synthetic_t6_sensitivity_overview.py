"""Create a compact overview of the Synthetic T=6 sensitivity analyses."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from make_synthetic_t6_ablation_paper_figures import (
    COLORS,
    MARKERS,
    METHOD_LABELS,
    TOKENS,
    _format_facet_value,
    configure_matplotlib,
    load_raw,
    mean_summary,
    save_figure,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "paper_figures_synT6_sensitivity_overview"

SPECS = [
    ("d0", "summary_synT6_abD0fix", ["800", "1000"], r"Initial log size $n_0$", r"$n_0$"),
    (
        "newdays",
        "summary_synT6_abNfix",
        ["30", "50", "100"],
        r"New true-DGP days $n_q^{\mathrm{new}}$",
        r"$n_q^{\mathrm{new}}$",
    ),
    ("m", "summary_synT6_abMfix", ["3", "6", "9"], r"Number of cities $M$", r"$M$"),
    ("seed", "summary_synT6_abSeedfix", ["2026", "2027", "2028"], "Run seed", "seed"),
]

LEARNING_METHODS = ["sacsa_reserve", "dfl_pto"]
LINESTYLES = ["-", "--", ":"]
LEGEND_KWARGS = {
    "d0": {"loc": "lower right"},
    "newdays": {"loc": "lower right"},
    "m": {"loc": "center right", "bbox_to_anchor": (0.98, 0.67)},
    "seed": {"loc": "lower right"},
}


def _sensitivity_summary(facet: str, summary_dir: str, keep_values: list[str]) -> pd.DataFrame:
    raw, _, _ = load_raw(ROOT / summary_dir, facet)
    keep = {_format_facet_value(value) for value in keep_values}
    raw = raw[raw["facet_value"].isin(keep)].copy()
    if raw.empty:
        raise SystemExit(f"No rows left for {facet}: {keep_values}")
    summary = mean_summary(raw)
    summary["ablation_facet"] = facet
    return summary


def main() -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.7), sharex=True)
    all_summary = []

    for ax, (facet, summary_dir, keep_values, panel_title, value_prefix) in zip(axes.ravel(), SPECS):
        summary = _sensitivity_summary(facet, summary_dir, keep_values)
        all_summary.append(summary)
        for method in LEARNING_METHODS:
            for value, linestyle in zip(keep_values, LINESTYLES):
                subset = summary[
                    (summary["method_key"] == method)
                    & (summary["facet_value"] == _format_facet_value(value))
                ].sort_values("round")
                if subset.empty:
                    continue
                ax.errorbar(
                    subset["round"],
                    subset["net_reward_mean"],
                    yerr=subset["net_reward_se"],
                    color=COLORS[method],
                    marker=MARKERS[method],
                    markerfacecolor=COLORS[method] if method == "sacsa_reserve" else TOKENS["panel"],
                    markeredgecolor=COLORS[method],
                    linestyle=linestyle,
                    linewidth=1.75 if method == "sacsa_reserve" else 1.2,
                    markersize=5.1,
                    capsize=1.8,
                    alpha=0.98,
                )
        value_handles = [
            Line2D([0], [0], color=TOKENS["muted"], linestyle=style, linewidth=1.25, label=f"{value_prefix}={value}")
            for value, style in zip(keep_values, LINESTYLES)
        ]
        ax.legend(handles=value_handles, fontsize=7.2, **LEGEND_KWARGS[facet])
        ax.set_title(panel_title, fontsize=9.4, weight="bold", color=TOKENS["ink"])
        ax.set_xlabel("Outer round q", fontsize=8.6)
        ax.set_ylabel("Mean net reward", fontsize=8.6)
        ax.grid(color=TOKENS["grid"], linewidth=0.65)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    method_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            markerfacecolor=COLORS[method] if method == "sacsa_reserve" else TOKENS["panel"],
            markeredgecolor=COLORS[method],
            linewidth=1.8 if method == "sacsa_reserve" else 1.2,
            label=METHOD_LABELS[method],
        )
        for method in LEARNING_METHODS
    ]
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=2,
        fontsize=8.2,
        columnspacing=1.2,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.13, hspace=0.36, wspace=0.24)

    source_dir = OUT_DIR / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_summary, ignore_index=True).to_csv(
        source_dir / "fig_synT6_sensitivity_overview_summary.csv", index=False
    )
    save_figure(fig, OUT_DIR, "fig_synT6_sensitivity_overview")
    print(f"[done] wrote overview figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
