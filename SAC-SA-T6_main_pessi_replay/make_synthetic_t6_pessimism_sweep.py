"""Summarize the pre-specified synthetic T=6 pessimism-strength sweep."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
LAMBDA_SPECS = [
    ("0p005", 0.005),
    ("0p02", 0.02),
    ("0p05", 0.05),
    ("0p10", 0.10),
]
VARIANTS = [
    ("full", "Full SAC-SA"),
    ("noreplay", "No real replay"),
]
FULL_COLOR = "#3F73C8"
NO_REPLAY_COLOR = "#78A3D8"
NO_PESS_COLOR = "#D55E00"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for lambda_tag, lambda_value in LAMBDA_SPECS:
        for variant, label in VARIANTS:
            pattern = f"results_synT6_comp_lp{lambda_tag}_{variant}_*sacsa_reserve"
            result_dirs = sorted(ROOT.glob(pattern))
            result_dirs = [
                path for path in result_dirs if (path / "round_eval_raw.csv").exists()
            ]
            if not result_dirs:
                raise FileNotFoundError(f"No completed results match {pattern}")
            for result_dir in result_dirs:
                with (result_dir / "config.json").open() as handle:
                    config = json.load(handle)
                training_seed = int(config["args"]["seed"])
                for row in read_csv(result_dir / "round_eval_raw.csv"):
                    if row.get("method_key") != "sacsa_reserve":
                        continue
                    rows.append(
                        {
                            **row,
                            "lambda_pess_setting": lambda_value,
                            "effective_lambda_pess": (
                                0.0 if variant == "nopess" else lambda_value
                            ),
                            "lambda_tag": lambda_tag,
                            "variant": variant,
                            "variant_label": label,
                            "training_seed": training_seed,
                            "result_dir": result_dir.name,
                        }
                    )
    no_pess_pattern = "results_synT6_comp_nopess_*sacsa_reserve"
    no_pess_dirs = [
        path
        for path in sorted(ROOT.glob(no_pess_pattern))
        if (path / "round_eval_raw.csv").exists()
    ]
    if not no_pess_dirs:
        raise FileNotFoundError(f"No completed results match {no_pess_pattern}")
    for result_dir in no_pess_dirs:
        with (result_dir / "config.json").open() as handle:
            config = json.load(handle)
        training_seed = int(config["args"]["seed"])
        for row in read_csv(result_dir / "round_eval_raw.csv"):
            if row.get("method_key") != "sacsa_reserve":
                continue
            rows.append(
                {
                    **row,
                    "lambda_pess_setting": 0.0,
                    "effective_lambda_pess": 0.0,
                    "lambda_tag": "0",
                    "variant": "nopess",
                    "variant_label": "No pessimism",
                    "training_seed": training_seed,
                    "result_dir": result_dir.name,
                }
            )
    return rows


def mean_summary(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[
        Tuple[float, str, int], Dict[int, List[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (
            float(row["lambda_pess_setting"]),
            str(row["variant"]),
            int(row["round"]),
        )
        groups[key][int(row["training_seed"])].append(float(row["net_reward"]))
    out: List[Dict[str, object]] = []
    for (lambda_value, variant, round_idx), seed_values in sorted(groups.items()):
        values = np.asarray(
            [value for current in seed_values.values() for value in current],
            dtype=np.float64,
        )
        seed_means = np.asarray(
            [np.mean(current) for current in seed_values.values()],
            dtype=np.float64,
        )
        out.append(
            {
                "lambda_pess_setting": lambda_value,
                "variant": variant,
                "round": round_idx,
                "episode_rows": len(values),
                "mean_net_reward": float(values.mean()),
                "episode_se": float(values.std(ddof=1) / np.sqrt(len(values))),
                "training_seed_count": len(seed_means),
                "training_seed_sd": (
                    float(seed_means.std(ddof=1)) if len(seed_means) > 1 else ""
                ),
            }
        )
    return out


def paired_gaps(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    values: Dict[
        Tuple[float, str, int], Dict[Tuple[int, int], float]
    ] = defaultdict(dict)
    for row in rows:
        key = (
            float(row["lambda_pess_setting"]),
            str(row["variant"]),
            int(row["round"]),
        )
        pair = (int(row["training_seed"]), int(row["eval_seed"]))
        values[key][pair] = float(row["net_reward"])

    out: List[Dict[str, object]] = []
    for _, lambda_value in LAMBDA_SPECS:
        for comparator in ("nopess", "noreplay"):
            for round_idx in (1, 2, 3):
                reference = values[(lambda_value, "full", round_idx)]
                comparison_lambda = 0.0 if comparator == "nopess" else lambda_value
                comparison = values[(comparison_lambda, comparator, round_idx)]
                common = sorted(set(reference) & set(comparison))
                if not common:
                    raise ValueError(
                        f"No paired rows for lambda={lambda_value}, "
                        f"comparator={comparator}, round={round_idx}"
                    )
                gaps_by_training_seed: Dict[int, List[float]] = defaultdict(list)
                for pair in common:
                    training_seed, _eval_seed = pair
                    gaps_by_training_seed[training_seed].append(
                        reference[pair] - comparison[pair]
                    )
                seed_means = np.asarray(
                    [
                        np.mean(seed_gaps)
                        for seed_gaps in gaps_by_training_seed.values()
                    ],
                    dtype=np.float64,
                )
                seed_mc_variances = [
                    np.var(seed_gaps, ddof=1) / len(seed_gaps)
                    for seed_gaps in gaps_by_training_seed.values()
                    if len(seed_gaps) > 1
                ]
                mean = float(seed_means.mean())
                se = float(
                    np.sqrt(np.sum(seed_mc_variances))
                    / len(gaps_by_training_seed)
                )
                out.append(
                    {
                        "lambda_pess_setting": lambda_value,
                        "round": round_idx,
                        "contrast": f"full_minus_{comparator}",
                        "paired_episode_rows": len(common),
                        "training_seed_count": len(seed_means),
                        "training_seed_sd": (
                            float(seed_means.std(ddof=1))
                            if len(seed_means) > 1
                            else ""
                        ),
                        "mean_gap": mean,
                        "paired_se": se,
                        "ci95_low": mean - 1.96 * se,
                        "ci95_high": mean + 1.96 * se,
                    }
                )
    return out


def plot(
    summary: Sequence[Dict[str, object]],
    gaps: Sequence[Dict[str, object]],
    out_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    no_pess = [
        row
        for row in summary
        if float(row["lambda_pess_setting"]) == 0.0
        and row["variant"] == "nopess"
    ]
    no_pess.sort(key=lambda row: int(row["round"]))
    no_pess_x = [int(row["round"]) for row in no_pess]
    no_pess_y = [float(row["mean_net_reward"]) for row in no_pess]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.6),
        sharex=True,
        sharey=True,
    )
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for axis, panel_label, (_, lambda_value) in zip(
        axes.flat,
        panel_labels,
        LAMBDA_SPECS,
    ):
        for variant, color, label, linestyle in (
            ("full", FULL_COLOR, "Full SAC-SA", "-"),
            ("noreplay", NO_REPLAY_COLOR, "No real replay", "-"),
        ):
            subset = [
                row
                for row in summary
                if float(row["lambda_pess_setting"]) == lambda_value
                and row["variant"] == variant
            ]
            subset.sort(key=lambda row: int(row["round"]))
            axis.plot(
                [int(row["round"]) for row in subset],
                [float(row["mean_net_reward"]) for row in subset],
                color=color,
                linestyle=linestyle,
                linewidth=1.45,
                label=label,
            )
        axis.plot(
            no_pess_x,
            no_pess_y,
            color=NO_PESS_COLOR,
            linestyle="--",
            linewidth=1.25,
            label="No pessimism",
        )
        axis.set_xticks((1, 2, 3))
        axis.set_title(
            rf"{panel_label} $\lambda_{{\mathrm{{pess}}}}={lambda_value:g}$",
            fontweight="bold",
        )
        axis.tick_params(axis="both", labelsize=7)

    for axis in axes[:, 0]:
        axis.set_ylabel("Mean net reward", fontweight="bold")
    for axis in axes[1, :]:
        axis.set_xlabel("Training round", fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        fontsize=7.0,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for stem in (
        "synthetic_t6_pessimism_sweep",
        "synthetic_t6_component_ablations",
    ):
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(
            out_dir / f"{stem}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    summary = mean_summary(rows)
    gaps = paired_gaps(rows)
    out_dir = ROOT / "paper_figures_synT6_pess_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "source_round_eval.csv", rows)
    write_csv(out_dir / "pessimism_round_summary.csv", summary)
    write_csv(out_dir / "component_paired_gaps.csv", gaps)
    plot(summary, gaps, out_dir)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
