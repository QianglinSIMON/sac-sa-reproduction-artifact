"""Create the synthetic T=6 actor-comparison figure and source tables."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ACTORS = [
    ("bd", "sacsa_reserve", "Beta-Dirichlet"),
    ("bln", "sac_logistic_normal_reserve", "Beta-Logistic-Normal"),
    ("ddir", "sacsa", "Direct Dirichlet"),
    ("dln", "sac_logistic_normal", "Direct Logistic-Normal"),
    ("gproj", "sac_gaussian_projection", "Gaussian Projection"),
]
OPERATIONAL_KEYS = {
    "no_subsidy",
    "random_pacing",
    "uniform_pacing",
    "gap_heuristic",
}
COLORS = ["#3F73C8", "#009E73", "#7E57C2", "#00A6D6", "#8C564B"]
MARKERS = ["D", "o", "s", "v", "^"]
TICK_LABELS = [
    "Beta-\nDir.",
    "Beta-\nLN",
    "Direct\nDir.",
    "Direct\nLN",
    "Gauss.\nProj.",
]


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.edgecolor": "#30343B",
            "axes.labelcolor": "#20242B",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": "#20242B",
            "ytick.color": "#20242B",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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


def is_complete(result_dir: Path) -> bool:
    event_path = result_dir / "run_events.jsonl"
    if not event_path.exists():
        return False
    lines = [line for line in event_path.read_text().splitlines() if line.strip()]
    return bool(lines and json.loads(lines[-1]).get("event") == "run_complete")


def load_rows() -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    labels: List[str] = []
    operational_reference: Dict[Tuple[str, int, int], float] = {}

    for actor_index, (tag, method_key, label) in enumerate(ACTORS):
        result_dir = ROOT / f"results_synT6_actor_{tag}_{method_key}"
        raw_path = result_dir / "round_eval_raw.csv"
        config_path = result_dir / "config.json"
        if not raw_path.exists() or not config_path.exists() or not is_complete(result_dir):
            raise FileNotFoundError(f"Missing completed actor result: {result_dir}")

        config = json.loads(config_path.read_text())["args"]
        required = {
            "M": 6,
            "T": 6,
            "daily_budget": 10.0,
            "q_max": 3,
            "initial_logged_days": 800,
            "new_logged_days": 100,
            "eval_episodes": 2000,
            "seed": 2026,
        }
        for key, expected in required.items():
            if float(config[key]) != float(expected):
                raise ValueError(
                    f"{result_dir.name}: {key}={config[key]} but expected {expected}"
                )

        source_rows = read_csv(raw_path)
        learned = [row for row in source_rows if row.get("method_key") == method_key]
        for round_idx in (1, 2, 3):
            count = sum(int(row["round"]) == round_idx for row in learned)
            if count != 2000:
                raise ValueError(
                    f"{result_dir.name}: round {round_idx} has {count} learned rows"
                )
        labels.append(label)
        for row in learned:
            rows.append(
                {
                    **row,
                    "comparison_key": tag,
                    "comparison_label": label,
                    "result_dir": result_dir.name,
                }
            )

        for row in source_rows:
            method = str(row.get("method_key", ""))
            if method not in OPERATIONAL_KEYS:
                continue
            pair_key = (method, int(row["round"]), int(row["eval_seed"]))
            value = float(row["net_reward"])
            if actor_index == 0:
                operational_reference[pair_key] = value
                rows.append(
                    {
                        **row,
                        "comparison_key": method,
                        "comparison_label": row["method"],
                        "result_dir": result_dir.name,
                    }
                )
            elif not math.isclose(
                operational_reference[pair_key],
                value,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    "Operational baseline differs across actor runs for "
                    f"{pair_key}: {operational_reference[pair_key]} vs {value}"
                )
    return rows, labels


def values(
    rows: Iterable[Dict[str, object]],
    label: str,
    round_idx: int,
) -> np.ndarray:
    result = [
        float(row["net_reward"])
        for row in rows
        if row.get("comparison_label") == label
        and int(row["round"]) == round_idx
    ]
    if not result:
        raise ValueError(f"No values for {label}, round {round_idx}")
    return np.asarray(result, dtype=np.float64)


def summarize(rows: Sequence[Dict[str, object]], labels: Sequence[str]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for label in labels:
        for round_idx in (1, 2, 3):
            x = values(rows, label, round_idx)
            output.append(
                {
                    "actor": label,
                    "round": round_idx,
                    "episodes": len(x),
                    "mean_net_reward": float(x.mean()),
                    "episode_se": float(x.std(ddof=1) / np.sqrt(len(x))),
                    "sd": float(x.std(ddof=1)),
                }
            )
    return output


def paired_contrasts(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    comparisons = [
        ("Beta-Dirichlet", "Beta-Logistic-Normal"),
        ("Beta-Dirichlet", "Direct Dirichlet"),
        ("Beta-Logistic-Normal", "Direct Logistic-Normal"),
        ("Beta-Dirichlet", "Gaussian Projection"),
    ]
    maps: Dict[str, Dict[int, float]] = {}
    for _, _, label in ACTORS:
        maps[label] = {
            int(row["eval_seed"]): float(row["net_reward"])
            for row in rows
            if row.get("comparison_label") == label and int(row["round"]) == 3
        }
    output: List[Dict[str, object]] = []
    for reference, comparator in comparisons:
        common = sorted(set(maps[reference]) & set(maps[comparator]))
        gaps = np.asarray(
            [maps[reference][seed] - maps[comparator][seed] for seed in common],
            dtype=np.float64,
        )
        se = float(gaps.std(ddof=1) / np.sqrt(len(gaps)))
        output.append(
            {
                "reference": reference,
                "comparator": comparator,
                "paired_episodes": len(gaps),
                "mean_gap": float(gaps.mean()),
                "paired_se": se,
                "ci95_low": float(gaps.mean() - 1.96 * se),
                "ci95_high": float(gaps.mean() + 1.96 * se),
            }
        )
    return output


def best_operational(rows: Sequence[Dict[str, object]]) -> Tuple[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    names: Dict[str, str] = {}
    for row in rows:
        key = str(row.get("method_key", ""))
        if key in OPERATIONAL_KEYS and int(row["round"]) == 3:
            grouped[key].append(float(row["net_reward"]))
            names[key] = str(row["method"])
    key = max(grouped, key=lambda item: float(np.mean(grouped[item])))
    return names[key], float(np.mean(grouped[key]))


def style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(1.0)
    axis.tick_params(labelsize=8)


def make_figure(
    rows: Sequence[Dict[str, object]],
    labels: Sequence[str],
    out_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(3.45, 5.75))

    final_values = [values(rows, label, 3) for label in labels]
    boxes = axes[0].boxplot(
        final_values,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
    )
    for index, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(COLORS[index])
        patch.set_edgecolor(COLORS[index])
        patch.set_alpha(0.68)
        for line in boxes["whiskers"][2 * index : 2 * index + 2]:
            line.set_color(COLORS[index])
        for line in boxes["caps"][2 * index : 2 * index + 2]:
            line.set_color(COLORS[index])
    for median in boxes["medians"]:
        median.set_color("#20242B")
        median.set_linewidth(1.3)
    for index, x in enumerate(final_values):
        axes[0].scatter(
            index + 1,
            float(x.mean()),
            marker="D",
            s=22,
            color="white",
            edgecolor=COLORS[index],
            linewidth=1.0,
            zorder=3,
        )
    axes[0].set_xticks(range(1, 6), TICK_LABELS)
    axes[0].set_ylabel("Net reward", fontweight="bold")
    axes[0].set_title("(a) Final round", fontsize=9.5, fontweight="bold")
    style_axis(axes[0])

    for index, label in enumerate(labels):
        means = [float(values(rows, label, round_idx).mean()) for round_idx in (1, 2, 3)]
        axes[1].plot(
            (1, 2, 3),
            means,
            color=COLORS[index],
            marker=MARKERS[index],
            markersize=3.5,
            linewidth=1.5 if index == 0 else 1.2,
            label=label,
        )
    _, operational_mean = best_operational(rows)
    axes[1].axhline(
        operational_mean,
        color="#7B8492",
        linestyle=(0, (4, 2)),
        linewidth=1.2,
        label="Best operational",
    )
    axes[1].set_xticks((1, 2, 3), ("1", "2", "3"))
    axes[1].set_xlabel("Training round", fontweight="bold")
    axes[1].set_ylabel("Mean net reward", fontweight="bold")
    axes[1].set_title("(b) Round trajectories", fontsize=9.5, fontweight="bold")
    axes[1].legend(
        ncol=2,
        fontsize=6.0,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        handlelength=1.8,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    style_axis(axes[1])

    figure.subplots_adjust(
        left=0.19,
        right=0.98,
        top=0.97,
        bottom=0.20,
        hspace=0.47,
    )
    stem = "synthetic_t6_actor_parameterization_column"
    figure.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_plotting()
    rows, labels = load_rows()
    out_dir = ROOT / "paper_figures_synT6_actor"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "source_round_eval.csv", rows)
    write_csv(out_dir / "actor_round_summary.csv", summarize(rows, labels))
    write_csv(out_dir / "actor_factorial_contrasts.csv", paired_contrasts(rows))
    write_csv(
        out_dir / "gaussian_projection_diagnostics.csv",
        [
            row
            for row in rows
            if row.get("comparison_label") == "Gaussian Projection"
        ],
    )
    make_figure(rows, labels, out_dir)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
