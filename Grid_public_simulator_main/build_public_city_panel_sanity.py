"""Validate the public-grid-calibrated city-panel true DGP before RL training."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List

import torch

from public_city_panel_dgp import (
    DYNAMIC_DIM,
    DYNAMIC_NAMES,
    STATIC_DIM,
    PublicCityPanelConfig,
    PublicCityPanelEnv,
    _get_calibration,
    _project_dynamic,
    _seed_to_pool_position,
    baseline_action,
    public_city_subsidy_effects,
)


METHODS = ("no_subsidy", "random_pacing", "uniform_pacing", "gap_pacing")


def _write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _episode(cfg: PublicCityPanelConfig, method: str, seed: int) -> Dict[str, object]:
    env = PublicCityPanelEnv(cfg, seed=seed, noise_split="evaluation")
    env.reset(seed)
    reward_total = 0.0
    response_total = 0.0
    spend_total = 0.0
    effect_total = 0.0
    reward_identity_error = 0.0
    paired_day_index = env.noise_idx
    projection_clip_counts = torch.zeros(DYNAMIC_DIM, dtype=torch.float64)
    projection_denominator = 0
    done = False
    while not done:
        action = baseline_action(env, method)
        _next_state, reward, done, info = env.step(action)
        response = float(info["Z"].sum().item())
        spend = float(info["spend"].item())
        reward_value = float(reward.item())
        reward_total += reward_value
        response_total += response
        spend_total += spend
        effect_total += float(info["subsidy_effect"].sum().item())
        reward_identity_error = max(reward_identity_error, abs(reward_value - (response - spend)))
        if int(info["paired_day_index"]) != paired_day_index:
            raise AssertionError("Initial state and residual path stopped using the same day index.")
        if bool(info["projection_applied"]):
            projection_clip_counts += info["projection_clip_mask"].double().sum(dim=0).cpu()
            projection_denominator += cfg.M
    row = {
        "method": method,
        "seed": seed,
        "net_reward": reward_total,
        "gross_response_score": response_total,
        "subsidy_effect": effect_total,
        "spend": spend_total,
        "final_budget": float(env.H.item()),
        "reward_identity_error": reward_identity_error,
        "budget_violation": float(spend_total > cfg.daily_budget + 1e-4),
        "paired_day_index": paired_day_index,
    }
    for index, name in enumerate(DYNAMIC_NAMES):
        row[f"projection_clip_rate_{name}"] = float(
            projection_clip_counts[index] / max(projection_denominator, 1)
        )
    return row


def _summaries(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    output: List[Dict[str, object]] = []
    for method in METHODS:
        values = grouped[method]
        summary: Dict[str, object] = {"method": method, "episodes": len(values)}
        metrics = [
            "net_reward",
            "gross_response_score",
            "subsidy_effect",
            "spend",
            "final_budget",
            *[f"projection_clip_rate_{name}" for name in DYNAMIC_NAMES],
        ]
        for metric in metrics:
            sample = [float(row[metric]) for row in values]
            summary[f"mean_{metric}"] = mean(sample)
            summary[f"se_{metric}"] = stdev(sample) / len(sample) ** 0.5 if len(sample) > 1 else 0.0
        summary["max_reward_identity_error"] = max(float(row["reward_identity_error"]) for row in values)
        summary["budget_violation_count"] = int(sum(float(row["budget_violation"]) for row in values))
        output.append(summary)
    return output


def _paired_zero_action_error(cfg: PublicCityPanelConfig, seeds: Iterable[int]) -> Dict[str, float]:
    max_dynamic_error = 0.0
    max_response_error = 0.0
    for seed in seeds:
        env = PublicCityPanelEnv(cfg, seed=seed, noise_split="evaluation")
        env.reset(seed)
        day_index = env.noise_idx
        for t in range(cfg.T):
            current = env.dynamic.clone()
            prediction = env.predictor(current, t)
            _next_state, _reward, done, info = env.step(torch.zeros(cfg.M, device=env.device))
            response_reference = (
                prediction[..., 0]
                + env.calibration.residual_z[day_index, t].to(env.device, env.dtype)
            ).clamp_min(0.0)
            max_response_error = max(
                max_response_error,
                float((info["Z"] - response_reference).abs().max().item()),
            )
            if not done:
                dynamic_reference = _project_dynamic(
                    prediction[..., 1:]
                    + env.calibration.residual_next[day_index, t].to(env.device, env.dtype),
                    cfg,
                )
                max_dynamic_error = max(
                    max_dynamic_error,
                    float((env.dynamic - dynamic_reference).abs().max().item()),
                )
    return {
        "paired_zero_action_max_dynamic_error": max_dynamic_error,
        "paired_zero_action_max_response_error": max_response_error,
    }


def _intervention_checks(cfg: PublicCityPanelConfig) -> Dict[str, float]:
    calibration = _get_calibration(cfg)
    dynamic = calibration.init_dynamic[calibration.split_indices["evaluation"][0]].to(dtype=cfg.dtype)
    zero = torch.zeros(cfg.M, dtype=cfg.dtype)
    low = torch.full((cfg.M,), 0.05, dtype=cfg.dtype)
    high = torch.full((cfg.M,), 0.10, dtype=cfg.dtype)
    effect_zero = public_city_subsidy_effects(dynamic, zero, cfg.T // 2, cfg)
    effect_low = public_city_subsidy_effects(dynamic, low, cfg.T // 2, cfg)
    effect_high = public_city_subsidy_effects(dynamic, high, cfg.T // 2, cfg)
    no_match_dynamic = dynamic.clone()
    no_match_dynamic[..., 2] = 0.0
    no_match_effect = public_city_subsidy_effects(no_match_dynamic, high, cfg.T // 2, cfg)
    first_increment = effect_low["z_effect"]
    second_increment = effect_high["z_effect"] - effect_low["z_effect"]
    return {
        "zero_action_max_absolute_effect": float(
            max(effect_zero["z_effect"].abs().max(), effect_zero["next_effect"].abs().max()).item()
        ),
        "positive_response_minimum": float(effect_low["z_effect"].min().item()),
        "diminishing_return_margin_minimum": float((first_increment - second_increment).min().item()),
        "demand_clearance_direction_maximum": float(effect_low["next_effect"][..., 0].max().item()),
        "supply_activation_direction_minimum": float(effect_low["next_effect"][..., 1].min().item()),
        "no_match_immediate_response_maximum": float(no_match_effect["z_effect"].max().item()),
    }


def _make_plot(summary: List[Dict[str, object]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [str(row["method"]).replace("_", " ").title() for row in summary]
    response = [float(row["mean_gross_response_score"]) for row in summary]
    reward = [float(row["mean_net_reward"]) for row in summary]
    spend = [float(row["mean_spend"]) for row in summary]
    x = list(range(len(labels)))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].bar([v - width / 2 for v in x], response, width, label="Gross response", color="#8EACE6")
    axes[0].bar([v + width / 2 for v in x], reward, width, label="Net reward", color="#F2CF5B")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Episode total")
    axes[0].legend(frameon=False)
    axes[1].scatter(spend, reward, s=70, color="#3A5A8C")
    for px, py, label in zip(spend, reward, labels):
        axes[1].annotate(label, (px, py), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Mean subsidy spend")
    axes[1].set_ylabel("Mean net reward")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "public_city_panel_environment_sanity.png", dpi=220)
    fig.savefig(out_dir / "public_city_panel_environment_sanity.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--daily_budget", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument(
        "--calibration_days",
        type=int,
        default=PublicCityPanelConfig.calibration_episodes,
    )
    parser.add_argument(
        "--calibration_seed",
        type=int,
        default=PublicCityPanelConfig.calibration_seed,
    )
    parser.add_argument("--calibration_workers", type=int, default=0)
    parser.add_argument(
        "--baseline_epochs",
        type=int,
        default=PublicCityPanelConfig.baseline_epochs,
    )
    parser.add_argument(
        "--baseline_batch_size",
        type=int,
        default=PublicCityPanelConfig.baseline_batch_size,
    )
    parser.add_argument(
        "--baseline_hidden_dims",
        default=",".join(str(value) for value in PublicCityPanelConfig.baseline_hidden_dims),
    )
    parser.add_argument(
        "--baseline_patience",
        type=int,
        default=PublicCityPanelConfig.baseline_patience,
    )
    parser.add_argument(
        "--baseline_learning_rate",
        type=float,
        default=PublicCityPanelConfig.baseline_learning_rate,
    )
    parser.add_argument(
        "--baseline_z_loss_weight",
        type=float,
        default=PublicCityPanelConfig.baseline_z_loss_weight,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", default="envcheck_public_city_panel")
    parser.add_argument("--make_plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PublicCityPanelConfig(
        M=args.M,
        T=args.T,
        daily_budget=args.daily_budget,
        calibration_episodes=args.calibration_days,
        calibration_seed=args.calibration_seed,
        calibration_workers=args.calibration_workers,
        baseline_epochs=args.baseline_epochs,
        baseline_batch_size=args.baseline_batch_size,
        baseline_hidden_dims=tuple(int(value) for value in args.baseline_hidden_dims.split(",") if value.strip()),
        baseline_patience=args.baseline_patience,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_z_loss_weight=args.baseline_z_loss_weight,
        device=args.device,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration = _get_calibration(cfg)

    split_sets = {name: set(indices.tolist()) for name, indices in calibration.split_indices.items()}
    split_disjoint = all(
        split_sets[left].isdisjoint(split_sets[right])
        for left in split_sets
        for right in split_sets
        if left < right
    )
    seed_mapping_duplicate_fractions = {}
    for name, indices in calibration.split_indices.items():
        draw_count = min(500, int(indices.numel()))
        positions = [
            _seed_to_pool_position(700_000 + offset, int(indices.numel()))
            for offset in range(draw_count)
        ]
        seed_mapping_duplicate_fractions[name] = 1.0 - len(set(positions)) / max(draw_count, 1)
    rows = [
        _episode(cfg, method, 100_000 + episode)
        for method in METHODS
        for episode in range(int(args.episodes))
    ]
    summary = _summaries(rows)
    checks = {
        "M": cfg.M,
        "T": cfg.T,
        "daily_budget": cfg.daily_budget,
        "state_dim": PublicCityPanelEnv(cfg).state_dim,
        "dynamic_dim": DYNAMIC_DIM,
        "static_dim": STATIC_DIM,
        "day_splits_are_disjoint": split_disjoint,
        **{
            f"with_replacement_duplicate_fraction_{name}": value
            for name, value in seed_mapping_duplicate_fractions.items()
        },
        "evaluation_days_used_for_baseline_fit": calibration.metrics["baseline_evaluation_days_used_for_fit"],
        **_paired_zero_action_error(cfg, range(91_000, 91_010)),
        **_intervention_checks(cfg),
        "maximum_reward_identity_error": max(float(row["reward_identity_error"]) for row in rows),
        "budget_violation_count": int(sum(float(row["budget_violation"]) for row in rows)),
    }
    if not split_disjoint:
        raise AssertionError("Day-level baseline splits overlap.")
    if checks["zero_action_max_absolute_effect"] != 0.0:
        raise AssertionError("The artificial intervention is not exactly zero at zero subsidy.")
    if checks["diminishing_return_margin_minimum"] < -1e-6:
        raise AssertionError("The subsidy response is not concave over the checked range.")
    if checks["no_match_immediate_response_maximum"] != 0.0:
        raise AssertionError("Subsidy generated an immediate response without a feasible match.")
    if checks["budget_violation_count"] != 0:
        raise AssertionError("A baseline violated the daily subsidy budget.")

    _write_rows(out_dir / "environment_baseline_raw.csv", rows)
    _write_rows(out_dir / "environment_baseline_summary.csv", summary)
    _write_rows(
        out_dir / "baseline_calibration_metrics.csv",
        [{"metric": name, "value": value} for name, value in sorted(calibration.metrics.items())],
    )
    with (out_dir / "environment_invariants.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2, sort_keys=True)
    if args.make_plots:
        _make_plot(summary, out_dir)
    print(json.dumps(checks, indent=2, sort_keys=True))
    for row in summary:
        print(
            f"{row['method']:>16s}  net={row['mean_net_reward']:.3f}  "
            f"gross={row['mean_gross_response_score']:.3f}  spend={row['mean_spend']:.3f}"
        )


if __name__ == "__main__":
    main()
