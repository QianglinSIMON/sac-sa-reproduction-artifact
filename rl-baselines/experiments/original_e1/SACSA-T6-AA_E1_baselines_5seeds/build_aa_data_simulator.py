from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from aa_data_based_simulator import (
    AADataBasedSubsidyEnv,
    AADataSimulatorConfig,
    calibrate_aa_simulator,
    make_sanity_plots,
    rollout_policy,
    write_calibration_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an AA data-based synthetic subsidy simulator calibration."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data_sim"))
    parser.add_argument("--out_dir", type=Path, default=Path("aa_data_based_simulator_outputs"))
    parser.add_argument("--target_t", type=int, default=6)
    parser.add_argument("--daily_budget", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--subsidy_response_scale", type=float, default=0.10)
    parser.add_argument("--overspend_waste", type=float, default=0.03)
    parser.add_argument("--gamma_x_ratio", type=float, default=0.60)
    parser.add_argument("--gamma_y_ratio", type=float, default=0.85)
    parser.add_argument("--target_gap_gross_uplift_per_budget", type=float, default=1.40)
    parser.add_argument("--center_residual", dest="center_residual", action="store_true")
    parser.add_argument("--no_center_residual", dest="center_residual", action="store_false")
    parser.set_defaults(center_residual=False)
    parser.add_argument("--sanity_episodes", type=int, default=500)
    parser.add_argument("--make_plots", action="store_true")
    return parser.parse_args()


def run_internal_sanity_checks(calibration, cfg: AADataSimulatorConfig, seed: int) -> None:
    env = AADataBasedSubsidyEnv(calibration, cfg, seed=seed, project_action=True)
    state = env.reset()
    zero_action = torch.zeros(env.M, device=state.device, dtype=state.dtype)
    _, reward, _, info = env.step(zero_action)
    if int(info["bootstrap_day_idx"]) != int(info["residual_idx"]):
        raise RuntimeError("AA bootstrap day and residual path day are not aligned.")
    if abs(float(info["spend"].detach().cpu())) > 1e-8:
        raise RuntimeError("Zero action should have zero spend.")
    if float(torch.abs(info["subsidy_z"]).max().detach().cpu()) > 1e-8:
        raise RuntimeError("Zero action should have zero subsidy response.")
    if not torch.isfinite(reward):
        raise RuntimeError("AA DGP sanity reward is nonfinite.")


def main() -> None:
    args = parse_args()
    cfg = AADataSimulatorConfig(
        daily_budget=args.daily_budget,
        target_t=args.target_t,
        subsidy_response_scale=args.subsidy_response_scale,
        overspend_waste=args.overspend_waste,
        gamma_x_ratio=args.gamma_x_ratio,
        gamma_y_ratio=args.gamma_y_ratio,
        target_gap_gross_uplift_per_budget=args.target_gap_gross_uplift_per_budget,
        center_residual=args.center_residual,
        device="cpu",
        dtype=torch.float32,
        seed=args.seed,
    )

    calibration, panel, coef_df, residual_df, diag_df = calibrate_aa_simulator(
        data_dir=args.data_dir,
        cfg=cfg,
    )
    write_calibration_outputs(args.out_dir, calibration, panel, coef_df, residual_df, diag_df)
    run_internal_sanity_checks(calibration, cfg, args.seed)

    policies = ["no_subsidy", "uniform_pacing", "demand_supply_gap", "random_pacing"]
    sanity_rows = [
        rollout_policy(
            calibration=calibration,
            cfg=cfg,
            policy=policy,
            n_episodes=args.sanity_episodes,
            seed=args.seed + 10000 * k,
        )
        for k, policy in enumerate(policies)
    ]
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(args.out_dir / "aa_dgp_sanity.csv", index=False)

    if args.make_plots:
        make_sanity_plots(args.out_dir, diag_df, sanity_df)

    print(f"[done] wrote AA data-based simulator artifacts to {args.out_dir}")
    print(
        "[panel] days={days} cities={cities} T={T} rows={rows}".format(
            days=calibration.n_days,
            cities=calibration.n_cities,
            T=calibration.T,
            rows=len(panel),
        )
    )
    print("[panel] aggregation=sum over source AA intervals")
    z_diag = diag_df[diag_df["equation"] == "gmv_z"]
    print(
        "[gmv_ols] mean_r2={:.4f} min_r2={:.4f} mean_rmse={:.4f}".format(
            float(z_diag["r2"].mean()),
            float(z_diag["r2"].min()),
            float(z_diag["rmse"].mean()),
        )
    )
    print(
        "[residual] mean_abs_z={:.4e} mean_abs_x={:.4e} mean_abs_y={:.4e}".format(
            float(abs(calibration.residual_mean_z).mean()),
            float(abs(calibration.residual_mean_x).mean()),
            float(abs(calibration.residual_mean_y).mean()),
        )
    )
    print("[sanity]")
    print(sanity_df.to_string(index=False))


if __name__ == "__main__":
    main()
