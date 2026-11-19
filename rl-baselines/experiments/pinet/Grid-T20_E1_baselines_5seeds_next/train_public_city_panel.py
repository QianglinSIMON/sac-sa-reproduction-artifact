"""CLI entry point for public city-panel SAC-SA and DFL-PTO experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from city_panel_training import (
    ExperimentConfig,
    PolicyConfig,
    SimulatorConfig,
    run_experiment,
)


def int_tuple(value: str):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def float_tuple(value: str):
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--method", choices=("sacsa_reserve", "dfl_pto"), required=True)
    argument_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    argument_parser.add_argument("--seed", type=int, default=2026)
    argument_parser.add_argument("--M", type=int, default=3)
    argument_parser.add_argument("--T", type=int, default=20)
    argument_parser.add_argument("--daily_budget", type=float, default=10.0)
    argument_parser.add_argument("--q_max", type=int, default=3)
    argument_parser.add_argument("--initial_logged_days", type=int, default=3000)
    argument_parser.add_argument("--new_logged_days", type=int, default=200)
    argument_parser.add_argument("--residual_calibration_days", type=int, default=800)
    argument_parser.add_argument("--validation_logged_days", type=int, default=600)
    argument_parser.add_argument("--holdout_logged_days", type=int, default=600)
    argument_parser.add_argument("--eval_episodes", type=int, default=1000)
    argument_parser.add_argument("--dgp_calibration_days", type=int, default=10000)
    argument_parser.add_argument("--dgp_baseline_epochs", type=int, default=100)
    argument_parser.add_argument("--logging_batch_size", type=int, default=512)

    argument_parser.add_argument("--sim_hidden_dims", type=int_tuple, default=(768, 512, 256))
    argument_parser.add_argument("--sim_head_dim", type=int, default=256)
    argument_parser.add_argument("--sim_epochs", type=int, default=300)
    argument_parser.add_argument("--sim_finetune_epochs", type=int, default=120)
    argument_parser.add_argument("--sim_batch_size", type=int, default=4096)
    argument_parser.add_argument("--sim_learning_rate", type=float, default=3e-4)
    argument_parser.add_argument("--sim_patience", type=int, default=30)
    argument_parser.add_argument("--bootstrap_models", type=int, default=8)
    argument_parser.add_argument("--bootstrap_epochs", type=int, default=100)
    argument_parser.add_argument("--no_center_residual", action="store_true")
    argument_parser.add_argument("--min_holdout_r2_z", type=float, default=0.75)
    argument_parser.add_argument("--min_holdout_r2_z_by_t", type=float, default=0.30)
    argument_parser.add_argument("--strict_simulator_quality", action="store_true")

    argument_parser.add_argument("--policy_hidden_dims", type=int_tuple, default=(512, 512, 256))
    argument_parser.add_argument("--synthetic_train_episodes", type=int, default=6000)
    argument_parser.add_argument("--synthetic_train_episodes_later_rounds", type=int, default=0)
    argument_parser.add_argument("--rollout_batch_size", type=int, default=256)
    argument_parser.add_argument("--updates_per_rollout_batch", type=int, default=128)
    argument_parser.add_argument("--policy_batch_size", type=int, default=1024)
    argument_parser.add_argument("--init_spend_share", type=float, default=0.10)
    argument_parser.add_argument("--lambda_pess", type=float, default=0.02)
    argument_parser.add_argument("--fixed_temperature", type=float, default=0.05)
    argument_parser.add_argument("--auto_alpha", action="store_true")
    argument_parser.add_argument("--no_real_online_update", action="store_true")
    argument_parser.add_argument("--real_gradient_steps_per_step", type=float, default=1.0)
    argument_parser.add_argument("--real_critic_warmup_updates", type=int, default=0)
    argument_parser.add_argument("--real_exploration_epsilon", type=float, default=0.40)
    argument_parser.add_argument("--behavior_support_lambda", type=float, default=0.05)
    argument_parser.add_argument("--support_prototype_count", type=int, default=512)
    argument_parser.add_argument("--simulator_selection_episodes", type=int, default=1200)
    argument_parser.add_argument("--simulator_selection_z_value", type=float, default=1.645)
    argument_parser.add_argument("--simulator_selection_tolerance", type=float, default=0.0)
    argument_parser.add_argument("--dfl_epochs", type=int, default=150)
    argument_parser.add_argument("--dfl_batch_size", type=int, default=1024)
    argument_parser.add_argument("--dfl_candidate_random_count", type=int, default=128)
    argument_parser.add_argument("--dfl_hidden_dims", type=int_tuple, default=(256, 256))
    argument_parser.add_argument("--dfl_learning_rate", type=float, default=1e-3)
    argument_parser.add_argument("--dfl_decision_weight", type=float, default=1.0)
    argument_parser.add_argument("--dfl_softmax_temperature", type=float, default=1.0)
    argument_parser.add_argument(
        "--dfl_budget_fractions", type=float_tuple, default=(0.25, 0.50, 0.75, 1.00)
    )
    argument_parser.add_argument("--out_dir", type=Path, required=True)
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    experiment = ExperimentConfig(
        M=args.M,
        T=args.T,
        daily_budget=args.daily_budget,
        seed=args.seed,
        device=args.device,
        q_max=args.q_max,
        initial_logged_days=args.initial_logged_days,
        new_logged_days=args.new_logged_days,
        residual_calibration_days=args.residual_calibration_days,
        validation_logged_days=args.validation_logged_days,
        holdout_logged_days=args.holdout_logged_days,
        eval_episodes=args.eval_episodes,
        dgp_calibration_days=args.dgp_calibration_days,
        dgp_baseline_epochs=args.dgp_baseline_epochs,
        logging_batch_size=args.logging_batch_size,
    )
    simulator = SimulatorConfig(
        hidden_dims=args.sim_hidden_dims,
        head_dim=args.sim_head_dim,
        epochs=args.sim_epochs,
        finetune_epochs=args.sim_finetune_epochs,
        batch_size=args.sim_batch_size,
        learning_rate=args.sim_learning_rate,
        patience=args.sim_patience,
        bootstrap_models=args.bootstrap_models,
        bootstrap_epochs=args.bootstrap_epochs,
        residual_center=not args.no_center_residual,
        min_holdout_r2_z=args.min_holdout_r2_z,
        min_holdout_r2_z_by_t=args.min_holdout_r2_z_by_t,
        strict_quality_gate=args.strict_simulator_quality,
    )
    policy = PolicyConfig(
        hidden_dims=args.policy_hidden_dims,
        train_episodes=args.synthetic_train_episodes,
        later_round_train_episodes=args.synthetic_train_episodes_later_rounds,
        rollout_batch_size=args.rollout_batch_size,
        updates_per_rollout_batch=args.updates_per_rollout_batch,
        batch_size=args.policy_batch_size,
        init_spend_share=args.init_spend_share,
        lambda_pess=args.lambda_pess,
        auto_alpha=args.auto_alpha,
        fixed_temperature=args.fixed_temperature,
        real_online_update=not args.no_real_online_update,
        real_gradient_steps_per_step=args.real_gradient_steps_per_step,
        real_critic_warmup_updates=args.real_critic_warmup_updates,
        real_exploration_epsilon=args.real_exploration_epsilon,
        behavior_support_lambda=args.behavior_support_lambda,
        support_prototype_count=args.support_prototype_count,
        simulator_selection_episodes=args.simulator_selection_episodes,
        simulator_selection_z_value=args.simulator_selection_z_value,
        simulator_selection_tolerance=args.simulator_selection_tolerance,
        dfl_epochs=args.dfl_epochs,
        dfl_batch_size=args.dfl_batch_size,
        dfl_candidate_random_count=args.dfl_candidate_random_count,
        dfl_hidden_dims=args.dfl_hidden_dims,
        dfl_learning_rate=args.dfl_learning_rate,
        dfl_decision_weight=args.dfl_decision_weight,
        dfl_softmax_temperature=args.dfl_softmax_temperature,
        dfl_budget_fractions=args.dfl_budget_fractions,
    )
    run_experiment(args.method, experiment, simulator, policy, args.out_dir)


if __name__ == "__main__":
    main()
