"""
Iterative synthetic experiment for the current SAC-SA paper logic.

This entrypoint keeps the old true-DGP code as the synthetic marketplace, but
does not train SAC-SA directly in that environment. Instead it follows the
paper's model-based loop:

    logged real days -> fitted response simulator -> residual-bootstrap rollout
    -> uncertainty-aware synthetic SAC-SA training -> per-round true-DGP evaluation
    -> append new on-policy logged days if continuing.

The learned simulator is a neural conditional-mean response model. Synthetic
rollouts add day-level nonparametric residual-bootstrap noise estimated from
the current real log. A bootstrap ensemble estimates epistemic model uncertainty
for pessimistic training and planning. Learned-simulator validation is used as
simulator-quality diagnostics; reported policy outcomes always use unpenalized
true-DGP rewards.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DGP_sim_env import (
    CITY_OVERSPEND_WASTE,
    DGP_COEF_KEYS,
    SUBSIDY_RESPONSE_SCALE,
    SimConfig,
    SubsidyAllocationEnv,
    generate_dgp_coefs,
    subsidy_effect_multipliers,
    subsidy_response,
)
from sacsa_rl_components import (
    BUDGET_SAFETY,
    EPS,
    ReplayBuffer,
    StateNormalizer,
    TD3Agent,
    TrainConfig,
    ensure_dir,
    ensure_feasible_action,
    json_safe,
    make_sacsa_reserve_agent,
    make_sacsa_agent,
    make_sac_logistic_normal_reserve_agent,
    make_sac_logistic_normal_agent,
    make_sac_gaussian_projection_agent,
    mlp,
    random_paced_action,
    save_json,
    set_seed,
    summarize_eval_rows,
    to_float,
    write_csv,
)


LEARNING_METHOD_NAMES = {
    "sacsa_reserve": "Proposed Beta-Dirichlet SAC-SA",
    "sac_logistic_normal_reserve": "Reserve-Gated Logistic-Normal SAC",
    "sacsa": "Original Augmented-Dirichlet SAC-SA",
    "sac_logistic_normal": "Original Logistic-Normal SAC",
    "sac_gaussian_projection": "Gaussian Projection SAC",
    "td3_projection": "TD3 + Feasible Shares",
    "dfl_pto": "DFL-PTO",
    "mpc3": "Finite-Candidate MPC-3",
}

ACTOR_CRITIC_METHODS = {
    "sacsa_reserve",
    "sac_logistic_normal_reserve",
    "sacsa",
    "sac_logistic_normal",
    "sac_gaussian_projection",
    "td3_projection",
}
PLANNING_METHODS = {"dfl_pto", "mpc3"}
UNCERTAINTY_PENALTY_SCOPE = "z"
MIXED_LOGGING_POLICY = (
    "no/low-random/medium-random/full-random/uniform/gap/gap-half paced actions"
)
RICH_LOGGING_POLICY = (
    "semi-balanced active-heavy mix over no/low/medium/high/full spend, smooth and concentrated "
    "Dirichlet-like shares, gap-random mixes, sparse top-k actions, and mid/late high-spend pacing"
)
TRUE_DGP_BASELINE_POLICIES = [
    ("no_subsidy", "No Subsidy"),
    ("random_pacing", "Random Pacing"),
    ("uniform_pacing", "Uniform Pacing"),
    ("gap_heuristic", "Demand-Supply Gap"),
]


def method_sort_key(name: str) -> int:
    order = {
        "Proposed Beta-Dirichlet SAC-SA": 0,
        "Reserve-Gated Logistic-Normal SAC": 1,
        "Original Augmented-Dirichlet SAC-SA": 2,
        "Original Logistic-Normal SAC": 3,
        "Gaussian Projection SAC": 4,
        "TD3 + Feasible Shares": 5,
        "DFL-PTO": 6,
        "Finite-Candidate MPC-3": 7,
        "Demand-Supply Gap": 8,
        "Uniform Pacing": 9,
        "Random Pacing": 10,
        "No Subsidy": 11,
    }
    return order.get(str(name), 99)


def logging_policy_description(logging_policy: str) -> str:
    return RICH_LOGGING_POLICY if str(logging_policy).lower() == "rich" else MIXED_LOGGING_POLICY


@dataclass
class LoggedTransition:
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: Optional[torch.Tensor]
    done: bool
    t: int
    w: torch.Tensor


LoggedTrajectory = List[LoggedTransition]


@dataclass
class FixedLoggedSplits:
    train_log: List[LoggedTrajectory]
    validation_log: List[LoggedTrajectory]
    holdout_log: List[LoggedTrajectory]
    validation_initial_states: List[torch.Tensor]


@dataclass
class SimulatorFitConfig:
    epochs: int = 50
    batch_size: int = 512
    lr: float = 1e-3
    hidden_dim: int = 256
    hidden_layers: Tuple[int, ...] = (512, 256, 128)
    head_hidden_dim: int = 0
    dropout: float = 0.05
    activation: str = "silu"
    weight_decay: float = 0.0
    z_loss_weight: float = 6.0
    x_loss_weight: float = 1.0
    y_loss_weight: float = 1.0
    c_loss_weight: float = 1.0


@dataclass
class PessimismConfig:
    use_pessimism: bool = True
    lambda_pess: float = 0.001
    b_boot: int = 5
    bootstrap_epochs: int = 20
    bootstrap_finetune_epochs: int = 5
    save_bootstrap_ensemble: bool = False
    uncertainty_penalty_scope: str = UNCERTAINTY_PENALTY_SCOPE


@dataclass
class CandidateConfig:
    budget_fracs: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    random_count: int = 64
    seed: int = 2026


@dataclass
class DFLConfig:
    epochs: int = 80
    batch_size: int = 512
    lr: float = 1e-3
    decision_weight: float = 1.0
    softmax_temperature: float = 1.0
    hidden_dim: int = 256


@dataclass
class InnerEarlyStopConfig:
    enabled: bool = False
    eval_every_episodes: int = 50
    patience: int = 3
    min_delta: float = 0.0


@dataclass
class RoundSimulatorBundle:
    main_models: List["ResponseModel"]
    bootstrap_ensemble: List[List["ResponseModel"]]
    train_residual_sampler: "ResidualBootstrapSampler"
    validation_residual_sampler: "ResidualBootstrapSampler"
    main_model_metrics: Dict[str, float]
    bootstrap_metrics: Dict[str, float]
    train_residual_metrics: Dict[str, float]
    validation_residual_metrics: Dict[str, float]
    holdout_metrics: Dict[str, float]
    interval_holdout_rows: List[Dict[str, Any]]
    sim_epochs_used: int
    bootstrap_epochs_used: int


def detach_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().to(dtype=torch.float32)


def state_time_features(t: int, T: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    tt = t + 1
    angle = 2.0 * math.pi * float(tt) / float(T)
    return torch.tensor(
        [math.sin(angle), math.cos(angle), float(T - t) / float(T)],
        device=device,
        dtype=dtype,
    )


def raw_state_from_parts(
    X: torch.Tensor,
    Y: torch.Tensor,
    C: torch.Tensor,
    O: torch.Tensor,
    H: torch.Tensor,
    t: int,
    T: int,
) -> torch.Tensor:
    time_feat = state_time_features(t, T, str(X.device), X.dtype)
    return torch.cat([X, Y, C, O, H.reshape(1), time_feat], dim=0)


def raw_state_from_batch_parts(
    X: torch.Tensor,
    Y: torch.Tensor,
    C: torch.Tensor,
    O: torch.Tensor,
    H: torch.Tensor,
    t: int,
    T: int,
) -> torch.Tensor:
    time_feat = state_time_features(t, T, str(X.device), X.dtype).expand(X.shape[0], 3)
    return torch.cat([X, Y, C, O, H.reshape(-1, 1), time_feat], dim=-1)


def normalize_state(state: torch.Tensor, normalizer: Optional[StateNormalizer]) -> torch.Tensor:
    if normalizer is None:
        return state
    return normalizer.transform(state.to(device=normalizer.mean.device))


def scale_action_for_model(action: torch.Tensor, daily_budget: float) -> torch.Tensor:
    """Use budget-normalized actions as simulator/score-model inputs only."""
    denom = max(float(daily_budget), EPS)
    return action / denom


def simulator_action_features(action: torch.Tensor, daily_budget: float) -> torch.Tensor:
    """Reality-facing subsidy features: A/B, (A/B)^2, and total spend/B."""
    share = scale_action_for_model(action, daily_budget)
    total_share = share.sum(dim=-1, keepdim=True)
    return torch.cat([share, share.pow(2), total_share], dim=-1)


def simulator_action_feature_dim(action_dim: int) -> int:
    return 2 * int(action_dim) + 1


def simulator_input_dim(state_dim: int, action_dim: int) -> int:
    return int(state_dim) + simulator_action_feature_dim(action_dim)


def simulator_state_features(
    raw_states: torch.Tensor,
    state_normalizer: Optional[StateNormalizer],
    daily_budget: float,
) -> torch.Tensor:
    if raw_states.ndim == 1:
        raw_states = raw_states.unsqueeze(0)
    state_in = normalize_state(raw_states, state_normalizer)
    h_idx = state_in.shape[-1] - 4
    state_in = state_in.clone()
    state_in[..., h_idx] = raw_states[..., h_idx] / max(float(daily_budget), EPS)
    return state_in


def split_raw_state(raw_state: torch.Tensor, M: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = raw_state[..., :M]
    Y = raw_state[..., M : 2 * M]
    C = raw_state[..., 2 * M : 3 * M]
    O = raw_state[..., 3 * M : 4 * M]
    H = raw_state[..., 4 * M]
    return X, Y, C, O, H


def normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = weights.sum(dim=-1, keepdim=True)
    uniform = torch.ones_like(weights) / max(weights.shape[-1], 1)
    return torch.where(total > EPS, weights / total.clamp_min(EPS), uniform)


@torch.no_grad()
def legacy_mixed_logging_action(
    raw_state: torch.Tensor,
    env: SubsidyAllocationEnv,
    random_logging_spend_scale: float = 1.0,
) -> torch.Tensor:
    """
    Fixed mixed logging policy used only to generate D0/D_val/D_hold.

    The mixture deliberately covers no-spend, low/medium/full random pacing,
    uniform pacing, and gap-directed pacing so the learned simulator sees a
    broader feasible-action support than a single random pacing rule.
    """
    device, dtype = env.device, env.dtype
    M = env.action_dim
    h = torch.as_tensor(env.H, device=device, dtype=dtype).clamp_min(0.0)
    if float(h.detach().cpu().item()) <= EPS:
        return torch.zeros(M, device=device, dtype=dtype)

    generator = getattr(env, "generator", None)
    u = float(torch.rand((), device=device, dtype=dtype, generator=generator).detach().cpu().item())
    remaining = max(int(env.T - env.t), 1)

    def random_dir() -> torch.Tensor:
        raw = torch.rand(M, device=device, dtype=dtype, generator=generator)
        return normalize_weights(raw)

    X, Y, _, _, _ = split_raw_state(raw_state.to(device=device, dtype=dtype), M)
    uniform_dir = torch.ones(M, device=device, dtype=dtype) / max(M, 1)
    gap_dir = normalize_weights((X - Y).clamp_min(0.0) + EPS)

    if u < 0.10:
        return torch.zeros(M, device=device, dtype=dtype)
    if u < 0.25:
        direction, paced_frac = random_dir(), 0.10
    elif u < 0.45:
        direction, paced_frac = random_dir(), 0.50
    elif u < 0.65:
        direction, paced_frac = random_dir(), 1.00
    elif u < 0.75:
        direction, paced_frac = uniform_dir, 1.00
    elif u < 0.90:
        direction, paced_frac = gap_dir, 1.00
    else:
        direction, paced_frac = gap_dir, 0.50

    spend = h * float(paced_frac) * float(random_logging_spend_scale) / float(remaining)
    spend = spend.clamp(min=0.0, max=h)
    return ensure_feasible_action(direction * spend, h)


@torch.no_grad()
def rich_mixed_logging_action(
    raw_state: torch.Tensor,
    env: SubsidyAllocationEnv,
    random_logging_spend_scale: float = 1.0,
) -> torch.Tensor:
    """
    Broader fixed logging support for D0/D_val/D_hold.

    The extra modes are reality-facing budget heuristics, not DGP-form hints.
    They deliberately cover reserve-heavy, medium/high/full spend, smooth,
    concentrated, gap-mixed, sparse top-k, and nonuniform timing actions using
    only observable state gaps, remaining budget, and the daily budget. The
    mixture puts more mass on nonzero spend trajectories, especially in the
    middle and late intervals where learned policies often concentrate budget.
    This gives the fitted simulator support around active budget-use policies
    without using the synthetic DGP equations.
    """
    device, dtype = env.device, env.dtype
    M = env.action_dim
    h = torch.as_tensor(env.H, device=device, dtype=dtype).clamp_min(0.0)
    if float(h.detach().cpu().item()) <= EPS:
        return torch.zeros(M, device=device, dtype=dtype)

    generator = getattr(env, "generator", None)
    u = float(torch.rand((), device=device, dtype=dtype, generator=generator).detach().cpu().item())
    remaining = max(int(env.T - env.t), 1)
    daily_budget = max(float(env.cfg.daily_budget), EPS)
    tau = int(env.t) + 1
    tau_frac = float(tau) / max(float(env.T), 1.0)

    def rand_float(low: float = 0.0, high: float = 1.0) -> float:
        draw = float(torch.rand((), device=device, dtype=dtype, generator=generator).detach().cpu().item())
        return float(low) + (float(high) - float(low)) * draw

    def random_dir(power: float = 1.0) -> torch.Tensor:
        raw = torch.rand(M, device=device, dtype=dtype, generator=generator).clamp_min(EPS)
        return normalize_weights(raw.pow(float(power)))

    def smooth_dir() -> torch.Tensor:
        # Smooth Dirichlet-like directions approximate SAC-style simplex shares.
        raw = torch.rand(M, device=device, dtype=dtype, generator=generator)
        raw = 0.65 * torch.ones_like(raw) + 0.35 * raw
        return normalize_weights(raw)

    def concentrated_dir() -> torch.Tensor:
        # Random but concentrated directions help cover actor policies that put
        # most of the spend on a small set of cities.
        return random_dir(power=5.0)

    def topk_dir(weights: torch.Tensor, k: int) -> torch.Tensor:
        k_eff = min(max(int(k), 1), M)
        _, idx = torch.topk(weights, k=k_eff)
        mask = torch.zeros_like(weights)
        mask[idx] = 1.0
        return normalize_weights(weights * mask + EPS * mask)

    def random_topk_dir(k: int) -> torch.Tensor:
        k_eff = min(max(int(k), 1), M)
        raw = torch.rand(M, device=device, dtype=dtype, generator=generator)
        _, idx = torch.topk(raw, k=k_eff)
        mask = torch.zeros_like(raw)
        mask[idx] = 1.0
        return normalize_weights(raw * mask + EPS * mask)

    X, Y, _, _, _ = split_raw_state(raw_state.to(device=device, dtype=dtype), M)
    gap = (X - Y).clamp_min(0.0)
    uniform_dir = torch.ones(M, device=device, dtype=dtype) / max(M, 1)
    gap_dir = normalize_weights(gap + EPS)
    top2_gap_dir = topk_dir(gap + EPS, k=min(2, M))
    top3_gap_dir = topk_dir(gap + EPS, k=min(3, M))

    def gap_random_mix_dir(rho: float) -> torch.Tensor:
        return normalize_weights(float(rho) * gap_dir + (1.0 - float(rho)) * random_dir())

    def paced_spend(frac: float) -> torch.Tensor:
        return (h * float(frac) * float(random_logging_spend_scale) / float(remaining)).clamp(min=0.0, max=h)

    def target_paced_spend(target_daily_fraction: float, aggressiveness: float = 1.0) -> torch.Tensor:
        spent_so_far = torch.as_tensor(daily_budget, device=device, dtype=dtype) - h
        target_total = torch.as_tensor(
            daily_budget * float(target_daily_fraction) * float(random_logging_spend_scale),
            device=device,
            dtype=dtype,
        )
        target_remaining = (target_total - spent_so_far).clamp_min(0.0)
        return (target_remaining * float(aggressiveness) / float(remaining)).clamp(min=0.0, max=h)

    def absolute_spend(frac_of_daily_budget: float, schedule_weight: float = 1.0) -> torch.Tensor:
        spend_val = daily_budget * float(frac_of_daily_budget) * float(schedule_weight) * float(random_logging_spend_scale)
        return torch.as_tensor(spend_val, device=device, dtype=dtype).clamp(min=0.0, max=h)

    def midlate_schedule_weight() -> float:
        if tau_frac < 0.35:
            return 0.05
        ramp = max((tau_frac - 0.35) / 0.65, 0.0)
        return min(1.65, 0.65 + 1.00 * ramp)

    def late_schedule_weight() -> float:
        if tau_frac < 0.60:
            return 0.03
        ramp = max((tau_frac - 0.60) / 0.40, 0.0)
        return min(1.75, 0.35 + 1.40 * ramp)

    # Draw the logging mode once per day rather than once per interval. This is
    # important for action support: a cautious SAC-style policy may follow an
    # entire low-spend trajectory, not just a few isolated low-spend steps.
    if int(env.t) == 0 or not isinstance(getattr(env, "_rich_logging_state", None), dict):
        mode_draw = rand_float()
        if mode_draw < 0.02:
            mode = "no_spend"
        elif mode_draw < 0.05:
            mode = "target_low_smooth"
        elif mode_draw < 0.08:
            mode = "target_low_gapmix"
        elif mode_draw < 0.17:
            mode = "target_medium_smooth"
        elif mode_draw < 0.28:
            mode = "target_medium_gapmix"
        elif mode_draw < 0.38:
            mode = "target_high_random"
        elif mode_draw < 0.48:
            mode = "target_high_gapmix"
        elif mode_draw < 0.56:
            mode = "uniform_full"
        elif mode_draw < 0.64:
            mode = "gap_full"
        elif mode_draw < 0.70:
            mode = "concentrated_medium"
        elif mode_draw < 0.76:
            mode = "random_topk_medium"
        elif mode_draw < 0.82:
            mode = "gap_topk_medium"
        elif mode_draw < 0.86:
            mode = "midlate_smooth_high"
        elif mode_draw < 0.89:
            mode = "midlate_gapmix_high"
        elif mode_draw < 0.90:
            mode = "midlate_topk_high"
        elif mode_draw < 0.95:
            mode = "late_gapmix_high"
        else:
            mode = "late_topk_high"
        setattr(
            env,
            "_rich_logging_state",
            {
                "mode": mode,
                "target_low": rand_float(0.10, 0.30),
                "target_medium": rand_float(0.45, 0.85),
                "target_high": rand_float(0.80, 1.00),
                "rho": rand_float(0.25, 0.85),
                "aggressiveness": rand_float(0.90, 1.50),
                "k": int(rand_float(1.0, min(4, M + 1))),
                "absolute_frac": rand_float(0.08, 0.22),
                "midlate_absolute_frac": rand_float(0.12, 0.28),
                "late_absolute_frac": rand_float(0.14, 0.30),
            },
        )

    logging_state = getattr(env, "_rich_logging_state")
    mode = str(logging_state.get("mode", "target_medium_smooth"))
    if mode == "no_spend":
        return torch.zeros(M, device=device, dtype=dtype)
    if mode == "target_low_smooth":
        direction = smooth_dir()
        spend = target_paced_spend(float(logging_state["target_low"]), float(logging_state["aggressiveness"]))
    elif mode == "target_low_gapmix":
        direction = gap_random_mix_dir(float(logging_state["rho"]))
        spend = target_paced_spend(float(logging_state["target_low"]), float(logging_state["aggressiveness"]))
    elif mode == "target_medium_smooth":
        direction = smooth_dir()
        spend = target_paced_spend(float(logging_state["target_medium"]), float(logging_state["aggressiveness"]))
    elif mode == "target_medium_gapmix":
        direction = gap_random_mix_dir(float(logging_state["rho"]))
        spend = target_paced_spend(float(logging_state["target_medium"]), float(logging_state["aggressiveness"]))
    elif mode == "target_high_random":
        direction = random_dir()
        spend = target_paced_spend(float(logging_state["target_high"]), float(logging_state["aggressiveness"]))
    elif mode == "target_high_gapmix":
        direction = gap_random_mix_dir(float(logging_state["rho"]))
        spend = target_paced_spend(float(logging_state["target_high"]), float(logging_state["aggressiveness"]))
    elif mode == "uniform_full":
        direction, spend = uniform_dir, paced_spend(1.00)
    elif mode == "gap_full":
        direction, spend = gap_dir, paced_spend(1.00)
    elif mode == "concentrated_medium":
        direction, spend = concentrated_dir(), absolute_spend(float(logging_state["absolute_frac"]))
    elif mode == "random_topk_medium":
        direction, spend = random_topk_dir(k=int(logging_state["k"])), absolute_spend(float(logging_state["absolute_frac"]))
    elif mode == "gap_topk_medium":
        direction = top2_gap_dir if int(logging_state["k"]) <= 2 else top3_gap_dir
        spend = absolute_spend(float(logging_state["absolute_frac"]))
    elif mode == "midlate_smooth_high":
        direction = smooth_dir()
        spend = absolute_spend(float(logging_state["midlate_absolute_frac"]), midlate_schedule_weight())
    elif mode == "midlate_gapmix_high":
        direction = gap_random_mix_dir(float(logging_state["rho"]))
        spend = absolute_spend(float(logging_state["midlate_absolute_frac"]), midlate_schedule_weight())
    elif mode == "midlate_topk_high":
        direction = top2_gap_dir if int(logging_state["k"]) <= 2 else top3_gap_dir
        spend = absolute_spend(float(logging_state["midlate_absolute_frac"]), midlate_schedule_weight())
    else:
        if mode == "late_topk_high":
            direction = top2_gap_dir if int(logging_state["k"]) <= 2 else top3_gap_dir
        else:
            direction = gap_random_mix_dir(float(logging_state["rho"]))
        spend = absolute_spend(float(logging_state["late_absolute_frac"]), late_schedule_weight())

    return ensure_feasible_action(direction * spend, h)


@torch.no_grad()
def mixed_logging_action(
    raw_state: torch.Tensor,
    env: SubsidyAllocationEnv,
    random_logging_spend_scale: float = 1.0,
    logging_policy: str = "legacy",
) -> torch.Tensor:
    if str(logging_policy).lower() == "rich":
        return rich_mixed_logging_action(raw_state, env, random_logging_spend_scale)
    return legacy_mixed_logging_action(raw_state, env, random_logging_spend_scale)


@torch.no_grad()
def policy_action(
    policy: Optional[nn.Module],
    raw_state: torch.Tensor,
    normalizer: Optional[StateNormalizer],
    env: SubsidyAllocationEnv,
    deterministic: bool,
    random_logging_spend_scale: float = 1.0,
    logging_policy: str = "legacy",
) -> torch.Tensor:
    if policy is None:
        return mixed_logging_action(raw_state, env, random_logging_spend_scale, logging_policy)
    if hasattr(policy, "select_action_from_raw_state"):
        return policy.select_action_from_raw_state(raw_state.to(env.device), env.t)
    state_for_policy = normalize_state(raw_state, normalizer).to(env.device)
    if hasattr(policy, "sample"):
        action, _, _ = policy.sample(state_for_policy.unsqueeze(0), deterministic=deterministic)
        return action.squeeze(0)
    return policy(state_for_policy.unsqueeze(0)).squeeze(0)


@torch.no_grad()
def collect_logged_trajectories(
    cfg: SimConfig,
    coefs: Dict[str, torch.Tensor],
    episodes: int,
    seed: int,
    normalizer: Optional[StateNormalizer],
    policy: Optional[nn.Module] = None,
    deterministic: bool = True,
    random_logging_spend_scale: float = 1.0,
    logging_policy: str = "legacy",
    audit: Optional[ExperimentAudit] = None,
    audit_phase: str = "logged_collection",
    progress_mode: str = "none",
    progress_desc: str = "collect logged trajectories",
) -> List[LoggedTrajectory]:
    trajectories: List[LoggedTrajectory] = []
    for ep in maybe_tqdm(range(episodes), progress_mode, progress_desc, total=episodes):
        env = SubsidyAllocationEnv(cfg=cfg, coefs=coefs, seed=seed + ep, project_action=False)
        raw_state = env.reset()
        traj: LoggedTrajectory = []
        done = False
        while not done:
            t = env.t
            action = policy_action(
                policy,
                raw_state,
                normalizer,
                env,
                deterministic,
                random_logging_spend_scale,
                logging_policy=logging_policy,
            )
            action = ensure_feasible_action(action.to(env.device), env.H)
            audit_action_tensor(audit, audit_phase, action, env.H, episode=ep, t=t)
            next_state, reward, done, info = env.step(action)
            if audit is not None and audit.enabled:
                if not torch.isfinite(raw_state).all():
                    audit.warn("nonfinite_state", "Logged raw state contains NaN or Inf.", phase=audit_phase, episode=ep, t=t)
                if next_state is not None and not torch.isfinite(next_state).all():
                    audit.warn("nonfinite_next_state", "Logged next state contains NaN or Inf.", phase=audit_phase, episode=ep, t=t)
                step_metrics = {
                    "net_reward": to_float(reward),
                    "response_score": float(info["Z"].detach().sum().cpu().item()),
                    "spend": to_float(info["spend"]),
                    "final_budget": to_float(info["remaining_budget"]),
                }
                audit_episode_metrics(audit, audit_phase, step_metrics, episode=ep, t=t, atol=1e-3)

            # W_it = (Z_it, X_i,t+1, Y_i,t+1, C_i,t+1). In the synthetic
            # setup, Z/X/Y are already log-scale synthetic indicators.
            w = torch.cat([info["Z"], env.X.detach(), env.Y.detach(), env.C.detach()], dim=0)
            traj.append(
                LoggedTransition(
                    state=detach_cpu(raw_state),
                    action=detach_cpu(action),
                    reward=to_float(reward),
                    next_state=None if next_state is None else detach_cpu(next_state),
                    done=done,
                    t=t,
                    w=detach_cpu(w),
                )
            )
            if next_state is not None:
                raw_state = next_state
        audit_episode_metrics(
            audit,
            audit_phase,
            {
                "net_reward": sum(to_float(tr.reward) for tr in traj),
                "response_score": sum(float(tr.w[: cfg.M].sum().detach().cpu().item()) for tr in traj),
                "spend": float(sum(float(tr.action.sum().item()) for tr in traj)),
                "final_budget": to_float(env.H),
            },
            episode=ep,
            atol=1e-3,
        )
        trajectories.append(traj)
    return trajectories


def generate_fixed_logged_splits(
    cfg: SimConfig,
    coefs: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    normalizer: Optional[StateNormalizer],
    train_log: Optional[List[LoggedTrajectory]] = None,
    audit: Optional[ExperimentAudit] = None,
    progress_mode: str = "none",
) -> FixedLoggedSplits:
    """Generate D0, D_val, and D_hold under the fixed randomized logging rule."""
    if train_log is None:
        train_log = collect_logged_trajectories(
            cfg,
            coefs,
            episodes=args.initial_logged_days,
            seed=args.seed + 1_000,
            normalizer=normalizer,
            policy=None,
            deterministic=True,
            random_logging_spend_scale=args.logging_spend_scale,
            logging_policy=args.logging_policy,
            audit=audit,
            audit_phase="initial_logged_collection",
            progress_mode=progress_mode,
            progress_desc="collect initial log",
        )
    validation_log = collect_logged_trajectories(
        cfg,
        coefs,
        episodes=args.validation_logged_days,
        seed=args.seed + 800_000,
        normalizer=normalizer,
        policy=None,
        deterministic=True,
        random_logging_spend_scale=args.logging_spend_scale,
        logging_policy=args.logging_policy,
        audit=audit,
        audit_phase="validation_logged_collection",
        progress_mode=progress_mode,
        progress_desc="collect validation log",
    )
    holdout_log = collect_logged_trajectories(
        cfg,
        coefs,
        episodes=args.holdout_logged_days,
        seed=args.seed + 900_000,
        normalizer=normalizer,
        policy=None,
        deterministic=True,
        random_logging_spend_scale=args.logging_spend_scale,
        logging_policy=args.logging_policy,
        audit=audit,
        audit_phase="holdout_logged_collection",
        progress_mode=progress_mode,
        progress_desc="collect holdout log",
    )
    return FixedLoggedSplits(
        train_log=train_log,
        validation_log=validation_log,
        holdout_log=holdout_log,
        validation_initial_states=[traj[0].state for traj in validation_log if traj],
    )


def flatten_transitions(trajectories: Sequence[LoggedTrajectory]) -> List[LoggedTransition]:
    return [tr for traj in trajectories for tr in traj]


def transitions_at_time(trajectories: Sequence[LoggedTrajectory], t: int) -> List[LoggedTransition]:
    return [tr for tr in flatten_transitions(trajectories) if int(tr.t) == int(t)]


def build_response_tensors_from_transitions(
    transitions: Sequence[LoggedTransition],
    state_normalizer: Optional[StateNormalizer],
    daily_budget: float,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for tr in transitions:
        raw_state = tr.state.to(device)
        action = tr.action.to(device)
        state_in = simulator_state_features(raw_state, state_normalizer, daily_budget).squeeze(0).to(device)
        action_in = simulator_action_features(action, daily_budget).to(device)
        xs.append(torch.cat([state_in, action_in], dim=0))
        ys.append(tr.w.to(device))
    if not xs:
        raise RuntimeError("No response-model transitions available for fitting.")
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def build_response_tensors(
    trajectories: Sequence[LoggedTrajectory],
    state_normalizer: Optional[StateNormalizer],
    daily_budget: float,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return build_response_tensors_from_transitions(
        flatten_transitions(trajectories),
        state_normalizer,
        daily_budget,
        device,
    )


def fit_state_normalizer_from_logged(
    trajectories: Sequence[LoggedTrajectory],
    M: int,
    device: str,
) -> StateNormalizer:
    states = [tr.state[: 4 * M] for tr in flatten_transitions(trajectories)]
    if not states:
        raise RuntimeError("No logged states available when fitting StateNormalizer.")
    mat = torch.stack(states, dim=0).to(device)
    mean = mat.mean(dim=0)
    std = mat.std(dim=0, unbiased=False)
    std = torch.nan_to_num(std, nan=0.0, posinf=1.0, neginf=1.0).clamp_min(1e-3)
    return StateNormalizer(M=M, mean=mean, std=std).to(device)


def activation_class(name: str) -> type[nn.Module]:
    key = str(name).strip().lower()
    if key in {"silu", "swish"}:
        return nn.SiLU
    if key == "gelu":
        return nn.GELU
    if key == "relu":
        return nn.ReLU
    if key == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported simulator activation: {name}")


def make_dropout_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: type[nn.Module],
    dropout: float = 0.0,
    activate_output: bool = False,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = int(input_dim)
    p_drop = max(float(dropout), 0.0)
    for hidden in hidden_dims:
        h = int(hidden)
        if h <= 0:
            continue
        layers.extend([nn.Linear(prev, h), activation()])
        if p_drop > 0.0:
            layers.append(nn.Dropout(p=p_drop))
        prev = h
    layers.append(nn.Linear(prev, int(output_dim)))
    if activate_output:
        layers.append(activation())
        if p_drop > 0.0:
            layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)


class ResponseModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        hidden_layers: Optional[Sequence[int]] = None,
        head_hidden_dim: int = 0,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()
        if output_dim % 4 != 0:
            raise ValueError(f"Response output_dim must be divisible by 4, got {output_dim}.")
        self.M = output_dim // 4
        layers = tuple(int(v) for v in (hidden_layers or (hidden_dim, hidden_dim)) if int(v) > 0)
        if not layers:
            layers = (int(hidden_dim),)
        act = activation_class(activation)
        trunk_out = int(layers[-1])
        trunk_hidden = layers[:-1]
        self.trunk = make_dropout_mlp(
            input_dim,
            trunk_hidden,
            trunk_out,
            activation=act,
            dropout=dropout,
            activate_output=True,
        )
        head_hidden = (int(head_hidden_dim),) if int(head_hidden_dim) > 0 else ()
        self.z_head = make_dropout_mlp(trunk_out, head_hidden, self.M, activation=act, dropout=dropout)
        self.x_head = make_dropout_mlp(trunk_out, head_hidden, self.M, activation=act, dropout=dropout)
        self.y_head = make_dropout_mlp(trunk_out, head_hidden, self.M, activation=act, dropout=dropout)
        self.c_head = make_dropout_mlp(trunk_out, head_hidden, self.M, activation=act, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat(
            [self.z_head(h), self.x_head(h), self.y_head(h), self.c_head(h)],
            dim=-1,
        )


def response_model_for_time(response_models: Sequence[ResponseModel], t: int) -> ResponseModel:
    if not response_models:
        raise ValueError("Empty per-time response model family.")
    if len(response_models) == 1:
        return response_models[0]
    idx = min(max(int(t), 0), len(response_models) - 1)
    return response_models[idx]


def bootstrap_ensemble_for_time(
    bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]],
    t: int,
) -> List[ResponseModel]:
    if not bootstrap_ensemble:
        return []
    idx = min(max(int(t), 0), len(bootstrap_ensemble) - 1)
    return list(bootstrap_ensemble[idx])


def response_scale_for_time(response_scale: Optional[torch.Tensor], t: int) -> Optional[torch.Tensor]:
    if response_scale is None:
        return None
    if response_scale.ndim == 1:
        return response_scale
    idx = min(max(int(t), 0), response_scale.shape[0] - 1)
    return response_scale[idx]


def response_block_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fit_cfg: SimulatorFitConfig,
    target_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if target.shape[-1] % 4 != 0:
        raise ValueError(f"Response target dimension must be divisible by 4, got {target.shape[-1]}.")
    m = target.shape[-1] // 4
    block_specs = (
        (slice(0, m), float(fit_cfg.z_loss_weight)),
        (slice(m, 2 * m), float(fit_cfg.x_loss_weight)),
        (slice(2 * m, 3 * m), float(fit_cfg.y_loss_weight)),
        (slice(3 * m, 4 * m), float(fit_cfg.c_loss_weight)),
    )
    losses: List[torch.Tensor] = []
    weights: List[float] = []
    for sl, weight in block_specs:
        if target_scale is None:
            scale = target[:, sl].std(dim=0, unbiased=False).clamp_min(1e-3)
        else:
            scale = target_scale[sl].to(device=target.device, dtype=target.dtype).clamp_min(1e-3)
        losses.append(((pred[:, sl] - target[:, sl]) / scale).pow(2).mean())
        weights.append(max(weight, 0.0))
    total_weight = max(float(sum(weights)), EPS)
    loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for block_loss, weight in zip(losses, weights):
        loss = loss + (weight / total_weight) * block_loss
    return loss


def _fit_response_model_from_transitions(
    transitions: Sequence[LoggedTransition],
    state_normalizer: Optional[StateNormalizer],
    state_dim: int,
    action_dim: int,
    daily_budget: float,
    fit_cfg: SimulatorFitConfig,
    device: str,
    seed: int,
    model_init: Optional[ResponseModel] = None,
    epochs_override: Optional[int] = None,
) -> Tuple[ResponseModel, float]:
    set_seed(seed)
    x, y = build_response_tensors_from_transitions(transitions, state_normalizer, daily_budget, device)
    target_scale = y.std(dim=0, unbiased=False).clamp_min(1e-3)
    if model_init is None:
        model = ResponseModel(
            simulator_input_dim(state_dim, action_dim),
            y.shape[-1],
            hidden_dim=fit_cfg.hidden_dim,
            hidden_layers=fit_cfg.hidden_layers,
            head_hidden_dim=fit_cfg.head_hidden_dim,
            dropout=fit_cfg.dropout,
            activation=fit_cfg.activation,
        ).to(device)
    else:
        model = copy.deepcopy(model_init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=fit_cfg.lr, weight_decay=fit_cfg.weight_decay)

    n = x.shape[0]
    best_epoch_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs = int(fit_cfg.epochs if epochs_override is None else epochs_override)
    val_size = min(max(n // 10, 1), n - 1, 4096) if n >= 20 else 0
    split_perm = torch.randperm(n, device=device)
    val_idx = split_perm[:val_size] if val_size > 0 else None
    train_idx_all = split_perm[val_size:] if val_size > 0 else split_perm
    train_n = int(train_idx_all.shape[0])
    model.train()
    for _ in range(max(epochs, 0)):
        perm = torch.randperm(train_n, device=device)
        epoch_losses: List[float] = []
        for start in range(0, train_n, fit_cfg.batch_size):
            idx = train_idx_all[perm[start : start + fit_cfg.batch_size]]
            pred = model(x[idx])
            loss = response_block_weighted_loss(pred, y[idx], fit_cfg, target_scale=target_scale)
            if not torch.isfinite(loss):
                raise FloatingPointError("Response model training produced a nonfinite loss.")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        if epoch_losses:
            epoch_loss = float(np.mean(epoch_losses))
            checkpoint_metric = epoch_loss
            if val_idx is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(x[val_idx])
                    val_loss = response_block_weighted_loss(val_pred, y[val_idx], fit_cfg, target_scale=target_scale)
                checkpoint_metric = float(val_loss.detach().cpu().item())
                model.train()
            if checkpoint_metric < best_epoch_loss:
                best_epoch_loss = checkpoint_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, best_epoch_loss if math.isfinite(best_epoch_loss) else float("nan")


def fit_pooled_response_models(
    trajectories: Sequence[LoggedTrajectory],
    T: int,
    state_normalizer: Optional[StateNormalizer],
    state_dim: int,
    action_dim: int,
    daily_budget: float,
    fit_cfg: SimulatorFitConfig,
    device: str,
    seed: int,
    previous_models: Optional[Sequence[ResponseModel]] = None,
    epochs_override: Optional[int] = None,
) -> Tuple[List[ResponseModel], Dict[str, float]]:
    """Fit one pooled response model with time features and expose interval lookup aliases."""
    flat = flatten_transitions(trajectories)
    if not flat:
        raise RuntimeError("No logged transitions available for pooled simulator fitting.")
    for t in range(int(T)):
        if not transitions_at_time(trajectories, t):
            raise RuntimeError(f"No logged transitions available for interval t={t}.")
    init_model = previous_models[0] if previous_models else None
    model, loss = _fit_response_model_from_transitions(
        flat,
        state_normalizer,
        state_dim,
        action_dim,
        daily_budget,
        fit_cfg,
        device,
        seed + 104729,
        model_init=init_model,
        epochs_override=epochs_override,
    )
    model.eval()
    models = [model for _ in range(int(T))]
    return models, {
        "main_sim_loss": float(loss),
        "main_sim_loss_std": 0.0,
        "main_sim_loss_min": float(loss),
        "main_sim_loss_max": float(loss),
        "simulator_type": "pooled_time_conditioned_multihead_nn",
        "simulator_input_dim": simulator_input_dim(state_dim, action_dim),
        "simulator_action_feature_dim": simulator_action_feature_dim(action_dim),
        "z_loss_weight": float(fit_cfg.z_loss_weight),
        "x_loss_weight": float(fit_cfg.x_loss_weight),
        "y_loss_weight": float(fit_cfg.y_loss_weight),
        "c_loss_weight": float(fit_cfg.c_loss_weight),
    }


def _compute_global_response_scale(
    trajectories: Sequence[LoggedTrajectory],
    output_dim: int,
    device: str,
) -> torch.Tensor:
    """Fixed W-scale used to put bootstrap uncertainty components on comparable units."""
    flat = flatten_transitions(trajectories)
    if not flat:
        return torch.ones(output_dim, device=device)
    w = torch.stack([tr.w for tr in flat], dim=0).to(device)
    scale = w.std(dim=0, unbiased=False)
    return torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-3)


def compute_response_scale_by_interval(
    trajectories: Sequence[LoggedTrajectory],
    T: int,
    output_dim: int,
    device: str,
) -> torch.Tensor:
    scales: List[torch.Tensor] = []
    global_scale = _compute_global_response_scale(trajectories, output_dim, device)
    for t in range(int(T)):
        transitions = transitions_at_time(trajectories, t)
        if not transitions:
            scales.append(global_scale.clone())
            continue
        w = torch.stack([tr.w for tr in transitions], dim=0).to(device)
        scale = w.std(dim=0, unbiased=False)
        scales.append(torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-3))
    return torch.stack(scales, dim=0)


def bootstrap_resample_trajectories(
    trajectories: Sequence[LoggedTrajectory],
    seed: int,
) -> List[LoggedTrajectory]:
    if not trajectories:
        raise ValueError("Cannot bootstrap-resample an empty trajectory list.")
    rng = random.Random(int(seed))
    return [trajectories[rng.randrange(len(trajectories))] for _ in range(len(trajectories))]


def fit_pooled_bootstrap_ensemble(
    trajectories: Sequence[LoggedTrajectory],
    T: int,
    state_normalizer: Optional[StateNormalizer],
    state_dim: int,
    action_dim: int,
    daily_budget: float,
    fit_cfg: SimulatorFitConfig,
    device: str,
    seed: int,
    b_boot: int,
    epochs: int,
    previous_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
    main_models: Optional[Sequence[ResponseModel]] = None,
) -> Tuple[List[List[ResponseModel]], Dict[str, float]]:
    """Fit a pooled day-bootstrap ensemble for epistemic uncertainty."""
    if b_boot <= 0:
        return [], {"bootstrap_loss_mean": float("nan"), "bootstrap_loss_std": float("nan")}
    flat = flatten_transitions(trajectories)
    if not flat:
        raise RuntimeError("No logged transitions available for pooled bootstrap ensemble.")
    losses: List[float] = []
    pooled_models: List[ResponseModel] = []
    for b in range(int(b_boot)):
        boot_data = bootstrap_resample_trajectories(trajectories, seed + 1009 * b)
        boot_transitions = flatten_transitions(boot_data)
        init_model: Optional[ResponseModel] = None
        if previous_ensemble is not None and previous_ensemble and b < len(previous_ensemble[0]):
            init_model = previous_ensemble[0][b]
        elif main_models is not None and main_models:
            init_model = main_models[0]
        model_b, loss_b = _fit_response_model_from_transitions(
            boot_transitions,
            state_normalizer,
            state_dim,
            action_dim,
            daily_budget,
            fit_cfg,
            device,
            seed + 7919 * (b + 1),
            model_init=init_model,
            epochs_override=epochs,
        )
        model_b.eval()
        pooled_models.append(model_b)
        losses.append(float(loss_b))
    ensemble_by_interval = [list(pooled_models) for _ in range(int(T))]
    return ensemble_by_interval, {
        "bootstrap_loss_mean": float(np.mean(losses)) if losses else float("nan"),
        "bootstrap_loss_std": float(np.std(losses)) if len(losses) > 1 else 0.0,
        "bootstrap_simulator_type": "pooled_time_conditioned_multihead_nn",
    }


class ResidualBootstrapSampler:
    """Day-level residual-path bootstrap for learned-simulator rollouts."""

    def __init__(
        self,
        residual_paths: torch.Tensor,
        residuals_by_t: Dict[int, torch.Tensor],
        output_dim: int,
        T: int,
        device: str,
        dtype: torch.dtype,
        seed: int,
    ):
        self.output_dim = int(output_dim)
        self.T = int(T)
        self.device = device
        self.dtype = dtype
        self.residual_paths = residual_paths.detach().to(device=device, dtype=dtype)
        self.residuals_by_t = {
            int(t): r.detach().to(device=device, dtype=dtype)
            for t, r in residuals_by_t.items()
            if r.numel() > 0
        }
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        self.current_path: Optional[torch.Tensor] = None
        self.interval_fallback_count = 0

    def fork(self, seed: int) -> "ResidualBootstrapSampler":
        return ResidualBootstrapSampler(
            residual_paths=self.residual_paths,
            residuals_by_t=self.residuals_by_t,
            output_dim=self.output_dim,
            T=self.T,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )

    def reset_episode(self) -> None:
        if self.residual_paths.numel() == 0 or self.residual_paths.shape[0] == 0:
            self.current_path = None
            return
        idx = torch.randint(0, self.residual_paths.shape[0], (), device=self.device, generator=self.generator)
        self.current_path = self.residual_paths[int(idx.detach().cpu().item())]

    def sample(self, t: int, shape: Tuple[int, ...] = ()) -> torch.Tensor:
        t = int(t)
        if self.residual_paths.numel() > 0 and self.residual_paths.shape[0] > 0:
            t_idx = min(max(t, 0), self.T - 1)
            if shape:
                self.interval_fallback_count += 1
            else:
                if self.current_path is None:
                    self.reset_episode()
                if self.current_path is not None:
                    return self.current_path[t_idx]

        pool = self.residuals_by_t.get(int(t))
        if pool is None or pool.shape[0] == 0:
            if shape and self.residual_paths.numel() > 0 and self.residual_paths.shape[0] > 0:
                t_idx = min(max(t, 0), self.T - 1)
                idx = torch.randint(0, self.residual_paths.shape[0], shape, device=self.device, generator=self.generator)
                return self.residual_paths[idx, t_idx, :]
            return torch.zeros((*shape, self.output_dim), device=self.device, dtype=self.dtype)
        idx = torch.randint(0, pool.shape[0], shape, device=self.device, generator=self.generator)
        return pool[idx]


@torch.no_grad()
def fit_residual_bootstrap_sampler(
    trajectories: Sequence[LoggedTrajectory],
    response_models: Sequence[ResponseModel],
    state_normalizer: Optional[StateNormalizer],
    daily_budget: float,
    device: str,
    seed: int,
    batch_size: int,
    center_residual: bool = True,
) -> Tuple[ResidualBootstrapSampler, Dict[str, float]]:
    flat = flatten_transitions(trajectories)
    if not flat:
        raise ValueError("Cannot fit residual bootstrap sampler from an empty log.")
    T = max((tr.t for tr in flat), default=-1) + 1

    raw_targets = torch.stack([tr.w for tr in flat], dim=0).to(device)
    times = torch.tensor([tr.t for tr in flat], device=device, dtype=torch.long)
    x, _ = build_response_tensors_from_transitions(flat, state_normalizer, daily_budget, device)

    pred = torch.empty_like(raw_targets)
    for t_val in torch.unique(times.detach().cpu()).tolist():
        mask = times == int(t_val)
        model_t = response_model_for_time(response_models, int(t_val))
        model_t.eval()
        x_t = x[mask]
        preds_t: List[torch.Tensor] = []
        for start in range(0, x_t.shape[0], max(int(batch_size), 1)):
            xb = x_t[start : start + max(int(batch_size), 1)]
            preds_t.append(model_t(xb).to(device))
        pred[mask] = torch.cat(preds_t, dim=0)
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Response model produced nonfinite residual-bootstrap predictions.")
    residuals = raw_targets - pred
    residuals_centered = residuals.clone()
    time_mean_vectors: List[torch.Tensor] = []
    for t_val in torch.unique(times.detach().cpu()).tolist():
        mask = times == int(t_val)
        if not bool(mask.any().detach().cpu().item()):
            continue
        mean_t = residuals[mask].mean(dim=0)
        residuals_centered[mask] = residuals[mask] - mean_t
        time_mean_vectors.append(mean_t)
    residuals_for_sampling = residuals_centered if bool(center_residual) else residuals

    residuals_by_t_lists: Dict[int, List[torch.Tensor]] = {}
    residual_paths: List[torch.Tensor] = []
    cursor = 0
    for traj in trajectories:
        if not traj:
            continue
        traj_resid = residuals_for_sampling[cursor : cursor + len(traj)]
        cursor += len(traj)
        if traj_resid.shape[0] != len(traj):
            continue
        path = torch.zeros((T, residuals.shape[-1]), device=device, dtype=residuals.dtype)
        seen = torch.zeros((T,), device=device, dtype=torch.bool)
        for local_idx, tr in enumerate(traj):
            t_val = int(tr.t)
            if 0 <= t_val < T:
                r = traj_resid[local_idx]
                path[t_val] = r
                seen[t_val] = True
                residuals_by_t_lists.setdefault(t_val, []).append(r)
        if bool(seen.all().detach().cpu().item()):
            residual_paths.append(path)

    residuals_by_t: Dict[int, torch.Tensor] = {}
    for t_val, vals in residuals_by_t_lists.items():
        if vals:
            residuals_by_t[int(t_val)] = torch.stack(vals, dim=0)
    for t_val in torch.unique(times.detach().cpu()).tolist():
        if int(t_val) not in residuals_by_t:
            mask = times == int(t_val)
            residuals_by_t[int(t_val)] = residuals_for_sampling[mask]
    residual_path_tensor = (
        torch.stack(residual_paths, dim=0)
        if residual_paths
        else torch.zeros((0, max(T, 1), raw_targets.shape[-1]), device=device, dtype=raw_targets.dtype)
    )

    mse_by_block = response_mse_blocks(pred, raw_targets)
    resid_var = residuals_for_sampling.pow(2).mean()
    if time_mean_vectors:
        time_mean_tensor = torch.stack(time_mean_vectors, dim=0)
        residual_mean_l1_raw = time_mean_tensor.abs().sum(dim=1).mean()
        residual_mean_linf_raw = time_mean_tensor.abs().max()
    else:
        residual_mean_l1_raw = torch.tensor(0.0, device=device, dtype=raw_targets.dtype)
        residual_mean_linf_raw = torch.tensor(0.0, device=device, dtype=raw_targets.dtype)
    metrics = {
        "center_residual": int(bool(center_residual)),
        "residual_pool_size": float(residuals.shape[0]),
        "residual_path_count": float(residual_path_tensor.shape[0]),
        "batched_residual_path_coherent": int(residual_path_tensor.shape[0] > 0),
        "residual_fallback_count": 0.0,
        "residual_mse_w_raw": to_float(mse_by_block["holdout_mse_w_raw"]),
        "residual_mse_z": to_float(mse_by_block["holdout_mse_z"]),
        "residual_mse_x_next": to_float(mse_by_block["holdout_mse_x_next"]),
        "residual_mse_y_next": to_float(mse_by_block["holdout_mse_y_next"]),
        "residual_mse_c_next": to_float(mse_by_block["holdout_mse_c_next"]),
        "residual_variance_mean": to_float(resid_var),
        "residual_mean_l1_raw": to_float(residual_mean_l1_raw),
        "residual_mean_linf_raw": to_float(residual_mean_linf_raw),
    }
    return (
        ResidualBootstrapSampler(
            residual_paths=residual_path_tensor,
            residuals_by_t=residuals_by_t,
            output_dim=raw_targets.shape[-1],
            T=max(T, 1),
            device=device,
            dtype=raw_targets.dtype,
            seed=seed,
        ),
        metrics,
    )


def response_mse_blocks(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    err2 = (pred - target).pow(2)
    m = target.shape[-1] // 4
    blocks = {
        "w_raw": slice(None),
        "z": slice(0, m),
        "x_next": slice(m, 2 * m),
        "y_next": slice(2 * m, 3 * m),
        "c_next": slice(3 * m, 4 * m),
    }
    out: Dict[str, float] = {}
    for name, sl in blocks.items():
        pred_b = pred[:, sl]
        target_b = target[:, sl]
        resid_b = pred_b - target_b
        mse = resid_b.pow(2).mean()
        var = target_b.var(unbiased=False)
        out[f"holdout_mse_{name}"] = to_float(mse)
        out[f"holdout_r2_{name}"] = to_float(1.0 - mse / var.clamp_min(EPS)) if to_float(var) > EPS else float("nan")
        out[f"holdout_residual_mean_{name}"] = to_float(resid_b.mean())
        out[f"holdout_residual_std_{name}"] = to_float(resid_b.std(unbiased=False))
        out[f"holdout_residual_p05_{name}"] = to_float(torch.quantile(resid_b.reshape(-1), 0.05))
        out[f"holdout_residual_p95_{name}"] = to_float(torch.quantile(resid_b.reshape(-1), 0.95))
    return out


def response_mse_blocks_with_prefix(pred: torch.Tensor, target: torch.Tensor, prefix: str = "holdout") -> Dict[str, float]:
    raw = response_mse_blocks(pred, target)
    if prefix == "holdout":
        return raw
    return {k.replace("holdout_", f"{prefix}_", 1): v for k, v in raw.items()}


@torch.no_grad()
def evaluate_response_holdout_mse(
    trajectories: Sequence[LoggedTrajectory],
    response_models: Sequence[ResponseModel],
    state_normalizer: Optional[StateNormalizer],
    daily_budget: float,
    device: str,
    batch_size: int,
) -> Dict[str, float]:
    """Evaluate raw W=(Z,X',Y',C') prediction MSE on a fixed holdout log."""
    flat = flatten_transitions(trajectories)
    if not flat:
        return {
            "holdout_mse_w_raw": float("nan"),
            "holdout_mse_z": float("nan"),
            "holdout_mse_x_next": float("nan"),
            "holdout_mse_y_next": float("nan"),
            "holdout_mse_c_next": float("nan"),
        }

    raw_targets = torch.stack([tr.w for tr in flat], dim=0).to(device)
    times = torch.tensor([tr.t for tr in flat], device=device, dtype=torch.long)
    x, _ = build_response_tensors_from_transitions(flat, state_normalizer, daily_budget, device)

    pred = torch.empty_like(raw_targets)
    for t_val in torch.unique(times.detach().cpu()).tolist():
        mask = times == int(t_val)
        model_t = response_model_for_time(response_models, int(t_val))
        model_t.eval()
        x_t = x[mask]
        preds_t: List[torch.Tensor] = []
        for start in range(0, x_t.shape[0], max(int(batch_size), 1)):
            xb = x_t[start : start + max(int(batch_size), 1)]
            preds_t.append(model_t(xb).to(device))
        pred[mask] = torch.cat(preds_t, dim=0)
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Response model produced nonfinite holdout predictions.")
    return response_mse_blocks(pred, raw_targets)


@torch.no_grad()
def evaluate_response_holdout_by_interval(
    trajectories: Sequence[LoggedTrajectory],
    response_models: Sequence[ResponseModel],
    state_normalizer: Optional[StateNormalizer],
    daily_budget: float,
    device: str,
    batch_size: int,
    round_idx: int,
) -> List[Dict[str, Any]]:
    """Per-interval one-step W diagnostics on the fixed holdout log."""
    rows: List[Dict[str, Any]] = []
    times_available = sorted({int(tr.t) for tr in flatten_transitions(trajectories)})
    for t_val in times_available:
        transitions_t = transitions_at_time(trajectories, t_val)
        if not transitions_t:
            continue
        raw_targets = torch.stack([tr.w for tr in transitions_t], dim=0).to(device)
        x_t, _ = build_response_tensors_from_transitions(transitions_t, state_normalizer, daily_budget, device)
        model_t = response_model_for_time(response_models, t_val)
        model_t.eval()
        preds_t: List[torch.Tensor] = []
        for start in range(0, x_t.shape[0], max(int(batch_size), 1)):
            xb = x_t[start : start + max(int(batch_size), 1)]
            preds_t.append(model_t(xb).to(device))
        pred = torch.cat(preds_t, dim=0)
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Response model produced nonfinite interval holdout predictions.")
        actions = torch.stack([tr.action for tr in transitions_t], dim=0).to(device)
        row: Dict[str, Any] = {
            "round": int(round_idx),
            "t": int(t_val),
            "tau_t": int(t_val) + 1,
            "holdout_transitions": len(transitions_t),
            "holdout_avg_spend": to_float(actions.sum(dim=-1).mean()),
            "holdout_avg_total_spend_share": to_float(actions.sum(dim=-1).mean() / max(float(daily_budget), EPS)),
        }
        metrics = response_mse_blocks(pred, raw_targets)
        row.update(metrics)
        for block in ["w_raw", "z", "x_next", "y_next", "c_next"]:
            row[f"mse_{block}"] = metrics.get(f"holdout_mse_{block}", float("nan"))
            row[f"r2_{block}"] = metrics.get(f"holdout_r2_{block}", float("nan"))
            row[f"residual_mean_{block}"] = metrics.get(f"holdout_residual_mean_{block}", float("nan"))
            row[f"residual_std_{block}"] = metrics.get(f"holdout_residual_std_{block}", float("nan"))
        rows.append(row)
    return rows


def summarize_interval_holdout_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Compact per-interval holdout diagnostics for round-level logs."""
    if not rows:
        return {
            "holdout_interval_count": 0.0,
            "holdout_mean_mse_z_by_t": float("nan"),
            "holdout_max_mse_z_by_t": float("nan"),
            "holdout_mean_r2_z_by_t": float("nan"),
            "holdout_min_r2_z_by_t": float("nan"),
            "holdout_worst_mse_z_t": float("nan"),
            "holdout_worst_r2_z_t": float("nan"),
        }

    def finite_values(key: str) -> List[Tuple[int, float]]:
        vals: List[Tuple[int, float]] = []
        for row in rows:
            val = to_float(row.get(key, float("nan")))
            if math.isfinite(val):
                vals.append((int(row.get("t", -1)), val))
        return vals

    out: Dict[str, float] = {"holdout_interval_count": float(len(rows))}
    for block in ["w_raw", "z", "x_next", "y_next", "c_next"]:
        mse_vals = finite_values(f"holdout_mse_{block}")
        r2_vals = finite_values(f"holdout_r2_{block}")
        if mse_vals:
            worst_mse_t, worst_mse = max(mse_vals, key=lambda item: item[1])
            out[f"holdout_mean_mse_{block}_by_t"] = float(np.mean([v for _, v in mse_vals]))
            out[f"holdout_max_mse_{block}_by_t"] = float(worst_mse)
            out[f"holdout_worst_mse_{block}_t"] = float(worst_mse_t)
        else:
            out[f"holdout_mean_mse_{block}_by_t"] = float("nan")
            out[f"holdout_max_mse_{block}_by_t"] = float("nan")
            out[f"holdout_worst_mse_{block}_t"] = float("nan")
        if r2_vals:
            worst_r2_t, worst_r2 = min(r2_vals, key=lambda item: item[1])
            out[f"holdout_mean_r2_{block}_by_t"] = float(np.mean([v for _, v in r2_vals]))
            out[f"holdout_min_r2_{block}_by_t"] = float(worst_r2)
            out[f"holdout_worst_r2_{block}_t"] = float(worst_r2_t)
        else:
            out[f"holdout_mean_r2_{block}_by_t"] = float("nan")
            out[f"holdout_min_r2_{block}_by_t"] = float("nan")
            out[f"holdout_worst_r2_{block}_t"] = float("nan")
    return out


def fit_round_simulator_bundle(
    logged: Sequence[LoggedTrajectory],
    validation_logged: Sequence[LoggedTrajectory],
    holdout_logged: Sequence[LoggedTrajectory],
    sim_cfg: SimConfig,
    state_normalizer: Optional[StateNormalizer],
    state_dim: int,
    action_dim: int,
    sim_fit_cfg: SimulatorFitConfig,
    pess_cfg: PessimismConfig,
    device: str,
    seed: int,
    q: int,
    sim_batch_size: int,
    sim_epochs_used: int,
    bootstrap_epochs_used: int,
    previous_models: Optional[Sequence[ResponseModel]],
    previous_ensemble: Optional[Sequence[Sequence[ResponseModel]]],
    center_residual: bool,
) -> RoundSimulatorBundle:
    """Fit/fine-tune the pooled time-conditioned simulator and diagnostics."""
    main_models, main_model_metrics = fit_pooled_response_models(
        logged,
        sim_cfg.Time_intervals,
        state_normalizer,
        state_dim,
        action_dim,
        sim_cfg.daily_budget,
        sim_fit_cfg,
        device,
        seed + 30_000 + q,
        previous_models=previous_models,
        epochs_override=sim_epochs_used,
    )
    bootstrap_ensemble, bootstrap_metrics = fit_pooled_bootstrap_ensemble(
        logged,
        sim_cfg.Time_intervals,
        state_normalizer,
        state_dim,
        action_dim,
        sim_cfg.daily_budget,
        sim_fit_cfg,
        device,
        seed + 32_000 + 100 * q,
        b_boot=pess_cfg.b_boot if pess_cfg.use_pessimism else 0,
        epochs=bootstrap_epochs_used,
        previous_ensemble=previous_ensemble,
        main_models=main_models,
    )
    train_residual_sampler, train_residual_metrics = fit_residual_bootstrap_sampler(
        logged,
        main_models,
        state_normalizer,
        sim_cfg.daily_budget,
        device,
        seed + 40_000 + 100 * q,
        sim_batch_size,
        center_residual=center_residual,
    )
    validation_residual_sampler, validation_residual_metrics = fit_residual_bootstrap_sampler(
        validation_logged,
        main_models,
        state_normalizer,
        sim_cfg.daily_budget,
        device,
        seed + 41_000 + 100 * q,
        sim_batch_size,
        center_residual=center_residual,
    )
    holdout_metrics = evaluate_response_holdout_mse(
        holdout_logged,
        main_models,
        state_normalizer,
        sim_cfg.daily_budget,
        device,
        sim_batch_size,
    )
    interval_holdout_rows = evaluate_response_holdout_by_interval(
        holdout_logged,
        main_models,
        state_normalizer,
        sim_cfg.daily_budget,
        device,
        sim_batch_size,
        round_idx=q,
    )
    holdout_metrics = {
        **holdout_metrics,
        **summarize_interval_holdout_rows(interval_holdout_rows),
    }
    return RoundSimulatorBundle(
        main_models=main_models,
        bootstrap_ensemble=bootstrap_ensemble,
        train_residual_sampler=train_residual_sampler,
        validation_residual_sampler=validation_residual_sampler,
        main_model_metrics=main_model_metrics,
        bootstrap_metrics=bootstrap_metrics,
        train_residual_metrics=train_residual_metrics,
        validation_residual_metrics=validation_residual_metrics,
        holdout_metrics=holdout_metrics,
        interval_holdout_rows=interval_holdout_rows,
        sim_epochs_used=sim_epochs_used,
        bootstrap_epochs_used=bootstrap_epochs_used if pess_cfg.use_pessimism else 0,
    )


class CandidateActionGenerator:
    """
    Shared finite feasible candidate set for DFL-PTO and MPC-3.
    """

    def __init__(self, action_dim: int, cfg: CandidateConfig, device: str, dtype: torch.dtype):
        self.M = action_dim
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        gen = torch.Generator(device=device)
        gen.manual_seed(int(cfg.seed))
        if cfg.random_count > 0:
            random_raw = torch.rand((cfg.random_count, action_dim), device=device, dtype=dtype, generator=gen)
            self.random_dirs = normalize_weights(random_raw)
            self.random_fracs = torch.rand((cfg.random_count,), device=device, dtype=dtype, generator=gen)
        else:
            self.random_dirs = torch.zeros((0, action_dim), device=device, dtype=dtype)
            self.random_fracs = torch.zeros((0,), device=device, dtype=dtype)
        self.budget_fracs = torch.tensor(tuple(cfg.budget_fracs), device=device, dtype=dtype).clamp(0.0, 1.0)

    def _deterministic_dirs(self, raw_state: torch.Tensor) -> torch.Tensor:
        X, Y, _, _, _ = split_raw_state(raw_state, self.M)
        ones = torch.ones(self.M, device=raw_state.device, dtype=raw_state.dtype)
        uniform = ones / max(self.M, 1)
        gap = normalize_weights((X - Y).clamp_min(0.0) + EPS)
        demand = normalize_weights(X.clamp_min(0.0) + EPS)
        shortage = normalize_weights(X.clamp_min(0.0) / Y.clamp_min(EPS))
        return torch.stack([uniform, gap, demand, shortage], dim=0)

    @torch.no_grad()
    def candidate_actions(self, raw_state: torch.Tensor) -> torch.Tensor:
        raw_state = raw_state.to(device=self.device, dtype=self.dtype)
        _, _, _, _, H = split_raw_state(raw_state, self.M)
        h = H.clamp_min(0.0)
        zero = torch.zeros((1, self.M), device=self.device, dtype=self.dtype)
        dirs = self._deterministic_dirs(raw_state)
        det = (self.budget_fracs[:, None, None] * h * dirs[None, :, :]).reshape(-1, self.M)
        rnd = self.random_fracs[:, None] * h * self.random_dirs
        out = torch.cat([zero, det, rnd], dim=0)
        return ensure_feasible_action(out, h)

    @torch.no_grad()
    def candidate_actions_batch(self, raw_states: torch.Tensor) -> torch.Tensor:
        raw_states = raw_states.to(device=self.device, dtype=self.dtype)
        X, Y, _, _, H = split_raw_state(raw_states, self.M)
        h = H.clamp_min(0.0)
        b = int(raw_states.shape[0])
        zero = torch.zeros((b, 1, self.M), device=self.device, dtype=self.dtype)

        uniform = torch.ones((b, self.M), device=self.device, dtype=self.dtype) / max(self.M, 1)
        gap = normalize_weights((X - Y).clamp_min(0.0) + EPS)
        demand = normalize_weights(X.clamp_min(0.0) + EPS)
        shortage = normalize_weights(X.clamp_min(0.0) / Y.clamp_min(EPS))
        dirs = torch.stack([uniform, gap, demand, shortage], dim=1)

        det = (
            self.budget_fracs.reshape(1, -1, 1, 1)
            * h.reshape(b, 1, 1, 1)
            * dirs.reshape(b, 1, 4, self.M)
        ).reshape(b, -1, self.M)
        rnd = self.random_fracs.reshape(1, -1, 1) * h.reshape(b, 1, 1) * self.random_dirs.reshape(1, -1, self.M)
        out = torch.cat([zero, det, rnd], dim=1)
        return ensure_feasible_action(out, h)


class OneStepScoreModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(input_dim, (hidden_dim, hidden_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def score_model_for_time(score_models: Sequence[OneStepScoreModel], t: int) -> OneStepScoreModel:
    if not score_models:
        raise ValueError("Empty per-time DFL score model family.")
    idx = min(max(int(t), 0), len(score_models) - 1)
    return score_models[idx]


def build_state_action_inputs(
    raw_states: torch.Tensor,
    actions: torch.Tensor,
    state_normalizer: Optional[StateNormalizer],
    daily_budget: float,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if raw_states.ndim == 1:
        raw_states = raw_states.unsqueeze(0)
    raw_states = raw_states.to(device=actions.device, dtype=actions.dtype)
    state_in = simulator_state_features(raw_states, state_normalizer, daily_budget).to(
        device=actions.device,
        dtype=actions.dtype,
    )
    actions_in = simulator_action_features(actions, daily_budget)
    if actions.ndim == 2:
        return torch.cat([state_in, actions_in], dim=-1), actions.shape[:-1]
    if actions.ndim == 3:
        b, k, m = actions.shape
        state_expanded = state_in[:, None, :].expand(b, k, state_in.shape[-1])
        return torch.cat([state_expanded.reshape(b * k, -1), actions_in.reshape(b * k, -1)], dim=-1), (b, k)
    raise ValueError(f"Unsupported action shape: {tuple(actions.shape)}")


@torch.no_grad()
def bootstrap_uncertainty_penalty(
    model_input: torch.Tensor,
    bootstrap_ensemble: Optional[Sequence[ResponseModel]],
    response_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """Z-block L1 bootstrap predictive std in fixed response-scale units."""
    if not bootstrap_ensemble or len(bootstrap_ensemble) < 2:
        return torch.zeros((model_input.shape[0],), device=model_input.device, dtype=model_input.dtype)
    preds = []
    for model in bootstrap_ensemble:
        model.eval()
        preds.append(model(model_input).to(device=model_input.device, dtype=model_input.dtype))
    stacked = torch.stack(preds, dim=0)
    if not torch.isfinite(stacked).all():
        raise FloatingPointError("Bootstrap ensemble produced nonfinite uncertainty predictions.")
    sd = stacked.std(dim=0, unbiased=True)
    if sd.shape[-1] % 4 != 0:
        raise ValueError(f"Response dimension must be 4*M for Z/X/Y/C blocks, got {sd.shape[-1]}.")
    z_dim = sd.shape[-1] // 4
    sd = sd[..., :z_dim]
    if response_scale is None:
        scale = torch.ones(sd.shape[-1], device=sd.device, dtype=sd.dtype)
    else:
        scale = response_scale.to(device=sd.device, dtype=sd.dtype)
        if scale.shape[-1] % 4 != 0:
            raise ValueError(f"Response scale dimension must be 4*M for Z/X/Y/C blocks, got {scale.shape[-1]}.")
        scale = scale[..., :z_dim].clamp_min(1e-3)
    return (sd / scale).abs().sum(dim=-1)


@torch.no_grad()
def learned_model_step_batch(
    cfg: SimConfig,
    raw_states: torch.Tensor,
    actions: torch.Tensor,
    t: int,
    response_models: Sequence[ResponseModel],
    state_normalizer: Optional[StateNormalizer],
    bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
    response_scale: Optional[torch.Tensor] = None,
    lambda_pess: float = 0.0,
    use_pessimism: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = actions.device
    dtype = actions.dtype
    raw_states = raw_states.to(device=device, dtype=dtype)
    actions = ensure_feasible_action(actions, raw_states[:, 4 * cfg.M])
    model_input, _ = build_state_action_inputs(raw_states, actions, state_normalizer, cfg.daily_budget)
    model_t = response_model_for_time(response_models, t)
    model_t.eval()
    w = model_t(model_input).to(device=device, dtype=dtype)
    if not torch.isfinite(w).all():
        raise FloatingPointError("Learned simulator produced nonfinite conditional-mean predictions.")
    z = w[:, : cfg.M]
    x_next = w[:, cfg.M : 2 * cfg.M]
    y_next = w[:, 2 * cfg.M : 3 * cfg.M]
    c_next = w[:, 3 * cfg.M : 4 * cfg.M]
    reward = z.sum(dim=-1) - actions.sum(dim=-1)
    uncertainty = bootstrap_uncertainty_penalty(
        model_input,
        bootstrap_ensemble_for_time(bootstrap_ensemble, t),
        response_scale_for_time(response_scale, t),
    )
    score = reward - float(lambda_pess) * uncertainty if use_pessimism else reward
    _, _, _, O, H = split_raw_state(raw_states, cfg.M)
    H_next = (H - actions.sum(dim=-1)).clamp_min(0.0)
    next_raw = raw_state_from_batch_parts(x_next, y_next, c_next, O, H_next, min(t + 1, cfg.Time_intervals - 1), cfg.Time_intervals)
    return next_raw, reward, score, uncertainty


@torch.no_grad()
def score_candidates_with_simulator(
    cfg: SimConfig,
    raw_states: torch.Tensor,
    actions: torch.Tensor,
    t: int,
    response_models: Sequence[ResponseModel],
    state_normalizer: Optional[StateNormalizer],
    bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
    response_scale: Optional[torch.Tensor] = None,
    lambda_pess: float = 0.0,
    use_pessimism: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, k, m = actions.shape
    state_flat = raw_states[:, None, :].expand(b, k, raw_states.shape[-1]).reshape(b * k, -1)
    action_flat = actions.reshape(b * k, m)
    _, reward, score, uncertainty = learned_model_step_batch(
        cfg,
        state_flat,
        action_flat,
        t,
        response_models,
        state_normalizer,
        bootstrap_ensemble=bootstrap_ensemble,
        response_scale=response_scale,
        lambda_pess=lambda_pess,
        use_pessimism=use_pessimism,
    )
    return reward.reshape(b, k), score.reshape(b, k), uncertainty.reshape(b, k)


@torch.no_grad()
def score_candidates_with_simulator_by_interval(
    cfg: SimConfig,
    raw_states: torch.Tensor,
    actions: torch.Tensor,
    times: torch.Tensor,
    response_models: Sequence[ResponseModel],
    state_normalizer: Optional[StateNormalizer],
    bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
    response_scale: Optional[torch.Tensor] = None,
    lambda_pess: float = 0.0,
    use_pessimism: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw_states.shape[0] != actions.shape[0] or raw_states.shape[0] != times.shape[0]:
        raise ValueError("raw_states, actions, and times must have the same batch dimension.")
    b, k, _ = actions.shape
    reward = torch.empty((b, k), device=actions.device, dtype=actions.dtype)
    score = torch.empty_like(reward)
    uncertainty = torch.empty_like(reward)
    for t_val in torch.unique(times.detach().to(device="cpu", dtype=torch.long)).tolist():
        mask = times == int(t_val)
        r_t, s_t, u_t = score_candidates_with_simulator(
            cfg,
            raw_states[mask],
            actions[mask],
            int(t_val),
            response_models,
            state_normalizer,
            bootstrap_ensemble=bootstrap_ensemble,
            response_scale=response_scale,
            lambda_pess=lambda_pess,
            use_pessimism=use_pessimism,
        )
        reward[mask] = r_t
        score[mask] = s_t
        uncertainty[mask] = u_t
    return reward, score, uncertainty


class LearnedSimulatorEnv:
    """
    Pooled time-conditioned learned simulator for synthetic rollouts.

    The shared response model predicts the conditional mean of W=(Z,X',Y',C')
    directly on the synthetic log scale. Each synthetic episode samples one
    logged-day residual path, preserving within-day residual dependence.
    """

    def __init__(
        self,
        cfg: SimConfig,
        initial_states: Sequence[torch.Tensor],
        response_models: Sequence[ResponseModel],
        state_normalizer: Optional[StateNormalizer],
        residual_sampler: ResidualBootstrapSampler,
        seed: int,
        bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
        response_scale: Optional[torch.Tensor] = None,
        lambda_pess: float = 0.0,
        use_pessimism: bool = False,
    ):
        self.cfg = cfg
        self.M = cfg.M
        self.T = cfg.Time_intervals
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.initial_states = [s.detach().to(device=self.device, dtype=self.dtype) for s in initial_states]
        self.response_models = list(response_models)
        self.state_normalizer = state_normalizer
        self.residual_sampler = residual_sampler
        self.bootstrap_ensemble = list(bootstrap_ensemble) if bootstrap_ensemble else []
        self.response_scale = response_scale
        self.lambda_pess = float(lambda_pess)
        self.use_pessimism = bool(use_pessimism)
        self.rng = random.Random(seed)

        self.t = 0
        self.X = torch.zeros(self.M, device=self.device, dtype=self.dtype)
        self.Y = torch.zeros(self.M, device=self.device, dtype=self.dtype)
        self.C = torch.zeros(self.M, device=self.device, dtype=self.dtype)
        self.O = torch.zeros(self.M, device=self.device, dtype=self.dtype)
        self.H = torch.tensor(float(cfg.daily_budget), device=self.device, dtype=self.dtype)

    @property
    def state_dim(self) -> int:
        return 4 * self.M + 1 + 3

    @property
    def action_dim(self) -> int:
        return self.M

    def _raw_state(self) -> torch.Tensor:
        return raw_state_from_parts(self.X, self.Y, self.C, self.O, self.H, self.t, self.T)

    def _policy_state(self) -> torch.Tensor:
        return normalize_state(self._raw_state(), self.state_normalizer).to(self.device)

    def reset(self) -> torch.Tensor:
        init = self.initial_states[self.rng.randrange(len(self.initial_states))].clone()
        self.residual_sampler.reset_episode()
        self.t = 0
        self.X = init[: self.M].clone()
        self.Y = init[self.M : 2 * self.M].clone()
        self.C = init[2 * self.M : 3 * self.M].clone()
        self.O = init[3 * self.M : 4 * self.M].clone()
        self.H = torch.tensor(float(self.cfg.daily_budget), device=self.device, dtype=self.dtype)
        return self._policy_state()

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        if self.t >= self.T:
            raise RuntimeError("Episode is already done. Call reset().")
        action = ensure_feasible_action(action.to(self.device), self.H)
        model_input, _ = build_state_action_inputs(
            self._raw_state(),
            action.unsqueeze(0),
            self.state_normalizer,
            self.cfg.daily_budget,
        )
        model_t = response_model_for_time(self.response_models, self.t)
        model_t.eval()
        w_mean = model_t(model_input).squeeze(0)
        if not torch.isfinite(w_mean).all():
            raise FloatingPointError("Learned simulator produced nonfinite conditional-mean predictions.")
        w = w_mean + self.residual_sampler.sample(self.t)
        if not torch.isfinite(w).all():
            raise FloatingPointError("Residual-bootstrap learned simulator produced nonfinite responses.")

        z = w[: self.M]
        x_next = w[self.M : 2 * self.M]
        y_next = w[2 * self.M : 3 * self.M]
        c_next = w[3 * self.M : 4 * self.M]

        raw_reward = z.sum() - action.sum()
        uncertainty = bootstrap_uncertainty_penalty(
            model_input,
            bootstrap_ensemble_for_time(self.bootstrap_ensemble, self.t),
            response_scale_for_time(self.response_scale, self.t),
        ).squeeze(0)
        reward = raw_reward - self.lambda_pess * uncertainty if self.use_pessimism else raw_reward

        self.X = x_next
        self.Y = y_next
        self.C = c_next
        self.H = torch.clamp(self.H - action.sum(), min=0.0)
        self.t += 1
        done = self.t >= self.T
        next_state = None if done else self._policy_state()
        info = {
            "Z": z.detach().clone(),
            "spend": action.sum().detach().clone(),
            "remaining_budget": self.H.detach().clone(),
            "reward": reward.detach().clone(),
            "raw_reward": raw_reward.detach().clone(),
            "uncertainty_penalty": uncertainty.detach().clone(),
            "t": self.t,
        }
        return next_state, reward.detach(), done, info


def make_torch_generator(device: str, seed: int) -> Tuple[torch.Generator, str]:
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen, str(device)
    except (RuntimeError, TypeError):
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        return gen, "cpu"


class BatchedLearnedSimulatorEnv:
    """
    Vectorized learned simulator for actor-critic synthetic rollouts.

    Each batch row is one synthetic market day. Residual paths are sampled once
    at reset and then indexed by time, preserving the day-level residual
    dependence used by the single-episode learned simulator.
    """

    def __init__(
        self,
        cfg: SimConfig,
        initial_states: Sequence[torch.Tensor],
        response_models: Sequence[ResponseModel],
        state_normalizer: Optional[StateNormalizer],
        residual_sampler: ResidualBootstrapSampler,
        seed: int,
        batch_size: int,
        bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
        response_scale: Optional[torch.Tensor] = None,
        lambda_pess: float = 0.0,
        use_pessimism: bool = False,
    ):
        if not initial_states:
            raise ValueError("BatchedLearnedSimulatorEnv requires at least one initial state.")
        self.cfg = cfg
        self.M = cfg.M
        self.T = cfg.Time_intervals
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.batch_size = int(batch_size)
        self.initial_state_tensor = torch.stack(
            [s.detach().to(device=self.device, dtype=self.dtype) for s in initial_states],
            dim=0,
        )
        self.response_models = list(response_models)
        self.state_normalizer = state_normalizer
        self.residual_sampler = residual_sampler
        self.bootstrap_ensemble = list(bootstrap_ensemble) if bootstrap_ensemble else []
        self.response_scale = response_scale
        self.lambda_pess = float(lambda_pess)
        self.use_pessimism = bool(use_pessimism)
        self.generator, self.generator_device = make_torch_generator(self.device, seed)

        self.t = 0
        self.X = torch.zeros((self.batch_size, self.M), device=self.device, dtype=self.dtype)
        self.Y = torch.zeros_like(self.X)
        self.C = torch.zeros_like(self.X)
        self.O = torch.zeros_like(self.X)
        self.H = torch.full((self.batch_size,), float(cfg.daily_budget), device=self.device, dtype=self.dtype)
        self.residual_path_idx: Optional[torch.Tensor] = None

    @property
    def state_dim(self) -> int:
        return 4 * self.M + 1 + 3

    @property
    def action_dim(self) -> int:
        return self.M

    def _randint(self, high: int, shape: Tuple[int, ...]) -> torch.Tensor:
        if high <= 0:
            raise ValueError("Cannot sample from an empty index set.")
        if self.generator_device == str(self.device):
            return torch.randint(0, high, shape, device=self.device, generator=self.generator)
        return torch.randint(0, high, shape, generator=self.generator).to(self.device)

    def _rand(self, shape: Tuple[int, ...]) -> torch.Tensor:
        if self.generator_device == str(self.device):
            return torch.rand(
                shape,
                device=self.device,
                dtype=self.dtype,
                generator=self.generator,
            )
        return torch.rand(
            shape,
            dtype=self.dtype,
            generator=self.generator,
        ).to(self.device)

    def _raw_state(self) -> torch.Tensor:
        return raw_state_from_batch_parts(self.X, self.Y, self.C, self.O, self.H, self.t, self.T)

    def _policy_state(self) -> torch.Tensor:
        return normalize_state(self._raw_state(), self.state_normalizer).to(device=self.device, dtype=self.dtype)

    def _sample_residual_paths(self) -> None:
        paths = self.residual_sampler.residual_paths
        if paths.numel() == 0 or paths.shape[0] == 0:
            self.residual_path_idx = None
            return
        self.residual_path_idx = self._randint(int(paths.shape[0]), (self.batch_size,))

    def _residuals_at_time(self, t: int) -> torch.Tensor:
        if self.residual_path_idx is not None:
            t_idx = min(max(int(t), 0), self.residual_sampler.T - 1)
            return self.residual_sampler.residual_paths[self.residual_path_idx, t_idx, :]
        return self.residual_sampler.sample(int(t), shape=(self.batch_size,))

    def reset(self) -> torch.Tensor:
        idx = self._randint(int(self.initial_state_tensor.shape[0]), (self.batch_size,))
        init = self.initial_state_tensor[idx].clone()
        self._sample_residual_paths()
        self.t = 0
        self.X = init[:, : self.M].clone()
        self.Y = init[:, self.M : 2 * self.M].clone()
        self.C = init[:, 2 * self.M : 3 * self.M].clone()
        self.O = init[:, 3 * self.M : 4 * self.M].clone()
        self.H = torch.full((self.batch_size,), float(self.cfg.daily_budget), device=self.device, dtype=self.dtype)
        return self._policy_state()

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        if self.t >= self.T:
            raise RuntimeError("Episode batch is already done. Call reset().")
        actions = ensure_feasible_action(actions.to(device=self.device, dtype=self.dtype), self.H)
        raw_state = self._raw_state()
        model_input, _ = build_state_action_inputs(
            raw_state,
            actions,
            self.state_normalizer,
            self.cfg.daily_budget,
        )
        model_t = response_model_for_time(self.response_models, self.t)
        model_t.eval()
        w_mean = model_t(model_input).to(device=self.device, dtype=self.dtype)
        if not torch.isfinite(w_mean).all():
            raise FloatingPointError("Batched learned simulator produced nonfinite conditional-mean predictions.")
        w = w_mean + self._residuals_at_time(self.t)
        if not torch.isfinite(w).all():
            raise FloatingPointError("Batched residual-bootstrap learned simulator produced nonfinite responses.")

        z = w[:, : self.M]
        x_next = w[:, self.M : 2 * self.M]
        y_next = w[:, 2 * self.M : 3 * self.M]
        c_next = w[:, 3 * self.M : 4 * self.M]
        spend = actions.sum(dim=-1)
        raw_reward = z.sum(dim=-1) - spend
        uncertainty = bootstrap_uncertainty_penalty(
            model_input,
            bootstrap_ensemble_for_time(self.bootstrap_ensemble, self.t),
            response_scale_for_time(self.response_scale, self.t),
        )
        reward = raw_reward - self.lambda_pess * uncertainty if self.use_pessimism else raw_reward

        self.X = x_next
        self.Y = y_next
        self.C = c_next
        self.H = (self.H - spend).clamp_min(0.0)
        self.t += 1
        done = self.t >= self.T
        next_state = None if done else self._policy_state()
        info = {
            "Z": z.detach(),
            "spend": spend.detach(),
            "remaining_budget": self.H.detach(),
            "reward": reward.detach(),
            "raw_reward": raw_reward.detach(),
            "uncertainty_penalty": uncertainty.detach(),
            "t": self.t,
        }
        return next_state, reward.detach(), done, info


class DFLPolicy:
    def __init__(
        self,
        score_models: Sequence[OneStepScoreModel],
        candidate_generator: CandidateActionGenerator,
        state_normalizer: Optional[StateNormalizer],
        daily_budget: float,
        bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
        response_scale: Optional[torch.Tensor] = None,
        lambda_pess: float = 0.0,
        use_pessimism: bool = False,
    ):
        self.score_models = list(score_models)
        self.candidate_generator = candidate_generator
        self.state_normalizer = state_normalizer
        self.daily_budget = float(daily_budget)
        self.bootstrap_ensemble = list(bootstrap_ensemble) if bootstrap_ensemble else []
        self.response_scale = response_scale
        self.lambda_pess = float(lambda_pess)
        self.use_pessimism = bool(use_pessimism)

    @torch.no_grad()
    def select_action_from_raw_state(self, raw_state: torch.Tensor, t: int) -> torch.Tensor:
        candidates = self.candidate_generator.candidate_actions(raw_state)
        model_input, shape = build_state_action_inputs(
            raw_state.unsqueeze(0),
            candidates.unsqueeze(0),
            self.state_normalizer,
            self.daily_budget,
        )
        score_model_t = score_model_for_time(self.score_models, t)
        score_model_t.eval()
        scores = score_model_t(model_input).reshape(shape)
        if self.use_pessimism:
            penalty = bootstrap_uncertainty_penalty(
                model_input,
                bootstrap_ensemble_for_time(self.bootstrap_ensemble, t),
                response_scale_for_time(self.response_scale, t),
            ).reshape(shape)
            scores = scores - self.lambda_pess * penalty
        idx = int(torch.argmax(scores.reshape(-1)).detach().cpu().item())
        return candidates[idx]

    def state_dict(self) -> Dict[str, Any]:
        return {"score_models": [m.state_dict() for m in self.score_models]}


class MPC3Policy:
    def __init__(
        self,
        cfg: SimConfig,
        response_models: Sequence[ResponseModel],
        state_normalizer: Optional[StateNormalizer],
        candidate_generator: CandidateActionGenerator,
        horizon: int = 3,
        bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
        response_scale: Optional[torch.Tensor] = None,
        lambda_pess: float = 0.0,
        use_pessimism: bool = False,
    ):
        self.cfg = cfg
        self.response_models = list(response_models)
        self.state_normalizer = state_normalizer
        self.candidate_generator = candidate_generator
        self.horizon = max(int(horizon), 1)
        self.bootstrap_ensemble = list(bootstrap_ensemble) if bootstrap_ensemble else []
        self.response_scale = response_scale
        self.lambda_pess = float(lambda_pess)
        self.use_pessimism = bool(use_pessimism)

    @torch.no_grad()
    def _greedy_actions(self, raw_states: torch.Tensor, t: int) -> torch.Tensor:
        candidates = self.candidate_generator.candidate_actions_batch(raw_states)
        _, scores, _ = score_candidates_with_simulator(
            self.cfg,
            raw_states,
            candidates,
            t,
            self.response_models,
            self.state_normalizer,
            bootstrap_ensemble=self.bootstrap_ensemble,
            response_scale=self.response_scale,
            lambda_pess=self.lambda_pess,
            use_pessimism=self.use_pessimism,
        )
        idx = scores.argmax(dim=-1)
        return candidates[torch.arange(candidates.shape[0], device=candidates.device), idx]

    @torch.no_grad()
    def select_action_from_raw_state(self, raw_state: torch.Tensor, t: int) -> torch.Tensor:
        raw_state = raw_state.to(device=self.cfg.device, dtype=self.cfg.dtype)
        candidates = self.candidate_generator.candidate_actions(raw_state)
        n = candidates.shape[0]
        raw_batch = raw_state.unsqueeze(0).expand(n, raw_state.shape[-1])
        next_states, _, total_score, _ = learned_model_step_batch(
            self.cfg,
            raw_batch,
            candidates,
            t,
            self.response_models,
            self.state_normalizer,
            bootstrap_ensemble=self.bootstrap_ensemble,
            response_scale=self.response_scale,
            lambda_pess=self.lambda_pess,
            use_pessimism=self.use_pessimism,
        )
        current_states = next_states
        current_t = t + 1
        for _ in range(1, self.horizon):
            if current_t >= self.cfg.Time_intervals:
                break
            continuation = self._greedy_actions(current_states, current_t)
            current_states, _, step_score, _ = learned_model_step_batch(
                self.cfg,
                current_states,
                continuation,
                current_t,
                self.response_models,
                self.state_normalizer,
                bootstrap_ensemble=self.bootstrap_ensemble,
                response_scale=self.response_scale,
                lambda_pess=self.lambda_pess,
                use_pessimism=self.use_pessimism,
            )
            total_score = total_score + step_score
            current_t += 1
        idx = int(torch.argmax(total_score).detach().cpu().item())
        return candidates[idx]


def fit_dfl_pto_policy(
    trajectories: Sequence[LoggedTrajectory],
    cfg: SimConfig,
    response_models: Sequence[ResponseModel],
    state_normalizer: Optional[StateNormalizer],
    candidate_generator: CandidateActionGenerator,
    state_dim: int,
    action_dim: int,
    dfl_cfg: DFLConfig,
    seed: int,
    q: int,
    log_every_epochs: int,
    bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
    response_scale: Optional[torch.Tensor] = None,
    lambda_pess: float = 0.0,
    use_pessimism: bool = False,
    early_stop_cfg: Optional[InnerEarlyStopConfig] = None,
    previous_policy: Optional[DFLPolicy] = None,
) -> Tuple[DFLPolicy, List[Dict[str, Any]]]:
    set_seed(seed)
    device = cfg.device
    score_models: List[OneStepScoreModel] = []
    logs: List[Dict[str, Any]] = []
    temp = max(float(dfl_cfg.softmax_temperature), 1e-4)
    early_stop_cfg = early_stop_cfg or InnerEarlyStopConfig(enabled=False)

    for t in range(cfg.Time_intervals):
        transitions_t = transitions_at_time(trajectories, t)
        if not transitions_t:
            raise RuntimeError(f"No logged transitions available for DFL interval t={t}.")
        raw_states = torch.stack([tr.state for tr in transitions_t], dim=0).to(device)
        actions = torch.stack([tr.action for tr in transitions_t], dim=0).to(device)
        rewards = torch.tensor([tr.reward for tr in transitions_t], device=device, dtype=torch.float32)
        score_model_t = OneStepScoreModel(simulator_input_dim(state_dim, action_dim), dfl_cfg.hidden_dim).to(device)
        if previous_policy is not None and t < len(previous_policy.score_models):
            score_model_t.load_state_dict(previous_policy.score_models[t].state_dict())
        opt = torch.optim.Adam(score_model_t.parameters(), lr=dfl_cfg.lr)
        n = raw_states.shape[0]
        last: Dict[str, float] = {}
        best_epoch_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_checks = 0
        stopped_epoch = dfl_cfg.epochs
        stopped = False

        for epoch in range(1, dfl_cfg.epochs + 1):
            perm = torch.randperm(n, device=device)
            epoch_losses: List[float] = []
            for start in range(0, n, dfl_cfg.batch_size):
                idx = perm[start : start + dfl_cfg.batch_size]
                s_batch = raw_states[idx]
                a_batch = actions[idx]
                r_batch = rewards[idx]
                x_logged, _ = build_state_action_inputs(s_batch, a_batch, state_normalizer, cfg.daily_budget)
                pred_logged = score_model_t(x_logged)
                pred_loss = F.mse_loss(pred_logged, r_batch)

                candidates = candidate_generator.candidate_actions_batch(s_batch)
                teacher_rewards, teacher_scores, teacher_penalty = score_candidates_with_simulator(
                    cfg,
                    s_batch,
                    candidates,
                    t,
                    response_models,
                    state_normalizer,
                    bootstrap_ensemble=bootstrap_ensemble,
                    response_scale=response_scale,
                    lambda_pess=lambda_pess,
                    use_pessimism=use_pessimism,
                )
                x_cand, shape = build_state_action_inputs(s_batch, candidates, state_normalizer, cfg.daily_budget)
                pred_scores = score_model_t(x_cand).reshape(shape)
                # The score model learns unpenalized one-step reward. The
                # pessimistic penalty is used only when ranking feasible
                # candidates for the teacher and final DFL policy action.
                pred_decision_scores = pred_scores - float(lambda_pess) * teacher_penalty if use_pessimism else pred_scores
                best_idx = teacher_scores.detach().argmax(dim=-1)
                decision_loss = F.cross_entropy(pred_decision_scores / temp, best_idx)
                loss = pred_loss + dfl_cfg.decision_weight * decision_loss
                candidate_spend = candidates.sum(dim=-1)
                selected_idx = pred_decision_scores.detach().argmax(dim=-1)
                teacher_spend = candidate_spend.gather(1, best_idx[:, None]).mean()
                selected_spend = candidate_spend.gather(1, selected_idx[:, None]).mean()
                teacher_score_range = (teacher_scores.max(dim=-1).values - teacher_scores.min(dim=-1).values).mean()
                teacher_penalty_mean = teacher_penalty.mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(score_model_t.parameters(), 5.0)
                opt.step()
                epoch_losses.append(to_float(loss))
                last = {
                    "time": t,
                    "dfl_loss": to_float(loss),
                    "dfl_pred_loss": to_float(pred_loss),
                    "dfl_decision_loss": to_float(decision_loss),
                    "dfl_teacher_candidate_spend": to_float(teacher_spend),
                    "dfl_selected_candidate_spend": to_float(selected_spend),
                    "dfl_candidate_score_range": to_float(teacher_score_range),
                    "dfl_candidate_penalty_mean": to_float(teacher_penalty_mean),
                }
            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            previous_best_epoch_loss = best_epoch_loss
            if math.isfinite(epoch_loss) and epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in score_model_t.state_dict().items()}
            should_check_stop = (
                early_stop_cfg.enabled
                and epoch % max(int(early_stop_cfg.eval_every_episodes), 1) == 0
                and math.isfinite(epoch_loss)
            )
            if should_check_stop:
                if epoch_loss < previous_best_epoch_loss - float(early_stop_cfg.min_delta):
                    bad_checks = 0
                else:
                    bad_checks += 1
                if bad_checks >= max(int(early_stop_cfg.patience), 1):
                    stopped_epoch = epoch
                    stopped = True
                    if best_state is not None:
                        score_model_t.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                    logs.append(
                        {
                            "round": q,
                            "method": LEARNING_METHOD_NAMES["dfl_pto"],
                            "method_key": "dfl_pto",
                            "episode": epoch,
                            "step": epoch,
                            "time": t,
                            "inner_early_stop_enabled": 1,
                            "inner_early_stopped": 1,
                            "inner_stop_epoch": stopped_epoch,
                            "inner_best_metric": best_epoch_loss,
                            **last,
                        }
                    )
                    break
            if epoch == dfl_cfg.epochs or epoch % max(log_every_epochs, 1) == 0:
                logs.append(
                    {
                        "round": q,
                        "method": LEARNING_METHOD_NAMES["dfl_pto"],
                        "method_key": "dfl_pto",
                        "episode": epoch,
                        "step": epoch,
                        "inner_early_stop_enabled": int(early_stop_cfg.enabled),
                        "inner_early_stopped": int(stopped),
                        "inner_stop_epoch": stopped_epoch if stopped else 0,
                        "inner_best_metric": best_epoch_loss if math.isfinite(best_epoch_loss) else epoch_loss,
                        **last,
                    }
                )
            if stopped:
                break
        if best_state is not None:
            score_model_t.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        score_model_t.eval()
        score_models.append(score_model_t)
    return (
        DFLPolicy(
            score_models,
            candidate_generator,
            state_normalizer,
            cfg.daily_budget,
            bootstrap_ensemble=bootstrap_ensemble,
            response_scale=response_scale,
            lambda_pess=lambda_pess,
            use_pessimism=use_pessimism,
        ),
        logs,
    )


def make_learning_agent(method: str, state_dim: int, action_dim: int, train_cfg: TrainConfig) -> Any:
    if method == "sacsa_reserve":
        return make_sacsa_reserve_agent(state_dim, action_dim, train_cfg)
    if method == "sac_logistic_normal_reserve":
        return make_sac_logistic_normal_reserve_agent(state_dim, action_dim, train_cfg)
    if method == "sacsa":
        return make_sacsa_agent(state_dim, action_dim, train_cfg)
    if method == "sac_logistic_normal":
        return make_sac_logistic_normal_agent(state_dim, action_dim, train_cfg)
    if method == "sac_gaussian_projection":
        return make_sac_gaussian_projection_agent(state_dim, action_dim, train_cfg)
    if method == "td3_projection":
        return TD3Agent(state_dim, action_dim, train_cfg)
    raise ValueError(f"Unknown learning method: {method}")


@torch.no_grad()
def agent_action(
    agent: Any,
    method: str,
    state: torch.Tensor,
    train_cfg: TrainConfig,
    deterministic: bool = False,
) -> torch.Tensor:
    if method == "td3_projection":
        noise_std = 0.0 if deterministic else train_cfg.exploration_noise
        return agent.act(state, noise_std=noise_std)
    return agent.act(state, deterministic=deterministic)


@torch.no_grad()
def agent_actions_batch(agent: Any, method: str, states: torch.Tensor, train_cfg: TrainConfig) -> torch.Tensor:
    if method == "td3_projection":
        if train_cfg.exploration_noise > 0:
            return agent.actor.sample_with_logit_noise(states, train_cfg.exploration_noise, train_cfg.noise_clip)
        return agent.actor(states)
    actions, _, _ = agent.actor.sample(states, deterministic=False)
    return actions


def collect_real_interaction_and_update_agent(
    cfg: SimConfig,
    coefs: Dict[str, torch.Tensor],
    episodes: int,
    seed: int,
    normalizer: Optional[StateNormalizer],
    agent: Any,
    method: str,
    train_cfg: TrainConfig,
    q: int,
    real_batch_size: int,
    real_update_after_steps: int,
    real_gradient_steps_per_step: int,
    real_exploration_epsilon: float = 0.0,
    logging_policy: str = "rich",
    logging_spend_scale: float = 1.0,
    audit: Optional[ExperimentAudit] = None,
    progress_mode: str = "none",
    progress_desc: str = "real environment online update",
) -> Tuple[List[LoggedTrajectory], List[Dict[str, Any]], Any]:
    """Collect true-DGP on-policy data and update actor-critic with true rewards.

    This is the model-free part of the hybrid pipeline. It does not use learned
    simulator residuals, bootstrap uncertainty, or pessimistic rewards.
    """
    if method not in ACTOR_CRITIC_METHODS:
        raise ValueError("Real online actor-critic update is only defined for actor-critic methods.")
    episodes = int(episodes)
    if episodes <= 0:
        return [], [], agent

    rb = ReplayBuffer(4 * cfg.M + 1 + 3, cfg.M, train_cfg.replay_size, train_cfg.device)
    logs: List[Dict[str, Any]] = []
    trajectories: List[LoggedTrajectory] = []
    global_step = 0
    updates_total = 0
    last_update_info: Dict[str, Any] = {}
    real_batch_size = max(int(real_batch_size), 1)
    real_update_after_steps = max(int(real_update_after_steps), 0)
    real_gradient_steps_per_step = max(int(real_gradient_steps_per_step), 0)
    real_exploration_epsilon = float(np.clip(real_exploration_epsilon, 0.0, 1.0))

    episode_iter = maybe_tqdm(range(1, episodes + 1), progress_mode, progress_desc, total=episodes)
    for episode in episode_iter:
        env = SubsidyAllocationEnv(cfg=cfg, coefs=coefs, seed=seed + episode - 1, project_action=False)
        raw_state = env.reset()
        traj: LoggedTrajectory = []
        ep_reward = 0.0
        ep_response_score = 0.0
        ep_spend = 0.0
        done = False
        while not done:
            t = env.t
            global_step += 1
            state = normalize_state(raw_state, normalizer).to(train_cfg.device)
            use_logging_exploration = (
                q > 1
                and real_exploration_epsilon > 0.0
                and random.random() < real_exploration_epsilon
            )
            if use_logging_exploration:
                action = mixed_logging_action(raw_state, env, logging_spend_scale, logging_policy)
                deploy_mode = "rich_logging_exploration"
            else:
                action = agent_action(agent, method, state, train_cfg, deterministic=True)
                deploy_mode = "deterministic_policy"
            action = ensure_feasible_action(action.to(env.device), env.H)
            audit_action_tensor(audit, "true_dgp_online_update", action, env.H, round=q, episode=episode, t=t)
            next_raw_state, reward, done, info = env.step(action)
            next_state = None if next_raw_state is None else normalize_state(next_raw_state, normalizer).to(train_cfg.device)
            rb.add(state, action.to(train_cfg.device), reward, next_state, done)

            w = torch.cat([info["Z"], env.X.detach(), env.Y.detach(), env.C.detach()], dim=0)
            traj.append(
                LoggedTransition(
                    state=detach_cpu(raw_state),
                    action=detach_cpu(action),
                    reward=to_float(reward),
                    next_state=None if next_raw_state is None else detach_cpu(next_raw_state),
                    done=done,
                    t=t,
                    w=detach_cpu(w),
                )
            )
            ep_reward += to_float(reward)
            ep_response_score += float(info["Z"].detach().sum().cpu().item())
            ep_spend += to_float(info["spend"])

            if (
                real_gradient_steps_per_step > 0
                and global_step >= real_update_after_steps
                and rb.size >= real_batch_size
            ):
                for _ in range(real_gradient_steps_per_step):
                    last_update_info = agent.update(rb.sample(real_batch_size))
                    updates_total += 1

            if next_raw_state is not None:
                raw_state = next_raw_state

        trajectories.append(traj)
        audit_episode_metrics(
            audit,
            "true_dgp_online_update",
            {
                "net_reward": ep_reward,
                "response_score": ep_response_score,
                "spend": ep_spend,
                "final_budget": to_float(env.H),
            },
            episode=episode,
            atol=1e-3,
        )
        if episode == episodes or episode % max(int(train_cfg.log_every_episodes), 1) == 0:
            logs.append(
                {
                    "round": q,
                    "method": LEARNING_METHOD_NAMES[method],
                    "method_key": method,
                    "episode": episode,
                    "step": global_step,
                    "training_source": "true_dgp_online",
                    "real_online_update": 1,
                    "real_reward_is_pessimistic": 0,
                    "real_finetune_return": ep_reward,
                    "real_finetune_response_score": ep_response_score,
                    "real_finetune_spend": ep_spend,
                    "real_finetune_final_budget": to_float(env.H),
                    "real_updates_total": updates_total,
                    "real_batch_size": real_batch_size,
                    "real_update_after_steps": real_update_after_steps,
                    "real_gradient_steps_per_step": real_gradient_steps_per_step,
                    "real_deploy_mode": deploy_mode,
                    "real_exploration_epsilon": real_exploration_epsilon,
                    **last_update_info,
                }
            )
    return trajectories, logs, agent


@torch.no_grad()
def random_paced_actions_batch(env: BatchedLearnedSimulatorEnv, safety: float = BUDGET_SAFETY) -> torch.Tensor:
    weights = env._rand((env.batch_size, env.action_dim))
    weights = normalize_weights(weights)
    planned_spend = (env.H / max(env.T - env.t, 1)).reshape(-1, 1)
    return weights * planned_spend * float(safety)


def train_agent_in_learned_simulator(
    agent: Any,
    method: str,
    env: LearnedSimulatorEnv,
    train_cfg: TrainConfig,
    q: int,
    early_stop_cfg: Optional[InnerEarlyStopConfig] = None,
    rollout_batch_size: int = 1,
    batch_rollout_update_cap: int = 128,
    audit: Optional[ExperimentAudit] = None,
    progress_mode: str = "none",
) -> Tuple[Any, List[Dict[str, Any]]]:
    if int(rollout_batch_size) > 1:
        return train_agent_in_batched_learned_simulator(
            agent,
            method,
            env,
            train_cfg,
            q,
            early_stop_cfg=early_stop_cfg,
            rollout_batch_size=int(rollout_batch_size),
            batch_rollout_update_cap=int(batch_rollout_update_cap),
            audit=audit,
            progress_mode=progress_mode,
        )

    rb = ReplayBuffer(env.state_dim, env.action_dim, train_cfg.replay_size, train_cfg.device)
    logs: List[Dict[str, Any]] = []
    global_step = 0
    last_update_info: Dict[str, Any] = {}
    early_stop_cfg = early_stop_cfg or InnerEarlyStopConfig(enabled=False)
    recent_returns: List[float] = []
    best_metric = -float("inf")
    bad_checks = 0
    stopped = False
    stop_episode = 0

    episode_iter = maybe_tqdm(
        range(1, train_cfg.train_episodes + 1),
        progress_mode,
        f"round {q} {method} train",
        total=train_cfg.train_episodes,
    )
    for episode in episode_iter:
        state = env.reset()
        ep_reward = 0.0
        ep_raw_reward = 0.0
        ep_uncertainty = 0.0
        done = False
        while not done:
            global_step += 1
            if global_step <= train_cfg.warmup_steps:
                action = random_paced_action(env, env.action_dim)
            else:
                action = agent_action(agent, method, state, train_cfg)
            action = ensure_feasible_action(action, env.H)
            audit_action_tensor(audit, "learned_simulator_training", action, env.H, round=q, episode=episode, step=global_step)
            next_state, reward, done, info = env.step(action)
            if audit is not None and audit.enabled and (not torch.isfinite(state).all() or not torch.isfinite(action).all()):
                audit.warn("nonfinite_training_tensor", "Training state/action contains NaN or Inf.", phase="learned_simulator_training", round=q, episode=episode)
            rb.add(state, action, reward, next_state, done)
            ep_reward += to_float(reward)
            ep_raw_reward += to_float(info.get("raw_reward", reward))
            ep_uncertainty += to_float(info.get("uncertainty_penalty", 0.0))

            if (
                global_step >= train_cfg.update_after_steps
                and rb.size >= train_cfg.batch_size
                and global_step % train_cfg.update_every_steps == 0
            ):
                for _ in range(train_cfg.gradient_steps_per_update):
                    last_update_info = agent.update(rb.sample(train_cfg.batch_size))

            if next_state is not None:
                state = next_state

        recent_returns.append(ep_reward)
        if (
            early_stop_cfg.enabled
            and episode % max(int(early_stop_cfg.eval_every_episodes), 1) == 0
            and recent_returns
        ):
            metric = float(np.mean(recent_returns[-max(int(early_stop_cfg.eval_every_episodes), 1) :]))
            if metric > best_metric + float(early_stop_cfg.min_delta):
                best_metric = metric
                bad_checks = 0
            else:
                bad_checks += 1
            if bad_checks >= max(int(early_stop_cfg.patience), 1):
                stopped = True
                stop_episode = episode

        if episode == train_cfg.train_episodes or episode % max(train_cfg.log_every_episodes, 1) == 0:
            logs.append(
                {
                    "round": q,
                    "method": LEARNING_METHOD_NAMES[method],
                    "method_key": method,
                    "episode": episode,
                    "step": global_step,
                    "synthetic_return": ep_reward,
                    "synthetic_raw_return": ep_raw_reward,
                    "synthetic_uncertainty_penalty": ep_uncertainty,
                    "inner_early_stop_enabled": int(early_stop_cfg.enabled),
                    "inner_early_stopped": int(stopped),
                    "inner_stop_episode": stop_episode,
                    "inner_best_metric": best_metric if math.isfinite(best_metric) else "",
                    "inner_bad_checks": bad_checks,
                    **last_update_info,
                }
            )
        if stopped:
            if not logs or int(logs[-1].get("episode", -1)) != episode:
                logs.append(
                    {
                        "round": q,
                        "method": LEARNING_METHOD_NAMES[method],
                        "method_key": method,
                        "episode": episode,
                        "step": global_step,
                        "synthetic_return": ep_reward,
                        "synthetic_raw_return": ep_raw_reward,
                        "synthetic_uncertainty_penalty": ep_uncertainty,
                        "inner_early_stop_enabled": 1,
                        "inner_early_stopped": 1,
                        "inner_stop_episode": stop_episode,
                        "inner_best_metric": best_metric,
                        "inner_bad_checks": bad_checks,
                        **last_update_info,
                    }
                )
            break
    return agent, logs


def train_agent_in_batched_learned_simulator(
    agent: Any,
    method: str,
    env: LearnedSimulatorEnv,
    train_cfg: TrainConfig,
    q: int,
    early_stop_cfg: Optional[InnerEarlyStopConfig] = None,
    rollout_batch_size: int = 1,
    batch_rollout_update_cap: int = 128,
    audit: Optional[ExperimentAudit] = None,
    progress_mode: str = "none",
) -> Tuple[Any, List[Dict[str, Any]]]:
    rb = ReplayBuffer(env.state_dim, env.action_dim, train_cfg.replay_size, train_cfg.device)
    logs: List[Dict[str, Any]] = []
    global_step = 0
    last_update_info: Dict[str, Any] = {}
    early_stop_cfg = early_stop_cfg or InnerEarlyStopConfig(enabled=False)
    recent_returns: List[float] = []
    best_metric = -float("inf")
    bad_checks = 0
    stopped = False
    stop_episode = 0
    total_episodes = int(train_cfg.train_episodes)
    batch_size = max(int(rollout_batch_size), 1)
    update_cap = max(int(batch_rollout_update_cap), int(train_cfg.gradient_steps_per_update))
    episode = 0
    next_log_episode = max(int(train_cfg.log_every_episodes), 1)
    next_early_stop_episode = max(int(early_stop_cfg.eval_every_episodes), 1)
    pbar = None
    if str(progress_mode).lower() == "tqdm":
        try:
            from tqdm.auto import tqdm  # type: ignore

            pbar = tqdm(total=total_episodes, desc=f"round {q} {method} batched train", leave=False)
        except Exception:
            pbar = None

    while episode < total_episodes:
        current_batch = min(batch_size, total_episodes - episode)
        boundary_candidates = [
            int(boundary)
            for boundary in (train_cfg.update_after_episodes, train_cfg.warmup_episodes)
            if episode < int(boundary) < episode + current_batch
        ]
        if boundary_candidates:
            current_batch = min(boundary_candidates) - episode
        episode_start = episode + 1
        batch_env = BatchedLearnedSimulatorEnv(
            cfg=env.cfg,
            initial_states=env.initial_states,
            response_models=env.response_models,
            state_normalizer=env.state_normalizer,
            residual_sampler=env.residual_sampler.fork(train_cfg.seed + 170_000 + 1000 * q + episode_start),
            seed=train_cfg.seed + 171_000 + 1000 * q + episode_start,
            batch_size=current_batch,
            bootstrap_ensemble=env.bootstrap_ensemble,
            response_scale=env.response_scale,
            lambda_pess=env.lambda_pess,
            use_pessimism=env.use_pessimism,
        )
        states = batch_env.reset()
        batch_in_warmup = episode_start <= int(train_cfg.warmup_episodes)
        ep_reward = torch.zeros((current_batch,), device=batch_env.device, dtype=batch_env.dtype)
        ep_raw_reward = torch.zeros_like(ep_reward)
        ep_uncertainty = torch.zeros_like(ep_reward)
        done = False

        while not done:
            if batch_in_warmup:
                actions = random_paced_actions_batch(batch_env)
            else:
                actions = agent_actions_batch(agent, method, states, train_cfg)
            actions = ensure_feasible_action(actions, batch_env.H)
            audit_action_tensor(audit, "batched_learned_simulator_training", actions, batch_env.H, round=q, episode_start=episode_start, t=batch_env.t)
            next_states, rewards, done, info = batch_env.step(actions)
            next_for_buffer = torch.zeros_like(states) if next_states is None else next_states
            done_tensor = torch.full((current_batch,), float(done), device=batch_env.device, dtype=batch_env.dtype)
            rb.add_batch(states, actions, rewards, next_for_buffer, done_tensor)

            ep_reward += rewards
            ep_raw_reward += info.get("raw_reward", rewards)
            ep_uncertainty += info.get("uncertainty_penalty", torch.zeros_like(rewards))
            global_step += current_batch

            if (
                episode_start > int(train_cfg.update_after_episodes)
                and rb.size >= train_cfg.batch_size
                and global_step % train_cfg.update_every_steps == 0
            ):
                updates_this_step = max(
                    int(train_cfg.gradient_steps_per_update),
                    min(update_cap, int(train_cfg.gradient_steps_per_update) * current_batch),
                )
                for _ in range(updates_this_step):
                    last_update_info = agent.update(rb.sample(train_cfg.batch_size))
                last_update_info = {
                    **last_update_info,
                    "batch_rollout_updates_this_step": updates_this_step,
                    "batch_rollout_update_cap": update_cap,
                }

            if next_states is not None:
                states = next_states

        episode += current_batch
        if pbar is not None:
            pbar.update(current_batch)
        recent_returns.extend(float(v) for v in ep_reward.detach().cpu().tolist())

        if early_stop_cfg.enabled:
            while episode >= next_early_stop_episode:
                window = max(int(early_stop_cfg.eval_every_episodes), 1)
                metric = float(np.mean(recent_returns[-window:])) if recent_returns else -float("inf")
                if metric > best_metric + float(early_stop_cfg.min_delta):
                    best_metric = metric
                    bad_checks = 0
                else:
                    bad_checks += 1
                if bad_checks >= max(int(early_stop_cfg.patience), 1):
                    stopped = True
                    stop_episode = episode
                    break
                next_early_stop_episode += window

        should_log = episode >= total_episodes or episode >= next_log_episode or stopped
        if should_log:
            logs.append(
                {
                    "round": q,
                    "method": LEARNING_METHOD_NAMES[method],
                    "method_key": method,
                    "episode": episode,
                    "step": global_step,
                    "synthetic_return": to_float(ep_reward.mean()),
                    "synthetic_raw_return": to_float(ep_raw_reward.mean()),
                    "synthetic_uncertainty_penalty": to_float(ep_uncertainty.mean()),
                    "rollout_batch_size": int(rollout_batch_size),
                    "actual_rollout_batch_size": int(current_batch),
                    "batched_rollout": 1,
                    "inner_early_stop_enabled": int(early_stop_cfg.enabled),
                    "inner_early_stopped": int(stopped),
                    "inner_stop_episode": stop_episode,
                    "inner_best_metric": best_metric if math.isfinite(best_metric) else "",
                    "inner_bad_checks": bad_checks,
                    **last_update_info,
                }
            )
            while next_log_episode <= episode:
                next_log_episode += max(int(train_cfg.log_every_episodes), 1)

        if stopped:
            break
    if pbar is not None:
        pbar.close()

    return agent, logs


@torch.no_grad()
def run_learned_simulator_policy_episode(
    env: LearnedSimulatorEnv,
    policy: Any,
    method: str,
    deterministic: bool = True,
) -> Dict[str, float]:
    state = env.reset()
    done = False
    total_reward = 0.0
    total_pess_reward = 0.0
    total_response_score = 0.0
    total_spend = 0.0
    total_uncertainty = 0.0

    while not done:
        if method == "no_subsidy":
            action = torch.zeros(env.M, device=env.device, dtype=env.dtype)
        elif method == "random_pacing":
            action = random_paced_action(env, env.M)
        elif method == "uniform_pacing":
            spend = env.H / max(env.T - env.t, 1)
            action = torch.ones(env.M, device=env.device, dtype=env.dtype) * (spend / env.M)
        elif method == "gap_heuristic":
            X, Y, _, _, _ = split_raw_state(env._raw_state(), env.M)
            weights = normalize_weights((X - Y).clamp_min(0.0) + EPS)
            spend = env.H / max(env.T - env.t, 1)
            action = spend * weights
        elif hasattr(policy, "select_action_from_raw_state"):
            action = policy.select_action_from_raw_state(env._raw_state(), env.t)
        elif method in ACTOR_CRITIC_METHODS and method != "td3_projection":
            action, _, _ = policy.sample(state.unsqueeze(0), deterministic=deterministic)
            action = action.squeeze(0)
        elif method == "td3_projection":
            action = policy(state.unsqueeze(0)).squeeze(0)
        else:
            raise ValueError(f"Unknown learned-simulator validation method: {method}")

        action = ensure_feasible_action(action, env.H)
        next_state, _, done, info = env.step(action)
        total_reward += to_float(info.get("raw_reward", info["reward"]))
        total_pess_reward += to_float(info["reward"])
        total_uncertainty += to_float(info.get("uncertainty_penalty", 0.0))
        total_response_score += float(info["Z"].detach().sum().cpu().item())
        total_spend += to_float(info["spend"])
        if next_state is not None:
            state = next_state

    return {
        "net_reward": total_reward,
        "net_reward_pess": total_pess_reward,
        "uncertainty_penalty": total_uncertainty,
        "response_score": total_response_score,
        "spend": total_spend,
        "final_budget": to_float(env.H),
    }


@torch.no_grad()
def evaluate_learned_simulator_summary(
    cfg: SimConfig,
    initial_states: Sequence[torch.Tensor],
    response_models: Sequence[ResponseModel],
    state_normalizer: Optional[StateNormalizer],
    actor: Any,
    method: str,
    episodes: int,
    seed: int,
    residual_sampler: ResidualBootstrapSampler,
    bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
    response_scale: Optional[torch.Tensor] = None,
    lambda_pess: float = 0.0,
    use_pessimism: bool = False,
) -> Dict[str, float]:
    rows: List[Dict[str, Any]] = []
    for ep in range(episodes):
        env = LearnedSimulatorEnv(
            cfg=cfg,
            initial_states=initial_states,
            response_models=response_models,
            state_normalizer=state_normalizer,
            residual_sampler=residual_sampler,
            seed=seed + ep,
            bootstrap_ensemble=bootstrap_ensemble,
            response_scale=response_scale,
            lambda_pess=lambda_pess,
            use_pessimism=use_pessimism,
        )
        rows.append(run_learned_simulator_policy_episode(env, actor, method, deterministic=True))
    summary = summarize_eval_rows(rows)
    return {
        k: v
        for k, v in summary.items()
        if k.endswith("_se")
        or k in {
            "net_reward",
            "net_reward_pess",
            "uncertainty_penalty",
            "response_score",
            "spend",
            "final_budget",
        }
    }


@torch.no_grad()
def evaluate_paired_validation_set(
    cfg: SimConfig,
    initial_states: Sequence[torch.Tensor],
    response_models: Sequence[ResponseModel],
    state_normalizer: Optional[StateNormalizer],
    actor: Any,
    method: str,
    episodes: int,
    seed: int,
    residual_sampler: ResidualBootstrapSampler,
    bootstrap_ensemble: Optional[Sequence[Sequence[ResponseModel]]] = None,
    response_scale: Optional[torch.Tensor] = None,
    lambda_pess: float = 0.0,
    use_pessimism: bool = False,
    residual_seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    policies: List[Tuple[str, str, Any]] = [
        ("policy", method, actor),
        ("no_subsidy", "no_subsidy", None),
        ("random", "random_pacing", None),
        ("uniform", "uniform_pacing", None),
        ("gap", "gap_heuristic", None),
    ]
    out: Dict[str, Dict[str, float]] = {}
    for label, method_key, policy_obj in policies:
        out[label] = evaluate_learned_simulator_summary(
            cfg=cfg,
            initial_states=initial_states,
            response_models=response_models,
            state_normalizer=state_normalizer,
            actor=policy_obj,
            method=method_key,
            episodes=episodes,
            seed=seed,
            residual_sampler=residual_sampler.fork(int(residual_seed)),
            bootstrap_ensemble=bootstrap_ensemble,
            response_scale=response_scale,
            lambda_pess=lambda_pess,
            use_pessimism=use_pessimism,
        )
    return out


@torch.no_grad()
def run_true_dgp_policy_episode(
    cfg: SimConfig,
    coefs: Dict[str, torch.Tensor],
    policy: Any,
    method: str,
    normalizer: Optional[StateNormalizer],
    seed: int,
    deterministic: bool = True,
    collect_steps: bool = False,
    audit: Optional[ExperimentAudit] = None,
    audit_phase: str = "true_dgp_eval",
    audit_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    env = SubsidyAllocationEnv(cfg=cfg, coefs=coefs, seed=seed, project_action=False)
    raw_state = env.reset()
    done = False
    total_reward = 0.0
    total_response_score = 0.0
    total_spend = 0.0
    projection_count = 0.0
    projection_distance = 0.0
    preprojection_budget_excess = 0.0
    step_rows: List[Dict[str, float]] = []

    while not done:
        t = env.t
        if method == "no_subsidy":
            action = torch.zeros(cfg.M, device=env.device, dtype=env.dtype)
        elif method == "random_pacing":
            action = random_paced_action(env, cfg.M)
        elif method == "uniform_pacing":
            spend = env.H / max(env.T - env.t, 1)
            action = torch.ones(cfg.M, device=env.device, dtype=env.dtype) * (spend / cfg.M)
        elif method == "gap_heuristic":
            X, Y, _, _, _ = split_raw_state(raw_state, cfg.M)
            weights = normalize_weights((X - Y).clamp_min(0.0) + EPS)
            spend = env.H / max(env.T - env.t, 1)
            action = spend * weights
        else:
            action = policy_action(policy, raw_state, normalizer, env, deterministic)
            projection_flag = getattr(policy, "last_projection_applied", None)
            projection_dist = getattr(policy, "last_projection_distance", None)
            projection_excess = getattr(policy, "last_preprojection_budget_excess", None)
            if isinstance(projection_flag, torch.Tensor):
                projection_count += to_float(projection_flag.mean())
            if isinstance(projection_dist, torch.Tensor):
                projection_distance += to_float(projection_dist.mean())
            if isinstance(projection_excess, torch.Tensor):
                preprojection_budget_excess += to_float(projection_excess.mean())
        action = ensure_feasible_action(action.to(env.device), env.H)
        context = dict(audit_context or {})
        context.update({"t": t, "method": method, "seed": seed})
        audit_action_tensor(audit, audit_phase, action, env.H, **context)
        next_state, reward, done, info = env.step(action)
        if audit is not None and audit.enabled:
            if not torch.isfinite(raw_state).all():
                audit.warn("nonfinite_state", "True-DGP raw state contains NaN or Inf.", phase=audit_phase, **context)
            if next_state is not None and not torch.isfinite(next_state).all():
                audit.warn("nonfinite_next_state", "True-DGP next state contains NaN or Inf.", phase=audit_phase, **context)
            audit_episode_metrics(
                audit,
                audit_phase,
                {
                    "net_reward": to_float(reward),
                    "response_score": float(info["Z"].detach().sum().cpu().item()),
                    "spend": to_float(info["spend"]),
                    "final_budget": to_float(info["remaining_budget"]),
                },
                **context,
            )
        total_reward += to_float(reward)
        total_response_score += float(info["Z"].detach().sum().cpu().item())
        total_spend += to_float(info["spend"])
        if collect_steps:
            step_rows.append(
                {
                    "t": t,
                    "step_net_reward": to_float(reward),
                    "step_response_score": float(info["Z"].detach().sum().cpu().item()),
                    "step_spend": to_float(info["spend"]),
                    "remaining_budget_after": to_float(info["remaining_budget"]),
                    "action_mean": to_float(action.mean()),
                    "action_max": to_float(action.max()),
                    "action_nonzero_share": to_float((action > EPS).to(dtype=action.dtype).mean()),
                }
            )
        if next_state is not None:
            raw_state = next_state

    out: Dict[str, Any] = {
        "net_reward": total_reward,
        "response_score": total_response_score,
        "spend": total_spend,
        "final_budget": to_float(env.H),
        "projection_rate": projection_count / max(int(cfg.Time_intervals), 1),
        "mean_projection_distance": projection_distance / max(int(cfg.Time_intervals), 1),
        "mean_preprojection_budget_excess": preprojection_budget_excess / max(int(cfg.Time_intervals), 1),
    }
    audit_episode_metrics(audit, audit_phase, out, method=method, seed=seed, **(audit_context or {}))
    if collect_steps:
        out["_step_rows"] = step_rows
    return out


@torch.no_grad()
def evaluate_true_dgp_policy_raw(
    cfg: SimConfig,
    coefs: Dict[str, torch.Tensor],
    policy: Any,
    method: str,
    method_name: str,
    episodes: int,
    seed: int,
    normalizer: Optional[StateNormalizer],
    seed_offset: int,
    checkpoint_episode: int,
    checkpoint_step: int,
    stage: str,
    step_rows_out: Optional[List[Dict[str, Any]]] = None,
    audit: Optional[ExperimentAudit] = None,
    progress_mode: str = "none",
    progress_desc: str = "true-DGP eval",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ep in maybe_tqdm(range(episodes), progress_mode, progress_desc, total=episodes):
        eval_seed = seed + seed_offset + ep
        out = run_true_dgp_policy_episode(
            cfg,
            coefs,
            policy,
            method,
            normalizer,
            eval_seed,
            deterministic=True,
            collect_steps=step_rows_out is not None,
            audit=audit,
            audit_phase=stage,
            audit_context={"eval_episode": ep, "checkpoint_episode": checkpoint_episode, "checkpoint_step": checkpoint_step},
        )
        episode_step_rows = out.pop("_step_rows", [])
        if step_rows_out is not None:
            for step_row in episode_step_rows:
                step_rows_out.append(
                    {
                        "stage": stage,
                        "method": method_name,
                        "method_key": method,
                        "eval_episode": ep,
                        "eval_seed": eval_seed,
                        "checkpoint_episode": checkpoint_episode,
                        "checkpoint_step": checkpoint_step,
                        **step_row,
                    }
                )
        rows.append(
            {
                "stage": stage,
                "method": method_name,
                "method_key": method,
                "eval_episode": ep,
                "eval_seed": eval_seed,
                "checkpoint_episode": checkpoint_episode,
                "checkpoint_step": checkpoint_step,
                **out,
            }
        )
    return rows


def evaluate_heuristic_raw_rows(
    cfg: SimConfig,
    coefs: Dict[str, torch.Tensor],
    normalizer: Optional[StateNormalizer],
    episodes: int,
    seed: int,
    seed_offset: int,
    stage: str = "final_true_dgp_eval",
    checkpoint_episode: int = 0,
    checkpoint_step: int = 0,
    step_rows_out: Optional[List[Dict[str, Any]]] = None,
    audit: Optional[ExperimentAudit] = None,
    progress_mode: str = "none",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, name in TRUE_DGP_BASELINE_POLICIES:
        rows.extend(
            evaluate_true_dgp_policy_raw(
                cfg=cfg,
                coefs=coefs,
                policy=None,
                method=method,
                method_name=name,
                episodes=episodes,
                seed=seed,
                normalizer=normalizer,
                seed_offset=seed_offset,
                checkpoint_episode=checkpoint_episode,
                checkpoint_step=checkpoint_step,
                stage=stage,
                step_rows_out=step_rows_out,
                audit=audit,
                progress_mode=progress_mode,
                progress_desc=f"{stage} {method}",
            )
        )
    return rows


def checkpoint_step_for_round(method: str, q: int, train_cfg: TrainConfig, args: argparse.Namespace) -> int:
    if method in ACTOR_CRITIC_METHODS:
        return int(q * train_cfg.train_steps)
    if method == "dfl_pto":
        return int(q * args.dfl_epochs)
    return int(q)


def evaluate_round_true_dgp_rows(
    cfg: SimConfig,
    coefs: Dict[str, torch.Tensor],
    normalizer: Optional[StateNormalizer],
    policy: Any,
    method: str,
    method_name: str,
    episodes: int,
    seed: int,
    seed_offset: int,
    round_idx: int,
    checkpoint_step: int,
    stage: str = "round_true_dgp_eval",
    step_rows_out: Optional[List[Dict[str, Any]]] = None,
    audit: Optional[ExperimentAudit] = None,
    progress_mode: str = "none",
) -> List[Dict[str, Any]]:
    rows = evaluate_heuristic_raw_rows(
        cfg,
        coefs,
        normalizer,
        episodes=episodes,
        seed=seed,
        seed_offset=seed_offset,
        stage=stage,
        checkpoint_episode=round_idx,
        checkpoint_step=checkpoint_step,
        step_rows_out=step_rows_out,
        audit=audit,
        progress_mode=progress_mode,
    )
    rows.extend(
        evaluate_true_dgp_policy_raw(
            cfg=cfg,
            coefs=coefs,
            policy=policy,
            method=method,
            method_name=method_name,
            episodes=episodes,
            seed=seed,
            normalizer=normalizer,
            seed_offset=seed_offset,
            checkpoint_episode=round_idx,
            checkpoint_step=checkpoint_step,
            stage=stage,
            step_rows_out=step_rows_out,
            audit=audit,
            progress_mode=progress_mode,
            progress_desc=f"{stage} {method}",
        )
    )
    for row in rows:
        row["round"] = round_idx
        row["evaluation_source"] = "independent_true_dgp_paired_seeds"
    if step_rows_out is not None:
        for step_row in step_rows_out:
            if "round" not in step_row:
                step_row["round"] = round_idx
            if "evaluation_source" not in step_row:
                step_row["evaluation_source"] = "independent_true_dgp_paired_seeds"
    return rows


def summarize_round_eval_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row.get("round", 0)), str(row.get("method", ""))), []).append(row)
    for (round_idx, method_name), group_rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        summary.append(
            {
                "round": round_idx,
                "method": method_name,
                "method_key": str(group_rows[0].get("method_key", "")) if group_rows else "",
                "eval_episodes": len(group_rows),
                **summarize_eval_rows(group_rows),
            }
        )
    return summary


def summarize_simulator_rollout_diagnostics(
    round_idx: int,
    source_method_key: str,
    val_set: Dict[str, Dict[str, float]],
    round_eval_rows: Sequence[Dict[str, Any]],
    holdout_metrics: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Compare learned-simulator rollout diagnostics with true-DGP round eval.

    The comparison is diagnostic only. Learned-simulator validation uses fixed
    held-out initial states and residual paths, while true-DGP evaluation uses
    independent paired seeds.
    """
    label_to_true_key = {
        "policy": source_method_key,
        "no_subsidy": "no_subsidy",
        "random": "random_pacing",
        "uniform": "uniform_pacing",
        "gap": "gap_heuristic",
    }
    true_summary = {
        str(row.get("method_key", "")): row
        for row in summarize_round_eval_rows(round_eval_rows)
    }
    out: List[Dict[str, Any]] = []
    for label, sim_summary in val_set.items():
        true_key = label_to_true_key.get(label, label)
        true_row = true_summary.get(true_key, {})
        out.append(
            {
                "round": round_idx,
                "source_method_key": source_method_key,
                "diagnostic_policy": label,
                "true_method_key": true_key,
                "simulator_diagnostic_source": "learned_simulator_validation_rollout",
                "true_eval_source": "independent_true_dgp_paired_seeds",
                "sim_net_reward": sim_summary.get("net_reward", float("nan")),
                "sim_net_reward_pess": sim_summary.get("net_reward_pess", float("nan")),
                "true_net_reward": true_row.get("net_reward", float("nan")),
                "sim_minus_true_net_reward": float(sim_summary.get("net_reward", float("nan")))
                - float(true_row.get("net_reward", float("nan"))),
                "sim_response_score": sim_summary.get("response_score", float("nan")),
                "true_response_score": true_row.get("response_score", float("nan")),
                "sim_minus_true_response_score": float(sim_summary.get("response_score", float("nan")))
                - float(true_row.get("response_score", float("nan"))),
                "sim_spend": sim_summary.get("spend", float("nan")),
                "true_spend": true_row.get("spend", float("nan")),
                "sim_minus_true_spend": float(sim_summary.get("spend", float("nan")))
                - float(true_row.get("spend", float("nan"))),
                **holdout_metrics,
            }
        )
    return out


def action_support_feature_vector(action: torch.Tensor, daily_budget: float) -> torch.Tensor:
    action = action.detach().to(dtype=torch.float32).clamp_min(0.0)
    denom = max(float(daily_budget), EPS)
    action_over_b = action / denom
    spend_over_b = action_over_b.sum()
    max_action_over_b = action_over_b.max() if action_over_b.numel() else torch.zeros((), dtype=torch.float32)
    p = action / action.sum().clamp_min(EPS)
    entropy = -(p * torch.log(p.clamp_min(EPS))).sum()
    hhi = p.pow(2).sum()
    nonzero_share = (action > EPS).to(dtype=torch.float32).mean()
    return torch.stack([spend_over_b, max_action_over_b, entropy, hhi, nonzero_share])


def action_support_metric_dict(prefix: str, values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
        }
    vals = values.detach().cpu().numpy().astype(np.float64)
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_p05": float(np.percentile(vals, 5)),
        f"{prefix}_p50": float(np.percentile(vals, 50)),
        f"{prefix}_p95": float(np.percentile(vals, 95)),
    }


@torch.no_grad()
def collect_policy_action_support_by_time(
    cfg: SimConfig,
    coefs: Dict[str, torch.Tensor],
    policy: Any,
    method: str,
    normalizer: Optional[StateNormalizer],
    episodes: int,
    seed: int,
    seed_offset: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    by_time: Dict[int, Dict[str, List[torch.Tensor]]] = {
        t: {"actions": [], "features": []} for t in range(int(cfg.Time_intervals))
    }
    for ep in range(int(episodes)):
        env = SubsidyAllocationEnv(cfg=cfg, coefs=coefs, seed=seed + seed_offset + ep, project_action=False)
        raw_state = env.reset()
        done = False
        while not done:
            t = int(env.t)
            if method == "no_subsidy":
                action = torch.zeros(cfg.M, device=env.device, dtype=env.dtype)
            elif method == "random_pacing":
                action = random_paced_action(env, cfg.M)
            elif method == "uniform_pacing":
                spend = env.H / max(env.T - env.t, 1)
                action = torch.ones(cfg.M, device=env.device, dtype=env.dtype) * (spend / cfg.M)
            elif method == "gap_heuristic":
                X, Y, _, _, _ = split_raw_state(raw_state, cfg.M)
                weights = normalize_weights((X - Y).clamp_min(0.0) + EPS)
                spend = env.H / max(env.T - env.t, 1)
                action = spend * weights
            else:
                action = policy_action(policy, raw_state, normalizer, env, deterministic=True)
            action = ensure_feasible_action(action.to(env.device), env.H)
            by_time[t]["actions"].append(detach_cpu(action / max(float(cfg.daily_budget), EPS)))
            by_time[t]["features"].append(action_support_feature_vector(action, cfg.daily_budget).cpu())
            next_state, _, done, _ = env.step(action)
            if next_state is not None:
                raw_state = next_state
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for t, parts in by_time.items():
        actions = parts["actions"]
        features = parts["features"]
        out[t] = {
            "actions": torch.stack(actions, dim=0) if actions else torch.empty(0, cfg.M),
            "features": torch.stack(features, dim=0) if features else torch.empty(0, 5),
        }
    return out


def compute_policy_support_diagnostics(
    logged: Sequence[LoggedTrajectory],
    cfg: SimConfig,
    coefs: Dict[str, torch.Tensor],
    policy: Any,
    method: str,
    method_name: str,
    normalizer: Optional[StateNormalizer],
    episodes: int,
    seed: int,
    seed_offset: int,
    round_idx: int,
) -> List[Dict[str, Any]]:
    logged_by_time: Dict[int, Dict[str, List[torch.Tensor]]] = {
        t: {"actions": [], "features": []} for t in range(int(cfg.Time_intervals))
    }
    for tr in flatten_transitions(logged):
        t = int(tr.t)
        action = tr.action.to(dtype=torch.float32)
        logged_by_time.setdefault(t, {"actions": [], "features": []})
        logged_by_time[t]["actions"].append(action / max(float(cfg.daily_budget), EPS))
        logged_by_time[t]["features"].append(action_support_feature_vector(action, cfg.daily_budget))

    policy_by_time = collect_policy_action_support_by_time(
        cfg=cfg,
        coefs=coefs,
        policy=policy,
        method=method,
        normalizer=normalizer,
        episodes=episodes,
        seed=seed,
        seed_offset=seed_offset,
    )

    feature_names = [
        "step_spend_over_B",
        "max_action_over_B",
        "action_entropy",
        "action_hhi",
        "action_nonzero_share",
    ]
    rows: List[Dict[str, Any]] = []
    for t in range(int(cfg.Time_intervals)):
        logged_actions = (
            torch.stack(logged_by_time.get(t, {}).get("actions", []), dim=0)
            if logged_by_time.get(t, {}).get("actions")
            else torch.empty(0, cfg.M)
        )
        logged_features = (
            torch.stack(logged_by_time.get(t, {}).get("features", []), dim=0)
            if logged_by_time.get(t, {}).get("features")
            else torch.empty(0, 5)
        )
        policy_actions = policy_by_time[t]["actions"]
        policy_features = policy_by_time[t]["features"]
        row: Dict[str, Any] = {
            "round": round_idx,
            "method": method_name,
            "method_key": method,
            "t": t,
            "logged_action_count": int(logged_actions.shape[0]),
            "policy_action_count": int(policy_actions.shape[0]),
            "daily_budget": float(cfg.daily_budget),
        }
        for idx, name in enumerate(feature_names):
            row.update(action_support_metric_dict(f"logged_{name}", logged_features[:, idx] if logged_features.numel() else torch.empty(0)))
            row.update(action_support_metric_dict(f"policy_{name}", policy_features[:, idx] if policy_features.numel() else torch.empty(0)))

        if logged_features.numel() and policy_features.numel():
            logged_p95 = torch.quantile(logged_features, 0.95, dim=0)
            logged_p05 = torch.quantile(logged_features, 0.05, dim=0)
            spend_oos = policy_features[:, 0] > logged_p95[0] + 1e-6
            max_oos = policy_features[:, 1] > logged_p95[1] + 1e-6
            entropy_oos = policy_features[:, 2] < logged_p05[2] - 1e-6
            hhi_oos = policy_features[:, 3] > logged_p95[3] + 1e-6
            risk = spend_oos | max_oos | entropy_oos | hhi_oos
            row["share_above_logged_p95_step_spend"] = float(spend_oos.to(dtype=torch.float32).mean().item())
            row["share_above_logged_p95_max_action"] = float(max_oos.to(dtype=torch.float32).mean().item())
            row["share_below_logged_p05_entropy"] = float(entropy_oos.to(dtype=torch.float32).mean().item())
            row["share_above_logged_p95_hhi"] = float(hhi_oos.to(dtype=torch.float32).mean().item())
            row["extrapolation_risk_share"] = float(risk.to(dtype=torch.float32).mean().item())
        else:
            row["share_above_logged_p95_step_spend"] = float("nan")
            row["share_above_logged_p95_max_action"] = float("nan")
            row["share_below_logged_p05_entropy"] = float("nan")
            row["share_above_logged_p95_hhi"] = float("nan")
            row["extrapolation_risk_share"] = float("nan")

        if logged_actions.numel() and policy_actions.numel():
            distances = torch.cdist(policy_actions.to(dtype=torch.float32), logged_actions.to(dtype=torch.float32), p=2)
            nearest = distances.min(dim=1).values
            row.update(action_support_metric_dict("policy_nearest_logged_action_l2", nearest))
        else:
            row.update(action_support_metric_dict("policy_nearest_logged_action_l2", torch.empty(0)))
        rows.append(row)
    return rows


def save_round_policy_checkpoint(
    out_dir: Path,
    q: int,
    method: str,
    policy: Any,
    agent: Optional[Any],
    train_cfg: TrainConfig,
    args: argparse.Namespace,
) -> None:
    """Persist the current round policy state independently of later rounds."""
    policy_dir = ensure_dir(out_dir / "policies" / f"round_{q}")
    save_json(
        policy_dir / "metadata.json",
        {
            "round": q,
            "method": method,
            "method_name": LEARNING_METHOD_NAMES[method],
            "checkpoint_step": checkpoint_step_for_round(method, q, train_cfg, args),
            "policy_checkpoint_role": "current_round_policy",
            "simulator_checkpoint": f"simulators/round_{q}",
            "actor_critic_warm_started_from_previous_round": int(method in ACTOR_CRITIC_METHODS and q >= 2),
            "actor_critic_real_online_update_enabled": int(method in ACTOR_CRITIC_METHODS and bool(getattr(args, "real_online_update", False))),
            "dfl_score_model_warm_started_from_previous_round": int(method == "dfl_pto" and q >= 2),
        },
    )
    if policy is not None and hasattr(policy, "state_dict"):
        torch.save(policy.state_dict(), policy_dir / "policy_state_dict.pt")
    if agent is None:
        return
    for name in ["actor", "actor_targ", "q1", "q2", "q1_targ", "q2_targ"]:
        module = getattr(agent, name, None)
        if module is not None and hasattr(module, "state_dict"):
            torch.save(module.state_dict(), policy_dir / f"{name}.pt")
    if hasattr(agent, "log_alpha"):
        torch.save(getattr(agent, "log_alpha").detach().cpu(), policy_dir / "log_alpha.pt")


def _mean_se(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, se


def summarize_step_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact per-time true-DGP diagnostics for reward and budget curves."""
    grouped: Dict[Tuple[int, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (int(row.get("round", 0)), str(row.get("method", "")), int(row.get("t", 0)))
        grouped.setdefault(key, []).append(row)

    metrics = [
        "step_net_reward",
        "step_response_score",
        "step_spend",
        "remaining_budget_after",
        "action_mean",
        "action_max",
        "action_nonzero_share",
    ]
    out: List[Dict[str, Any]] = []
    for (round_idx, method_name, t), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        item: Dict[str, Any] = {
            "round": round_idx,
            "method": method_name,
            "method_key": str(group[0].get("method_key", "")),
            "t": t,
            "episodes": len(group),
        }
        for metric in metrics:
            mean, se = _mean_se([float(r.get(metric, float("nan"))) for r in group])
            item[metric] = mean
            item[f"{metric}_se"] = se
        out.append(item)
    return out


def make_train_config(args: argparse.Namespace, device: str) -> TrainConfig:
    return TrainConfig(
        seed=args.seed,
        device=device,
        M=args.M,
        T=args.T,
        daily_budget=args.daily_budget,
        algo=args.method,
        train_episodes=args.synthetic_train_episodes,
        warmup_episodes=args.warmup_episodes,
        update_after_episodes=args.update_after_episodes,
        eval_every_episodes=max(args.synthetic_train_episodes, 1),
        eval_episodes=args.eval_episodes,
        log_every_episodes=args.log_every_episodes,
        verbose_episode_log=False,
        update_every_steps=args.update_every_steps,
        gradient_steps_per_update=args.gradient_steps_per_update,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        use_state_norm=not args.no_state_norm,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        gamma=args.gamma,
        tau=args.tau,
        grad_clip_norm=args.grad_clip_norm,
        critic_loss=args.critic_loss,
        huber_delta=args.huber_delta,
        auto_alpha=args.auto_alpha,
        fixed_temperature=args.fixed_temperature,
        init_temperature=args.init_temperature,
        target_entropy_scale=args.target_entropy_scale,
        alpha_logp_clip=args.alpha_logp_clip,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
        sac_entropy_space=args.sac_entropy_space,
        dirichlet_min_alpha=args.dirichlet_min_alpha,
        dirichlet_max_alpha=args.dirichlet_max_alpha,
        dirichlet_logit_clip=args.dirichlet_logit_clip,
        stop_on_nonfinite=not args.no_stop_on_nonfinite,
        dirichlet_init_total_spend_share=args.dirichlet_init_total_spend_share,
        dirichlet_init_concentration=args.dirichlet_init_concentration,
        ln_init_total_spend_share=args.ln_init_total_spend_share,
        gaussian_projection_init_log_std=args.gaussian_projection_init_log_std,
        td3_init_total_spend_share=args.td3_init_total_spend_share,
        out_dir=args.out_dir,
        save_models=args.save_models,
        make_plots=False,
        save_final_eval_raw=True,
        make_final_boxplots=False,
        final_eval_seed_offset=args.final_eval_seed_offset,
    )


def make_sim_config(args: argparse.Namespace, device: str) -> SimConfig:
    return SimConfig(
        M=args.M,
        Time_intervals=args.T,
        D=max(args.initial_logged_days + args.q_max * args.new_logged_days, 1),
        daily_budget=args.daily_budget,
        sigma_x=args.sigma_x,
        sigma_y=args.sigma_y,
        sigma_c=args.sigma_c,
        sigma2_z=args.sigma2_z,
        rho_corr=args.rho_corr,
        device=device,
        dtype=torch.float32,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    raw_argv = list(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Iterative learned-simulator SAC-SA synthetic experiment.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--method", type=str, default="sacsa", choices=sorted(LEARNING_METHOD_NAMES))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--daily_budget", type=float, default=10.0)
    parser.add_argument(
        "--logging_spend_scale",
        type=float,
        default=1.0,
        help=(
            "Spend multiplier for nonzero actions in the fixed mixed logging policy only; "
            "on-policy collection uses the learned policy directly."
        ),
    )
    parser.add_argument("--sigma_x", type=float, default=1.0)
    parser.add_argument("--sigma_y", type=float, default=1.0)
    parser.add_argument("--sigma_c", type=float, default=1.0)
    parser.add_argument("--sigma2_z", type=float, default=1.5)
    parser.add_argument("--rho_corr", type=float, default=0.5)

    parser.add_argument("--initial_logged_days", type=int, default=80)
    parser.add_argument("--new_logged_days", type=int, default=80)
    parser.add_argument("--q_max", type=int, default=3)

    parser.add_argument(
        "--validation_logged_days",
        type=int,
        default=None,
        help="Fixed held-out logged days used only for learned-simulator validation residual paths. Defaults to --holdout_logged_days.",
    )
    parser.add_argument("--holdout_logged_days", type=int, default=100)
    parser.add_argument("--sim_epochs", type=int, default=50)
    parser.add_argument("--sim_update_mode", type=str, default="finetune", choices=["finetune", "refit"])
    parser.add_argument("--sim_finetune_epochs", type=int, default=20)
    parser.add_argument("--sim_batch_size", type=int, default=512)
    parser.add_argument("--sim_lr", type=float, default=1e-3)
    parser.add_argument("--sim_hidden_dim", type=int, default=256, help="Legacy fallback: used as [d,d] only when --sim_hidden_layers is omitted and this option is explicitly set.")
    parser.add_argument("--sim_hidden_layers", type=int, nargs="+", default=None, help="Pooled simulator trunk hidden layers, e.g. 512 256 128.")
    parser.add_argument("--sim_head_hidden_dim", type=int, default=0, help="Optional per-block head hidden width. 0 means linear block heads.")
    parser.add_argument("--sim_dropout", type=float, default=0.05)
    parser.add_argument("--sim_activation", type=str, default="silu", choices=["silu", "swish", "gelu", "relu", "tanh"])
    parser.add_argument("--sim_weight_decay", type=float, default=1e-5)
    parser.add_argument("--sim_z_loss_weight", type=float, default=6.0)
    parser.add_argument("--sim_x_loss_weight", type=float, default=1.0)
    parser.add_argument("--sim_y_loss_weight", type=float, default=1.0)
    parser.add_argument("--sim_c_loss_weight", type=float, default=1.0)
    parser.add_argument("--logging_policy", type=str, default="rich", choices=["legacy", "rich"])
    parser.set_defaults(center_residual=True)
    parser.add_argument(
        "--center_residual",
        dest="center_residual",
        action="store_true",
        help="Center residual-bootstrap noise by interval before synthetic rollouts.",
    )
    parser.add_argument(
        "--no_center_residual",
        dest="center_residual",
        action="store_false",
        help="Use raw fitted residuals in residual-bootstrap rollouts for ablation.",
    )
    parser.set_defaults(use_pessimism=True)
    parser.add_argument("--use_pessimism", dest="use_pessimism", action="store_true")
    parser.add_argument("--no_pessimism", dest="use_pessimism", action="store_false")
    parser.add_argument("--lambda_pess", type=float, default=0.001)
    parser.add_argument("--b_boot", type=int, default=5)
    parser.add_argument("--bootstrap_epochs", type=int, default=20)
    parser.add_argument("--bootstrap_finetune_epochs", type=int, default=5)
    parser.add_argument("--save_bootstrap_ensemble", action="store_true")

    parser.add_argument("--candidate_random_count", type=int, default=64)
    parser.add_argument("--candidate_budget_fracs", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--dfl_epochs", type=int, default=80)
    parser.add_argument("--dfl_batch_size", type=int, default=512)
    parser.add_argument("--dfl_lr", type=float, default=1e-3)
    parser.add_argument("--dfl_decision_weight", type=float, default=1.0)
    parser.add_argument("--dfl_softmax_temperature", type=float, default=1.0)
    parser.add_argument("--mpc_horizon", type=int, default=3)

    parser.add_argument("--synthetic_train_episodes", type=int, default=500)
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=1,
        help=(
            "Parallel learned-simulator rollout batch size for actor-critic methods. "
            "1 preserves the original single-episode rollout path. With "
            "--auto_accelerator_defaults on CUDA, the default is promoted to 32."
        ),
    )
    parser.add_argument(
        "--batch_rollout_update_cap",
        type=int,
        default=128,
        help="Maximum actor-critic gradient updates after one batched rollout step.",
    )
    parser.add_argument("--inner_early_stop", action="store_true")
    parser.add_argument("--inner_eval_every_episodes", type=int, default=50)
    parser.add_argument("--inner_patience", type=int, default=3)
    parser.add_argument("--inner_min_delta", type=float, default=0.0)
    parser.add_argument("--warmup_episodes", type=int, default=20)
    parser.add_argument("--update_after_episodes", type=int, default=20)
    parser.add_argument("--update_every_steps", type=int, default=1)
    parser.add_argument("--gradient_steps_per_update", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_size", type=int, default=200_000)
    parser.set_defaults(real_online_update=True)
    parser.add_argument(
        "--real_online_update",
        dest="real_online_update",
        action="store_true",
        help=(
            "For actor-critic methods, use newly collected true-DGP transitions "
            "for a small unpenalized online replay update before the round evaluation."
        ),
    )
    parser.add_argument(
        "--no_real_online_update",
        dest="real_online_update",
        action="store_false",
        help="Only collect true-DGP on-policy logs for the next simulator round; do not update actor-critic online.",
    )
    parser.add_argument("--real_batch_size", type=int, default=64)
    parser.add_argument("--real_update_after_steps", type=int, default=0)
    parser.add_argument("--real_gradient_steps_per_step", type=int, default=1)
    parser.add_argument(
        "--real_exploration_epsilon",
        type=float,
        default=0.0,
        help="For q>1 real online collection, mix rich logging exploration with this probability; q=1 remains deterministic.",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--lr_alpha", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--critic_loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.set_defaults(auto_alpha=True)
    parser.add_argument("--auto_alpha", dest="auto_alpha", action="store_true")
    parser.add_argument("--no_auto_alpha", dest="auto_alpha", action="store_false")
    parser.add_argument(
        "--fixed_temperature",
        type=float,
        default=None,
        help="Use a fixed SAC temperature; forces --no_auto_alpha and sets init/min/max temperature to this value.",
    )
    parser.add_argument("--init_temperature", type=float, default=0.05)
    parser.add_argument("--target_entropy_scale", type=float, default=2.5)
    parser.add_argument("--alpha_logp_clip", type=float, default=20.0)
    parser.add_argument("--min_temperature", type=float, default=1e-5)
    parser.add_argument("--max_temperature", type=float, default=0.05)
    parser.add_argument("--sac_entropy_space", type=str, default="share_clipped", choices=["share_clipped", "share_unclipped", "action"])
    parser.add_argument("--dirichlet_min_alpha", type=float, default=1e-3)
    parser.add_argument("--dirichlet_max_alpha", type=float, default=1e4)
    parser.add_argument("--dirichlet_logit_clip", type=float, default=20.0)
    parser.add_argument("--dirichlet_init_total_spend_share", type=float, default=0.8)
    parser.add_argument("--dirichlet_init_concentration", type=float, default=30.0)
    parser.add_argument("--ln_init_total_spend_share", type=float, default=0.8)
    parser.add_argument(
        "--gaussian_projection_init_log_std",
        type=float,
        default=-4.0,
        help="Initial Gaussian proposal log standard deviation before Euclidean projection.",
    )
    parser.add_argument("--td3_init_total_spend_share", type=float, default=0.8)
    parser.add_argument("--no_stop_on_nonfinite", action="store_true")

    parser.set_defaults(cuda_preflight=True)
    parser.add_argument("--cuda_preflight", dest="cuda_preflight", action="store_true")
    parser.add_argument("--no_cuda_preflight", dest="cuda_preflight", action="store_false")
    parser.add_argument("--allow_cpu_fallback", action="store_true")
    parser.add_argument("--torch_num_threads", type=int, default=0)
    parser.add_argument("--torch_num_interop_threads", type=int, default=0)
    parser.add_argument("--matmul_precision", type=str, default="high", choices=["highest", "high", "medium"])
    parser.set_defaults(auto_accelerator_defaults=True)
    parser.add_argument(
        "--auto_accelerator_defaults",
        dest="auto_accelerator_defaults",
        action="store_true",
        help=(
            "On CUDA, promote unchanged CPU-oriented batch defaults to "
            "GPU-friendly values such as rollout_batch_size=32 and "
            "batch_size=1024."
        ),
    )
    parser.add_argument(
        "--no_auto_accelerator_defaults",
        dest="auto_accelerator_defaults",
        action="store_false",
        help="Disable automatic CUDA batch-size promotion.",
    )
    parser.add_argument("--progress", type=str, default="compact", choices=["none", "compact", "tqdm"])
    parser.set_defaults(audit_checks=True)
    parser.add_argument("--audit_checks", dest="audit_checks", action="store_true")
    parser.add_argument("--no_audit_checks", dest="audit_checks", action="store_false")
    parser.add_argument("--strict_audit", action="store_true")
    parser.set_defaults(dgp_sanity=True)
    parser.add_argument("--dgp_sanity", dest="dgp_sanity", action="store_true")
    parser.add_argument("--no_dgp_sanity", dest="dgp_sanity", action="store_false")
    parser.add_argument("--dgp_sanity_episodes", type=int, default=64)
    parser.add_argument(
        "--save_step_diagnostics",
        action="store_true",
        help="Save per-episode true-DGP step logs and per-time reward/spend/action summaries.",
    )
    parser.add_argument("--validation_episodes", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--no_state_norm", action="store_true")
    parser.add_argument("--log_every_episodes", type=int, default=50)
    parser.add_argument("--final_eval_seed_offset", type=int, default=20_000)
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--out_dir", type=str, default="results_iterative_synthetic_sacsa")
    args = parser.parse_args()
    has_hidden_layers = any(arg == "--sim_hidden_layers" or arg.startswith("--sim_hidden_layers=") for arg in raw_argv)
    has_legacy_hidden_dim = any(arg == "--sim_hidden_dim" or arg.startswith("--sim_hidden_dim=") for arg in raw_argv)
    if args.sim_hidden_layers is None:
        if has_legacy_hidden_dim and not has_hidden_layers:
            args.sim_hidden_layers = [int(args.sim_hidden_dim), int(args.sim_hidden_dim)]
        else:
            args.sim_hidden_layers = [512, 256, 128]
    args.sim_hidden_layers = [int(v) for v in args.sim_hidden_layers if int(v) > 0]
    if not args.sim_hidden_layers:
        raise SystemExit("--sim_hidden_layers must contain at least one positive width.")
    if args.fixed_temperature is not None:
        if float(args.fixed_temperature) <= 0.0:
            raise SystemExit("--fixed_temperature must be positive.")
        args.auto_alpha = False
        args.init_temperature = float(args.fixed_temperature)
        args.min_temperature = float(args.fixed_temperature)
        args.max_temperature = float(args.fixed_temperature)
    return args


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row)) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(row), ensure_ascii=True) + "\n")


def _metric_for_print(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return str(value)
    if abs(v) >= 1000 or (0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.4f}"


def _cuda_memory_snapshot(device: str) -> Dict[str, float]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {}
    torch_device = torch.device(device)
    idx = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    return {
        "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(idx) / (1024 ** 2)),
        "gpu_memory_reserved_mb": float(torch.cuda.memory_reserved(idx) / (1024 ** 2)),
        "gpu_max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(idx) / (1024 ** 2)),
    }


def maybe_tqdm(iterable: Iterable[Any], mode: str, desc: str, total: Optional[int] = None) -> Iterable[Any]:
    if str(mode).lower() != "tqdm":
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:
        return iterable


class ProgressReporter:
    """Small run-event logger with compact human-readable stage output."""

    def __init__(self, out_dir: Path, mode: str, device: str):
        self.out_dir = out_dir
        self.mode = str(mode).lower()
        self.device = device
        self.path = out_dir / "run_events.jsonl"
        self.start_time = time.perf_counter()
        if self.path.exists():
            self.path.unlink()

    def emit(self, event: str, print_keys: Optional[Sequence[str]] = None, **payload: Any) -> None:
        row = {
            "event": event,
            "elapsed_sec": time.perf_counter() - self.start_time,
            **_cuda_memory_snapshot(self.device),
            **payload,
        }
        append_jsonl(self.path, row)
        if self.mode == "none":
            return
        keys = list(print_keys or [])
        if not keys:
            keys = [
                "round",
                "method",
                "logged_days",
                "holdout_r2_z",
                "sim_minus_true_net_reward",
                "true_net_reward",
                "spend",
                "delta_vs_best_operational",
                "duration_sec",
                "gpu_max_memory_allocated_mb",
            ]
        parts = [f"[{row['elapsed_sec']:8.1f}s] {event}"]
        for key in keys:
            if key in row and row[key] not in (None, ""):
                parts.append(f"{key}={_metric_for_print(row[key])}")
        print(" | ".join(parts), flush=True)

    @contextmanager
    def stage(self, name: str, **payload: Any) -> Iterator[None]:
        stage_start = time.perf_counter()
        self.emit(f"{name}_start", **payload)
        try:
            yield
        except Exception as exc:
            self.emit(
                f"{name}_failed",
                duration_sec=time.perf_counter() - stage_start,
                error_type=type(exc).__name__,
                error=str(exc),
                **payload,
            )
            raise
        self.emit(f"{name}_done", duration_sec=time.perf_counter() - stage_start, **payload)


class ExperimentAudit:
    """Lightweight invariant checks that do not change experiment semantics."""

    def __init__(self, out_dir: Path, enabled: bool = True, strict: bool = False):
        self.enabled = bool(enabled)
        self.strict = bool(strict)
        self.path = out_dir / "audit_warnings.jsonl"
        self.count = 0
        if self.enabled and self.path.exists():
            self.path.unlink()

    def warn(self, code: str, message: str, **context: Any) -> None:
        if not self.enabled:
            return
        row = {"code": code, "message": message, **context}
        append_jsonl(self.path, row)
        self.count += 1
        if self.strict:
            raise AssertionError(f"{code}: {message}")


def audit_action_tensor(
    audit: Optional[ExperimentAudit],
    phase: str,
    action: torch.Tensor,
    budget: torch.Tensor | float,
    *,
    atol: float = 1e-4,
    **context: Any,
) -> None:
    if audit is None or not audit.enabled:
        return
    a = action.detach()
    b = torch.as_tensor(budget, device=a.device, dtype=a.dtype).detach()
    if not torch.isfinite(a).all():
        audit.warn("nonfinite_action", "Action contains NaN or Inf.", phase=phase, **context)
        return
    if torch.any(a < -float(atol)):
        audit.warn(
            "negative_action",
            "Action contains entries below zero tolerance.",
            phase=phase,
            min_action=float(a.min().detach().cpu().item()),
            **context,
        )
    spend = a.sum(dim=-1)
    if b.ndim == 0:
        b_cmp = b
    else:
        b_cmp = b.reshape(spend.shape)
    over = spend - b_cmp
    if torch.any(over > float(atol)):
        audit.warn(
            "budget_violation",
            "Action spend exceeds remaining budget tolerance.",
            phase=phase,
            max_spend=float(spend.max().detach().cpu().item()),
            max_budget=float(b_cmp.max().detach().cpu().item()),
            max_over=float(over.max().detach().cpu().item()),
            **context,
        )


def audit_episode_metrics(
    audit: Optional[ExperimentAudit],
    phase: str,
    metrics: Dict[str, Any],
    *,
    atol: float = 1e-3,
    **context: Any,
) -> None:
    if audit is None or not audit.enabled:
        return
    for key in ("net_reward", "response_score", "spend", "final_budget"):
        value = metrics.get(key)
        try:
            v = float(value)
        except (TypeError, ValueError):
            audit.warn("missing_metric", f"Metric {key} is missing or not numeric.", phase=phase, **context)
            continue
        if not math.isfinite(v):
            audit.warn("nonfinite_metric", f"Metric {key} is not finite.", phase=phase, metric=key, value=value, **context)
    final_budget = float(metrics.get("final_budget", 0.0))
    if final_budget < -atol:
        audit.warn("negative_final_budget", "Final budget is negative.", phase=phase, final_budget=final_budget, **context)
    nr = metrics.get("net_reward")
    rs = metrics.get("response_score")
    spend = metrics.get("spend")
    try:
        residual = float(nr) - (float(rs) - float(spend))
    except (TypeError, ValueError):
        return
    if math.isfinite(residual) and abs(residual) > atol:
        audit.warn(
            "reward_identity_mismatch",
            "net_reward differs from response_score - spend.",
            phase=phase,
            residual=residual,
            **context,
        )


def write_dgp_sanity_outputs(
    out_dir: Path,
    cfg: SimConfig,
    coefs: Dict[str, torch.Tensor],
    baseline_rows: Sequence[Dict[str, Any]],
) -> None:
    rows: List[Dict[str, Any]] = [
        {"section": "config", "name": "M", "value": cfg.M},
        {"section": "config", "name": "T", "value": cfg.Time_intervals},
        {"section": "config", "name": "daily_budget", "value": cfg.daily_budget},
        {"section": "config", "name": "reward_definition", "value": "sum(Z)-sum(A)"},
        {"section": "config", "name": "response_score_scale", "value": "log_scale_synthetic_index_not_original_currency_gmv"},
        {"section": "config", "name": "subsidy_response_scale", "value": SUBSIDY_RESPONSE_SCALE},
        {"section": "config", "name": "city_overspend_waste", "value": CITY_OVERSPEND_WASTE},
    ]
    for key in DGP_COEF_KEYS:
        val = coefs[key].detach().cpu().float()
        rows.append(
            {
                "section": "coefficient",
                "name": key,
                "min": float(val.min().item()),
                "max": float(val.max().item()),
                "mean": float(val.mean().item()),
                "std": float(val.std(unbiased=False).item()),
            }
        )

    a_grid = torch.linspace(0.0, float(cfg.daily_budget), 101, device=cfg.device, dtype=cfg.dtype)
    psi = subsidy_response(a_grid).detach().cpu()
    for idx, (a_val, psi_val) in enumerate(zip(a_grid.detach().cpu().tolist(), psi.tolist())):
        rows.append({"section": "subsidy_response_curve", "name": "psi_A", "grid_index": idx, "A": a_val, "value": psi_val})
    write_csv(out_dir / "dgp_sanity.csv", rows)

    baseline_summary: List[Dict[str, Any]] = []
    for method in sorted({str(r.get("method", "")) for r in baseline_rows}, key=method_sort_key):
        method_rows = [r for r in baseline_rows if str(r.get("method", "")) == method]
        baseline_summary.append({"method": method, "method_key": method_rows[0].get("method_key", ""), **summarize_eval_rows(method_rows)})
    write_csv(out_dir / "dgp_baseline_policy_summary.csv", baseline_summary)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(a_grid.detach().cpu().numpy(), psi.numpy())
        axes[0].set_title("DGP concave subsidy transform")
        axes[0].set_xlabel("A")
        axes[0].set_ylabel("psi(A)")

        gaps = [0.0, 0.5, 1.0]
        for gap in gaps:
            X = torch.full_like(a_grid, float(gap))
            Y = torch.zeros_like(a_grid)
            O = torch.zeros_like(a_grid)
            gate, _, _ = subsidy_effect_multipliers(X, Y, O)
            axes[1].plot(a_grid.detach().cpu().numpy(), (psi.to(a_grid.device) * gate).detach().cpu().numpy(), label=f"gap={gap}")
        axes[1].set_title("Response term by positive gap")
        axes[1].set_xlabel("A")
        axes[1].set_ylabel("psi(A) * gap gate")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "dgp_response_curves.png", dpi=220)
        plt.close(fig)
    except Exception as exc:
        append_jsonl(out_dir / "audit_warnings.jsonl", {"code": "dgp_plot_failed", "message": str(exc)})


def configure_torch_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    if args.torch_num_interop_threads and args.torch_num_interop_threads > 0:
        try:
            torch.set_num_interop_threads(int(args.torch_num_interop_threads))
        except RuntimeError:
            # PyTorch only allows setting interop threads before parallel work.
            pass
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)
    return {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": int(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "matmul_precision": args.matmul_precision,
    }


def apply_accelerator_defaults(args: argparse.Namespace, device: str) -> Dict[str, Any]:
    """
    Promote conservative CPU-oriented defaults to GPU-friendly batch sizes.

    Only values that still equal the parser defaults are changed, so an explicit
    user choice is respected. The goal is to route actor-critic synthetic
    rollouts through the vectorized learned-simulator path on CUDA without
    changing the algorithm or the final evaluation protocol.
    """
    if not bool(getattr(args, "auto_accelerator_defaults", True)):
        return {"auto_accelerator_defaults": 0, "auto_accelerator_overrides": {}}

    overrides: Dict[str, Dict[str, Any]] = {}

    def promote(name: str, default_value: Any, new_value: Any) -> None:
        current = getattr(args, name)
        if current == default_value:
            setattr(args, name, new_value)
            overrides[name] = {"from": default_value, "to": new_value}

    if str(device).startswith("cuda"):
        promote("rollout_batch_size", 1, 32)
        promote("sim_batch_size", 512, 2048)
        promote("dfl_batch_size", 512, 2048)
        promote("batch_size", 256, 1024)
        promote("real_batch_size", 64, 128)

    return {
        "auto_accelerator_defaults": 1,
        "auto_accelerator_device": str(device),
        "auto_accelerator_overrides": overrides,
    }


def cuda_preflight(device: str) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    torch_device = torch.device(device)
    device_index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    arch = f"sm_{capability[0]}{capability[1]}"
    arch_list = list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else []
    if arch_list and arch not in arch_list:
        raise RuntimeError(
            f"CUDA preflight failed: GPU {name} requires {arch}, but this PyTorch wheel reports "
            f"arch list {arch_list}. Install a PyTorch wheel that supports this GPU architecture."
        )
    try:
        x = torch.randn((1024, 1024), device=torch_device, dtype=torch.float32)
        y = x @ x
        _ = float(y.mean().detach().cpu().item())
        torch.cuda.synchronize(device_index)
    except Exception as exc:
        raise RuntimeError(
            "CUDA preflight failed during a real matmul. The driver may be usable, but the "
            "installed PyTorch CUDA wheel is not compatible with this GPU/runtime."
        ) from exc
    return {
        "gpu_name": name,
        "gpu_device_index": int(device_index),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "required_arch": arch,
        "arch_list": arch_list,
        "arch_list_contains_required": int(arch in arch_list) if arch_list else 0,
        "matmul_preflight_ok": 1,
    }


def main() -> None:
    args = parse_args()
    runtime_info = configure_torch_runtime(args)
    if args.validation_logged_days is None:
        args.validation_logged_days = args.holdout_logged_days
    if args.validation_logged_days <= 0:
        raise ValueError("--validation_logged_days must be positive.")
    if args.holdout_logged_days <= 0:
        raise ValueError("--holdout_logged_days must be positive.")
    if args.use_pessimism and args.b_boot < 2:
        raise ValueError("--b_boot must be at least 2 when --use_pessimism is enabled.")
    if args.lambda_pess < 0:
        raise ValueError("--lambda_pess must be nonnegative.")
    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            if args.allow_cpu_fallback:
                print("CUDA requested but not available. Falling back to CPU.", flush=True)
                device = "cpu"
            else:
                raise RuntimeError("CUDA requested but not available. Use --allow_cpu_fallback to run on CPU.")
        elif args.cuda_preflight:
            runtime_info.update(cuda_preflight(device))
    runtime_info["requested_device"] = args.device
    runtime_info["effective_device"] = device
    runtime_info.update(apply_accelerator_defaults(args, device))

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    reporter = ProgressReporter(out_dir, args.progress, device)
    audit = ExperimentAudit(out_dir, enabled=bool(args.audit_checks), strict=bool(args.strict_audit))
    reporter.emit(
        "run_start",
        method=args.method,
        seed=args.seed,
        device=device,
        progress=args.progress,
        audit_checks=int(args.audit_checks),
        strict_audit=int(args.strict_audit),
    )
    train_cfg = make_train_config(args, device)
    sim_cfg = make_sim_config(args, device)
    sim_fit_cfg = SimulatorFitConfig(
        epochs=args.sim_epochs,
        batch_size=args.sim_batch_size,
        lr=args.sim_lr,
        hidden_dim=args.sim_hidden_dim,
        hidden_layers=tuple(args.sim_hidden_layers),
        head_hidden_dim=args.sim_head_hidden_dim,
        dropout=args.sim_dropout,
        activation=args.sim_activation,
        weight_decay=args.sim_weight_decay,
        z_loss_weight=args.sim_z_loss_weight,
        x_loss_weight=args.sim_x_loss_weight,
        y_loss_weight=args.sim_y_loss_weight,
        c_loss_weight=args.sim_c_loss_weight,
    )
    pess_cfg = PessimismConfig(
        use_pessimism=bool(args.use_pessimism),
        lambda_pess=float(args.lambda_pess),
        b_boot=int(args.b_boot),
        bootstrap_epochs=int(args.bootstrap_epochs),
        bootstrap_finetune_epochs=int(args.bootstrap_finetune_epochs),
        save_bootstrap_ensemble=bool(args.save_bootstrap_ensemble),
    )
    candidate_cfg = CandidateConfig(
        budget_fracs=tuple(float(x) for x in args.candidate_budget_fracs),
        random_count=args.candidate_random_count,
        seed=args.seed + 90_000,
    )
    dfl_cfg = DFLConfig(
        epochs=args.dfl_epochs,
        batch_size=args.dfl_batch_size,
        lr=args.dfl_lr,
        decision_weight=args.dfl_decision_weight,
        softmax_temperature=args.dfl_softmax_temperature,
        hidden_dim=args.hidden_dim,
    )
    inner_early_stop_cfg = InnerEarlyStopConfig(
        enabled=bool(args.inner_early_stop),
        eval_every_episodes=int(args.inner_eval_every_episodes),
        patience=int(args.inner_patience),
        min_delta=float(args.inner_min_delta),
    )
    save_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "train_cfg": json_safe(asdict(train_cfg)),
            "sim_cfg": json_safe(asdict(sim_cfg)),
            "sim_fit_cfg": json_safe(asdict(sim_fit_cfg)),
            "pessimism_cfg": json_safe(asdict(pess_cfg)),
            "candidate_cfg": json_safe(asdict(candidate_cfg)),
            "dfl_cfg": json_safe(asdict(dfl_cfg)),
            "inner_early_stop_cfg": json_safe(asdict(inner_early_stop_cfg)),
            "logging_policy": args.logging_policy,
            "mixed_logging_policy": logging_policy_description(args.logging_policy),
            "runtime": json_safe(runtime_info),
        },
    )

    gen = torch.Generator(device=sim_cfg.device)
    gen.manual_seed(sim_cfg.seed)
    coefs = generate_dgp_coefs(sim_cfg, generator=gen)

    reporter.emit(
        "setup",
        print_keys=["method", "M", "T", "q_max", "initial_logged_days", "new_logged_days", "device"],
        method=args.method,
        M=sim_cfg.M,
        T=sim_cfg.Time_intervals,
        q_max=args.q_max,
        initial_logged_days=args.initial_logged_days,
        new_logged_days=args.new_logged_days,
        validation_logged_days=args.validation_logged_days,
        holdout_logged_days=args.holdout_logged_days,
        rollout="residual_path_bootstrap",
        use_pessimism=int(pess_cfg.use_pessimism),
        lambda_pess=pess_cfg.lambda_pess,
        b_boot=pess_cfg.b_boot,
        uncertainty_penalty_scope=pess_cfg.uncertainty_penalty_scope,
        sim_update_mode=args.sim_update_mode,
        logging_spend_scale=args.logging_spend_scale,
        logging_policy=args.logging_policy,
        logging_policy_description=logging_policy_description(args.logging_policy),
        sim_hidden_layers=tuple(args.sim_hidden_layers),
        sim_dropout=args.sim_dropout,
        device=device,
    )

    normalizer: Optional[StateNormalizer] = None
    if args.dgp_sanity:
        with reporter.stage("dgp_sanity", episodes=args.dgp_sanity_episodes):
            dgp_baseline_rows = evaluate_heuristic_raw_rows(
                sim_cfg,
                coefs,
                normalizer=None,
                episodes=max(int(args.dgp_sanity_episodes), 0),
                seed=args.seed,
                seed_offset=310_000,
                stage="dgp_sanity",
                audit=audit,
                progress_mode=args.progress,
            )
            write_dgp_sanity_outputs(out_dir, sim_cfg, coefs, dgp_baseline_rows)

    with reporter.stage("collect_initial_log", episodes=args.initial_logged_days):
        initial_logged = collect_logged_trajectories(
            sim_cfg,
            coefs,
            episodes=args.initial_logged_days,
            seed=args.seed + 1_000,
            normalizer=normalizer,
            policy=None,
            deterministic=True,
            random_logging_spend_scale=args.logging_spend_scale,
            logging_policy=args.logging_policy,
            audit=audit,
            audit_phase="initial_logged_collection",
            progress_mode=args.progress,
            progress_desc="collect initial log",
        )
        if train_cfg.use_state_norm:
            normalizer = fit_state_normalizer_from_logged(initial_logged, sim_cfg.M, device)
            save_json(out_dir / "state_normalizer.json", normalizer.state_dict())
    with reporter.stage("collect_validation_holdout_logs", validation_logged_days=args.validation_logged_days, holdout_logged_days=args.holdout_logged_days):
        fixed_logs = generate_fixed_logged_splits(
            sim_cfg,
            coefs,
            args,
            normalizer,
            train_log=initial_logged,
            audit=audit,
            progress_mode=args.progress,
        )
    logged = fixed_logs.train_log
    validation_logged = fixed_logs.validation_log
    validation_initial_states = fixed_logs.validation_initial_states
    holdout_logged = fixed_logs.holdout_log

    state_dim = 4 * sim_cfg.M + 1 + 3
    action_dim = sim_cfg.M
    response_scale = compute_response_scale_by_interval(logged, sim_cfg.Time_intervals, 4 * sim_cfg.M, device)
    save_json(out_dir / "response_scale.json", {"scale": response_scale.detach().cpu().tolist()})
    agent = make_learning_agent(args.method, state_dim, action_dim, train_cfg) if args.method in ACTOR_CRITIC_METHODS else None
    current_policy: Optional[Any] = None
    j_prev: Optional[float] = None
    j_val_pess_prev: Optional[float] = None
    validation_diag_delta_prev: Optional[float] = None

    round_rows: List[Dict[str, Any]] = []
    synthetic_training_rows: List[Dict[str, Any]] = []
    round_eval_rows: List[Dict[str, Any]] = []
    round_eval_step_rows: List[Dict[str, Any]] = []
    simulator_diagnostic_rows: List[Dict[str, Any]] = []
    interval_simulator_rows: List[Dict[str, Any]] = []
    main_models_prev: Optional[List[ResponseModel]] = None
    bootstrap_ensemble_prev: Optional[List[List[ResponseModel]]] = None
    dfl_policy_prev: Optional[DFLPolicy] = None
    completed_rounds = 0

    for q in range(1, args.q_max + 1):
        warm_start = q >= 2 and args.sim_update_mode == "finetune"
        sim_epochs_used = int(args.sim_finetune_epochs if warm_start else args.sim_epochs)
        bootstrap_epochs_used = int(
            args.bootstrap_finetune_epochs if warm_start and bootstrap_ensemble_prev is not None else args.bootstrap_epochs
        )
        with reporter.stage("round_simulator_fit", round=q, logged_days=len(logged), sim_epochs=sim_epochs_used, bootstrap_epochs=bootstrap_epochs_used):
            sim_bundle = fit_round_simulator_bundle(
                logged,
                validation_logged,
                holdout_logged,
                sim_cfg,
                normalizer,
                state_dim,
                action_dim,
                sim_fit_cfg,
                pess_cfg,
                device,
                args.seed,
                q,
                args.sim_batch_size,
                sim_epochs_used,
                bootstrap_epochs_used,
                main_models_prev if warm_start else None,
                bootstrap_ensemble_prev if warm_start else None,
                center_residual=bool(args.center_residual),
            )
        main_models = sim_bundle.main_models
        bootstrap_ensemble = sim_bundle.bootstrap_ensemble
        reporter.emit(
            "round_simulator_metrics",
            print_keys=[
                "round",
                "logged_days",
                "holdout_r2_z",
                "holdout_mse_z",
                "holdout_min_r2_z_by_t",
                "holdout_max_mse_z_by_t",
                "residual_path_count",
            ],
            round=q,
            logged_days=len(logged),
            **sim_bundle.main_model_metrics,
            **sim_bundle.train_residual_metrics,
            **sim_bundle.holdout_metrics,
        )

        initial_states = [traj[0].state for traj in logged if traj]
        learned_env = LearnedSimulatorEnv(
            cfg=sim_cfg,
            initial_states=initial_states,
            response_models=main_models,
            state_normalizer=normalizer,
            residual_sampler=sim_bundle.train_residual_sampler,
            seed=args.seed + 50_000 + q,
            bootstrap_ensemble=bootstrap_ensemble,
            response_scale=response_scale,
            lambda_pess=pess_cfg.lambda_pess,
            use_pessimism=pess_cfg.use_pessimism,
        )

        candidate_generator = CandidateActionGenerator(action_dim, candidate_cfg, device, torch.float32)
        with reporter.stage("round_policy_training", round=q, method=args.method, synthetic_train_episodes=args.synthetic_train_episodes):
            if args.method in ACTOR_CRITIC_METHODS:
                if agent is None:
                    raise RuntimeError("Actor-critic method requires an initialized agent.")
                agent, train_rows = train_agent_in_learned_simulator(
                    agent,
                    args.method,
                    learned_env,
                    train_cfg,
                    q,
                    early_stop_cfg=inner_early_stop_cfg,
                    rollout_batch_size=args.rollout_batch_size,
                    batch_rollout_update_cap=args.batch_rollout_update_cap,
                    audit=audit,
                    progress_mode=args.progress,
                )
                current_policy = agent.actor
            elif args.method == "dfl_pto":
                current_policy, train_rows = fit_dfl_pto_policy(
                    logged,
                    sim_cfg,
                    main_models,
                    normalizer,
                    candidate_generator,
                    state_dim,
                    action_dim,
                    dfl_cfg,
                    args.seed + 80_000 + q,
                    q,
                    args.log_every_episodes,
                    bootstrap_ensemble=bootstrap_ensemble,
                    response_scale=response_scale,
                    lambda_pess=pess_cfg.lambda_pess,
                    use_pessimism=pess_cfg.use_pessimism,
                    early_stop_cfg=inner_early_stop_cfg,
                    previous_policy=dfl_policy_prev if warm_start else None,
                )
            elif args.method == "mpc3":
                current_policy = MPC3Policy(
                    cfg=sim_cfg,
                    response_models=main_models,
                    state_normalizer=normalizer,
                    candidate_generator=candidate_generator,
                    horizon=args.mpc_horizon,
                    bootstrap_ensemble=bootstrap_ensemble,
                    response_scale=response_scale,
                    lambda_pess=pess_cfg.lambda_pess,
                    use_pessimism=pess_cfg.use_pessimism,
                )
                train_rows = [
                    {
                        "round": q,
                        "method": LEARNING_METHOD_NAMES[args.method],
                        "method_key": args.method,
                        "episode": 0,
                        "step": 0,
                        "mpc_horizon": args.mpc_horizon,
                        "candidate_count": 1 + 4 * len(candidate_cfg.budget_fracs) + candidate_cfg.random_count,
                    }
                ]
            else:
                raise ValueError(f"Unknown method: {args.method}")
        synthetic_training_rows.extend(train_rows)
        new_logged_for_next_round: Optional[List[LoggedTrajectory]] = None
        if args.method in ACTOR_CRITIC_METHODS and bool(args.real_online_update) and args.new_logged_days > 0:
            if agent is None:
                raise RuntimeError("Real online update requires an initialized actor-critic agent.")
            with reporter.stage("round_real_online_update", round=q, method=args.method, episodes=args.new_logged_days):
                new_logged_for_next_round, real_update_rows, agent = collect_real_interaction_and_update_agent(
                    cfg=sim_cfg,
                    coefs=coefs,
                    episodes=args.new_logged_days,
                    seed=args.seed + 70_000 + q * 1_000,
                    normalizer=normalizer,
                    agent=agent,
                    method=args.method,
                    train_cfg=train_cfg,
                    q=q,
                    real_batch_size=args.real_batch_size,
                    real_update_after_steps=args.real_update_after_steps,
                    real_gradient_steps_per_step=args.real_gradient_steps_per_step,
                    real_exploration_epsilon=args.real_exploration_epsilon,
                    logging_policy=args.logging_policy,
                    logging_spend_scale=args.logging_spend_scale,
                    audit=audit,
                    progress_mode=args.progress,
                    progress_desc=f"round {q} true-DGP online update",
                )
            synthetic_training_rows.extend(real_update_rows)
            current_policy = agent.actor
        save_round_policy_checkpoint(out_dir, q, args.method, current_policy, agent, train_cfg, args)

        # Keep validation episode draws fixed across outer rounds. The fitted
        # simulator and held-out residual values change with q, but the
        # validation initial-state indices and residual-path indices do not.
        validation_seed = args.seed + 60_000
        validation_residual_seed = args.seed + 61_000
        with reporter.stage("round_validation_diagnostic", round=q, episodes=args.validation_episodes):
            val_set = evaluate_paired_validation_set(
                cfg=sim_cfg,
                initial_states=validation_initial_states,
                response_models=main_models,
                state_normalizer=normalizer,
                actor=current_policy,
                method=args.method,
                episodes=args.validation_episodes,
                seed=validation_seed,
                residual_sampler=sim_bundle.validation_residual_sampler,
                bootstrap_ensemble=bootstrap_ensemble,
                response_scale=response_scale,
                lambda_pess=pess_cfg.lambda_pess,
                use_pessimism=pess_cfg.use_pessimism,
                residual_seed=validation_residual_seed,
            )
        val = val_set["policy"]
        val_no_subsidy = val_set["no_subsidy"]
        val_random = val_set["random"]
        val_uniform = val_set["uniform"]
        val_gap = val_set["gap"]

        def diagnostic_score(summary: Dict[str, float]) -> float:
            if pess_cfg.use_pessimism:
                return float(summary.get("net_reward_pess", summary["net_reward"]))
            return float(summary["net_reward"])

        j_q = float(val["net_reward"])
        j_val_pess = float(val.get("net_reward_pess", j_q))
        policy_diag_score = diagnostic_score(val)
        no_subsidy_diag_score = diagnostic_score(val_no_subsidy)
        random_diag_score = diagnostic_score(val_random)
        uniform_diag_score = diagnostic_score(val_uniform)
        gap_diag_score = diagnostic_score(val_gap)
        reference_diag_scores = {
            "no_subsidy": no_subsidy_diag_score,
            "random_pacing": random_diag_score,
            "uniform_pacing": uniform_diag_score,
            "gap_heuristic": gap_diag_score,
        }
        validation_diag_reference, validation_diag_reference_value = max(reference_diag_scores.items(), key=lambda kv: kv[1])
        validation_diag_delta_vs_best_operational = policy_diag_score - validation_diag_reference_value
        validation_diag_delta_vs_no_subsidy = policy_diag_score - no_subsidy_diag_score
        validation_diag_delta_vs_random = policy_diag_score - random_diag_score
        validation_diag_delta_vs_uniform = policy_diag_score - uniform_diag_score
        validation_diag_delta_vs_gap = policy_diag_score - gap_diag_score
        delta = None if j_prev is None else j_q - j_prev
        delta_val_pess = None if j_val_pess_prev is None else j_val_pess - j_val_pess_prev
        delta_validation_diag = (
            None
            if validation_diag_delta_prev is None
            else validation_diag_delta_vs_best_operational - validation_diag_delta_prev
        )

        row = {
            "round": q,
            "logged_days": len(logged),
            "new_logged_days": args.new_logged_days,
            **sim_bundle.main_model_metrics,
            **sim_bundle.bootstrap_metrics,
            **sim_bundle.train_residual_metrics,
            **{f"validation_{k}": v for k, v in sim_bundle.validation_residual_metrics.items()},
            **sim_bundle.holdout_metrics,
            "validation_diag_policy_return": j_q,
            "validation_diag_policy_pess_return": j_val_pess,
            "validation_diag_no_subsidy_return": float(val_no_subsidy["net_reward"]),
            "validation_diag_no_subsidy_pess_return": diagnostic_score(val_no_subsidy),
            "validation_diag_random_return": float(val_random["net_reward"]),
            "validation_diag_random_pess_return": diagnostic_score(val_random),
            "validation_diag_uniform_return": float(val_uniform["net_reward"]),
            "validation_diag_uniform_pess_return": diagnostic_score(val_uniform),
            "validation_diag_gap_return": float(val_gap["net_reward"]),
            "validation_diag_gap_pess_return": diagnostic_score(val_gap),
            "validation_diag_policy_score": policy_diag_score,
            "validation_diag_no_subsidy_score": no_subsidy_diag_score,
            "validation_diag_random_score": random_diag_score,
            "validation_diag_uniform_score": uniform_diag_score,
            "validation_diag_gap_score": gap_diag_score,
            "validation_diag_reference": validation_diag_reference,
            "validation_diag_reference_value": validation_diag_reference_value,
            "validation_diag_delta_vs_best_operational": validation_diag_delta_vs_best_operational,
            "validation_diag_delta_vs_no_subsidy": validation_diag_delta_vs_no_subsidy,
            "validation_diag_delta_vs_random": validation_diag_delta_vs_random,
            "validation_diag_delta_vs_uniform": validation_diag_delta_vs_uniform,
            "validation_diag_delta_vs_gap": validation_diag_delta_vs_gap,
            "validation_diag_policy_return_change_from_prev_round": delta,
            "validation_diag_policy_pess_return_change_from_prev_round": delta_val_pess,
            "validation_diag_delta_vs_best_operational_change_from_prev_round": delta_validation_diag,
            "validation_source": "learned_simulator_holdout_residual_path_bootstrap",
            "validation_role": "simulator_quality_diagnostic_only",
            "validation_initial_source": "fixed_validation_log_initial_states",
            "validation_initial_count": len(validation_initial_states),
            "validation_logged_days": len(validation_logged),
            "rollout_batch_size": int(args.rollout_batch_size),
            "batch_rollout_update_cap": int(args.batch_rollout_update_cap),
            "validation_residual_source": "fixed_validation_log",
            "residual_bootstrap_mode": "day_path",
            "center_residual": int(bool(args.center_residual)),
            "logging_policy": args.logging_policy,
            "mixed_logging_policy": logging_policy_description(args.logging_policy),
            "real_online_update": int(bool(args.real_online_update) and args.method in ACTOR_CRITIC_METHODS),
            "real_online_update_role": "true_dgp_unpenalized_actor_critic_finetune" if args.method in ACTOR_CRITIC_METHODS else "not_applicable",
            "real_batch_size": int(args.real_batch_size),
            "real_update_after_steps": int(args.real_update_after_steps),
            "real_gradient_steps_per_step": int(args.real_gradient_steps_per_step),
            "real_exploration_epsilon": float(args.real_exploration_epsilon),
            "training_residual_seed": args.seed + 40_000 + 100 * q,
            "validation_residual_seed": validation_residual_seed,
            "holdout_logged_days": len(holdout_logged),
            "use_pessimism": int(pess_cfg.use_pessimism),
            "lambda_pess": pess_cfg.lambda_pess,
            "b_boot": pess_cfg.b_boot if pess_cfg.use_pessimism else 0,
            "uncertainty_penalty_scope": pess_cfg.uncertainty_penalty_scope,
            "sim_update_mode": args.sim_update_mode,
            "sim_train_epochs_used": sim_bundle.sim_epochs_used,
            "bootstrap_epochs_used": sim_bundle.bootstrap_epochs_used,
            "sim_hidden_layers": "-".join(str(v) for v in args.sim_hidden_layers),
            "sim_head_hidden_dim": args.sim_head_hidden_dim,
            "sim_dropout": args.sim_dropout,
            "sim_activation": args.sim_activation,
            "sim_weight_decay": args.sim_weight_decay,
            "auto_alpha": int(bool(args.auto_alpha)),
            "fixed_temperature": "" if args.fixed_temperature is None else float(args.fixed_temperature),
            "init_temperature": float(args.init_temperature),
            "min_temperature": float(args.min_temperature),
            "max_temperature": float(args.max_temperature),
            "val_uncertainty_penalty": val.get("uncertainty_penalty", 0.0),
            "mean_policy_uncertainty": val.get("uncertainty_penalty", 0.0) / max(sim_cfg.Time_intervals, 1),
            **{f"val_{k}": v for k, v in val.items()},
        }
        round_rows.append(row)
        reporter.emit(
            "round_validation_summary",
            print_keys=[
                "round",
                "logged_days",
                "holdout_r2_z",
                "holdout_mse_z",
                "holdout_min_r2_z_by_t",
                "holdout_max_mse_z_by_t",
                "validation_diag_delta_vs_best_operational",
                "val_spend",
                "mean_policy_uncertainty",
            ],
            **row,
        )

        checkpoint_step = checkpoint_step_for_round(args.method, q, train_cfg, args)
        current_round_step_rows: Optional[List[Dict[str, Any]]] = [] if args.save_step_diagnostics else None
        with reporter.stage("round_true_dgp_eval", round=q, episodes=args.eval_episodes):
            current_round_eval_rows = evaluate_round_true_dgp_rows(
                cfg=sim_cfg,
                coefs=coefs,
                normalizer=normalizer,
                policy=current_policy,
                method=args.method,
                method_name=LEARNING_METHOD_NAMES[args.method],
                episodes=args.eval_episodes,
                seed=args.seed,
                seed_offset=args.final_eval_seed_offset,
                round_idx=q,
                checkpoint_step=checkpoint_step,
                stage="round_true_dgp_eval",
                step_rows_out=current_round_step_rows,
                audit=audit,
                progress_mode=args.progress,
            )
        round_eval_rows.extend(current_round_eval_rows)
        if current_round_step_rows is not None:
            round_eval_step_rows.extend(current_round_step_rows)
        current_round_summary = summarize_round_eval_rows(current_round_eval_rows)
        policy_eval = next((r for r in current_round_summary if str(r.get("method_key", "")) == args.method), {})
        operational_rows = [r for r in current_round_summary if str(r.get("method_key", "")) in {"no_subsidy", "random_pacing", "uniform_pacing", "gap_heuristic"}]
        best_operational_net = max((to_float(r.get("net_reward")) for r in operational_rows), default=float("nan"))
        reporter.emit(
            "round_true_dgp_summary",
            print_keys=["round", "method", "true_net_reward", "spend", "delta_vs_best_operational"],
            round=q,
            method=args.method,
            true_dgp_eval_source="independent_true_dgp_paired_seeds",
            true_net_reward=policy_eval.get("net_reward", ""),
            spend=policy_eval.get("spend", ""),
            delta_vs_best_operational=(
                to_float(policy_eval.get("net_reward")) - best_operational_net
                if math.isfinite(to_float(policy_eval.get("net_reward"))) and math.isfinite(best_operational_net)
                else ""
            ),
            round_eval_summary=current_round_summary,
        )
        simulator_diagnostic_rows.extend(
            summarize_simulator_rollout_diagnostics(
                round_idx=q,
                source_method_key=args.method,
                val_set=val_set,
                round_eval_rows=current_round_eval_rows,
                holdout_metrics=sim_bundle.holdout_metrics,
            )
        )
        interval_simulator_rows.extend(sim_bundle.interval_holdout_rows)
        completed_rounds = q
        write_csv(out_dir / "iterative_rounds.csv", round_rows)
        write_csv(out_dir / "synthetic_training_rows.csv", synthetic_training_rows)
        write_csv(
            out_dir / "round_eval_raw.csv",
            sorted(
                round_eval_rows,
                key=lambda r: (int(r.get("round", 0)), str(r["method"]), int(r["eval_episode"])),
            ),
        )
        write_csv(out_dir / "round_eval_summary.csv", summarize_round_eval_rows(round_eval_rows))
        write_csv(out_dir / "simulator_diagnostics.csv", simulator_diagnostic_rows)
        write_csv(out_dir / "simulator_vs_true_policy_calibration.csv", simulator_diagnostic_rows)
        write_csv(out_dir / "interval_simulator_diagnostics.csv", interval_simulator_rows)
        if args.save_step_diagnostics:
            write_csv(
                out_dir / "round_eval_step_raw.csv",
                sorted(
                    round_eval_step_rows,
                    key=lambda r: (
                        int(r.get("round", 0)),
                        str(r.get("method", "")),
                        int(r.get("eval_episode", 0)),
                        int(r.get("t", 0)),
                    ),
                ),
            )
            write_csv(out_dir / "round_eval_step_summary.csv", summarize_step_rows(round_eval_step_rows))

        sim_round_dir = ensure_dir(out_dir / "simulators" / f"round_{q}")
        pooled_state = main_models[0].state_dict()
        torch.save(pooled_state, sim_round_dir / "pooled_time_conditioned_response_model.pt")
        # Backward-compatible aliases for older analysis scripts. The maintained
        # simulator is pooled and time-conditioned; these are not independent
        # per-interval models.
        torch.save(pooled_state, sim_round_dir / "pooled_response_model.pt")
        torch.save([pooled_state], sim_round_dir / "main_response_models.pt")
        save_json(
            sim_round_dir / "compatibility_aliases.json",
            {
                "pooled_time_conditioned_response_model.pt": "canonical pooled time-conditioned simulator checkpoint",
                "pooled_response_model.pt": "backward-compatible alias of the canonical pooled simulator",
                "main_response_models.pt": "backward-compatible one-element list alias; no independent per-time models are saved",
            },
        )
        save_json(
            sim_round_dir / "metadata.json",
            {
                "round": q,
                "simulator_type": "pooled_time_conditioned_multihead_nn",
                "simulator_checkpoint": "pooled_time_conditioned_response_model.pt",
                "interval_checkpoint_copies_saved": False,
                "interval_checkpoint_note": "time heterogeneity is represented by tau_t inputs to one shared pooled NN, not by separate per-interval simulator checkpoints",
                "time_intervals": sim_cfg.Time_intervals,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "simulator_input_dim": simulator_input_dim(state_dim, action_dim),
                "sim_hidden_layers": list(args.sim_hidden_layers),
                "sim_head_hidden_dim": args.sim_head_hidden_dim,
                "sim_dropout": args.sim_dropout,
                "sim_activation": args.sim_activation,
                "sim_weight_decay": args.sim_weight_decay,
                "simulator_action_features": ["A_over_B", "A_over_B_squared", "total_spend_over_B"],
                "center_residual": bool(args.center_residual),
                "time_feature_tau_convention": "tau=t_idx+1; sin(2*pi*tau/T), cos(2*pi*tau/T), (T-tau+1)/T",
                "output_dim": 4 * sim_cfg.M,
                "sim_cfg": json_safe(asdict(sim_cfg)),
                "state_normalizer": None if normalizer is None else normalizer.state_dict(),
                "response_scale": response_scale.detach().cpu().tolist(),
                "main_model_metrics": sim_bundle.main_model_metrics,
                "bootstrap_metrics": sim_bundle.bootstrap_metrics,
                "holdout_metrics": sim_bundle.holdout_metrics,
                "sim_update_mode": args.sim_update_mode,
                "sim_train_epochs_used": sim_bundle.sim_epochs_used,
                "bootstrap_epochs_used": sim_bundle.bootstrap_epochs_used,
                "use_pessimism": pess_cfg.use_pessimism,
                "lambda_pess": pess_cfg.lambda_pess,
                "b_boot": pess_cfg.b_boot if pess_cfg.use_pessimism else 0,
                "uncertainty_penalty_scope": pess_cfg.uncertainty_penalty_scope,
                "logging_policy": args.logging_policy,
                "mixed_logging_policy": logging_policy_description(args.logging_policy),
            },
        )
        if pess_cfg.save_bootstrap_ensemble:
            boot_dir = ensure_dir(sim_round_dir / "bootstrap_ensemble")
            for t_idx, models_t in enumerate(bootstrap_ensemble):
                time_dir = ensure_dir(boot_dir / f"t{t_idx}")
                for b_idx, model_b in enumerate(models_t):
                    torch.save(model_b.state_dict(), time_dir / f"bootstrap_response_model_{b_idx}.pt")

        if q == args.q_max:
            break

        if new_logged_for_next_round is None:
            with reporter.stage("collect_on_policy_log", round=q, episodes=args.new_logged_days):
                new_logged_for_next_round = collect_logged_trajectories(
                    sim_cfg,
                    coefs,
                    episodes=args.new_logged_days,
                    seed=args.seed + 70_000 + q * 1_000,
                    normalizer=normalizer,
                    policy=current_policy,
                    deterministic=False,
                    random_logging_spend_scale=args.logging_spend_scale,
                    logging_policy=args.logging_policy,
                    audit=audit,
                    audit_phase="on_policy_logged_collection",
                    progress_mode=args.progress,
                    progress_desc=f"round {q} collect on-policy log",
                )
        else:
            reporter.emit(
                "collect_on_policy_log_reused_from_real_online_update",
                print_keys=["round", "episodes"],
                round=q,
                episodes=len(new_logged_for_next_round),
            )
        new_logged = new_logged_for_next_round
        logged.extend(new_logged)
        j_prev = j_q
        j_val_pess_prev = j_val_pess
        validation_diag_delta_prev = validation_diag_delta_vs_best_operational
        main_models_prev = main_models
        bootstrap_ensemble_prev = bootstrap_ensemble
        if args.method == "dfl_pto" and isinstance(current_policy, DFLPolicy):
            dfl_policy_prev = current_policy

    with reporter.stage("write_final_outputs", completed_rounds=completed_rounds):
        write_csv(out_dir / "iterative_rounds.csv", round_rows)
        write_csv(out_dir / "synthetic_training_rows.csv", synthetic_training_rows)
        round_eval_rows = sorted(
            round_eval_rows,
            key=lambda r: (int(r.get("round", 0)), str(r["method"]), int(r["eval_episode"])),
        )
        round_eval_summary = summarize_round_eval_rows(round_eval_rows)
        write_csv(out_dir / "round_eval_raw.csv", round_eval_rows)
        write_csv(out_dir / "round_eval_summary.csv", round_eval_summary)
        write_csv(out_dir / "simulator_diagnostics.csv", simulator_diagnostic_rows)
        write_csv(out_dir / "simulator_vs_true_policy_calibration.csv", simulator_diagnostic_rows)
        write_csv(out_dir / "interval_simulator_diagnostics.csv", interval_simulator_rows)
        if args.save_step_diagnostics:
            round_eval_step_rows = sorted(
                round_eval_step_rows,
                key=lambda r: (
                    int(r.get("round", 0)),
                    str(r.get("method", "")),
                    int(r.get("eval_episode", 0)),
                    int(r.get("t", 0)),
                ),
            )
            write_csv(out_dir / "round_eval_step_raw.csv", round_eval_step_rows)
            write_csv(out_dir / "round_eval_step_summary.csv", summarize_step_rows(round_eval_step_rows))

    if completed_rounds <= 0:
        raise RuntimeError("No outer-loop round completed.")
    final_rows: List[Dict[str, Any]] = []
    for row in round_eval_rows:
        if int(row.get("round", 0)) != int(completed_rounds):
            continue
        final_row = dict(row)
        final_row["stage"] = "final_true_dgp_eval"
        final_rows.append(final_row)
    final_rows = sorted(final_rows, key=lambda r: (str(r["method"]), int(r["eval_episode"])))
    write_csv(out_dir / "final_eval_raw.csv", final_rows)
    if args.save_step_diagnostics:
        final_step_rows: List[Dict[str, Any]] = []
        for row in round_eval_step_rows:
            if int(row.get("round", 0)) != int(completed_rounds):
                continue
            final_step_row = dict(row)
            final_step_row["stage"] = "final_true_dgp_eval"
            final_step_rows.append(final_step_row)
        write_csv(out_dir / "final_eval_step_raw.csv", final_step_rows)
        write_csv(out_dir / "final_eval_step_summary.csv", summarize_step_rows(final_step_rows))

    summary_rows: List[Dict[str, Any]] = []
    for method in sorted({str(r["method"]) for r in final_rows}):
        rows = [r for r in final_rows if str(r["method"]) == method]
        summary_rows.append({"method": method, "eval_episodes": len(rows), **summarize_eval_rows(rows)})
    write_csv(out_dir / "final_eval_summary_table.csv", summary_rows)

    with reporter.stage("policy_support_diagnostics", completed_rounds=completed_rounds, episodes=args.eval_episodes):
        policy_support_rows = compute_policy_support_diagnostics(
            logged=logged,
            cfg=sim_cfg,
            coefs=coefs,
            policy=current_policy,
            method=args.method,
            method_name=LEARNING_METHOD_NAMES[args.method],
            normalizer=normalizer,
            episodes=args.eval_episodes,
            seed=args.seed,
            seed_offset=args.final_eval_seed_offset,
            round_idx=completed_rounds,
        )
    write_csv(out_dir / "policy_support_diagnostics.csv", policy_support_rows)

    if args.save_models:
        if current_policy is not None and hasattr(current_policy, "state_dict"):
            torch.save(current_policy.state_dict(), out_dir / f"last_{args.method}_policy.pt")
        if agent is not None:
            torch.save(agent.actor.state_dict(), out_dir / f"last_{args.method}_actor.pt")

    save_json(out_dir / "audit_summary.json", {"audit_checks": bool(args.audit_checks), "strict_audit": bool(args.strict_audit), "warning_count": audit.count})
    final_policy_summary = next((r for r in summary_rows if str(r.get("method", "")) == LEARNING_METHOD_NAMES[args.method]), {})
    reporter.emit(
        "run_complete",
        print_keys=["completed_rounds", "final_net_reward", "final_spend", "audit_warning_count", "out_dir"],
        completed_rounds=completed_rounds,
        final_net_reward=final_policy_summary.get("net_reward", ""),
        final_spend=final_policy_summary.get("spend", ""),
        audit_warning_count=audit.count,
        out_dir=str(out_dir.resolve()),
    )


if __name__ == "__main__":
    main()
