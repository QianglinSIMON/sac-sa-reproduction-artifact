"""Training and evaluation layer for the frozen public city-panel true DGP.

Pipeline
--------
rich true-DGP logging -> learned neural simulator -> residual bootstrap ->
Beta-Dirichlet SAC-SA or DFL-PTO -> independent true-DGP evaluation.

No true-DGP mechanism is defined or changed in this file.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from city_panel_models import (
    EPS,
    FeatureNormalizer,
    LearnedSimulatorNet,
    OneStepScoreNet,
    ReplayBuffer,
    SACAgent,
    SACConfig,
    feasible_action,
    remaining_budget,
)
from public_city_panel_dgp import (
    DYNAMIC_DIM,
    STATIC_DIM,
    BatchedPublicCityPanelEnv,
    PublicCityPanelConfig,
    PublicCityPanelEnv,
    _project_dynamic,
    _state_from_dynamic,
    baseline_action,
    baseline_action_from_state,
)


BASELINE_METHODS = (
    "no_subsidy",
    "uniform_pacing",
    "gap_pacing",
    "random_pacing",
)
PROTOCOL_ID = "public-city-simsel-v1"
METHOD_LABELS = {
    "sacsa_reserve": "Proposed Beta-Dirichlet SAC-SA",
    "dfl_pto": "DFL-PTO",
    "no_subsidy": "No Subsidy",
    "uniform_pacing": "Uniform Pacing",
    "gap_pacing": "Demand-Supply Gap",
    "random_pacing": "Random Pacing",
}


@dataclass
class ExperimentConfig:
    M: int = 3
    T: int = 20
    daily_budget: float = 10.0
    seed: int = 2026
    device: str = "cpu"
    q_max: int = 3
    initial_logged_days: int = 3000
    new_logged_days: int = 200
    residual_calibration_days: int = 800
    validation_logged_days: int = 600
    holdout_logged_days: int = 600
    eval_episodes: int = 1000
    dgp_calibration_days: int = 10000
    dgp_baseline_epochs: int = 100
    logging_batch_size: int = 512


@dataclass
class SimulatorConfig:
    hidden_dims: Tuple[int, ...] = (768, 512, 256)
    head_dim: int = 256
    epochs: int = 300
    finetune_epochs: int = 120
    batch_size: int = 4096
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 30
    z_loss_weight: float = 4.0
    bootstrap_models: int = 8
    bootstrap_epochs: int = 100
    residual_center: bool = True
    min_holdout_r2_z: float = 0.75
    min_holdout_r2_z_by_t: float = 0.30
    strict_quality_gate: bool = False


@dataclass
class PolicyConfig:
    hidden_dims: Tuple[int, ...] = (512, 512, 256)
    train_episodes: int = 6000
    later_round_train_episodes: int = 0
    rollout_batch_size: int = 256
    updates_per_rollout_batch: int = 128
    warmup_episodes: int = 512
    batch_size: int = 1024
    replay_size: int = 750_000
    init_spend_share: float = 0.10
    lambda_pess: float = 0.02
    auto_alpha: bool = False
    fixed_temperature: float = 0.05
    real_online_update: bool = True
    real_gradient_steps_per_step: float = 1.0
    real_critic_warmup_updates: int = 0
    real_exploration_epsilon: float = 0.40
    behavior_support_lambda: float = 0.05
    support_prototype_count: int = 512
    simulator_selection_episodes: int = 1200
    simulator_selection_z_value: float = 1.645
    simulator_selection_tolerance: float = 0.0
    dfl_epochs: int = 150
    dfl_batch_size: int = 1024
    dfl_candidate_random_count: int = 128
    dfl_hidden_dims: Tuple[int, ...] = (256, 256)
    dfl_learning_rate: float = 1e-3
    dfl_decision_weight: float = 1.0
    dfl_softmax_temperature: float = 1.0
    dfl_budget_fractions: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)


@dataclass
class TrajectoryData:
    states: torch.Tensor
    actions: torch.Tensor
    z: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    design_spend_fraction: torch.Tensor
    design_mode: torch.Tensor
    design_dominant_city: torch.Tensor
    design_profile: torch.Tensor
    is_logging_action: torch.Tensor
    seeds: torch.Tensor
    noise_indices: torch.Tensor

    @property
    def days(self) -> int:
        return int(self.states.shape[0])

    @property
    def T(self) -> int:
        return int(self.states.shape[1])

    def to(self, device: torch.device | str) -> "TrajectoryData":
        values: Dict[str, torch.Tensor] = {}
        for name in self.__dataclass_fields__:
            values[name] = getattr(self, name).to(device)
        return TrajectoryData(**values)

    @staticmethod
    def concatenate(parts: Sequence["TrajectoryData"]) -> "TrajectoryData":
        if not parts:
            raise ValueError("Cannot concatenate an empty trajectory collection.")
        return TrajectoryData(
            **{
                name: torch.cat([getattr(part, name).cpu() for part in parts], dim=0)
                for name in parts[0].__dataclass_fields__
            }
        )


@dataclass
class OutputNormalizer:
    z_mean: torch.Tensor
    z_scale: torch.Tensor
    dynamic_mean: torch.Tensor
    dynamic_scale: torch.Tensor

    def to(self, device: torch.device | str) -> "OutputNormalizer":
        self.z_mean = self.z_mean.to(device)
        self.z_scale = self.z_scale.to(device)
        self.dynamic_mean = self.dynamic_mean.to(device)
        self.dynamic_scale = self.dynamic_scale.to(device)
        return self


@dataclass
class SimulatorBundle:
    model: LearnedSimulatorNet
    ensemble: List[LearnedSimulatorNet]
    state_normalizer: FeatureNormalizer
    output_normalizer: OutputNormalizer
    residual_z: torch.Tensor
    residual_dynamic: torch.Tensor
    initial_states: torch.Tensor
    z_bias_by_t: torch.Tensor
    dynamic_bias_by_t: torch.Tensor
    reward_center_by_t: torch.Tensor
    reward_scale: float
    O: torch.Tensor
    panel_config: PublicCityPanelConfig
    diagnostics: Dict[str, float]
    support_mean_by_t: torch.Tensor
    support_scale_by_t: torch.Tensor
    support_prototypes_by_t: torch.Tensor
    support_threshold_by_t: torch.Tensor


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


class EventLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()

    def log(self, event: str, **values: Any) -> None:
        row = {"elapsed_sec": time.monotonic() - self.start, "event": event, **values}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        fields = " | ".join(f"{key}={value}" for key, value in values.items())
        print(f"[{row['elapsed_sec']:9.1f}s] {event}" + (f" | {fields}" if fields else ""), flush=True)


def make_panel_config(config: ExperimentConfig) -> PublicCityPanelConfig:
    return PublicCityPanelConfig(
        M=config.M,
        T=config.T,
        daily_budget=config.daily_budget,
        device=config.device,
        calibration_episodes=config.dgp_calibration_days,
        baseline_epochs=config.dgp_baseline_epochs,
    )


def dynamic_from_state(state: torch.Tensor, M: int) -> torch.Tensor:
    """Invert the dynamic blocks written by public_city_panel_dgp._state_from_dynamic."""
    if state.shape[-1] != (DYNAMIC_DIM + 1 + STATIC_DIM) * M + 5:
        raise ValueError(f"Unexpected state dimension {state.shape[-1]} for M={M}.")
    blocks = [state[..., 0:M], state[..., M : 2 * M]]
    # Block 2 is derived pressure; blocks 3..9 contain dynamic coordinates 2..8.
    blocks.extend(state[..., block * M : (block + 1) * M] for block in range(3, 10))
    return torch.stack(blocks, dim=-1)


def action_feature_dim(M: int) -> int:
    return 2 * int(M) + 3


def action_features(action: torch.Tensor, state: torch.Tensor, budget: float) -> torch.Tensor:
    spend = action.sum(dim=-1, keepdim=True)
    shares = torch.where(spend > EPS, action / spend.clamp_min(EPS), torch.zeros_like(action))
    hhi = shares.square().sum(dim=-1, keepdim=True)
    max_share = shares.max(dim=-1, keepdim=True).values
    city_reference = max(float(budget) / max(action.shape[-1], 1), EPS)
    features = torch.cat(
        [action / city_reference, shares, spend / max(float(budget), EPS), hhi, max_share],
        dim=-1,
    )
    if features.shape[-1] != action_feature_dim(action.shape[-1]):
        raise RuntimeError("Action-feature layout no longer matches action_feature_dim().")
    return features


def simulator_inputs(
    state: torch.Tensor,
    action: torch.Tensor,
    state_normalizer: FeatureNormalizer,
    budget: float,
    time_index: torch.Tensor | int,
    T: int,
) -> torch.Tensor:
    index = torch.as_tensor(time_index, device=state.device, dtype=torch.long)
    if index.ndim == 0:
        index = index.expand(state.shape[:-1])
    one_hot = torch.nn.functional.one_hot(index.clamp(0, T - 1), num_classes=T).to(state.dtype)
    return torch.cat(
        [state_normalizer.transform(state), action_features(action, state, budget), one_hot],
        dim=-1,
    )


def state_action_support_features(
    state: torch.Tensor,
    action: torch.Tensor,
    state_normalizer: FeatureNormalizer,
    budget: float,
) -> torch.Tensor:
    """Features used consistently by support regularization and diagnostics."""
    return torch.cat(
        [
            state_normalizer.transform(state),
            action_features(action, state, budget),
        ],
        dim=-1,
    )


def balance_support_feature_groups(
    standardized: torch.Tensor,
    state_dim: int,
) -> torch.Tensor:
    """Give state and action blocks comparable total weight in distances."""
    action_dim = int(standardized.shape[-1] - state_dim)
    output = standardized.clone()
    output[..., :state_dim] /= math.sqrt(max(state_dim, 1))
    output[..., state_dim:] /= math.sqrt(max(action_dim, 1))
    return output


@torch.no_grad()
def fit_support_reference(
    train: TrajectoryData,
    state_normalizer: FeatureNormalizer,
    config: ExperimentConfig,
    prototype_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit per-period empirical state-action support references.

    The threshold is the 95th percentile leave-one-out nearest-neighbor
    distance among logged prototypes.  A policy is penalized only for distance
    beyond this empirical within-support radius, rather than for every
    deviation from a behavior-policy mean.
    """
    device = torch.device(config.device)
    count = min(max(int(prototype_count), 2), train.days)
    means: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    prototypes: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 44_771)
    for t in range(config.T):
        features = state_action_support_features(
            train.states[:, t].to(device),
            train.actions[:, t].to(device),
            state_normalizer,
            config.daily_budget,
        )
        mean = features.mean(dim=0)
        scale = features.std(dim=0, unbiased=False).clamp_min(0.05)
        standardized = balance_support_feature_groups(
            (features - mean) / scale,
            train.states.shape[-1],
        )
        indices = torch.randperm(train.days, generator=generator)[:count].to(device)
        selected = standardized[indices]
        distances = torch.cdist(selected, selected)
        distances.fill_diagonal_(float("inf"))
        nearest = distances.min(dim=1).values
        means.append(mean)
        scales.append(scale)
        prototypes.append(selected)
        thresholds.append(torch.quantile(nearest, 0.95))
    return (
        torch.stack(means),
        torch.stack(scales),
        torch.stack(prototypes),
        torch.stack(thresholds),
    )


def _temporal_profiles(T: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(T, device=device, dtype=torch.float32)
    tau = (t + 0.5) / float(T)
    profiles = [
        torch.ones(T, device=device),
        torch.exp(-0.5 * ((tau - 0.25) / 0.12).square()),
        torch.exp(-0.5 * ((tau - 0.75) / 0.14).square()),
        torch.exp(-0.5 * ((tau - 0.25) / 0.12).square())
        + 1.2 * torch.exp(-0.5 * ((tau - 0.75) / 0.14).square()),
        torch.linspace(2.0, 0.2, T, device=device),
        torch.linspace(0.2, 2.0, T, device=device),
    ]
    # Smooth profiles alone leave little support for policies that concentrate
    # spend in one decision period.  Add one pulse profile per period while
    # retaining a small background allocation over the rest of the day.
    for peak in range(T):
        pulse = torch.full((T,), 0.35 / max(T - 1, 1), device=device)
        pulse[peak] = 0.65
        profiles.append(pulse)
    # Short blocks provide repeated support for moderately concentrated spend
    # at every period; isolated pulses alone would remain below a per-period
    # empirical 95th percentile in a long horizon.
    for center in range(T):
        active = sorted({max(center - 1, 0), center, min(center + 1, T - 1)})
        inactive_count = max(T - len(active), 1)
        block = torch.full((T,), 0.25 / inactive_count, device=device)
        block[active] = 0.75 / len(active)
        profiles.append(block)
    return torch.stack([profile / profile.sum().clamp_min(EPS) for profile in profiles])


def _logging_designs(
    n: int,
    T: int,
    M: int,
    budget: float,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Balanced strata over spend, timing, concentration, and dominant city.

    A seeded permutation of the full factorial design is used for each cycle.
    This keeps every marginal represented even when a logging batch contains
    fewer days than the complete design.  In particular, batching must not
    repeatedly restart at the first temporal profiles.
    """
    spend_levels = torch.tensor(
        [0.0, 0.15, 0.30, 0.50, 0.70, 0.90, 1.0], device=device
    )
    temporal_profiles = _temporal_profiles(T, device)
    allocation_mode_count = 8
    factorial = np.asarray(
        list(
            product(
                range(spend_levels.numel()),
                range(allocation_mode_count),
                range(M),
                range(temporal_profiles.shape[0]),
            )
        ),
        dtype=np.int64,
    )
    numpy_generator = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    remaining = int(n)
    while remaining > 0:
        permutation = numpy_generator.permutation(len(factorial))
        take = min(remaining, len(factorial))
        selected.append(factorial[permutation[:take]])
        remaining -= take
    design = np.concatenate(selected, axis=0)
    spend_index = torch.as_tensor(design[:, 0], device=device)
    mode = torch.as_tensor(design[:, 1], device=device)
    dominant = torch.as_tensor(design[:, 2], device=device)
    profile_index = torch.as_tensor(design[:, 3], device=device)
    spend_fraction = spend_levels[spend_index]
    profiles = temporal_profiles[profile_index]
    random_weights = torch.empty((n, T, M), device=device)
    low_concentration = 0.20
    mid_concentration = 1.0
    high_concentration = 4.0
    for mode_id, concentration in ((2, low_concentration), (3, mid_concentration), (5, high_concentration)):
        mask = mode == mode_id
        count = int(mask.sum().item())
        if count:
            gamma = torch.as_tensor(
                numpy_generator.gamma(
                    shape=concentration,
                    scale=1.0,
                    size=(count, T, M),
                ),
                device=device,
                dtype=torch.float32,
            ).clamp_min(EPS)
            random_weights[mask] = gamma / gamma.sum(dim=-1, keepdim=True)
    for mode_id in (0, 1, 6, 7):
        random_weights[mode == mode_id] = 1.0 / M
    dominant_mask = mode == 4
    random_weights[dominant_mask] = 0.15 / max(M - 1, 1)
    if dominant_mask.any():
        rows = torch.nonzero(dominant_mask, as_tuple=False).flatten()
        random_weights[rows, :, dominant[rows]] = 0.85
    return {
        "spend_fraction": spend_fraction,
        "mode": mode,
        "dominant": dominant,
        "profile": profile_index,
        "profiles": profiles,
        "random_weights": random_weights,
        "target_spend": spend_fraction * float(budget),
    }


def collect_rich_logging(
    panel_config: PublicCityPanelConfig,
    days: int,
    seed_offset: int,
    noise_split: str,
    batch_size: int = 512,
    policy: Optional[Any] = None,
    policy_normalizer: Optional[FeatureNormalizer] = None,
    exploration_epsilon: float = 0.0,
) -> TrajectoryData:
    """Collect day trajectories with explicit logging-support strata."""
    if days <= 0:
        raise ValueError("Logging days must be positive.")
    device = torch.device(panel_config.device)
    chunks: list[TrajectoryData] = []
    full_design = _logging_designs(
        int(days),
        int(panel_config.T),
        int(panel_config.M),
        float(panel_config.daily_budget),
        int(seed_offset),
        device,
    )
    for start in range(0, int(days), int(batch_size)):
        count = min(int(batch_size), int(days) - start)
        seeds = [int(seed_offset + start + i) for i in range(count)]
        env = BatchedPublicCityPanelEnv(panel_config, seeds, noise_split=noise_split)
        state = env.reset()
        design = {
            name: value[start : start + count]
            for name, value in full_design.items()
        }
        use_logging = torch.ones(count, dtype=torch.bool, device=device)
        if policy is not None:
            generator = torch.Generator(device=device).manual_seed(seed_offset + start + 991)
            use_logging = torch.rand(count, generator=generator, device=device) < float(exploration_epsilon)
        state_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        z_rows: list[torch.Tensor] = []
        reward_rows: list[torch.Tensor] = []
        next_rows: list[torch.Tensor] = []
        done_rows: list[torch.Tensor] = []
        for t in range(env.T):
            period_spend = design["target_spend"] * design["profiles"][:, t]
            gap = env.gap_weights()
            shares = design["random_weights"][:, t].clone()
            shares[design["mode"] == 1] = gap[design["mode"] == 1]
            hybrid = design["mode"] == 5
            shares[hybrid] = 0.5 * shares[hybrid] + 0.5 * gap[hybrid]
            demand = design["mode"] == 6
            if demand.any():
                X = state[demand, : env.M].clamp_min(EPS)
                shares[demand] = X / X.sum(dim=-1, keepdim=True).clamp_min(EPS)
            shortage_ratio = design["mode"] == 7
            if shortage_ratio.any():
                X = state[shortage_ratio, : env.M].clamp_min(0.0)
                Y = state[shortage_ratio, env.M : 2 * env.M].clamp_min(0.0)
                weights = (X + 1.0) / (Y + 1.0)
                shares[shortage_ratio] = weights / weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(EPS)
            shares = shares / shares.sum(dim=-1, keepdim=True).clamp_min(EPS)
            logging_action = shares * torch.minimum(period_spend, env.H).unsqueeze(-1)
            logging_action[design["spend_fraction"] == 0.0] = 0.0
            if policy is None:
                action = logging_action
            else:
                assert policy_normalizer is not None
                normalized = policy_normalizer.transform(state)
                if hasattr(policy, "act_batch"):
                    with torch.no_grad():
                        policy_action = policy.act_batch(state, t)
                else:
                    with torch.no_grad():
                        policy_action, _, _, _ = policy.actor.sample(normalized, deterministic=True)
                action = torch.where(use_logging.unsqueeze(-1), logging_action, policy_action)
            action = feasible_action(action, state)
            next_state, reward, done, info = env.step(action)
            terminal = torch.full((count,), float(done), device=device)
            next_value = torch.zeros_like(state) if next_state is None else next_state
            state_rows.append(state.cpu())
            action_rows.append(action.cpu())
            z_rows.append(info["Z"].cpu())
            reward_rows.append(reward.cpu())
            next_rows.append(next_value.cpu())
            done_rows.append(terminal.cpu())
            if next_state is not None:
                state = next_state
        chunks.append(
            TrajectoryData(
                states=torch.stack(state_rows, dim=1),
                actions=torch.stack(action_rows, dim=1),
                z=torch.stack(z_rows, dim=1),
                rewards=torch.stack(reward_rows, dim=1),
                next_states=torch.stack(next_rows, dim=1),
                dones=torch.stack(done_rows, dim=1),
                design_spend_fraction=design["spend_fraction"].cpu(),
                design_mode=design["mode"].cpu(),
                design_dominant_city=design["dominant"].cpu(),
                design_profile=design["profile"].cpu(),
                is_logging_action=use_logging.cpu(),
                seeds=torch.tensor(seeds, dtype=torch.long),
                noise_indices=env.noise_indices.detach().cpu().clone(),
            )
        )
    return TrajectoryData.concatenate(chunks)


def data_split_diagnostics(
    named_data: Dict[str, TrajectoryData],
) -> List[Dict[str, Any]]:
    """Audit residual-day pools and pairwise overlap across downstream splits."""
    rows: list[dict[str, Any]] = []
    names = list(named_data)
    index_sets = {
        name: set(int(value) for value in data.noise_indices.tolist())
        for name, data in named_data.items()
    }
    for name in names:
        data = named_data[name]
        rows.append(
            {
                "record_type": "split",
                "split_a": name,
                "split_b": "",
                "days": data.days,
                "unique_seeds": int(torch.unique(data.seeds).numel()),
                "unique_residual_days": len(index_sets[name]),
                "paired_initial_state_and_residual": 1,
                "overlap_residual_days": 0,
            }
        )
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = index_sets[left].intersection(index_sets[right])
            rows.append(
                {
                    "record_type": "overlap",
                    "split_a": left,
                    "split_b": right,
                    "days": "",
                    "unique_seeds": "",
                    "unique_residual_days": "",
                    "paired_initial_state_and_residual": 1,
                    "overlap_residual_days": len(overlap),
                }
            )
    return rows


def logging_coverage_rows(data: TrajectoryData, budget: float) -> List[Dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spend = data.actions.sum(dim=-1)
    shares = torch.where(
        spend.unsqueeze(-1) > EPS,
        data.actions / spend.unsqueeze(-1).clamp_min(EPS),
        torch.zeros_like(data.actions),
    )
    hhi = shares.square().sum(dim=-1)
    dominant = shares.argmax(dim=-1)
    spend_bins = torch.bucketize(spend / max(float(budget), EPS), torch.tensor([0.01, 0.03, 0.06, 0.10, 0.20]))
    concentration_bins = torch.bucketize(hhi, torch.tensor([0.40, 0.60, 0.80]))
    for t in range(data.T):
        for spend_bin in range(6):
            for concentration_bin in range(4):
                mask = (spend_bins[:, t] == spend_bin) & (concentration_bins[:, t] == concentration_bin)
                if not mask.any():
                    continue
                for city in range(data.actions.shape[-1]):
                    city_mask = mask & (dominant[:, t] == city)
                    if city_mask.any():
                        rows.append(
                            {
                                "t": t,
                                "spend_bin": spend_bin,
                                "concentration_bin": concentration_bin,
                                "dominant_city": city,
                                "count": int(city_mask.sum()),
                                "mean_step_spend": float(spend[city_mask, t].mean()),
                            }
                        )
    return rows


def logging_design_summary_rows(data: TrajectoryData) -> List[Dict[str, Any]]:
    """Marginal counts for the explicit historical logging design."""
    rows: list[dict[str, Any]] = []
    logging_mask = data.is_logging_action.bool()
    fields = {
        "spend_fraction": data.design_spend_fraction,
        "allocation_mode": data.design_mode,
        "dominant_city": data.design_dominant_city,
        "temporal_profile": data.design_profile,
    }
    for dimension, values in fields.items():
        selected = values[logging_mask]
        for value in torch.unique(selected, sorted=True):
            rows.append(
                {
                    "dimension": dimension,
                    "level": float(value) if value.is_floating_point() else int(value),
                    "count": int((selected == value).sum()),
                    "logging_days": int(logging_mask.sum()),
                    "on_policy_days": int((~logging_mask).sum()),
                }
            )
    return rows


def logging_time_support_rows(
    data: TrajectoryData,
    budget: float,
) -> List[Dict[str, Any]]:
    """Compact per-period audit of the realized state-action logging support."""
    spend = data.actions.sum(dim=-1)
    positive = spend > EPS
    shares = torch.where(
        positive.unsqueeze(-1),
        data.actions / spend.unsqueeze(-1).clamp_min(EPS),
        torch.zeros_like(data.actions),
    )
    hhi = shares.square().sum(dim=-1)
    dominant = shares.argmax(dim=-1)
    rows: list[dict[str, Any]] = []
    for t in range(data.T):
        positive_t = positive[:, t]
        positive_spend = spend[positive_t, t]
        positive_hhi = hhi[positive_t, t]
        row: dict[str, Any] = {
            "t": t,
            "days": data.days,
            "logging_day_share": float(data.is_logging_action.float().mean()),
            "positive_action_share": float(positive_t.float().mean()),
            "step_spend_p05_fraction_of_budget": 0.0,
            "step_spend_p50_fraction_of_budget": 0.0,
            "step_spend_p95_fraction_of_budget": 0.0,
            "allocation_hhi_p05": 0.0,
            "allocation_hhi_p50": 0.0,
            "allocation_hhi_p95": 0.0,
        }
        if positive_spend.numel():
            for label, quantile in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95)):
                row[f"step_spend_{label}_fraction_of_budget"] = float(
                    torch.quantile(positive_spend, quantile) / max(float(budget), EPS)
                )
                row[f"allocation_hhi_{label}"] = float(
                    torch.quantile(positive_hhi, quantile)
                )
        for city in range(data.actions.shape[-1]):
            row[f"dominant_city_{city}_positive_count"] = int(
                (positive_t & (dominant[:, t] == city)).sum()
            )
        rows.append(row)
    return rows


def _fit_output_normalizer(data: TrajectoryData, M: int) -> OutputNormalizer:
    z = data.z.reshape(-1, M).float()
    mask = ~data.dones.bool()
    dynamic = dynamic_from_state(data.next_states[mask], M).reshape(-1, M * DYNAMIC_DIM).float()
    return OutputNormalizer(
        z_mean=z.mean(dim=0),
        z_scale=z.std(dim=0, unbiased=False).clamp_min(1e-3),
        dynamic_mean=dynamic.mean(dim=0),
        dynamic_scale=dynamic.std(dim=0, unbiased=False).clamp_min(1e-3),
    )


def _model_loss(
    model: LearnedSimulatorNet,
    inputs: torch.Tensor,
    z_target: torch.Tensor,
    dynamic_target: torch.Tensor,
    transition_mask: torch.Tensor,
    z_weight: float,
) -> torch.Tensor:
    z_pred, dynamic_pred = model(inputs)
    z_loss = torch.nn.functional.smooth_l1_loss(z_pred, z_target)
    if transition_mask.any():
        dynamic_loss = torch.nn.functional.smooth_l1_loss(
            dynamic_pred[transition_mask], dynamic_target[transition_mask]
        )
    else:
        dynamic_loss = torch.zeros((), device=inputs.device)
    return float(z_weight) * z_loss + dynamic_loss


def _flat_model_tensors(
    data: TrajectoryData,
    state_normalizer: FeatureNormalizer,
    output_normalizer: OutputNormalizer,
    M: int,
    budget: float,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    states = data.states.reshape(-1, data.states.shape[-1]).to(device)
    actions = data.actions.reshape(-1, M).to(device)
    z = data.z.reshape(-1, M).to(device)
    next_states = data.next_states.reshape(-1, data.states.shape[-1]).to(device)
    transition_mask = (~data.dones.reshape(-1).bool()).to(device)
    next_dynamic = dynamic_from_state(next_states, M).reshape(-1, M * DYNAMIC_DIM)
    time_index = torch.arange(data.T, device=device).repeat(data.days)
    inputs = simulator_inputs(states, actions, state_normalizer, budget, time_index, data.T)
    z_target = (z - output_normalizer.z_mean) / output_normalizer.z_scale
    dynamic_target = (
        next_dynamic - output_normalizer.dynamic_mean
    ) / output_normalizer.dynamic_scale
    return inputs, z_target, dynamic_target, transition_mask


def _fit_one_simulator(
    train: TrajectoryData,
    validation: TrajectoryData,
    state_normalizer: FeatureNormalizer,
    output_normalizer: OutputNormalizer,
    config: ExperimentConfig,
    simulator_config: SimulatorConfig,
    epochs: int,
    seed: int,
    initial_model: Optional[LearnedSimulatorNet] = None,
    bootstrap_days: Optional[torch.Tensor] = None,
) -> LearnedSimulatorNet:
    device = torch.device(config.device)
    torch.manual_seed(int(seed))
    feature_dim = action_feature_dim(config.M)
    input_dim = train.states.shape[-1] + feature_dim + config.T
    model = LearnedSimulatorNet(
        input_dim,
        config.M,
        DYNAMIC_DIM,
        simulator_config.hidden_dims,
        simulator_config.head_dim,
        state_dim=train.states.shape[-1],
        action_feature_dim=feature_dim,
    ).to(device)
    if initial_model is not None:
        model.load_state_dict(copy.deepcopy(initial_model.state_dict()))
    if bootstrap_days is not None:
        indices = bootstrap_days.long().cpu()
        train = TrajectoryData(
            **{name: getattr(train, name)[indices] for name in train.__dataclass_fields__}
        )
    train_tensors = _flat_model_tensors(
        train, state_normalizer, output_normalizer, config.M, config.daily_budget, device
    )
    validation_tensors = _flat_model_tensors(
        validation, state_normalizer, output_normalizer, config.M, config.daily_budget, device
    )
    dataset = TensorDataset(*train_tensors)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=simulator_config.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=simulator_config.learning_rate,
        weight_decay=simulator_config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    for _epoch in range(int(epochs)):
        model.train()
        for batch in loader:
            loss = _model_loss(model, *batch, simulator_config.z_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _model_loss(model, *validation_tensors, simulator_config.z_loss_weight)
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= simulator_config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def _predict_raw(
    model: LearnedSimulatorNet,
    data: TrajectoryData,
    state_normalizer: FeatureNormalizer,
    output_normalizer: OutputNormalizer,
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(config.device)
    states = data.states.reshape(-1, data.states.shape[-1]).to(device)
    actions = data.actions.reshape(-1, config.M).to(device)
    time_index = torch.arange(data.T, device=device).repeat(data.days)
    inputs = simulator_inputs(
        states, actions, state_normalizer, config.daily_budget, time_index, data.T
    )
    z_standardized, dynamic_standardized = model(inputs)
    z = z_standardized * output_normalizer.z_scale + output_normalizer.z_mean
    dynamic = (
        dynamic_standardized * output_normalizer.dynamic_scale
        + output_normalizer.dynamic_mean
    )
    return z.reshape(data.days, data.T, config.M).cpu(), dynamic.reshape(
        data.days, data.T, config.M, DYNAMIC_DIM
    ).cpu()


def _r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target = target.double().reshape(-1)
    prediction = prediction.double().reshape(-1)
    denominator = ((target - target.mean()) ** 2).sum()
    if denominator <= 1e-12:
        return float("nan")
    return float(1.0 - ((target - prediction) ** 2).sum() / denominator)


def simulator_metrics(
    model: LearnedSimulatorNet,
    data: TrajectoryData,
    state_normalizer: FeatureNormalizer,
    output_normalizer: OutputNormalizer,
    config: ExperimentConfig,
    z_bias_by_t: Optional[torch.Tensor] = None,
    dynamic_bias_by_t: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    z_pred, dynamic_pred = _predict_raw(model, data, state_normalizer, output_normalizer, config)
    if z_bias_by_t is not None:
        z_pred = z_pred + z_bias_by_t.cpu().unsqueeze(0)
    if dynamic_bias_by_t is not None:
        dynamic_pred = dynamic_pred + dynamic_bias_by_t.cpu().unsqueeze(0)
    metrics: Dict[str, float] = {
        "holdout_r2_z": _r2(data.z, z_pred),
        "holdout_mse_z": float((data.z - z_pred).square().mean()),
    }
    per_time: list[dict[str, float]] = []
    r2_values: list[float] = []
    mse_values: list[float] = []
    for t in range(config.T):
        r2_t = _r2(data.z[:, t], z_pred[:, t])
        mse_t = float((data.z[:, t] - z_pred[:, t]).square().mean())
        per_time.append({"t": t, "r2_z": r2_t, "mse_z": mse_t})
        if math.isfinite(r2_t):
            r2_values.append(r2_t)
        mse_values.append(mse_t)
    metrics["holdout_min_r2_z_by_t"] = min(r2_values) if r2_values else float("nan")
    metrics["holdout_max_mse_z_by_t"] = max(mse_values)
    city_r2 = [_r2(data.z[..., city], z_pred[..., city]) for city in range(config.M)]
    valid_city_r2 = [value for value in city_r2 if math.isfinite(value)]
    metrics["holdout_min_r2_z_by_city"] = (
        min(valid_city_r2) if valid_city_r2 else float("nan")
    )
    mask = ~data.dones.bool()
    true_dynamic = dynamic_from_state(data.next_states[mask], config.M)
    metrics["holdout_r2_next_dynamic"] = _r2(true_dynamic, dynamic_pred[mask])
    metrics["holdout_mse_next_dynamic"] = float(
        (true_dynamic - dynamic_pred[mask]).square().mean()
    )
    spend_fraction = data.actions.sum(dim=-1) / max(float(config.daily_budget), EPS)
    spend_edges = (0.0, 0.02, 0.05, 0.10, 0.20, float("inf"))
    for bin_index, (lower, upper) in enumerate(zip(spend_edges[:-1], spend_edges[1:])):
        if bin_index == 0:
            bin_mask = spend_fraction <= upper
        else:
            bin_mask = (spend_fraction > lower) & (spend_fraction <= upper)
        count = int(bin_mask.sum())
        metrics[f"holdout_spend_bin_{bin_index}_count"] = count
        if count:
            metrics[f"holdout_spend_bin_{bin_index}_r2_z"] = _r2(
                data.z[bin_mask], z_pred[bin_mask]
            )
            metrics[f"holdout_spend_bin_{bin_index}_mse_z"] = float(
                (data.z[bin_mask] - z_pred[bin_mask]).square().mean()
            )
    return metrics, per_time


def fit_simulator_bundle(
    train: TrajectoryData,
    residual_calibration: TrajectoryData,
    validation: TrajectoryData,
    holdout: TrajectoryData,
    config: ExperimentConfig,
    simulator_config: SimulatorConfig,
    support_prototype_count: int,
    round_index: int,
    previous_model: Optional[LearnedSimulatorNet] = None,
    fixed_state_normalizer: Optional[FeatureNormalizer] = None,
    fixed_output_normalizer: Optional[OutputNormalizer] = None,
    fixed_reward_center_by_t: Optional[torch.Tensor] = None,
    fixed_reward_scale: Optional[float] = None,
) -> Tuple[SimulatorBundle, List[Dict[str, float]]]:
    device = torch.device(config.device)
    state_normalizer = fixed_state_normalizer or FeatureNormalizer.fit(
        train.states.reshape(-1, train.states.shape[-1])
    )
    output_normalizer = fixed_output_normalizer or _fit_output_normalizer(train, config.M)
    state_normalizer = state_normalizer.to(device)
    output_normalizer = output_normalizer.to(device)
    epochs = simulator_config.epochs if round_index == 1 else simulator_config.finetune_epochs
    model = _fit_one_simulator(
        train,
        validation,
        state_normalizer,
        output_normalizer,
        config,
        simulator_config,
        epochs,
        config.seed + 1000 * round_index,
        initial_model=previous_model,
    )
    ensemble: list[LearnedSimulatorNet] = []
    generator = torch.Generator().manual_seed(config.seed + 7000 * round_index)
    for member in range(simulator_config.bootstrap_models):
        bootstrap_days = torch.randint(
            train.days, (train.days,), generator=generator
        )
        ensemble.append(
            _fit_one_simulator(
                train,
                validation,
                state_normalizer,
                output_normalizer,
                config,
                simulator_config,
                simulator_config.bootstrap_epochs,
                config.seed + round_index * 10_000 + member,
                initial_model=model,
                bootstrap_days=bootstrap_days,
            )
        )
    residual_z_prediction, residual_dynamic_prediction = _predict_raw(
        model, residual_calibration, state_normalizer, output_normalizer, config
    )
    residual_z = residual_calibration.z - residual_z_prediction
    true_next_dynamic = dynamic_from_state(residual_calibration.next_states, config.M)
    residual_dynamic = true_next_dynamic - residual_dynamic_prediction
    residual_dynamic[residual_calibration.dones.bool()] = 0.0
    z_bias_by_t = torch.zeros_like(residual_z.mean(dim=0))
    dynamic_bias_by_t = torch.zeros_like(residual_dynamic.mean(dim=0))
    if simulator_config.residual_center:
        z_bias_by_t = residual_z.mean(dim=0)
        residual_z = residual_z - z_bias_by_t.unsqueeze(0)
        for t in range(config.T - 1):
            dynamic_bias_by_t[t] = residual_dynamic[:, t].mean(dim=0)
            residual_dynamic[:, t] -= dynamic_bias_by_t[t].unsqueeze(0)
    metrics, per_time = simulator_metrics(
        model,
        holdout,
        state_normalizer,
        output_normalizer,
        config,
        z_bias_by_t,
        dynamic_bias_by_t,
    )
    metrics["residual_calibration_mean_abs_z_bias"] = float(z_bias_by_t.abs().mean())
    metrics["residual_calibration_max_abs_z_bias"] = float(z_bias_by_t.abs().max())
    reward_center_by_t = (
        train.rewards.mean(dim=0).to(device)
        if fixed_reward_center_by_t is None
        else fixed_reward_center_by_t.to(device)
    )
    reward_scale = (
        max(float(train.rewards.std(unbiased=False)), 1.0)
        if fixed_reward_scale is None
        else float(fixed_reward_scale)
    )
    prototype = PublicCityPanelEnv(make_panel_config(config), config.seed, noise_split="train")
    (
        support_mean_by_t,
        support_scale_by_t,
        support_prototypes_by_t,
        support_threshold_by_t,
    ) = fit_support_reference(
        train,
        state_normalizer,
        config,
        support_prototype_count,
    )
    bundle = SimulatorBundle(
        model=model,
        ensemble=ensemble,
        state_normalizer=state_normalizer,
        output_normalizer=output_normalizer,
        residual_z=residual_z.to(device),
        residual_dynamic=residual_dynamic.to(device),
        initial_states=residual_calibration.states[:, 0].to(device),
        z_bias_by_t=z_bias_by_t.to(device),
        dynamic_bias_by_t=dynamic_bias_by_t.to(device),
        reward_center_by_t=reward_center_by_t,
        reward_scale=reward_scale,
        O=prototype.O.detach().clone(),
        panel_config=prototype.panel_cfg,
        diagnostics=metrics,
        support_mean_by_t=support_mean_by_t,
        support_scale_by_t=support_scale_by_t,
        support_prototypes_by_t=support_prototypes_by_t,
        support_threshold_by_t=support_threshold_by_t,
    )
    return bundle, per_time


class LearnedSimulatorBatch:
    def __init__(
        self,
        bundle: SimulatorBundle,
        batch_size: int,
        seed: int,
        lambda_pess: float,
        behavior_support_lambda: float = 0.0,
    ):
        self.bundle = bundle
        self.batch_size = int(batch_size)
        self.device = bundle.initial_states.device
        self.M = int(bundle.panel_config.M)
        self.T = int(bundle.panel_config.T)
        self.daily_budget = float(bundle.panel_config.daily_budget)
        self.lambda_pess = float(lambda_pess)
        self.behavior_support_lambda = float(behavior_support_lambda)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.t = 0
        self.path_indices = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.raw_state = torch.zeros(
            (self.batch_size, bundle.initial_states.shape[-1]), device=self.device
        )

    def reset(self) -> torch.Tensor:
        self.t = 0
        self.path_indices = torch.randint(
            self.bundle.initial_states.shape[0],
            (self.batch_size,),
            generator=self.generator,
            device=self.device,
        )
        self.raw_state = self.bundle.initial_states[self.path_indices].clone()
        return self.bundle.state_normalizer.transform(self.raw_state)

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        action = feasible_action(action, self.raw_state)
        inputs = simulator_inputs(
            self.raw_state,
            action,
            self.bundle.state_normalizer,
            self.daily_budget,
            self.t,
            self.T,
        )
        standardized_z: list[torch.Tensor] = []
        standardized_dynamic: list[torch.Tensor] = []
        models = [self.bundle.model, *self.bundle.ensemble]
        for model in models:
            z_member, dynamic_member = model(inputs)
            standardized_z.append(z_member)
            standardized_dynamic.append(dynamic_member)
        z_stack = torch.stack(standardized_z)
        dynamic_stack = torch.stack(standardized_dynamic)
        z_raw_stack = (
            z_stack * self.bundle.output_normalizer.z_scale
            + self.bundle.output_normalizer.z_mean
        )
        dynamic_raw_stack = (
            dynamic_stack * self.bundle.output_normalizer.dynamic_scale
            + self.bundle.output_normalizer.dynamic_mean
        )
        # Residual paths are calibrated against the main model.  Keep that
        # model as the conditional mean; bootstrap members quantify epistemic
        # uncertainty only and must not shift the residual center.
        z_mean = z_raw_stack[0] + self.bundle.z_bias_by_t[self.t]
        dynamic_mean = dynamic_raw_stack[0].reshape(
            self.batch_size, self.M, DYNAMIC_DIM
        ) + self.bundle.dynamic_bias_by_t[self.t]
        z = (
            z_mean + self.bundle.residual_z[self.path_indices, self.t]
        ).clamp_min(0.0)
        uncertainty_stack = z_raw_stack[1:] if z_raw_stack.shape[0] > 1 else z_raw_stack
        uncertainty = uncertainty_stack.std(dim=0, unbiased=False).square().sum(dim=-1).sqrt()
        spend = action.sum(dim=-1)
        raw_reward = z.sum(dim=-1) - spend
        pessimistic_raw_reward = raw_reward - self.lambda_pess * uncertainty
        support_features = state_action_support_features(
            self.raw_state,
            action,
            self.bundle.state_normalizer,
            self.daily_budget,
        )
        standardized_support = balance_support_feature_groups(
            (
                support_features - self.bundle.support_mean_by_t[self.t]
            ) / self.bundle.support_scale_by_t[self.t],
            self.raw_state.shape[-1],
        )
        support_distance = torch.cdist(
            standardized_support,
            self.bundle.support_prototypes_by_t[self.t],
        ).min(dim=1).values
        support_excess = (
            support_distance - self.bundle.support_threshold_by_t[self.t]
        ).clamp_min(0.0)
        learning_reward = (
            pessimistic_raw_reward - self.bundle.reward_center_by_t[self.t]
        ) / self.bundle.reward_scale
        learning_reward = (
            learning_reward - self.behavior_support_lambda * support_excess
        )
        H = (remaining_budget(self.raw_state) - spend).clamp_min(0.0)
        self.t += 1
        done_flag = self.t >= self.T
        done = torch.full((self.batch_size,), float(done_flag), device=self.device)
        if done_flag:
            next_raw = torch.zeros_like(self.raw_state)
        else:
            dynamic = dynamic_mean + self.bundle.residual_dynamic[
                self.path_indices, self.t - 1
            ]
            dynamic = _project_dynamic(dynamic, self.bundle.panel_config)
            next_raw = _state_from_dynamic(
                dynamic, self.bundle.O, H, self.t, self.T
            )
        self.raw_state = next_raw
        return (
            self.bundle.state_normalizer.transform(next_raw),
            learning_reward,
            done,
            {
                "raw_reward": raw_reward,
                "pessimistic_raw_reward": pessimistic_raw_reward,
                "uncertainty": uncertainty,
                "support_distance": support_distance,
                "support_excess": support_excess,
                "z": z,
                "spend": spend,
            },
        )


def random_feasible_actions(state: torch.Tensor, T: int, t: int, generator: torch.Generator) -> torch.Tensor:
    batch = state.shape[0]
    # Infer action dimension from the city-block count: state_dim = 17 M + 5.
    action_dim = int((state.shape[-1] - 5) // (DYNAMIC_DIM + 1 + STATIC_DIM))
    weights = torch.rand((batch, action_dim), device=state.device, generator=generator)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
    spend_fraction = torch.rand((batch, 1), device=state.device, generator=generator)
    paced = remaining_budget(state).unsqueeze(-1) / max(T - t, 1)
    spend = torch.minimum(remaining_budget(state).unsqueeze(-1), paced * (0.25 + 2.5 * spend_fraction))
    return feasible_action(weights * spend, state)


def train_sac_policy(
    bundle: SimulatorBundle,
    config: ExperimentConfig,
    policy_config: PolicyConfig,
    agent: Optional[SACAgent],
    round_index: int,
) -> Tuple[SACAgent, List[Dict[str, float]]]:
    device = torch.device(config.device)
    state_dim = int(bundle.initial_states.shape[-1])
    if agent is None:
        sac_config = SACConfig(
            auto_alpha=policy_config.auto_alpha,
            fixed_temperature=policy_config.fixed_temperature,
        )
        agent = SACAgent(
            state_dim,
            config.M,
            device,
            sac_config,
            policy_config.hidden_dims,
            policy_config.init_spend_share,
        )
    replay = ReplayBuffer(
        policy_config.replay_size, state_dim, config.M, device
    )
    rows: list[dict[str, float]] = []
    target_episodes = int(policy_config.train_episodes)
    if round_index > 1 and int(policy_config.later_round_train_episodes) > 0:
        target_episodes = int(policy_config.later_round_train_episodes)
    trained = 0
    rollout_index = 0
    while trained < target_episodes:
        batch_size = min(
            policy_config.rollout_batch_size,
            target_episodes - trained,
        )
        env = LearnedSimulatorBatch(
            bundle,
            batch_size,
            seed=config.seed + round_index * 100_000 + rollout_index,
            lambda_pess=policy_config.lambda_pess,
            behavior_support_lambda=policy_config.behavior_support_lambda,
        )
        state = env.reset()
        generator = torch.Generator(device=device).manual_seed(
            config.seed + round_index * 200_000 + rollout_index
        )
        episode_raw_return = torch.zeros(batch_size, device=device)
        episode_support_excess = torch.zeros(batch_size, device=device)
        for t in range(config.T):
            if trained < policy_config.warmup_episodes:
                action = random_feasible_actions(env.raw_state, config.T, t, generator)
            else:
                with torch.no_grad():
                    action, _, _, _ = agent.actor.sample(state)
            next_state, reward, done, info = env.step(action)
            replay.add_batch(state, action, reward, next_state, done)
            episode_raw_return += info["raw_reward"]
            episode_support_excess += info["support_excess"]
            state = next_state
        updates = 0
        if replay.size >= policy_config.batch_size:
            updates = policy_config.updates_per_rollout_batch
            last_metrics: Dict[str, float] = {}
            for _ in range(updates):
                last_metrics = agent.update(replay.sample(policy_config.batch_size))
        else:
            last_metrics = {}
        trained += batch_size
        rollout_index += 1
        rows.append(
            {
                "round": round_index,
                "episodes": trained,
                "mean_synthetic_raw_return": float(episode_raw_return.mean()),
                "mean_episode_support_excess": float(
                    episode_support_excess.mean()
                ),
                "updates": updates,
                **last_metrics,
            }
        )
    return agent, rows


def candidate_actions(
    raw_state: torch.Tensor,
    t: int,
    config: ExperimentConfig,
    random_count: int,
    seed: int,
    budget_fractions: Sequence[float] = (0.25, 0.50, 0.75, 1.00),
) -> torch.Tensor:
    """DFL candidate set shared with the synthetic and AA experiments."""
    single = raw_state.ndim == 1
    states = raw_state.unsqueeze(0) if single else raw_state
    batch = states.shape[0]
    H = remaining_budget(states)
    _ = t
    X = states[..., : config.M]
    Y = states[..., config.M : 2 * config.M]

    def normalize_direction(weights: torch.Tensor) -> torch.Tensor:
        weights = torch.nan_to_num(
            weights, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        total = weights.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / config.M)
        return torch.where(total > EPS, weights / total.clamp_min(EPS), uniform)

    gap = normalize_direction((X - Y).clamp_min(0.0))
    demand = normalize_direction(X)
    shortage = normalize_direction(X.clamp_min(0.0) / Y.clamp_min(EPS))
    allocations = torch.stack(
        [torch.full_like(gap, 1.0 / config.M), gap, demand, shortage], dim=1
    )
    fractions = torch.as_tensor(
        tuple(float(value) for value in budget_fractions),
        device=states.device,
        dtype=states.dtype,
    ).clamp(0.0, 1.0)
    deterministic = (
        H[:, None, None, None]
        * fractions[None, :, None, None]
        * allocations[:, None, :, :]
    ).reshape(batch, -1, config.M)
    generator = torch.Generator(device=states.device).manual_seed(int(seed))
    random_directions = torch.rand(
        (int(random_count), config.M), device=states.device, generator=generator
    )
    random_directions = random_directions / random_directions.sum(
        dim=-1, keepdim=True
    ).clamp_min(EPS)
    random_fractions = torch.rand(
        (int(random_count),), device=states.device, generator=generator
    )
    random = (
        H[:, None, None]
        * random_fractions[None, :, None]
        * random_directions[None, :, :]
    )
    output = torch.cat(
        [torch.zeros((batch, 1, config.M), device=states.device, dtype=states.dtype),
         deterministic,
         random],
        dim=1,
    )
    return output.squeeze(0) if single else output


@torch.no_grad()
def simulator_immediate_reward_and_uncertainty(
    bundle: SimulatorBundle,
    raw_state: torch.Tensor,
    action: torch.Tensor,
    budget: float,
    t: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conditional-mean reward and response uncertainty for DFL candidates.

    A calibrated residual path is common to all candidate actions at a fixed
    state and time. Adding it would shift every candidate score by the same
    scalar and therefore cannot change the teacher argmax.
    """
    shape = action.shape[:-1]
    states = raw_state
    while states.ndim < action.ndim:
        states = states.unsqueeze(-2)
    states = states.expand(*shape, raw_state.shape[-1])
    inputs = simulator_inputs(
        states.reshape(-1, states.shape[-1]),
        action.reshape(-1, action.shape[-1]),
        bundle.state_normalizer,
        budget,
        t,
        bundle.panel_config.T,
    )
    z_members: list[torch.Tensor] = []
    for model in [bundle.model, *bundle.ensemble]:
        z_standardized, _ = model(inputs)
        z_members.append(
            z_standardized * bundle.output_normalizer.z_scale
            + bundle.output_normalizer.z_mean
        )
    z_stack = torch.stack(z_members)
    z_mean = z_stack[0] + bundle.z_bias_by_t[t]
    reward = z_mean.sum(dim=-1) - action.reshape(-1, action.shape[-1]).sum(dim=-1)
    epistemic_stack = z_stack[1:] if z_stack.shape[0] > 1 else z_stack
    uncertainty = epistemic_stack.std(dim=0, unbiased=False).square().sum(dim=-1).sqrt()
    return reward.reshape(shape), uncertainty.reshape(shape)


class DFLPolicy:
    def __init__(
        self,
        models: Sequence[OneStepScoreNet],
        bundle: SimulatorBundle,
        config: ExperimentConfig,
        policy_config: PolicyConfig,
    ):
        self.models = list(models)
        self.bundle = bundle
        self.config = config
        self.policy_config = policy_config

    @torch.no_grad()
    def act_batch(self, raw_state: torch.Tensor, t: int) -> torch.Tensor:
        candidates = candidate_actions(
            raw_state,
            t,
            self.config,
            self.policy_config.dfl_candidate_random_count,
            self.config.seed + 90_000,
            self.policy_config.dfl_budget_fractions,
        )
        states = raw_state.unsqueeze(1).expand(
            raw_state.shape[0], candidates.shape[1], raw_state.shape[-1]
        )
        inputs = torch.cat(
            [
                self.bundle.state_normalizer.transform(states),
                action_features(candidates, states, self.config.daily_budget),
            ],
            dim=-1,
        )
        scores = self.models[t](inputs.reshape(-1, inputs.shape[-1])).reshape(
            raw_state.shape[0], candidates.shape[1]
        )
        _, uncertainty = simulator_immediate_reward_and_uncertainty(
            self.bundle, raw_state, candidates, self.config.daily_budget, t
        )
        decision_scores = scores - (
            float(self.policy_config.lambda_pess)
            * uncertainty
        )
        index = decision_scores.argmax(dim=-1)
        return candidates[torch.arange(raw_state.shape[0], device=raw_state.device), index]

    @torch.no_grad()
    def act(self, raw_state: torch.Tensor, t: int) -> torch.Tensor:
        return self.act_batch(raw_state.unsqueeze(0), t).squeeze(0)


def train_dfl_policy(
    logs: TrajectoryData,
    bundle: SimulatorBundle,
    config: ExperimentConfig,
    policy_config: PolicyConfig,
    round_index: int,
    previous_policy: Optional[DFLPolicy] = None,
) -> Tuple[DFLPolicy, List[Dict[str, float]]]:
    device = torch.device(config.device)
    torch.manual_seed(int(config.seed + 80_000 + round_index))
    models: list[OneStepScoreNet] = []
    rows: list[dict[str, float]] = []
    input_dim = logs.states.shape[-1] + action_feature_dim(config.M)
    for t in range(config.T):
        model = OneStepScoreNet(input_dim, policy_config.dfl_hidden_dims).to(device)
        if previous_policy is not None and t < len(previous_policy.models):
            model.load_state_dict(copy.deepcopy(previous_policy.models[t].state_dict()))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(policy_config.dfl_learning_rate)
        )
        states = logs.states[:, t].to(device)
        actions = logs.actions[:, t].to(device)
        rewards = logs.rewards[:, t].to(device)
        candidate_seed = config.seed + 90_000
        candidates = candidate_actions(
            states,
            t,
            config,
            policy_config.dfl_candidate_random_count,
            candidate_seed,
            policy_config.dfl_budget_fractions,
        )
        teacher_rows: list[torch.Tensor] = []
        uncertainty_rows: list[torch.Tensor] = []
        teacher_chunk_size = max(32, min(int(policy_config.dfl_batch_size), 256))
        for start in range(0, states.shape[0], teacher_chunk_size):
            stop = min(start + teacher_chunk_size, states.shape[0])
            teacher_reward, teacher_uncertainty = simulator_immediate_reward_and_uncertainty(
                bundle,
                states[start:stop],
                candidates[start:stop],
                config.daily_budget,
                t,
            )
            teacher_rows.append(
                (
                    teacher_reward
                    - float(policy_config.lambda_pess) * teacher_uncertainty
                ).argmax(dim=-1)
            )
            uncertainty_rows.append(teacher_uncertainty)
        teacher = torch.cat(teacher_rows, dim=0)
        candidate_uncertainty = torch.cat(uncertainty_rows, dim=0)
        dataset = TensorDataset(
            states, actions, rewards, candidates, teacher, candidate_uncertainty
        )
        loader = DataLoader(dataset, batch_size=policy_config.dfl_batch_size, shuffle=True)
        last_loss = 0.0
        for epoch in range(policy_config.dfl_epochs):
            for (
                state_batch,
                action_batch,
                reward_batch,
                candidate_batch,
                teacher_batch,
                uncertainty_batch,
            ) in loader:
                logged_inputs = torch.cat(
                    [
                        bundle.state_normalizer.transform(state_batch),
                        action_features(action_batch, state_batch, config.daily_budget),
                    ],
                    dim=-1,
                )
                prediction = model(logged_inputs).squeeze(-1)
                supervised = torch.nn.functional.mse_loss(prediction, reward_batch)
                expanded_states = state_batch.unsqueeze(1).expand(
                    state_batch.shape[0], candidate_batch.shape[1], state_batch.shape[-1]
                )
                candidate_inputs = torch.cat(
                    [
                        bundle.state_normalizer.transform(expanded_states),
                        action_features(candidate_batch, expanded_states, config.daily_budget),
                    ],
                    dim=-1,
                )
                candidate_scores = model(
                    candidate_inputs.reshape(-1, candidate_inputs.shape[-1])
                ).reshape(state_batch.shape[0], candidate_batch.shape[1])
                decision_scores = candidate_scores - (
                    float(policy_config.lambda_pess) * uncertainty_batch
                )
                decision_loss = torch.nn.functional.cross_entropy(
                    decision_scores
                    / max(float(policy_config.dfl_softmax_temperature), 1e-4),
                    teacher_batch,
                )
                loss = supervised + float(policy_config.dfl_decision_weight) * decision_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                last_loss = float(loss.detach())
        model.eval()
        models.append(model)
        rows.append(
            {
                "t": t,
                "final_dfl_loss": last_loss,
                "dfl_candidate_count": int(candidates.shape[1]),
                "dfl_model_parameters": sum(p.numel() for p in model.parameters()),
                "dfl_teacher_mean_spend": float(
                    candidates.sum(dim=-1)
                    .gather(1, teacher.unsqueeze(-1))
                    .mean()
                    .detach()
                ),
                "dfl_supervised_loss": float(supervised.detach()),
                "dfl_decision_loss": float(decision_loss.detach()),
                "dfl_decision_weight": float(policy_config.dfl_decision_weight),
                "dfl_softmax_temperature": float(policy_config.dfl_softmax_temperature),
            }
        )
    return DFLPolicy(models, bundle, config, policy_config), rows


def evaluate_true_policy(
    panel_config: PublicCityPanelConfig,
    policy: Any,
    method: str,
    episodes: int,
    seed_offset: int,
    policy_normalizer: Optional[FeatureNormalizer] = None,
    batch_size: int = 512,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Evaluate a frozen policy on the reporting-only target-DGP split."""
    raw_rows: list[dict[str, float]] = []
    for start in range(0, episodes, batch_size):
        count = min(batch_size, episodes - start)
        seeds = [seed_offset + start + i for i in range(count)]
        env = BatchedPublicCityPanelEnv(panel_config, seeds, noise_split="evaluation")
        state = env.reset()
        total_reward = torch.zeros(count, device=env.device)
        total_response = torch.zeros(count, device=env.device)
        total_spend = torch.zeros(count, device=env.device)
        for t in range(env.T):
            if method in BASELINE_METHODS:
                generator = torch.Generator(device=env.device).manual_seed(
                    seed_offset + start + t
                )
                action = baseline_action_from_state(
                    method,
                    env.H,
                    env.dynamic[..., 0],
                    env.dynamic[..., 1],
                    env.T,
                    t,
                    generator,
                )
            elif isinstance(policy, DFLPolicy):
                with torch.no_grad():
                    action = policy.act_batch(state, t)
            else:
                assert policy_normalizer is not None
                normalized = policy_normalizer.transform(state)
                with torch.no_grad():
                    action, _, _, _ = policy.actor.sample(normalized, deterministic=True)
            next_state, reward, _done, info = env.step(feasible_action(action, state))
            total_reward += reward
            total_response += info["Z"].sum(dim=-1)
            total_spend += info["spend"]
            if next_state is not None:
                state = next_state
        for index in range(count):
            raw_rows.append(
                {
                    "episode": start + index,
                    "net_reward": float(total_reward[index]),
                    "response_score": float(total_response[index]),
                    "spend": float(total_spend[index]),
                    "final_budget": float(panel_config.daily_budget - total_spend[index]),
                }
            )
    summary: Dict[str, float] = {}
    for key in ("net_reward", "response_score", "spend", "final_budget"):
        values = torch.tensor([row[key] for row in raw_rows], dtype=torch.float64)
        summary[key] = float(values.mean())
        summary[f"{key}_se"] = float(values.std(unbiased=True) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return summary, raw_rows


def paired_simulator_policy_selection(
    bundle: SimulatorBundle,
    candidate: Any,
    candidate_method: str,
    reference: Any,
    reference_method: str,
    episodes: int,
    seed_offset: int,
    lambda_pess: float,
    behavior_support_lambda: float,
    z_value: float,
    tolerance: float,
    comparison: str,
    round_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Select checkpoints using paired conservative learned-simulator rollouts."""
    candidate_summary, candidate_raw = learned_policy_rollouts(
        bundle,
        candidate,
        candidate_method,
        episodes,
        seed_offset,
        lambda_pess,
        behavior_support_lambda,
    )
    reference_summary, reference_raw = learned_policy_rollouts(
        bundle,
        reference,
        reference_method,
        episodes,
        seed_offset,
        lambda_pess,
        behavior_support_lambda,
    )
    candidate_values = torch.tensor(
        [row["sim_selection_score"] for row in candidate_raw], dtype=torch.float64
    )
    reference_values = torch.tensor(
        [row["sim_selection_score"] for row in reference_raw], dtype=torch.float64
    )
    delta = candidate_values - reference_values
    mean_delta = float(delta.mean())
    se_delta = (
        float(delta.std(unbiased=True) / math.sqrt(delta.numel()))
        if delta.numel() > 1
        else 0.0
    )
    lower_bound = mean_delta - float(z_value) * se_delta
    upper_bound = mean_delta + float(z_value) * se_delta
    accepted = lower_bound >= float(tolerance)
    return accepted, {
        "round": round_index,
        "comparison": comparison,
        "selection_source": "paired_learned_simulator_conservative_return",
        "simulator_episodes": int(episodes),
        "target_dgp_episodes": 0,
        "candidate_method": candidate_method,
        "reference_method": reference_method,
        "candidate_sim_net_reward": candidate_summary["sim_net_reward"],
        "reference_sim_net_reward": reference_summary["sim_net_reward"],
        "candidate_sim_pessimistic_net_reward": candidate_summary[
            "sim_pessimistic_net_reward"
        ],
        "reference_sim_pessimistic_net_reward": reference_summary[
            "sim_pessimistic_net_reward"
        ],
        "candidate_sim_selection_score": candidate_summary["sim_selection_score"],
        "reference_sim_selection_score": reference_summary["sim_selection_score"],
        "paired_delta_mean": mean_delta,
        "paired_delta_se": se_delta,
        "paired_delta_lower_bound": lower_bound,
        "paired_delta_upper_bound": upper_bound,
        "accepted": int(accepted),
    }


@torch.no_grad()
def learned_policy_rollouts(
    bundle: SimulatorBundle,
    policy: Any,
    method: str,
    episodes: int,
    seed: int,
    lambda_pess: float,
    behavior_support_lambda: float = 0.0,
    batch_size: int = 512,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    returns: list[torch.Tensor] = []
    pessimistic_returns: list[torch.Tensor] = []
    selection_scores: list[torch.Tensor] = []
    spends: list[torch.Tensor] = []
    for start in range(0, int(episodes), int(batch_size)):
        count = min(int(batch_size), int(episodes) - start)
        env = LearnedSimulatorBatch(
            bundle,
            count,
            seed + start,
            lambda_pess,
            behavior_support_lambda=behavior_support_lambda,
        )
        state = env.reset()
        total_return = torch.zeros(count, device=env.device)
        total_pessimistic_return = torch.zeros(count, device=env.device)
        total_selection_score = torch.zeros(count, device=env.device)
        total_spend = torch.zeros(count, device=env.device)
        generator = torch.Generator(device=env.device).manual_seed(seed + start + 17)
        for t in range(env.T):
            if method in BASELINE_METHODS:
                H = remaining_budget(env.raw_state)
                action = baseline_action_from_state(
                    method,
                    H,
                    env.raw_state[:, : env.M],
                    env.raw_state[:, env.M : 2 * env.M],
                    env.T,
                    t,
                    generator,
                )
            elif isinstance(policy, DFLPolicy):
                action = policy.act_batch(env.raw_state, t)
            else:
                action, _, _, _ = policy.actor.sample(state, deterministic=True)
            state, _learning_reward, _done, info = env.step(action)
            total_return += info["raw_reward"]
            total_pessimistic_return += info["pessimistic_raw_reward"]
            total_selection_score += (
                info["pessimistic_raw_reward"]
                - bundle.reward_scale * behavior_support_lambda * info["support_excess"]
            )
            total_spend += info["spend"]
        returns.append(total_return.cpu())
        pessimistic_returns.append(total_pessimistic_return.cpu())
        selection_scores.append(total_selection_score.cpu())
        spends.append(total_spend.cpu())
    values = torch.cat(returns).double()
    pessimistic_values = torch.cat(pessimistic_returns).double()
    score_values = torch.cat(selection_scores).double()
    spend_values = torch.cat(spends).double()
    summary = {
        "sim_net_reward": float(values.mean()),
        "sim_net_reward_se": float(values.std(unbiased=True) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
        "sim_pessimistic_net_reward": float(pessimistic_values.mean()),
        "sim_selection_score": float(score_values.mean()),
        "sim_spend": float(spend_values.mean()),
    }
    raw_rows = [
        {
            "episode": int(index),
            "sim_net_reward": float(values[index]),
            "sim_pessimistic_net_reward": float(pessimistic_values[index]),
            "sim_selection_score": float(score_values[index]),
            "sim_spend": float(spend_values[index]),
        }
        for index in range(values.numel())
    ]
    return summary, raw_rows


@torch.no_grad()
def evaluate_learned_policy(
    bundle: SimulatorBundle,
    policy: Any,
    method: str,
    episodes: int,
    seed: int,
    lambda_pess: float,
    behavior_support_lambda: float = 0.0,
    batch_size: int = 512,
) -> Dict[str, float]:
    summary, _ = learned_policy_rollouts(
        bundle,
        policy,
        method,
        episodes,
        seed,
        lambda_pess,
        behavior_support_lambda,
        batch_size,
    )
    return summary


@torch.no_grad()
def policy_support_rows(
    policy: Any,
    train_logs: TrajectoryData,
    holdout_logs: TrajectoryData,
    bundle: SimulatorBundle,
    config: ExperimentConfig,
    policy_config: PolicyConfig,
    round_index: int,
) -> List[Dict[str, float]]:
    rows: list[dict[str, float]] = []
    device = bundle.initial_states.device
    for t in range(config.T):
        logged_action = train_logs.actions[:, t].to(device)
        query_state = holdout_logs.states[:, t].to(device)
        if isinstance(policy, DFLPolicy):
            query_action = policy.act_batch(query_state, t)
        else:
            normalized = bundle.state_normalizer.transform(query_state)
            query_action, _, _, _ = policy.actor.sample(normalized, deterministic=True)
        query_features = state_action_support_features(
            query_state,
            query_action,
            bundle.state_normalizer,
            config.daily_budget,
        )
        query_standardized = balance_support_feature_groups(
            (query_features - bundle.support_mean_by_t[t])
            / bundle.support_scale_by_t[t],
            query_state.shape[-1],
        )
        distances: list[torch.Tensor] = []
        for start in range(0, query_standardized.shape[0], 128):
            distances.append(
                torch.cdist(
                    query_standardized[start : start + 128],
                    bundle.support_prototypes_by_t[t],
                ).min(dim=1).values
            )
        nearest = torch.cat(distances)
        logged_spend = logged_action.sum(dim=-1)
        query_spend = query_action.sum(dim=-1)
        logged_action_features = action_features(
            logged_action,
            train_logs.states[:, t].to(device),
            config.daily_budget,
        )
        query_action_features = action_features(
            query_action,
            query_state,
            config.daily_budget,
        )
        logged_hhi = logged_action_features[:, -2]
        query_hhi = query_action_features[:, -2]
        rows.append(
            {
                "round": round_index,
                "t": t,
                "mean_nearest_logged_action_distance": float(nearest.mean()),
                "p95_nearest_logged_action_distance": float(torch.quantile(nearest, 0.95)),
                "logged_support_distance_threshold": float(
                    bundle.support_threshold_by_t[t]
                ),
                "share_outside_logged_state_action_support": float(
                    (nearest > bundle.support_threshold_by_t[t]).float().mean()
                ),
                "share_above_logged_p95_step_spend": float(
                    (query_spend > torch.quantile(logged_spend, 0.95)).float().mean()
                ),
                "share_above_logged_p95_concentration": float(
                    (query_hhi > torch.quantile(logged_hhi, 0.95)).float().mean()
                ),
                "mean_policy_step_spend": float(query_spend.mean()),
                "logged_p95_step_spend": float(torch.quantile(logged_spend, 0.95)),
            }
        )
    return rows


def real_online_update(
    agent: SACAgent,
    logs: TrajectoryData,
    bundle: SimulatorBundle,
    config: ExperimentConfig,
    policy_config: PolicyConfig,
) -> List[Dict[str, float]]:
    device = torch.device(config.device)
    state = bundle.state_normalizer.transform(
        logs.states.reshape(-1, logs.states.shape[-1]).to(device)
    )
    next_state = bundle.state_normalizer.transform(
        logs.next_states.reshape(-1, logs.states.shape[-1]).to(device)
    )
    action = logs.actions.reshape(-1, config.M).to(device)
    t_index = torch.arange(config.T, device=device).repeat(logs.days)
    reward = (
        logs.rewards.reshape(-1).to(device) - bundle.reward_center_by_t[t_index]
    ) / bundle.reward_scale
    done = logs.dones.reshape(-1).to(device)
    replay = ReplayBuffer(len(state), state.shape[-1], config.M, device)
    replay.add_batch(state, action, reward, next_state, done)
    update_ratio = float(policy_config.real_gradient_steps_per_step)
    if update_ratio < 0.0:
        raise ValueError("real_gradient_steps_per_step must be nonnegative.")
    updates = int(round(len(state) * update_ratio))
    rows: list[dict[str, float]] = []
    if replay.size < policy_config.batch_size or updates <= 0:
        return rows
    critic_warmup = min(
        max(int(policy_config.real_critic_warmup_updates), 0),
        updates,
    )
    for update in range(updates):
        metrics = agent.update(
            replay.sample(policy_config.batch_size),
            update_actor=update >= critic_warmup,
        )
        if update % max(updates // 20, 1) == 0 or update == updates - 1:
            rows.append({"update": update + 1, **metrics})
    return rows


def run_experiment(
    method: str,
    config: ExperimentConfig,
    simulator_config: SimulatorConfig,
    policy_config: PolicyConfig,
    out_dir: Path,
) -> None:
    if method not in {"sacsa_reserve", "dfl_pto"}:
        raise ValueError("method must be sacsa_reserve or dfl_pto")
    if config.M < 2 or config.T <= 0 or config.daily_budget <= 0.0:
        raise ValueError("The experiment requires M>=2, T>0, and a positive budget.")
    if min(
        config.initial_logged_days,
        config.new_logged_days,
        config.residual_calibration_days,
        config.validation_logged_days,
        config.holdout_logged_days,
        config.eval_episodes,
    ) <= 0:
        raise ValueError("All logging and evaluation sample sizes must be positive.")
    if not 0.0 <= policy_config.real_exploration_epsilon <= 1.0:
        raise ValueError("real_exploration_epsilon must lie in [0,1].")
    if policy_config.behavior_support_lambda < 0.0:
        raise ValueError("behavior_support_lambda must be nonnegative.")
    if policy_config.support_prototype_count < 2:
        raise ValueError("support_prototype_count must be at least two.")
    if policy_config.simulator_selection_episodes < 2:
        raise ValueError("simulator_selection_episodes must be at least two.")
    out_dir.mkdir(parents=True, exist_ok=True)
    events = EventLogger(out_dir / "run_events.jsonl")
    write_json(
        out_dir / "config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "method": method,
            "experiment": asdict(config),
            "simulator": asdict(simulator_config),
            "policy": asdict(policy_config),
            "checkpoint_selection": {
                "source": "paired_learned_simulator_conservative_return",
                "target_dgp_episodes": 0,
                "simulator_episodes": policy_config.simulator_selection_episodes,
                "seed_role": "run_seed",
            },
        },
    )
    panel_config = make_panel_config(config)
    events.log(
        "run_start",
        protocol_id=PROTOCOL_ID,
        method=method,
        M=config.M,
        T=config.T,
        q_max=config.q_max,
        checkpoint_selection="learned_simulator_only",
        target_dgp_selection_episodes=0,
    )
    events.log("collect_initial_log_start", days=config.initial_logged_days)
    train_logs = collect_rich_logging(
        panel_config,
        config.initial_logged_days,
        config.seed * 1_000_000 + 10_000,
        "train",
        config.logging_batch_size,
    )
    residual_logs = collect_rich_logging(
        panel_config,
        config.residual_calibration_days,
        config.seed * 1_000_000 + 100_000,
        "calibration",
        config.logging_batch_size,
    )
    validation_logs = collect_rich_logging(
        panel_config,
        config.validation_logged_days,
        config.seed * 1_000_000 + 200_000,
        "validation",
        config.logging_batch_size,
    )
    holdout_logs = collect_rich_logging(
        panel_config,
        config.holdout_logged_days,
        config.seed * 1_000_000 + 300_000,
        "holdout",
        config.logging_batch_size,
    )
    split_rows = data_split_diagnostics(
        {
            "train": train_logs,
            "residual_calibration": residual_logs,
            "model_validation": validation_logs,
            "simulator_holdout": holdout_logs,
        }
    )
    write_csv(out_dir / "data_split_diagnostics.csv", split_rows)
    overlap_rows = [
        row
        for row in split_rows
        if row["record_type"] == "overlap" and int(row["overlap_residual_days"]) > 0
    ]
    if overlap_rows:
        raise RuntimeError(
            "Downstream data splits share frozen-DGP residual day indices; "
            "inspect data_split_diagnostics.csv."
        )
    write_csv(out_dir / "logging_coverage.csv", logging_coverage_rows(train_logs, config.daily_budget))
    write_csv(out_dir / "logging_design_summary.csv", logging_design_summary_rows(train_logs))
    write_csv(
        out_dir / "logging_time_support.csv",
        logging_time_support_rows(train_logs, config.daily_budget),
    )
    events.log("collect_initial_log_done", days=train_logs.days)

    baseline_summaries: list[dict[str, Any]] = []
    baseline_raw: list[dict[str, Any]] = []
    for baseline in BASELINE_METHODS:
        summary, raw = evaluate_true_policy(
            panel_config,
            None,
            baseline,
            config.eval_episodes,
            config.seed * 1_000_000 + 900_000,
        )
        baseline_summaries.append({"method": baseline, "round": 0, **summary})
        baseline_raw.extend({"method": baseline, "round": 0, **row} for row in raw)
    write_csv(out_dir / "baseline_summary.csv", baseline_summaries)
    previous_model: Optional[LearnedSimulatorNet] = None
    fixed_state_normalizer: Optional[FeatureNormalizer] = None
    fixed_output_normalizer: Optional[OutputNormalizer] = None
    fixed_reward_center_by_t: Optional[torch.Tensor] = None
    fixed_reward_scale: Optional[float] = None
    agent: Optional[SACAgent] = None
    dfl_policy: Optional[DFLPolicy] = None
    round_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    support_diagnostic_rows: list[dict[str, Any]] = []
    quality_gate_rows: list[dict[str, Any]] = []
    policy_selection_rows: list[dict[str, Any]] = []
    final_raw: list[dict[str, Any]] = baseline_raw
    for q in range(1, config.q_max + 1):
        events.log("round_simulator_fit_start", round=q, logged_days=train_logs.days)
        bundle, per_time = fit_simulator_bundle(
            train_logs,
            residual_logs,
            validation_logs,
            holdout_logs,
            config,
            simulator_config,
            policy_config.support_prototype_count,
            q,
            previous_model,
            fixed_state_normalizer,
            fixed_output_normalizer,
            fixed_reward_center_by_t,
            fixed_reward_scale,
        )
        previous_model = bundle.model
        if fixed_state_normalizer is None:
            fixed_state_normalizer = bundle.state_normalizer
            fixed_output_normalizer = bundle.output_normalizer
            fixed_reward_center_by_t = bundle.reward_center_by_t
            fixed_reward_scale = bundle.reward_scale
        round_baseline_calibration: list[dict[str, Any]] = []
        for baseline_row in baseline_summaries:
            baseline_method = str(baseline_row["method"])
            baseline_sim = evaluate_learned_policy(
                bundle,
                None,
                baseline_method,
                min(config.eval_episodes, 1000),
                config.seed * 1_000_000 + q * 1000 + 710_000,
                0.0,
            )
            true_value = float(baseline_row["net_reward"])
            bias = float(baseline_sim["sim_net_reward"] - true_value)
            round_baseline_calibration.append(
                {
                    "round": q,
                    "method": baseline_method,
                    **baseline_sim,
                    "true_net_reward": true_value,
                    "true_spend": baseline_row["spend"],
                    "sim_minus_true_net_reward": bias,
                    "absolute_relative_net_reward_bias": abs(bias) / max(abs(true_value), 1.0),
                }
            )
        calibration_rows.extend(round_baseline_calibration)
        quality_pass = (
            float(bundle.diagnostics["holdout_r2_z"]) >= simulator_config.min_holdout_r2_z
            and float(bundle.diagnostics["holdout_min_r2_z_by_t"])
            >= simulator_config.min_holdout_r2_z_by_t
        )
        quality_gate_rows.append(
            {
                "round": q,
                "pass": int(quality_pass),
                "holdout_r2_z": bundle.diagnostics["holdout_r2_z"],
                "holdout_min_r2_z_by_t": bundle.diagnostics["holdout_min_r2_z_by_t"],
                "required_r2_z": simulator_config.min_holdout_r2_z,
                "required_min_r2_z_by_t": simulator_config.min_holdout_r2_z_by_t,
            }
        )
        write_csv(out_dir / "simulator_quality_gate.csv", quality_gate_rows)
        write_csv(out_dir / "simulator_vs_true_policy_calibration.csv", calibration_rows)
        if not quality_pass:
            events.log(
                "simulator_quality_warning",
                round=q,
                holdout_r2_z=bundle.diagnostics["holdout_r2_z"],
                holdout_min_r2_z_by_t=bundle.diagnostics["holdout_min_r2_z_by_t"],
            )
            if simulator_config.strict_quality_gate:
                raise RuntimeError(
                    "Learned simulator failed the configured logged-data holdout quality gate; "
                    "policy training was intentionally not started."
                )
        write_csv(
            out_dir / f"simulator_per_time_q{q}.csv",
            [{"round": q, **row} for row in per_time],
        )
        events.log("round_simulator_fit_done", round=q, **bundle.diagnostics)
        if method == "sacsa_reserve":
            events.log("round_policy_training_start", round=q)
            incumbent_agent = copy.deepcopy(agent) if agent is not None else None
            agent, training_rows = train_sac_policy(
                bundle, config, policy_config, agent, q
            )
            write_csv(out_dir / f"sac_training_q{q}.csv", training_rows)
            if incumbent_agent is not None:
                accepted, selection_row = paired_simulator_policy_selection(
                    bundle,
                    agent,
                    method,
                    incumbent_agent,
                    method,
                    policy_config.simulator_selection_episodes,
                    config.seed * 1_000_000 + q * 10_000 + 820_000,
                    policy_config.lambda_pess,
                    policy_config.behavior_support_lambda,
                    policy_config.simulator_selection_z_value,
                    policy_config.simulator_selection_tolerance,
                    "synthetic_candidate_vs_incumbent",
                    q,
                )
                policy_selection_rows.append(selection_row)
                if not accepted:
                    agent = incumbent_agent
                    events.log(
                        "round_synthetic_policy_rolled_back",
                        round=q,
                        selection_source=selection_row["selection_source"],
                        paired_delta_mean=selection_row["paired_delta_mean"],
                        paired_delta_lower_bound=selection_row[
                            "paired_delta_lower_bound"
                        ],
                    )
            policy: Any = agent
        else:
            events.log("round_policy_training_start", round=q)
            incumbent_dfl = dfl_policy
            candidate_dfl, training_rows = train_dfl_policy(
                train_logs,
                bundle,
                config,
                policy_config,
                round_index=q,
                previous_policy=incumbent_dfl,
            )
            write_csv(out_dir / f"dfl_training_q{q}.csv", training_rows)
            dfl_policy = candidate_dfl
            if incumbent_dfl is not None:
                accepted, selection_row = paired_simulator_policy_selection(
                    bundle,
                    candidate_dfl,
                    method,
                    incumbent_dfl,
                    method,
                    policy_config.simulator_selection_episodes,
                    config.seed * 1_000_000 + q * 10_000 + 820_000,
                    policy_config.lambda_pess,
                    policy_config.behavior_support_lambda,
                    policy_config.simulator_selection_z_value,
                    policy_config.simulator_selection_tolerance,
                    "dfl_candidate_vs_incumbent",
                    q,
                )
                policy_selection_rows.append(selection_row)
                if not accepted:
                    dfl_policy = incumbent_dfl
                    events.log(
                        "round_dfl_policy_rolled_back",
                        round=q,
                        selection_source=selection_row["selection_source"],
                        paired_delta_mean=selection_row["paired_delta_mean"],
                        paired_delta_lower_bound=selection_row[
                            "paired_delta_lower_bound"
                        ],
                    )
            policy = dfl_policy
        events.log("round_policy_training_done", round=q)

        new_logs = collect_rich_logging(
            panel_config,
            config.new_logged_days,
            config.seed * 1_000_000 + q * 10_000,
            "train",
            config.logging_batch_size,
            policy=policy,
            policy_normalizer=bundle.state_normalizer,
            exploration_epsilon=policy_config.real_exploration_epsilon,
        )
        if method == "sacsa_reserve" and policy_config.real_online_update:
            assert agent is not None
            pre_real_agent = copy.deepcopy(agent)
            events.log("round_real_online_update_start", round=q)
            real_rows = real_online_update(
                agent, new_logs, bundle, config, policy_config
            )
            write_csv(out_dir / f"real_update_q{q}.csv", real_rows)
            events.log(
                "round_real_online_update_done",
                round=q,
                gradient_updates=int(
                    round(
                        new_logs.days
                        * config.T
                        * float(policy_config.real_gradient_steps_per_step)
                    )
                ),
                critic_warmup_updates=int(policy_config.real_critic_warmup_updates),
            )
            accepted, selection_row = paired_simulator_policy_selection(
                bundle,
                agent,
                method,
                pre_real_agent,
                method,
                policy_config.simulator_selection_episodes,
                config.seed * 1_000_000 + q * 10_000 + 830_000,
                policy_config.lambda_pess,
                policy_config.behavior_support_lambda,
                policy_config.simulator_selection_z_value,
                policy_config.simulator_selection_tolerance,
                "post_real_update_vs_pre_real",
                q,
            )
            policy_selection_rows.append(selection_row)
            if not accepted:
                agent = pre_real_agent
                policy = agent
                events.log(
                    "round_real_online_update_rolled_back",
                    round=q,
                    selection_source=selection_row["selection_source"],
                    paired_delta_mean=selection_row["paired_delta_mean"],
                    paired_delta_lower_bound=selection_row[
                        "paired_delta_lower_bound"
                    ],
                )

        simulator_baselines = []
        for baseline in BASELINE_METHODS:
            baseline_summary = evaluate_learned_policy(
                bundle,
                None,
                baseline,
                policy_config.simulator_selection_episodes,
                config.seed * 1_000_000 + q * 10_000 + 840_000,
                policy_config.lambda_pess,
                policy_config.behavior_support_lambda,
            )
            simulator_baselines.append({"method": baseline, **baseline_summary})
        best_simulator_baseline = max(
            simulator_baselines, key=lambda row: float(row["sim_selection_score"])
        )
        _, selection_row = paired_simulator_policy_selection(
            bundle,
            policy,
            method,
            None,
            str(best_simulator_baseline["method"]),
            policy_config.simulator_selection_episodes,
            config.seed * 1_000_000 + q * 10_000 + 840_000,
            policy_config.lambda_pess,
            policy_config.behavior_support_lambda,
            policy_config.simulator_selection_z_value,
            policy_config.simulator_selection_tolerance,
            "selected_policy_vs_best_simulator_operational_diagnostic",
            q,
        )
        policy_selection_rows.append(selection_row)

        summary, raw = evaluate_true_policy(
            panel_config,
            policy,
            method,
            config.eval_episodes,
            config.seed * 1_000_000 + 900_000,
            policy_normalizer=bundle.state_normalizer,
        )
        learned_summary = evaluate_learned_policy(
            bundle,
            policy,
            method,
            min(config.eval_episodes, 1000),
            config.seed * 1_000_000 + q * 1000 + 700_000,
            policy_config.lambda_pess,
        )
        calibration_rows.append(
            {
                "round": q,
                "method": method,
                **learned_summary,
                "true_net_reward": summary["net_reward"],
                "true_spend": summary["spend"],
                "sim_minus_true_net_reward": learned_summary["sim_net_reward"] - summary["net_reward"],
            }
        )
        support_diagnostic_rows.extend(
            policy_support_rows(
                policy,
                train_logs,
                holdout_logs,
                bundle,
                config,
                policy_config,
                q,
            )
        )
        best_baseline = max(row["net_reward"] for row in baseline_summaries)
        row = {
            "method": method,
            "round": q,
            "logged_days": train_logs.days,
            "delta_vs_best_operational": summary["net_reward"] - best_baseline,
            **summary,
            **bundle.diagnostics,
        }
        round_rows.append(row)
        final_raw.extend({"method": method, "round": q, **item} for item in raw)
        events.log(
            "round_true_dgp_summary",
            round=q,
            true_net_reward=summary["net_reward"],
            spend=summary["spend"],
            delta_vs_best_operational=row["delta_vs_best_operational"],
        )
        train_logs = TrajectoryData.concatenate([train_logs, new_logs])
        write_csv(
            out_dir / f"logging_coverage_q{q}.csv",
            logging_coverage_rows(train_logs, config.daily_budget),
        )
        write_csv(
            out_dir / f"logging_design_summary_q{q}.csv",
            logging_design_summary_rows(train_logs),
        )
        write_csv(
            out_dir / f"logging_time_support_q{q}.csv",
            logging_time_support_rows(train_logs, config.daily_budget),
        )
        write_csv(out_dir / "round_eval_summary.csv", round_rows)
        write_csv(out_dir / "simulator_vs_true_policy_calibration.csv", calibration_rows)
        write_csv(out_dir / "policy_support_diagnostics.csv", support_diagnostic_rows)
        write_csv(
            out_dir / "simulator_policy_selection_diagnostics.csv",
            policy_selection_rows,
        )
    if any(int(row.get("target_dgp_episodes", -1)) != 0 for row in policy_selection_rows):
        raise RuntimeError("Checkpoint selection unexpectedly consumed target-DGP episodes.")
    write_csv(out_dir / "final_eval_raw.csv", final_raw)
    write_csv(out_dir / "final_eval_summary.csv", [*baseline_summaries, *round_rows])
    if agent is not None:
        torch.save(agent.actor.state_dict(), out_dir / "sacsa_actor.pt")
    if dfl_policy is not None:
        torch.save([model.state_dict() for model in dfl_policy.models], out_dir / "dfl_models.pt")
    events.log("run_complete", completed_rounds=config.q_max)
