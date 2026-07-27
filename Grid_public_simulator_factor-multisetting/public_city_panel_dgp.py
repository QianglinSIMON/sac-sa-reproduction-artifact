"""Public-grid-calibrated city-time panel true DGP.

The construction has two deliberately separate layers:

1. The original 9x9 microscopic order-driver simulator generates independent
   no-subsidy city-day trajectories. These trajectories are aggregated before
   dispatch at the city-time level.
2. A fixed baseline neural transition model is fitted on a day-level training
   split only. Complete day-level residual paths are retained and paired with
   their own initial states. A bounded, concave, zero-at-zero subsidy
   intervention is then added to this frozen baseline model.

The resulting environment is a controlled, public-grid-calibrated
semi-synthetic DGP. It is not a claim that the artificial subsidy effect was
identified from the public simulator.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


_ENV_PATH = Path(__file__).with_name("Env-DGP.py")
_SPEC = importlib.util.spec_from_file_location("public_grid_base_env", _ENV_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load base public grid simulator from {_ENV_PATH}.")
BASE_ENV = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BASE_ENV)


EPS = 1e-6
CONTEXT_NAMES = (
    "matchability",
    "mean_remaining_wait",
    "expiring_orders",
    "returning_drivers",
    "mean_pickup_distance",
    "feasible_pair_count",
    "mean_candidate_trip_score",
)
DYNAMIC_NAMES = ("active_orders", "available_drivers", *CONTEXT_NAMES)
OUTPUT_NAMES = ("gross_response", *[f"next_{name}" for name in DYNAMIC_NAMES])
DYNAMIC_DIM = len(DYNAMIC_NAMES)
OUTPUT_DIM = len(OUTPUT_NAMES)
STATIC_FEATURE_NAMES = (
    "order_scale",
    "driver_scale",
    "reward_scale",
    "first_demand_peak_time",
    "second_demand_peak_time",
    "centered_supply",
    "mismatched_supply",
)
STATIC_DIM = len(STATIC_FEATURE_NAMES)
POLICY_TIME_DIM = 4
_CALIBRATION_CACHE_VERSION = 8
_CALIBRATION_CACHE: Dict[Tuple[Any, ...], "BaselineCalibration"] = {}

ORDER_REFERENCE_SCALE = 100.0
DRIVER_REFERENCE_SCALE = 25.0


@dataclass(frozen=True)
class CitySpec:
    num_orders: int
    num_drivers: int
    reward_scale: float
    demand_center_1: Tuple[float, float, float]
    demand_center_2: Tuple[float, float, float]
    driver_mode: str
    driver_center: Tuple[float, float]
    gamma_z: float


DEFAULT_CITY_SPECS: Tuple[CitySpec, ...] = (
    CitySpec(70, 30, 0.92, (2.5, 2.5, 3.0), (5.2, 5.2, 13.0), "center", (4.0, 4.0), 5.5),
    CitySpec(90, 27, 0.98, (3.0, 3.0, 2.5), (6.0, 6.0, 14.0), "uniform", (4.0, 4.0), 12.0),
    CitySpec(105, 25, 1.03, (3.5, 2.5, 4.0), (5.6, 6.2, 14.0), "uniform", (4.0, 4.0), 3.8),
    CitySpec(130, 21, 1.10, (2.0, 6.0, 4.0), (6.7, 2.4, 15.0), "mismatch", (1.5, 7.0), 13.0),
    CitySpec(150, 18, 1.16, (6.4, 2.0, 5.0), (2.1, 6.8, 15.5), "mismatch", (1.5, 7.0), 7.0),
    CitySpec(115, 23, 1.07, (6.2, 2.0, 4.5), (2.0, 6.4, 15.0), "mismatch", (6.8, 6.8), 9.0),
)


@dataclass
class PublicCityPanelConfig:
    M: int = 3
    T: int = int(BASE_ENV.NUMBER_OF_TIME_STEPS)
    daily_budget: float = 10.0
    subsidy_reference_budget: float = 10.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    calibration_episodes: int = 10000
    calibration_seed: int = 12345
    calibration_workers: int = 0
    baseline_hidden_dims: Tuple[int, ...] = (256, 256, 128)
    baseline_epochs: int = 100
    baseline_batch_size: int = 8192
    baseline_learning_rate: float = 1e-3
    baseline_weight_decay: float = 1e-5
    baseline_patience: int = 15
    baseline_z_loss_weight: float = 4.0
    driver_cap_multiplier: float = 1.25
    subsidy_response_scale: float = 0.15
    subsidy_intensity_cap: float = 0.60
    gap_response_scale: float = 0.10
    balance_response_scale: float = 0.45
    demand_clearance_ratio: float = 0.55
    supply_activation_ratio: float = 0.25
    matchability_response_ratio: float = 0.20
    waiting_response_ratio: float = 0.15
    expiring_response_ratio: float = 0.30
    returning_response_ratio: float = 0.15
    pickup_response_ratio: float = 0.10
    feasible_pair_response_ratio: float = 0.25
    avg_trip_response_score: float = 5.0
    city_specs: Tuple[CitySpec, ...] = DEFAULT_CITY_SPECS


@dataclass
class BaselineCalibration:
    init_dynamic: torch.Tensor
    city_O: torch.Tensor
    model_state: Dict[str, torch.Tensor]
    input_mean: torch.Tensor
    input_scale: torch.Tensor
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    residual_z: torch.Tensor
    residual_next: torch.Tensor
    split_indices: Dict[str, torch.Tensor]
    metrics: Dict[str, float]


class BaselineTransitionMLP(nn.Module):
    """Shared baseline trunk with separate response and transition heads."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        if int(output_dim) != OUTPUT_DIM:
            raise ValueError(f"Baseline model requires output_dim={OUTPUT_DIM}.")
        layers: List[nn.Module] = []
        width = int(input_dim)
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            width = hidden
        self.trunk = nn.Sequential(*layers)
        head_width = max(64, width // 2)
        self.response_head = nn.Sequential(
            nn.Linear(width, head_width),
            nn.SiLU(),
            nn.Linear(head_width, 1),
        )
        self.transition_head = nn.Sequential(
            nn.Linear(width, head_width),
            nn.SiLU(),
            nn.Linear(head_width, DYNAMIC_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        return torch.cat([self.response_head(features), self.transition_head(features)], dim=-1)


def _city_specs_for_cfg(cfg: PublicCityPanelConfig) -> Tuple[CitySpec, ...]:
    m = int(cfg.M)
    if m < 1 or m > len(cfg.city_specs):
        raise ValueError(f"Public-city calibration requires 1 <= M <= {len(cfg.city_specs)}; got {m}.")
    return tuple(cfg.city_specs[:m])


def _validate_panel_config(cfg: PublicCityPanelConfig) -> None:
    _city_specs_for_cfg(cfg)
    if int(cfg.T) != int(BASE_ENV.NUMBER_OF_TIME_STEPS):
        raise ValueError(f"Public city-panel benchmark requires T={BASE_ENV.NUMBER_OF_TIME_STEPS}.")
    if float(cfg.daily_budget) < 0.0:
        raise ValueError("daily_budget must be nonnegative.")
    if float(cfg.subsidy_reference_budget) <= 0.0:
        raise ValueError("subsidy_reference_budget must be positive.")
    if int(cfg.calibration_episodes) < 100:
        raise ValueError("At least 100 microscopic calibration days are required.")
    if not cfg.baseline_hidden_dims or any(int(width) <= 0 for width in cfg.baseline_hidden_dims):
        raise ValueError("baseline_hidden_dims must contain positive widths.")
    if int(cfg.baseline_epochs) <= 0 or int(cfg.baseline_batch_size) <= 0:
        raise ValueError("Baseline epochs and batch size must be positive.")
    if float(cfg.baseline_learning_rate) <= 0.0 or float(cfg.baseline_z_loss_weight) <= 0.0:
        raise ValueError("Baseline learning rate and Z loss weight must be positive.")
    if float(cfg.driver_cap_multiplier) < 1.0:
        raise ValueError("driver_cap_multiplier must be at least one.")
    nonnegative_fields = (
        "subsidy_response_scale",
        "subsidy_intensity_cap",
        "gap_response_scale",
        "balance_response_scale",
        "demand_clearance_ratio",
        "supply_activation_ratio",
        "matchability_response_ratio",
        "waiting_response_ratio",
        "expiring_response_ratio",
        "returning_response_ratio",
        "pickup_response_ratio",
        "feasible_pair_response_ratio",
    )
    for name in nonnegative_fields:
        if float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be nonnegative.")


def _time_profile(t: int, T: int) -> float:
    tau = float(t + 1) / float(T)
    morning = math.exp(-0.5 * ((tau - 0.25) / 0.11) ** 2)
    evening = 1.20 * math.exp(-0.5 * ((tau - 0.75) / 0.13) ** 2)
    return 0.20 + morning + evening


def _city_subsidy_time_profile(spec: CitySpec, t: int) -> float:
    morning = math.exp(-0.5 * ((float(t) - float(spec.demand_center_1[2])) / 2.4) ** 2)
    evening = math.exp(-0.5 * ((float(t) - float(spec.demand_center_2[2])) / 2.8) ** 2)
    return 0.45 + 0.65 * morning + 0.75 * evening


def _time_features_tensor(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = []
    for t in range(int(T)):
        angle = 2.0 * math.pi * float(t + 1) / float(T)
        rows.append(
            [
                _time_profile(t, T),
                math.sin(angle),
                math.cos(angle),
                float(T - t) / float(T),
            ]
        )
    return torch.tensor(rows, device=device, dtype=dtype)


def _baseline_time_features_tensor(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Smooth time profile plus an interval indicator for the fixed baseline model."""
    smooth = _time_features_tensor(T, device, dtype)
    interval = torch.eye(int(T), device=device, dtype=dtype)
    return torch.cat([smooth, interval], dim=-1)


def _baseline_input_dim(cfg: PublicCityPanelConfig) -> int:
    return DYNAMIC_DIM + STATIC_DIM + POLICY_TIME_DIM + int(cfg.T)


def city_supply_demand_pressure(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return (X.clamp_min(0.0) / (X.clamp_min(0.0) + Y.clamp_min(0.0) + EPS)).clamp(0.0, 1.0)


def _micro_config(spec: CitySpec, cfg: PublicCityPanelConfig) -> Any:
    return BASE_ENV.MicroGridConfig(
        T=int(cfg.T),
        num_orders=int(spec.num_orders),
        num_drivers=int(spec.num_drivers),
        max_manhattan_distance=int(BASE_ENV.MAX_MANHATTAN_DISTANCE),
        base_reward_per_trip=float(BASE_ENV.BASE_REWARD_PER_TRIP),
        reward_for_distance=float(BASE_ENV.REWARD_FOR_DISTANCE_PARAMETER),
        reward_scale=float(spec.reward_scale),
        demand_center_1=tuple(spec.demand_center_1),
        demand_center_2=tuple(spec.demand_center_2),
        driver_mode=str(spec.driver_mode),
        driver_center=tuple(spec.driver_center),
    )


def _split_day_indices(n: int, seed: int) -> Dict[str, torch.Tensor]:
    if int(n) < 100:
        raise ValueError("At least 100 microscopic calibration days are required.")
    names = (
        "train",
        "calibration",
        "validation",
        "holdout",
        "unused_reserved",
        "evaluation",
    )
    # Every downstream role receives a disjoint pool. The reserved pool is not
    # used for checkpoint selection, which is performed only in the learned simulator.
    fractions = (0.50, 0.08, 0.08, 0.08, 0.08, 0.18)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 991)
    permutation = torch.randperm(int(n), generator=generator)
    counts = [int(round(frac * n)) for frac in fractions[:-1]]
    counts.append(int(n) - sum(counts))
    result: Dict[str, torch.Tensor] = {}
    start = 0
    for name, count in zip(names, counts):
        result[name] = permutation[start : start + count].clone()
        start += count
    if start != int(n) or any(indices.numel() == 0 for indices in result.values()):
        raise RuntimeError("Invalid public-city day-level calibration split.")
    return result


def _simulate_baseline_chunk(
    cfg: PublicCityPanelConfig,
    start_episode: int,
    end_episode: int,
) -> Tuple[torch.Tensor, ...]:
    n = int(end_episode - start_episode)
    T = int(cfg.T)
    M = int(cfg.M)
    tensors = [torch.zeros((n, T, M), dtype=torch.float32) for _ in range(11)]
    X, Y, Q, W, E, R, P, F, V, Z, completed = tensors
    specs = _city_specs_for_cfg(cfg)
    for local_episode, global_episode in enumerate(range(int(start_episode), int(end_episode))):
        for m, spec in enumerate(specs):
            micro_seed = int(cfg.calibration_seed) + 104_729 * int(global_episode) + 1_009 * int(m)
            day = BASE_ENV.simulate_micro_city_day(_micro_config(spec, cfg), micro_seed)
            X[local_episode, :, m] = day["active_orders"]
            Y[local_episode, :, m] = day["available_drivers"]
            Q[local_episode, :, m] = day["matchability"]
            W[local_episode, :, m] = day["mean_remaining_wait"]
            E[local_episode, :, m] = day["expiring_orders"]
            R[local_episode, :, m] = day["returning_drivers"]
            P[local_episode, :, m] = day["mean_pickup_distance"]
            F[local_episode, :, m] = day["feasible_pair_count"]
            V[local_episode, :, m] = day["mean_candidate_trip_score"]
            Z[local_episode, :, m] = day["gross_response"]
            completed[local_episode, :, m] = day["completed_orders"]
    return X, Y, Q, W, E, R, P, F, V, Z, completed


def _simulate_baseline_chunk_worker(args: Tuple[PublicCityPanelConfig, int, int]) -> Tuple[torch.Tensor, ...]:
    torch.set_num_threads(1)
    return _simulate_baseline_chunk(*args)


def _simulate_baseline_panel(cfg: PublicCityPanelConfig) -> Tuple[torch.Tensor, ...]:
    n = int(cfg.calibration_episodes)
    requested = int(cfg.calibration_workers) if int(cfg.calibration_workers) > 0 else int(
        os.environ.get("PUBLIC_CITY_CALIBRATION_WORKERS", str(min(max(os.cpu_count() or 1, 1), 8)))
    )
    workers = min(max(requested, 1), n)
    if workers <= 1 or n < 500:
        return _simulate_baseline_chunk(cfg, 0, n)
    chunk_size = int(math.ceil(n / workers))
    tasks = [(cfg, start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]
    try:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            parts = list(executor.map(_simulate_baseline_chunk_worker, tasks))
    except (OSError, RuntimeError, PermissionError):
        return _simulate_baseline_chunk(cfg, 0, n)
    return tuple(torch.cat([part[i] for part in parts], dim=0) for i in range(11))


def _dynamic_tensor(panel: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    X, Y, Q, W, E, R, P, F, V = panel[:9]
    return torch.stack([X, Y, Q, W, E, R, P, F, V], dim=-1)


def _city_O(cfg: PublicCityPanelConfig) -> torch.Tensor:
    rows = []
    for spec in _city_specs_for_cfg(cfg):
        rows.append(
            [
                float(spec.num_orders) / ORDER_REFERENCE_SCALE,
                float(spec.num_drivers) / DRIVER_REFERENCE_SCALE,
                float(spec.reward_scale),
                float(spec.demand_center_1[2]) / max(float(cfg.T - 1), 1.0),
                float(spec.demand_center_2[2]) / max(float(cfg.T - 1), 1.0),
                float(spec.driver_mode == "center"),
                float(spec.driver_mode == "mismatch"),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def _baseline_inputs(dynamic: torch.Tensor, city_O: torch.Tensor, T: int) -> torch.Tensor:
    n, horizon, m, _ = dynamic.shape
    if horizon != int(T):
        raise ValueError("Baseline dynamic tensor has the wrong horizon.")
    time_features = _baseline_time_features_tensor(T, dynamic.device, dynamic.dtype)
    baseline_time_dim = POLICY_TIME_DIM + int(T)
    time_features = time_features.view(1, T, 1, baseline_time_dim).expand(n, T, m, baseline_time_dim)
    if city_O.shape != (m, STATIC_DIM):
        raise ValueError(f"Expected fixed city features with shape {(m, STATIC_DIM)}, got {tuple(city_O.shape)}.")
    city_features = city_O.to(dynamic).view(1, 1, m, STATIC_DIM).expand(n, T, m, STATIC_DIM)
    return torch.cat([dynamic, city_features, time_features], dim=-1)


def _baseline_targets(dynamic: torch.Tensor, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    targets = torch.zeros((*Z.shape, OUTPUT_DIM), dtype=Z.dtype, device=Z.device)
    masks = torch.zeros_like(targets)
    targets[..., 0] = Z
    masks[..., 0] = 1.0
    targets[:, :-1, :, 1:] = dynamic[:, 1:, :, :]
    masks[:, :-1, :, 1:] = 1.0
    return targets, masks


def _rows_for_days(tensor: torch.Tensor, day_indices: torch.Tensor) -> torch.Tensor:
    return tensor[day_indices].reshape(-1, tensor.shape[-1])


def _target_moments(targets: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    count = masks.sum(dim=0).clamp_min(1.0)
    mean = (targets * masks).sum(dim=0) / count
    var = ((targets - mean).pow(2) * masks).sum(dim=0) / count
    return mean, var.sqrt().clamp_min(1e-4)


def _predict_rows(
    model: BaselineTransitionMLP,
    inputs: torch.Tensor,
    input_mean: torch.Tensor,
    input_scale: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, inputs.shape[0], int(batch_size)):
            batch = inputs[start : start + int(batch_size)]
            encoded = (batch - input_mean) / input_scale
            outputs.append(model(encoded) * target_scale + target_mean)
    return torch.cat(outputs, dim=0)


def _standardized_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    pred_standard = (pred - target_mean) / target_scale
    target_standard = (target - target_mean) / target_scale
    weighted_mask = mask * loss_weights
    return (
        (pred_standard - target_standard).pow(2) * weighted_mask
    ).sum() / weighted_mask.sum().clamp_min(1.0)


def _fit_baseline_model(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    split_indices: Dict[str, torch.Tensor],
    cfg: PublicCityPanelConfig,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    train_inputs = _rows_for_days(inputs, split_indices["train"])
    train_targets = _rows_for_days(targets, split_indices["train"])
    train_masks = _rows_for_days(masks, split_indices["train"])
    calibration_inputs = _rows_for_days(inputs, split_indices["calibration"])
    calibration_targets = _rows_for_days(targets, split_indices["calibration"])
    calibration_masks = _rows_for_days(masks, split_indices["calibration"])

    input_mean = train_inputs.mean(dim=0)
    input_scale = train_inputs.std(dim=0, unbiased=False).clamp_min(1e-4)
    target_mean, target_scale = _target_moments(train_targets, train_masks)

    device_name = str(cfg.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.calibration_seed) + 7_919)
    torch.manual_seed(int(cfg.calibration_seed) + 7_919)
    model = BaselineTransitionMLP(inputs.shape[-1], cfg.baseline_hidden_dims, OUTPUT_DIM).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.baseline_learning_rate),
        weight_decay=float(cfg.baseline_weight_decay),
    )
    input_mean_d = input_mean.to(device)
    input_scale_d = input_scale.to(device)
    target_mean_d = target_mean.to(device)
    target_scale_d = target_scale.to(device)
    loss_weights = torch.ones(OUTPUT_DIM, device=device)
    loss_weights[0] = float(cfg.baseline_z_loss_weight)

    best_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale = 0
    batch_size = max(1, int(cfg.baseline_batch_size))
    for epoch in range(max(1, int(cfg.baseline_epochs))):
        model.train()
        permutation = torch.randperm(train_inputs.shape[0], generator=generator)
        for start in range(0, permutation.numel(), batch_size):
            idx = permutation[start : start + batch_size]
            x = train_inputs[idx].to(device)
            y = train_targets[idx].to(device)
            mask = train_masks[idx].to(device)
            pred_standard = model((x - input_mean_d) / input_scale_d)
            target_standard = (y - target_mean_d) / target_scale_d
            weighted_mask = mask * loss_weights
            loss = ((pred_standard - target_standard).pow(2) * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        calibration_pred = _predict_rows(
            model,
            calibration_inputs.to(device),
            input_mean_d,
            input_scale_d,
            target_mean_d,
            target_scale_d,
            batch_size,
        ).cpu()
        calibration_loss = float(
            _standardized_weighted_mse(
                calibration_pred,
                calibration_targets,
                calibration_masks,
                target_mean,
                target_scale,
                loss_weights.cpu(),
            ).item()
        )
        if calibration_loss < best_loss - 1e-6:
            best_loss = calibration_loss
            best_epoch = int(epoch + 1)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= max(1, int(cfg.baseline_patience)):
                break
    if best_state is None:
        raise RuntimeError("Baseline MLP training did not produce a finite checkpoint.")
    return (
        best_state,
        input_mean.cpu(),
        input_scale.cpu(),
        target_mean.cpu(),
        target_scale.cpu(),
        {
            "baseline_model_selection_standardized_weighted_mse": best_loss,
            "baseline_best_epoch": float(best_epoch),
        },
    )


def _r2_mse(target: torch.Tensor, prediction: torch.Tensor) -> Tuple[float, float]:
    # Metrics are cheap relative to fitting; float64 avoids cancellation when
    # a response block has little variation within a time slice.
    target = target.reshape(-1).to(dtype=torch.float64)
    prediction = prediction.reshape(-1).to(dtype=torch.float64)
    mse = (target - prediction).pow(2).mean()
    variance = (target - target.mean()).pow(2).mean().clamp_min(1e-12)
    return float((1.0 - mse / variance).item()), float(mse.item())


def _model_from_calibration(calibration: BaselineCalibration, cfg: PublicCityPanelConfig, device: torch.device) -> BaselineTransitionMLP:
    model = BaselineTransitionMLP(_baseline_input_dim(cfg), cfg.baseline_hidden_dims, OUTPUT_DIM)
    model.load_state_dict(calibration.model_state)
    model.to(device=device, dtype=cfg.dtype)
    model.eval()
    return model


def _all_baseline_predictions(
    calibration: BaselineCalibration,
    cfg: PublicCityPanelConfig,
    inputs: torch.Tensor,
) -> torch.Tensor:
    model = _model_from_calibration(calibration, cfg, torch.device("cpu"))
    flat = inputs.reshape(-1, inputs.shape[-1])
    pred = _predict_rows(
        model,
        flat,
        calibration.input_mean,
        calibration.input_scale,
        calibration.target_mean,
        calibration.target_scale,
        cfg.baseline_batch_size,
    )
    return pred.reshape(*inputs.shape[:-1], OUTPUT_DIM)


def _build_calibration(cfg: PublicCityPanelConfig) -> BaselineCalibration:
    split_indices = _split_day_indices(int(cfg.calibration_episodes), int(cfg.calibration_seed))
    panel = _simulate_baseline_panel(cfg)
    dynamic = _dynamic_tensor(panel)
    Z = panel[9]
    city_O = _city_O(cfg)
    inputs = _baseline_inputs(dynamic, city_O, int(cfg.T))
    targets, masks = _baseline_targets(dynamic, Z)
    model_state, input_mean, input_scale, target_mean, target_scale, train_metrics = _fit_baseline_model(
        inputs, targets, masks, split_indices, cfg
    )
    calibration = BaselineCalibration(
        init_dynamic=dynamic[:, 0].contiguous(),
        city_O=city_O.contiguous(),
        model_state=model_state,
        input_mean=input_mean,
        input_scale=input_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        residual_z=torch.empty(0),
        residual_next=torch.empty(0),
        split_indices=split_indices,
        metrics={},
    )
    prediction = _all_baseline_predictions(calibration, cfg, inputs)
    residual_z = Z - prediction[..., 0]
    residual_next = dynamic[:, 1:] - prediction[:, :-1, :, 1:]

    metrics: Dict[str, float] = {
        "calibration_episodes": float(cfg.calibration_episodes),
        "baseline_fit_days": float(split_indices["train"].numel()),
        "baseline_model_selection_days": float(split_indices["calibration"].numel()),
        "baseline_parameter_count": float(sum(value.numel() for value in model_state.values())),
        "baseline_time_indicator_dim": float(cfg.T),
        "baseline_response_transition_separate_heads": 1.0,
        **train_metrics,
    }
    for split_name, indices in split_indices.items():
        r2_z, mse_z = _r2_mse(Z[indices], prediction[indices, ..., 0])
        metrics[f"baseline_{split_name}_r2_z"] = r2_z
        metrics[f"baseline_{split_name}_mse_z"] = mse_z
        if split_name != "evaluation":
            r2_next, mse_next = _r2_mse(dynamic[indices, 1:], prediction[indices, :-1, :, 1:])
            metrics[f"baseline_{split_name}_r2_next_all"] = r2_next
            metrics[f"baseline_{split_name}_mse_next_all"] = mse_next
        interval_r2 = []
        interval_mse = []
        for t in range(int(cfg.T)):
            r2_t, mse_t = _r2_mse(Z[indices, t], prediction[indices, t, :, 0])
            interval_r2.append(r2_t)
            interval_mse.append(mse_t)
        metrics[f"baseline_{split_name}_min_r2_z_by_t"] = float(min(interval_r2))
        metrics[f"baseline_{split_name}_max_mse_z_by_t"] = float(max(interval_mse))
        city_r2 = [
            _r2_mse(Z[indices, :, m], prediction[indices, :, m, 0])[0]
            for m in range(int(cfg.M))
        ]
        metrics[f"baseline_{split_name}_min_r2_z_by_city"] = float(min(city_r2))
        metrics[f"residual_pool_{split_name}_days"] = float(indices.numel())
    metrics.update(
        {
            "baseline_mean_z": float(Z.mean().item()),
            "baseline_mean_x": float(dynamic[..., 0].mean().item()),
            "baseline_mean_y": float(dynamic[..., 1].mean().item()),
            "baseline_mean_matchability": float(dynamic[..., 2].mean().item()),
            "baseline_split_before_fit": 1.0,
            "baseline_evaluation_days_used_for_fit": 0.0,
        }
    )
    calibration.residual_z = residual_z.contiguous()
    calibration.residual_next = residual_next.contiguous()
    calibration.metrics = metrics
    return calibration


def _micro_source_digest() -> str:
    return hashlib.sha1(_ENV_PATH.read_bytes()).hexdigest()[:12]


def _cache_key(cfg: PublicCityPanelConfig) -> Tuple[Any, ...]:
    specs = tuple(tuple(asdict(spec).values()) for spec in _city_specs_for_cfg(cfg))
    return (
        _CALIBRATION_CACHE_VERSION,
        _micro_source_digest(),
        int(cfg.M),
        int(cfg.T),
        int(cfg.calibration_episodes),
        int(cfg.calibration_seed),
        tuple(int(v) for v in cfg.baseline_hidden_dims),
        int(cfg.baseline_epochs),
        int(cfg.baseline_batch_size),
        int(cfg.baseline_patience),
        float(cfg.baseline_learning_rate),
        float(cfg.baseline_weight_decay),
        float(cfg.baseline_z_loss_weight),
        str(cfg.device).split(":", 1)[0],
        specs,
    )


def _cache_path(key: Tuple[Any, ...]) -> Path:
    root = Path(os.environ.get("PUBLIC_CITY_CALIBRATION_CACHE_DIR", ".cache/public_city_panel"))
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:20]
    return root / f"calibration_v{_CALIBRATION_CACHE_VERSION}_{digest}.pt"


def _calibration_payload(calibration: BaselineCalibration) -> Dict[str, Any]:
    return {
        "version": _CALIBRATION_CACHE_VERSION,
        "init_dynamic": calibration.init_dynamic,
        "city_O": calibration.city_O,
        "model_state": calibration.model_state,
        "input_mean": calibration.input_mean,
        "input_scale": calibration.input_scale,
        "target_mean": calibration.target_mean,
        "target_scale": calibration.target_scale,
        "residual_z": calibration.residual_z,
        "residual_next": calibration.residual_next,
        "split_indices": calibration.split_indices,
        "metrics": calibration.metrics,
    }


def _calibration_from_payload(payload: Dict[str, Any]) -> BaselineCalibration:
    if int(payload.get("version", -1)) != _CALIBRATION_CACHE_VERSION:
        raise ValueError("Public-city calibration cache version mismatch.")
    return BaselineCalibration(
        init_dynamic=payload["init_dynamic"],
        city_O=payload["city_O"],
        model_state=payload["model_state"],
        input_mean=payload["input_mean"],
        input_scale=payload["input_scale"],
        target_mean=payload["target_mean"],
        target_scale=payload["target_scale"],
        residual_z=payload["residual_z"],
        residual_next=payload["residual_next"],
        split_indices=payload["split_indices"],
        metrics=payload["metrics"],
    )


def _load_cache(path: Path) -> Optional[BaselineCalibration]:
    if not path.exists():
        return None
    try:
        return _calibration_from_payload(torch.load(path, map_location="cpu", weights_only=False))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return None


def _get_calibration(cfg: PublicCityPanelConfig) -> BaselineCalibration:
    _validate_panel_config(cfg)
    key = _cache_key(cfg)
    if key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[key]
    path = _cache_path(key)
    cached = _load_cache(path)
    if cached is not None:
        cached.metrics["persistent_calibration_cache_hit"] = 1.0
        _CALIBRATION_CACHE[key] = cached
        return cached
    lock_path = path.with_suffix(path.suffix + ".lock")
    acquired = False
    deadline = time.time() + 7_200.0
    while not acquired:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            acquired = True
        except FileExistsError:
            cached = _load_cache(path)
            if cached is not None:
                cached.metrics["persistent_calibration_cache_hit"] = 1.0
                _CALIBRATION_CACHE[key] = cached
                return cached
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for public-city calibration cache {path}.")
            time.sleep(1.0)
    try:
        calibration = _build_calibration(cfg)
        calibration.metrics["persistent_calibration_cache_hit"] = 0.0
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        torch.save(_calibration_payload(calibration), temporary)
        os.replace(temporary, path)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    _CALIBRATION_CACHE[key] = calibration
    return calibration


class BaselinePredictor:
    def __init__(self, calibration: BaselineCalibration, cfg: PublicCityPanelConfig, device: torch.device):
        self.calibration = calibration
        self.cfg = cfg
        self.device = device
        self.model = _model_from_calibration(calibration, cfg, device)
        self.input_mean = calibration.input_mean.to(device=device, dtype=cfg.dtype)
        self.input_scale = calibration.input_scale.to(device=device, dtype=cfg.dtype)
        self.target_mean = calibration.target_mean.to(device=device, dtype=cfg.dtype)
        self.target_scale = calibration.target_scale.to(device=device, dtype=cfg.dtype)
        self.O = calibration.city_O.to(device=device, dtype=cfg.dtype)
        self.time_features = _baseline_time_features_tensor(cfg.T, device, cfg.dtype)

    def __call__(self, dynamic: torch.Tensor, t: int) -> torch.Tensor:
        if dynamic.shape[-2:] != (int(self.cfg.M), DYNAMIC_DIM):
            raise ValueError(f"Expected dynamic shape (...,{self.cfg.M},{DYNAMIC_DIM}), got {tuple(dynamic.shape)}.")
        prefix = dynamic.shape[:-2]
        O = self.O.view(*([1] * len(prefix)), self.cfg.M, STATIC_DIM).expand(
            *prefix, self.cfg.M, STATIC_DIM
        )
        time_dim = POLICY_TIME_DIM + int(self.cfg.T)
        time = self.time_features[int(t)].view(*([1] * len(prefix)), 1, time_dim).expand(
            *prefix, self.cfg.M, time_dim
        )
        inputs = torch.cat([dynamic, O, time], dim=-1)
        standardized = (inputs - self.input_mean) / self.input_scale
        with torch.no_grad():
            pred = self.model(standardized.reshape(-1, standardized.shape[-1]))
        return (pred * self.target_scale + self.target_mean).reshape(*prefix, self.cfg.M, OUTPUT_DIM)


def public_city_subsidy_effects(
    dynamic: torch.Tensor,
    action: torch.Tensor,
    t: int,
    cfg: PublicCityPanelConfig,
) -> Dict[str, torch.Tensor]:
    """Fixed bounded intervention; every effect is exactly zero at action=0."""
    X = dynamic[..., 0]
    Y = dynamic[..., 1]
    Q = dynamic[..., 2]
    W = dynamic[..., 3]
    E = dynamic[..., 4]
    P = dynamic[..., 6]
    V = dynamic[..., 8]
    action_scale = torch.full_like(action, float(cfg.subsidy_reference_budget) / max(int(cfg.M), 1))
    intensity = (action / action_scale.clamp_min(EPS)).clamp(0.0, float(cfg.subsidy_intensity_cap))
    scale = float(cfg.subsidy_response_scale)
    psi = scale * torch.log1p(intensity / max(scale, EPS))
    imbalance = (X - Y) / (X + Y + EPS)
    gap_multiplier = 1.0 + float(cfg.gap_response_scale) * imbalance.clamp_min(0.0)
    balance_multiplier = 1.0 + float(cfg.balance_response_scale) * (1.0 - imbalance.abs()).clamp(0.0, 1.0)
    pressure = city_supply_demand_pressure(X, Y)
    pressure_multiplier = 0.75 + 0.50 * pressure
    specs = _city_specs_for_cfg(cfg)
    gamma_base = torch.tensor([spec.gamma_z for spec in specs], device=dynamic.device, dtype=dynamic.dtype)
    gamma_time = torch.tensor(
        [_city_subsidy_time_profile(spec, int(t)) for spec in specs],
        device=dynamic.device,
        dtype=dynamic.dtype,
    )
    gamma_z = gamma_base * gamma_time
    raw_z_effect = gamma_z * psi * gap_multiplier * balance_multiplier * pressure_multiplier
    # Z is an aggregate gross-response/GMV index, not a completed-order count.
    # Hence it is not clipped by a matching-count ceiling. The intervention is
    # nevertheless zero when the current city has no feasible order-driver pair.
    operational_gate = ((X > EPS) & (Y > EPS) & (Q > EPS) & (V > EPS)).to(dynamic.dtype)
    candidate_gross_scale = X.clamp_min(0.0) * Q.clamp(0.0, 1.0) * V.clamp_min(0.0)
    z_effect = raw_z_effect * operational_gate
    trip_uplift = z_effect / max(float(cfg.avg_trip_response_score), EPS)

    next_effect = torch.zeros_like(dynamic)
    next_effect[..., 0] = -float(cfg.demand_clearance_ratio) * trip_uplift
    next_effect[..., 1] = float(cfg.supply_activation_ratio) * trip_uplift
    next_effect[..., 2] = (
        float(cfg.matchability_response_ratio)
        * trip_uplift
        / (X + 1.0)
        * (1.0 - Q).clamp_min(0.0)
    )
    next_effect[..., 3] = -float(cfg.waiting_response_ratio) * psi * W
    next_effect[..., 4] = -float(cfg.expiring_response_ratio) * trip_uplift * E / (X + 1.0)
    next_effect[..., 5] = float(cfg.returning_response_ratio) * trip_uplift
    next_effect[..., 6] = -float(cfg.pickup_response_ratio) * psi * P
    next_effect[..., 7] = (
        float(cfg.feasible_pair_response_ratio)
        * trip_uplift
        * (1.0 + Q * Y.clamp_min(0.0))
    )
    return {
        "intensity": intensity,
        "psi": psi,
        "gamma_z": gamma_z,
        "candidate_gross_scale": candidate_gross_scale,
        "z_effect": z_effect,
        "next_effect": next_effect,
    }


def _project_dynamic(dynamic: torch.Tensor, cfg: PublicCityPanelConfig) -> torch.Tensor:
    result = dynamic.clone()
    specs = _city_specs_for_cfg(cfg)
    orders_cap = torch.tensor([spec.num_orders for spec in specs], device=result.device, dtype=result.dtype)
    drivers_cap = torch.tensor(
        [float(cfg.driver_cap_multiplier) * spec.num_drivers for spec in specs],
        device=result.device,
        dtype=result.dtype,
    )
    result[..., 0] = result[..., 0].clamp_min(0.0).minimum(orders_cap)
    result[..., 1] = result[..., 1].clamp_min(0.0).minimum(drivers_cap)
    result[..., 2] = result[..., 2].clamp(0.0, 1.0)
    result[..., 3] = result[..., 3].clamp(0.0, 3.0)
    result[..., 4] = result[..., 4].clamp_min(0.0).minimum(result[..., 0])
    result[..., 5] = result[..., 5].clamp_min(0.0).minimum(drivers_cap)
    result[..., 6] = result[..., 6].clamp(0.0, float(BASE_ENV.MAX_MANHATTAN_DISTANCE + 1))
    result[..., 7] = result[..., 7].clamp_min(0.0).minimum(result[..., 0] * result[..., 1])
    max_trip_score = max(
        (1.0 + 2.0 * (BASE_ENV.NUMBER_OF_GRID_TILES_X - 1)) * float(spec.reward_scale)
        for spec in specs
    )
    result[..., 8] = result[..., 8].clamp(0.0, float(max_trip_score))
    return result


def _seed_to_pool_position(seed: int, pool_size: int) -> int:
    if int(pool_size) <= 0:
        raise ValueError("Residual-path pool must be nonempty.")
    mask = (1 << 64) - 1
    value = (int(seed) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    # Residual paths are intentionally bootstrapped with replacement. A hash
    # makes the draw reproducible; reset then uses the same selected day for
    # both the initial state and the complete residual path.
    return int(value % int(pool_size))


def _panel_config_from_sim_cfg(sim_cfg: Any) -> PublicCityPanelConfig:
    if isinstance(sim_cfg, PublicCityPanelConfig):
        return sim_cfg
    kwargs: Dict[str, Any] = {
        "M": int(getattr(sim_cfg, "M", PublicCityPanelConfig.M)),
        "T": int(getattr(sim_cfg, "Time_intervals", PublicCityPanelConfig.T)),
        "daily_budget": float(getattr(sim_cfg, "daily_budget", PublicCityPanelConfig.daily_budget)),
        "device": str(getattr(sim_cfg, "device", PublicCityPanelConfig.device)),
        "dtype": getattr(sim_cfg, "dtype", torch.float32),
        "calibration_episodes": int(getattr(sim_cfg, "public_micro_calibration_days", PublicCityPanelConfig.calibration_episodes)),
    }
    optional_fields = (
        "subsidy_response_scale",
        "subsidy_intensity_cap",
        "gap_response_scale",
        "balance_response_scale",
        "demand_clearance_ratio",
        "supply_activation_ratio",
        "matchability_response_ratio",
        "waiting_response_ratio",
        "expiring_response_ratio",
        "returning_response_ratio",
        "pickup_response_ratio",
        "feasible_pair_response_ratio",
        "driver_cap_multiplier",
    )
    for field in optional_fields:
        source_name = f"public_{field}"
        if hasattr(sim_cfg, source_name):
            kwargs[field] = float(getattr(sim_cfg, source_name))
    for field in (
        "baseline_epochs",
        "baseline_batch_size",
        "baseline_patience",
        "calibration_workers",
        "calibration_seed",
    ):
        source_name = f"public_{field}"
        if hasattr(sim_cfg, source_name):
            kwargs[field] = int(getattr(sim_cfg, source_name))
    if hasattr(sim_cfg, "public_baseline_hidden_dims"):
        kwargs["baseline_hidden_dims"] = tuple(int(v) for v in getattr(sim_cfg, "public_baseline_hidden_dims"))
    for field in (
        "baseline_learning_rate",
        "baseline_weight_decay",
        "baseline_z_loss_weight",
    ):
        source_name = f"public_{field}"
        if hasattr(sim_cfg, source_name):
            kwargs[field] = float(getattr(sim_cfg, source_name))
    return PublicCityPanelConfig(**kwargs)


def _state_from_dynamic(
    dynamic: torch.Tensor,
    O: torch.Tensor,
    H: torch.Tensor,
    t: int,
    T: int,
) -> torch.Tensor:
    X = dynamic[..., 0]
    Y = dynamic[..., 1]
    pressure = city_supply_demand_pressure(X, Y)
    blocks = [X, Y, pressure]
    blocks.extend(dynamic[..., index] for index in range(2, DYNAMIC_DIM))
    if O.shape != (dynamic.shape[-2], STATIC_DIM):
        raise ValueError(
            f"Expected fixed city features with shape {(dynamic.shape[-2], STATIC_DIM)}, got {tuple(O.shape)}."
        )
    O_expanded = O.view(*([1] * len(dynamic.shape[:-2])), dynamic.shape[-2], STATIC_DIM).expand(
        *dynamic.shape[:-2], dynamic.shape[-2], STATIC_DIM
    )
    blocks.extend(O_expanded[..., index] for index in range(STATIC_DIM))
    prefix = dynamic.shape[:-2]
    time = _time_features_tensor(T, dynamic.device, dynamic.dtype)[int(min(t, T - 1))]
    time = time.view(*([1] * len(prefix)), POLICY_TIME_DIM).expand(*prefix, POLICY_TIME_DIM)
    return torch.cat([*blocks, H.reshape(*prefix, 1), time], dim=-1)


class PublicCityPanelEnv:
    """Scalar recursive public-grid-calibrated city-panel true DGP."""

    def __init__(self, sim_cfg: Any, seed: int = 0, noise_split: str = "train"):
        self.cfg = sim_cfg
        self.panel_cfg = _panel_config_from_sim_cfg(sim_cfg)
        _validate_panel_config(self.panel_cfg)
        self.device = torch.device(self.panel_cfg.device)
        self.dtype = self.panel_cfg.dtype
        self.M = int(self.panel_cfg.M)
        self.T = int(self.panel_cfg.T)
        self.action_dim = self.M
        self.daily_budget = float(self.panel_cfg.daily_budget)
        self.calibration = _get_calibration(self.panel_cfg)
        self.predictor = BaselinePredictor(self.calibration, self.panel_cfg, self.device)
        self.O = self.calibration.city_O.to(self.device, self.dtype)
        self.generator = torch.Generator(device=self.device)
        self.seed = int(seed)
        self.noise_split = str(noise_split).lower()
        if self.noise_split not in self.calibration.split_indices:
            raise ValueError(f"Unknown residual split {noise_split!r}.")
        self.t = 0
        self.noise_idx = 0
        self.H = torch.tensor(self.daily_budget, device=self.device, dtype=self.dtype)
        self.dynamic = torch.zeros((self.M, DYNAMIC_DIM), device=self.device, dtype=self.dtype)

    @property
    def state_dim(self) -> int:
        return (DYNAMIC_DIM + 1 + STATIC_DIM) * self.M + 1 + 4

    def calibration_summary(self) -> Dict[str, float]:
        return dict(self.calibration.metrics)

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            self.seed = int(seed)
        self.generator.manual_seed(self.seed)
        pool = self.calibration.split_indices[self.noise_split]
        position = _seed_to_pool_position(self.seed, int(pool.numel()))
        self.noise_idx = int(pool[position].item())
        self.t = 0
        self.H = torch.tensor(self.daily_budget, device=self.device, dtype=self.dtype)
        self.dynamic = self.calibration.init_dynamic[self.noise_idx].to(self.device, self.dtype).clone()
        self.dynamic = _project_dynamic(self.dynamic, self.panel_cfg)
        return self._state()

    def _state(self) -> torch.Tensor:
        return _state_from_dynamic(self.dynamic, self.O, self.H, self.t, self.T)

    def step(self, action: torch.Tensor):
        action = torch.as_tensor(action, device=self.device, dtype=self.dtype).flatten()
        if action.numel() != self.M:
            raise ValueError(f"Action must have length {self.M}, got {action.numel()}.")
        if torch.any(action < -1e-6):
            raise ValueError("Action contains negative city subsidies.")
        action = action.clamp_min(0.0)
        if float(action.sum().item()) > float(self.H.item()) + 1e-5:
            raise ValueError("City-level subsidy action exceeds remaining budget.")
        current = self.dynamic.clone()
        prediction = self.predictor(current, self.t)
        residual_z = self.calibration.residual_z[self.noise_idx, self.t].to(self.device, self.dtype)
        z0 = (prediction[..., 0] + residual_z).clamp_min(0.0)
        effect = public_city_subsidy_effects(current, action, self.t, self.panel_cfg)
        z = (z0 + effect["z_effect"]).clamp_min(0.0)
        spend = action.sum()
        reward = z.sum() - spend
        self.H = (self.H - spend).clamp_min(0.0)
        self.t += 1
        done = self.t >= self.T
        projection_clip_mask = torch.zeros_like(current, dtype=torch.bool)
        if not done:
            residual_next = self.calibration.residual_next[self.noise_idx, self.t - 1].to(self.device, self.dtype)
            baseline_next = prediction[..., 1:] + residual_next
            raw_next = baseline_next + effect["next_effect"]
            self.dynamic = _project_dynamic(raw_next, self.panel_cfg)
            projection_clip_mask = (self.dynamic - raw_next).abs() > 1e-6
            next_state = self._state()
        else:
            next_state = None
        info = {
            "Z": z.detach().clone(),
            "Z0": z0.detach().clone(),
            "subsidy_effect": effect["z_effect"].detach().clone(),
            "next_state_effect": effect["next_effect"].detach().clone(),
            "spend": spend.detach().clone(),
            "spend_by_city": action.detach().clone(),
            "remaining_budget": self.H.detach().clone(),
            "reward": reward.detach().clone(),
            "noise_split": self.noise_split,
            "noise_idx": int(self.noise_idx),
            "paired_day_index": int(self.noise_idx),
            "initial_state_residual_pairing": True,
            "city_dynamic_state": current.detach().clone(),
            "subsidy_intensity": effect["intensity"].detach().clone(),
            "subsidy_gamma_z": effect["gamma_z"].detach().clone(),
            "candidate_gross_scale": effect["candidate_gross_scale"].detach().clone(),
            "projection_clip_mask": projection_clip_mask.detach().clone(),
            "projection_applied": bool(not done),
        }
        return next_state, reward, done, info

    def gap_weights(self) -> torch.Tensor:
        return _normalize_weights((self.dynamic[:, 0] - self.dynamic[:, 1]).clamp_min(0.0) + EPS)

    def zone_gap_weights(self) -> torch.Tensor:
        return self.gap_weights()


class BatchedPublicCityPanelEnv:
    """Vectorized environment with the same recursion as PublicCityPanelEnv."""

    def __init__(self, sim_cfg: Any, seeds: Sequence[int], noise_split: str = "train"):
        if not seeds:
            raise ValueError("BatchedPublicCityPanelEnv requires at least one seed.")
        prototype = PublicCityPanelEnv(sim_cfg, int(seeds[0]), noise_split=noise_split)
        self.cfg = sim_cfg
        self.panel_cfg = prototype.panel_cfg
        self.device = prototype.device
        self.dtype = prototype.dtype
        self.M = prototype.M
        self.T = prototype.T
        self.action_dim = self.M
        self.daily_budget = prototype.daily_budget
        self.calibration = prototype.calibration
        self.predictor = BaselinePredictor(self.calibration, self.panel_cfg, self.device)
        self.O = self.calibration.city_O.to(self.device, self.dtype)
        self.noise_split = str(noise_split).lower()
        self.seeds = [int(seed) for seed in seeds]
        pool = self.calibration.split_indices[self.noise_split]
        self.noise_indices = torch.tensor(
            [int(pool[_seed_to_pool_position(seed, int(pool.numel()))].item()) for seed in self.seeds],
            dtype=torch.long,
        )
        self.batch_size = len(self.seeds)
        self.t = 0
        self.H = torch.full((self.batch_size,), self.daily_budget, device=self.device, dtype=self.dtype)
        self.dynamic = torch.zeros((self.batch_size, self.M, DYNAMIC_DIM), device=self.device, dtype=self.dtype)
        self.reset()

    @property
    def state_dim(self) -> int:
        return (DYNAMIC_DIM + 1 + STATIC_DIM) * self.M + 1 + 4

    def reset(self) -> torch.Tensor:
        self.t = 0
        self.H.fill_(self.daily_budget)
        self.dynamic = self.calibration.init_dynamic[self.noise_indices].to(self.device, self.dtype).clone()
        self.dynamic = _project_dynamic(self.dynamic, self.panel_cfg)
        return self._state()

    def _state(self) -> torch.Tensor:
        return _state_from_dynamic(self.dynamic, self.O, self.H, self.t, self.T)

    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, device=self.device, dtype=self.dtype)
        if actions.shape != (self.batch_size, self.M):
            raise ValueError(f"Expected action shape {(self.batch_size, self.M)}, got {tuple(actions.shape)}.")
        if torch.any(actions < -1e-6):
            raise ValueError("Actions contain negative city subsidies.")
        actions = actions.clamp_min(0.0)
        if torch.any(actions.sum(dim=-1) > self.H + 1e-5):
            raise ValueError("At least one city-level action exceeds remaining budget.")
        current = self.dynamic.clone()
        prediction = self.predictor(current, self.t)
        residual_z = self.calibration.residual_z[self.noise_indices, self.t].to(self.device, self.dtype)
        z0 = (prediction[..., 0] + residual_z).clamp_min(0.0)
        effect = public_city_subsidy_effects(current, actions, self.t, self.panel_cfg)
        z = (z0 + effect["z_effect"]).clamp_min(0.0)
        spend = actions.sum(dim=-1)
        reward = z.sum(dim=-1) - spend
        self.H = (self.H - spend).clamp_min(0.0)
        self.t += 1
        done = self.t >= self.T
        projection_clip_mask = torch.zeros_like(current, dtype=torch.bool)
        if not done:
            residual_next = self.calibration.residual_next[self.noise_indices, self.t - 1].to(self.device, self.dtype)
            raw_next = prediction[..., 1:] + residual_next + effect["next_effect"]
            self.dynamic = _project_dynamic(raw_next, self.panel_cfg)
            projection_clip_mask = (self.dynamic - raw_next).abs() > 1e-6
            next_state = self._state()
        else:
            next_state = None
        info = {
            "Z": z,
            "Z0": z0,
            "subsidy_effect": effect["z_effect"],
            "next_state_effect": effect["next_effect"],
            "spend": spend,
            "spend_by_city": actions,
            "remaining_budget": self.H,
            "reward": reward,
            "noise_split": self.noise_split,
            "noise_idx": self.noise_indices.clone(),
            "paired_day_index": self.noise_indices.clone(),
            "initial_state_residual_pairing": True,
            "city_dynamic_state": current,
            "candidate_gross_scale": effect["candidate_gross_scale"],
            "projection_clip_mask": projection_clip_mask,
            "projection_applied": bool(not done),
        }
        return next_state, reward, done, info

    def gap_weights(self) -> torch.Tensor:
        return _normalize_weights((self.dynamic[..., 0] - self.dynamic[..., 1]).clamp_min(0.0) + EPS)

    def zone_gap_weights(self) -> torch.Tensor:
        return self.gap_weights()


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.clamp_min(0.0)
    denominator = weights.sum(dim=-1, keepdim=True)
    uniform = torch.ones_like(weights) / max(weights.shape[-1], 1)
    return torch.where(denominator > EPS, weights / denominator.clamp_min(EPS), uniform)


def baseline_action_from_state(
    method: str,
    budget_left: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    T: int,
    t: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Construct the shared scalar or batched operational-baseline action."""
    method = str(method).lower()
    if X.shape != Y.shape or X.ndim < 1:
        raise ValueError("Baseline X and Y tensors must have the same nonempty shape.")
    M = int(X.shape[-1])
    period_budget = torch.as_tensor(
        budget_left, device=X.device, dtype=X.dtype
    ) / float(max(int(T) - int(t), 1))
    while period_budget.ndim < X.ndim:
        period_budget = period_budget.unsqueeze(-1)
    if method in {"no_subsidy", "no subsidy"}:
        return torch.zeros_like(X)
    if method in {"uniform_pacing", "uniform pacing"}:
        return torch.full_like(X, 1.0 / M) * period_budget
    if method in {"gap_pacing", "demand_supply_gap", "demand-supply gap"}:
        return period_budget * _normalize_weights((X - Y).clamp_min(0.0) + EPS)
    if method in {"random_pacing", "random pacing"}:
        if generator is None:
            raise ValueError("Random pacing requires an explicit torch.Generator.")
        weights = torch.rand(X.shape, device=X.device, dtype=X.dtype, generator=generator)
        return period_budget * _normalize_weights(weights)
    raise ValueError(f"Unknown baseline method {method!r}.")


def baseline_action(env: PublicCityPanelEnv, method: str) -> torch.Tensor:
    return baseline_action_from_state(
        method,
        env.H,
        env.dynamic[..., 0],
        env.dynamic[..., 1],
        env.T,
        env.t,
        env.generator,
    )


def rollout_baseline(config: Optional[PublicCityPanelConfig], method: str, seed: int) -> Dict[str, float]:
    panel_cfg = config or PublicCityPanelConfig()
    env = PublicCityPanelEnv(panel_cfg, seed=int(seed), noise_split="evaluation")
    env.reset(seed)
    total_reward = 0.0
    total_response = 0.0
    total_spend = 0.0
    done = False
    while not done:
        action = baseline_action(env, method)
        _next_state, reward, done, info = env.step(action)
        total_reward += float(reward.item())
        total_response += float(info["Z"].sum().item())
        total_spend += float(info["spend"].item())
    return {
        "method": method,
        "seed": float(seed),
        "net_reward": total_reward,
        "gross_response_score": total_response,
        "spend": total_spend,
        "final_budget": float(env.H.item()),
    }
