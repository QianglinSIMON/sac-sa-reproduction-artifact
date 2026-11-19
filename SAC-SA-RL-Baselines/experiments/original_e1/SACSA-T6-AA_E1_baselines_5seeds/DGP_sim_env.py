from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from aa_data_based_simulator import (
    AA_AGGREGATION_METHOD,
    AADataBasedSubsidyEnv,
    AADataCalibration,
    AADataSimulatorConfig,
    calibrate_aa_simulator,
    write_calibration_outputs,
)


SUBSIDY_RESPONSE_SCALE = 0.10
CITY_OVERSPEND_WASTE = 0.03
FLOAT_DTYPES = {torch.float32, torch.float64}

# Keep the historical coefficient names because iterative_synthetic_sacsa.py
# writes DGP summaries by looping over DGP_COEF_KEYS. In the AA-data DGP these
# tensors are diagnostic summaries of the calibrated AA equations rather than
# randomly generated synthetic coefficients.
DGP_COEF_KEYS = (
    "alpha_x",
    "alpha_y",
    "alpha_z",
    "alpha_c",
    "phi_x",
    "phi_y",
    "phi_u",
    "rho_c",
    "lambda_x",
    "lambda_y",
    "gamma_x",
    "gamma_y",
    "gamma_z",
    "chi_c",
    "beta_O",
    "beta_1",
    "beta_2",
    "beta_c",
)

AA_DGP_M = 2
AA_DGP_T = 6
AA_DGP_BUDGET = 10.0


@dataclass
class SimConfig:
    """
    AA data-calibrated true DGP used by the maintained training pipeline.

    The DGP is calibrated from two anonymized AA dispatch panels. Taiyuan
    T=24 is summed in groups of four, and Hangzhou T=48 in groups of eight,
    yielding an M=2, T=6 panel. X is call volume, Y is online supply time,
    and Z is GMV on the anonymized modeling scale.

    The true DGP is fixed: baseline demand/supply/GMV dynamics are fitted from
    the AA panel, residual-bootstrap episodes sample an observed day with
    replacement, and a researcher-defined subsidy intervention is added on top.
    This module is therefore a controlled AA data-based simulator, not a
    causal estimate of subsidy effects in the original marketplace logs.
    """

    M: int = AA_DGP_M
    Time_intervals: int = AA_DGP_T
    D: int = 30
    daily_budget: float = AA_DGP_BUDGET

    # Kept for CLI compatibility with the synthetic DGP. They do not control
    # the AA bootstrap shocks, which come from calibrated residual paths.
    sigma_x: float = 1.0
    sigma_y: float = 1.0
    sigma_c: float = 1.0
    sigma2_z: float = 1.5
    rho_corr: float = 0.5

    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 2026


_CALIBRATION_CACHE: Dict[Tuple[int, str, float], AADataCalibration] = {}


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _aa_output_dir() -> Path:
    return _module_dir() / "aa_data_based_simulator_outputs"


def _aa_artifact_path() -> Path:
    return _aa_output_dir() / "aa_calibration_artifact.npz"


def _aa_runtime_config(
    cfg: SimConfig,
    seed: Optional[int] = None,
    daily_budget: Optional[float] = None,
) -> AADataSimulatorConfig:
    return AADataSimulatorConfig(
        daily_budget=float(cfg.daily_budget if daily_budget is None else daily_budget),
        target_t=int(cfg.Time_intervals),
        subsidy_response_scale=SUBSIDY_RESPONSE_SCALE,
        overspend_waste=CITY_OVERSPEND_WASTE,
        center_residual=False,
        device=cfg.device,
        dtype=cfg.dtype,
        seed=cfg.seed if seed is None else int(seed),
    )


def make_generator(device: str = "cpu", seed: Optional[int] = None) -> torch.Generator:
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))
    return gen


def assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"{name} contains nonfinite values.")


def budget_atol(reference: torch.Tensor, dtype: torch.dtype) -> float:
    scale = float(reference.detach().abs().max().cpu().item())
    if dtype == torch.float64:
        return 1e-10 * max(1.0, scale)
    return 1e-5 * max(1.0, scale)


def validate_sim_config(cfg: SimConfig) -> None:
    if cfg.dtype not in FLOAT_DTYPES:
        raise ValueError("dtype must be torch.float32 or torch.float64.")
    if cfg.M != AA_DGP_M:
        raise ValueError(f"The AA data-based DGP is fixed at M={AA_DGP_M}; got M={cfg.M}.")
    if cfg.Time_intervals != AA_DGP_T:
        raise ValueError(
            f"The AA data-based DGP is fixed at T={AA_DGP_T}; got T={cfg.Time_intervals}."
        )
    if cfg.D <= 0:
        raise ValueError("D must be positive.")
    if not math.isfinite(float(cfg.daily_budget)) or cfg.daily_budget <= 0:
        raise ValueError("daily_budget must be positive and finite.")


def _ensure_calibration(cfg: SimConfig) -> AADataCalibration:
    validate_sim_config(cfg)
    # The AA baseline dynamics and subsidy-response coefficients are calibrated
    # once at the reference budget. Runtime budget sweeps change only the
    # feasible daily budget constraint, not the underlying true DGP.
    key = (int(cfg.Time_intervals), AA_AGGREGATION_METHOD, float(AA_DGP_BUDGET))
    cached = _CALIBRATION_CACHE.get(key)
    if cached is not None:
        return cached

    artifact = _aa_artifact_path()
    calibration: Optional[AADataCalibration] = None
    if artifact.exists():
        loaded = AADataCalibration.load_npz(artifact)
        loaded_agg = str(loaded.metadata.get("aggregation_method", ""))
        if (
            loaded.T == cfg.Time_intervals
            and abs(loaded.daily_budget - AA_DGP_BUDGET) <= 1e-8
            and loaded_agg == AA_AGGREGATION_METHOD
        ):
            calibration = loaded

    if calibration is None:
        aa_cfg = _aa_runtime_config(cfg, daily_budget=AA_DGP_BUDGET)
        data_dir = _module_dir() / "data_sim"
        calibration, panel, coef_df, residual_df, diag_df = calibrate_aa_simulator(
            data_dir=data_dir,
            cfg=aa_cfg,
        )
        write_calibration_outputs(_aa_output_dir(), calibration, panel, coef_df, residual_df, diag_df)

    _CALIBRATION_CACHE[key] = calibration
    return calibration


def _to_tensor(x, cfg: SimConfig) -> torch.Tensor:
    return torch.as_tensor(x, device=cfg.device, dtype=cfg.dtype)


def validate_dgp_coefs(coefs: Dict[str, torch.Tensor], cfg: SimConfig) -> None:
    validate_sim_config(cfg)
    expected_shape = (cfg.Time_intervals,)
    missing = [key for key in DGP_COEF_KEYS if key not in coefs]
    if missing:
        raise ValueError(f"DGP coefficients are missing keys: {missing}.")
    extra = [key for key in coefs if key not in DGP_COEF_KEYS]
    if extra:
        raise ValueError(f"DGP coefficients contain unexpected keys: {extra}.")
    expected_device = torch.device(cfg.device)
    for key in DGP_COEF_KEYS:
        value = coefs[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"DGP coefficient {key} must be a torch.Tensor.")
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"DGP coefficient {key} must have shape {expected_shape}.")
        if value.dtype != cfg.dtype:
            raise ValueError(f"DGP coefficient {key} has dtype {value.dtype}, expected {cfg.dtype}.")
        if value.device.type != expected_device.type:
            raise ValueError(f"DGP coefficient {key} is on {value.device}, expected {expected_device}.")
        if expected_device.index is not None and value.device.index != expected_device.index:
            raise ValueError(f"DGP coefficient {key} is on {value.device}, expected {expected_device}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"DGP coefficient {key} contains nonfinite values.")


def subsidy_response(A: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(A).all():
        raise ValueError("Subsidy response received nonfinite action entries.")
    if torch.any(A < 0):
        raise ValueError("Subsidy response received negative action entries.")
    return SUBSIDY_RESPONSE_SCALE * torch.log1p(A / SUBSIDY_RESPONSE_SCALE)


def subsidy_effect_multipliers(
    X: torch.Tensor,
    Y: torch.Tensor,
    O: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ = O
    cfg = SimConfig(M=AA_DGP_M, Time_intervals=AA_DGP_T, daily_budget=AA_DGP_BUDGET, device=X.device.type, dtype=X.dtype)
    cal = _ensure_calibration(cfg)
    gap_scale = torch.as_tensor(float(cal.gap_scale), device=X.device, dtype=X.dtype).clamp_min(1e-8)
    positive_gap_std = torch.clamp(X - Y, min=0.0) / gap_scale
    gate = 0.10 + 1.90 * torch.sigmoid(6.0 * (positive_gap_std - 0.50))
    return gate, 0.60 * gate, 0.85 * gate


def context_from_xy(
    X: torch.Tensor,
    Y: torch.Tensor,
    cfg: Optional[SimConfig] = None,
) -> torch.Tensor:
    """AA state context C=(X-Y)_+/gap_scale, matching the true AA environment."""
    if cfg is None:
        cfg = SimConfig(M=AA_DGP_M, Time_intervals=AA_DGP_T, daily_budget=AA_DGP_BUDGET, device=X.device.type, dtype=X.dtype)
    cal = _ensure_calibration(cfg)
    gap_scale = torch.as_tensor(float(cal.gap_scale), device=X.device, dtype=X.dtype).clamp_min(1e-8)
    return torch.clamp(X - Y, min=0.0) / gap_scale


def generate_dgp_coefs(
    cfg: SimConfig,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    _ = generator
    cal = _ensure_calibration(cfg)
    coef_x = cal.coef_x.mean(axis=0)
    coef_y = cal.coef_y.mean(axis=0)
    coef_z = cal.coef_z.mean(axis=0)
    city_mean_gmv = cal.panel_z.mean(axis=(0, 2))
    city_o = (city_mean_gmv - city_mean_gmv.mean()) / max(float(city_mean_gmv.std()), 1e-8)
    beta_o = torch.full((cfg.Time_intervals,), float(city_o.mean()), device=cfg.device, dtype=cfg.dtype)
    zeros = torch.zeros((cfg.Time_intervals,), device=cfg.device, dtype=cfg.dtype)
    ones = torch.ones((cfg.Time_intervals,), device=cfg.device, dtype=cfg.dtype)
    coefs = {
        "alpha_x": _to_tensor(coef_x[:, 0], cfg),
        "alpha_y": _to_tensor(coef_y[:, 0], cfg),
        "alpha_z": _to_tensor(coef_z[:, 0], cfg),
        "alpha_c": zeros.clone(),
        "phi_x": _to_tensor(coef_x[:, 1], cfg),
        "phi_y": _to_tensor(coef_y[:, 2], cfg),
        "phi_u": zeros.clone(),
        "rho_c": zeros.clone(),
        "lambda_x": _to_tensor(coef_x[:, 2], cfg),
        "lambda_y": _to_tensor(coef_y[:, 1], cfg),
        "gamma_x": _to_tensor(cal.gamma_x.mean(axis=0), cfg),
        "gamma_y": _to_tensor(cal.gamma_y.mean(axis=0), cfg),
        "gamma_z": _to_tensor(cal.gamma_z.mean(axis=0), cfg),
        "chi_c": zeros.clone(),
        "beta_O": beta_o,
        "beta_1": _to_tensor(coef_z[:, 1], cfg),
        "beta_2": _to_tensor(coef_z[:, 2], cfg),
        "beta_c": ones.clone() * float(cal.gap_scale),
    }
    validate_dgp_coefs(coefs, cfg)
    return coefs


def generate_random_subsidies(
    cfg: SimConfig,
    generator: Optional[torch.Generator] = None,
    logging_budget_fraction: float = 1.0,
    spend_all_budget: bool = True,
) -> torch.Tensor:
    validate_sim_config(cfg)
    gen = generator or make_generator(cfg.device, cfg.seed)
    raw = torch.rand((cfg.D, cfg.M, cfg.Time_intervals), device=cfg.device, dtype=cfg.dtype, generator=gen)
    raw_sum = raw.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    daily_cap = float(cfg.daily_budget) * float(logging_budget_fraction)
    if spend_all_budget:
        spend = torch.ones((cfg.D, 1, 1), device=cfg.device, dtype=cfg.dtype)
    else:
        spend = torch.rand((cfg.D, 1, 1), device=cfg.device, dtype=cfg.dtype, generator=gen)
    return spend * daily_cap * raw / raw_sum


def check_budget(A: torch.Tensor, cfg: SimConfig, atol: Optional[float] = None) -> None:
    expected_shape = (cfg.D, cfg.M, cfg.Time_intervals)
    if A.ndim != 3:
        raise ValueError(f"Subsidy tensor must be 3-dimensional, got ndim={A.ndim}.")
    if tuple(A.shape) != expected_shape:
        raise ValueError(f"Subsidy tensor must have shape {expected_shape}, got {tuple(A.shape)}.")
    if not torch.is_floating_point(A):
        raise ValueError("Subsidy tensor must use a floating-point dtype.")
    if not torch.isfinite(A).all():
        raise ValueError("Subsidy tensor has nonfinite entries.")
    if torch.any(A < 0):
        raise ValueError("Subsidy tensor has negative entries.")
    daily_spend = A.sum(dim=(1, 2))
    cap = torch.tensor(float(cfg.daily_budget), device=A.device, dtype=A.dtype)
    if atol is None:
        atol = budget_atol(cap, A.dtype)
    if torch.any(daily_spend > cap + atol):
        raise ValueError("Subsidy tensor violates daily budget.")


def simulate_dgp(
    cfg: SimConfig,
    A: Optional[torch.Tensor] = None,
    coefs: Optional[Dict[str, torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    validate_sim_config(cfg)
    if coefs is None:
        coefs = generate_dgp_coefs(cfg, generator=generator)
    validate_dgp_coefs(coefs, cfg)
    if A is None:
        A = generate_random_subsidies(cfg, generator=generator)
    else:
        A = A.to(device=cfg.device, dtype=cfg.dtype)
    check_budget(A, cfg)

    O = torch.zeros((cfg.D, cfg.M), device=cfg.device, dtype=cfg.dtype)
    X = torch.zeros((cfg.D, cfg.M, cfg.Time_intervals + 1), device=cfg.device, dtype=cfg.dtype)
    Y = torch.zeros_like(X)
    C = torch.zeros_like(X)
    Z = torch.zeros((cfg.D, cfg.M, cfg.Time_intervals), device=cfg.device, dtype=cfg.dtype)
    Q = torch.zeros_like(Z)
    U = torch.zeros_like(Z)
    Ez = torch.zeros_like(Z)
    Ex = torch.zeros_like(Z)
    Ey = torch.zeros_like(Z)
    Ec = torch.zeros_like(Z)

    for d in range(cfg.D):
        env = SubsidyAllocationEnv(cfg=cfg, coefs=coefs, seed=cfg.seed + d, project_action=True)
        state = env.reset()
        O[d] = env.O
        X[d, :, 0] = env.X
        Y[d, :, 0] = env.Y
        C[d, :, 0] = env.C
        for t in range(cfg.Time_intervals):
            _, _, _, info = env.step(A[d, :, t])
            Z[d, :, t] = info["Z"]
            Q[d, :, t] = info["Q"]
            U[d, :, t] = info["U"]
            X[d, :, t + 1] = env.X
            Y[d, :, t + 1] = env.Y
            C[d, :, t + 1] = env.C
            Ez[d, :, t] = info["Z"] - info["baseline_z"]
            Ex[d, :, t] = env.X - X[d, :, t]
            Ey[d, :, t] = env.Y - Y[d, :, t]
    return {"O": O, "Q": Q, "U": U, "A": A, "Z": Z, "X": X, "Y": Y, "C": C, "Ez": Ez, "Ex": Ex, "Ey": Ey, "Ec": Ec, "coefs": coefs}


class SubsidyAllocationEnv:
    """
    Step-wise AA data-based true environment for dynamic subsidy allocation.

    Policy state:
      [X_t, Y_t, C_t, O, H_t,
       sin(2pi (t+1)/T), cos(2pi (t+1)/T), remaining_time_fraction]
    """

    def __init__(
        self,
        cfg: SimConfig,
        coefs: Optional[Dict[str, torch.Tensor]] = None,
        seed: Optional[int] = None,
        project_action: bool = False,
    ):
        validate_sim_config(cfg)
        self.cfg = cfg
        self.M = cfg.M
        self.T = cfg.Time_intervals
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.project_action = bool(project_action)
        env_seed = cfg.seed if seed is None else int(seed)
        self.generator = make_generator(cfg.device, env_seed)
        if coefs is None:
            coefs = generate_dgp_coefs(cfg)
        validate_dgp_coefs(coefs, cfg)
        self.coefs = coefs
        self.calibration = _ensure_calibration(cfg)
        aa_cfg = _aa_runtime_config(cfg, seed=env_seed)
        self._inner = AADataBasedSubsidyEnv(
            calibration=self.calibration,
            cfg=aa_cfg,
            seed=env_seed,
            project_action=project_action,
        )
        self.t = 0
        self.H: torch.Tensor
        self.O: torch.Tensor
        self.X: torch.Tensor
        self.Y: torch.Tensor
        self.C: torch.Tensor
        self._has_reset = False

    @property
    def state_dim(self) -> int:
        return 4 * self.M + 1 + 3

    @property
    def action_dim(self) -> int:
        return self.M

    def _sync_from_inner(self) -> None:
        self.t = int(self._inner.t)
        self.H = self._inner.H.detach().clone()
        self.X = torch.as_tensor(self._inner.X, device=self.device, dtype=self.dtype)
        self.Y = torch.as_tensor(self._inner.Y, device=self.device, dtype=self.dtype)
        self.C = torch.as_tensor(self._inner.C, device=self.device, dtype=self.dtype)
        self.O = torch.as_tensor(self._inner.O, device=self.device, dtype=self.dtype)

    def reset(self) -> torch.Tensor:
        state = self._inner.reset()
        self._sync_from_inner()
        self._has_reset = True
        return state

    def step(self, action: torch.Tensor):
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")
        x_before = self.X.detach().clone()
        y_before = self.Y.detach().clone()
        next_state, reward, done, info = self._inner.step(action)
        self._sync_from_inner()
        q = torch.minimum(x_before, y_before)
        u = torch.clamp(x_before - y_before, min=0.0)
        info["Q"] = q
        info["U"] = u
        info["matched"] = q
        info["excess"] = u
        return next_state, reward, done, info


if __name__ == "__main__":
    cfg = SimConfig()
    coefs = generate_dgp_coefs(cfg)
    env = SubsidyAllocationEnv(cfg, coefs=coefs, seed=cfg.seed, project_action=True)
    state = env.reset()
    total = 0.0
    done = False
    while not done:
        action = torch.ones(cfg.M, device=cfg.device, dtype=cfg.dtype) * (env.H / max(env.T - env.t, 1) / cfg.M)
        state, reward, done, info = env.step(action)
        total += float(reward.detach().cpu())
    print({"M": cfg.M, "T": cfg.Time_intervals, "B": cfg.daily_budget, "episode_reward": round(total, 4)})
