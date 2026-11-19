from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch


CITY_SPECS = {
    "taiyuan": {
        "file": "V2_taiyuan_serial_order_dispatch_AA.csv",
        "source_t": 24,
    },
    "hangzhou": {
        "file": "V2_hangzhou_serial_order_dispatch_AA.csv",
        "source_t": 48,
    },
}

AA_AGGREGATION_METHOD = "sum"


@dataclass
class AADataSimulatorConfig:
    """Runtime config for the AA data-based synthetic true environment."""

    daily_budget: float = 10.0
    target_t: int = 6
    subsidy_response_scale: float = 0.10
    overspend_waste: float = 0.03
    gamma_x_ratio: float = 0.60
    gamma_y_ratio: float = 0.85
    target_gap_gross_uplift_per_budget: float = 1.40
    center_residual: bool = False
    state_floor: float = 1e-3
    state_ceiling_multiplier: float = 1.75
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 2026


@dataclass
class AADataCalibration:
    city_names: List[str]
    dates: np.ndarray
    target_t: int
    daily_budget: float
    panel_x: np.ndarray
    panel_y: np.ndarray
    panel_z: np.ndarray
    coef_z: np.ndarray
    coef_x: np.ndarray
    coef_y: np.ndarray
    resid_z: np.ndarray
    resid_x: np.ndarray
    resid_y: np.ndarray
    residual_mean_z: np.ndarray
    residual_mean_x: np.ndarray
    residual_mean_y: np.ndarray
    gamma_z: np.ndarray
    gamma_x: np.ndarray
    gamma_y: np.ndarray
    gap_scale: float
    x_clip: np.ndarray
    y_clip: np.ndarray
    z_clip: np.ndarray
    metadata: Dict[str, Any]

    @property
    def n_days(self) -> int:
        return int(self.panel_x.shape[0])

    @property
    def n_cities(self) -> int:
        return int(self.panel_x.shape[1])

    @property
    def T(self) -> int:
        return int(self.target_t)

    def residual_arrays(self, center: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not center:
            return self.resid_z, self.resid_x, self.resid_y
        rz = self.resid_z - self.residual_mean_z.reshape(1, self.n_cities, self.T)
        rx = self.resid_x - self.residual_mean_x.reshape(1, self.n_cities, self.T)
        ry = self.resid_y - self.residual_mean_y.reshape(1, self.n_cities, self.T)
        return rz, rx, ry

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            city_names=np.asarray(self.city_names, dtype=object),
            dates=self.dates,
            target_t=np.asarray(self.target_t),
            daily_budget=np.asarray(self.daily_budget),
            panel_x=self.panel_x,
            panel_y=self.panel_y,
            panel_z=self.panel_z,
            coef_z=self.coef_z,
            coef_x=self.coef_x,
            coef_y=self.coef_y,
            resid_z=self.resid_z,
            resid_x=self.resid_x,
            resid_y=self.resid_y,
            residual_mean_z=self.residual_mean_z,
            residual_mean_x=self.residual_mean_x,
            residual_mean_y=self.residual_mean_y,
            gamma_z=self.gamma_z,
            gamma_x=self.gamma_x,
            gamma_y=self.gamma_y,
            gap_scale=np.asarray(self.gap_scale),
            x_clip=self.x_clip,
            y_clip=self.y_clip,
            z_clip=self.z_clip,
            metadata=np.asarray([self.metadata], dtype=object),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "AADataCalibration":
        data = np.load(path, allow_pickle=True)
        return cls(
            city_names=[str(x) for x in data["city_names"].tolist()],
            dates=data["dates"],
            target_t=int(data["target_t"]),
            daily_budget=float(data["daily_budget"]),
            panel_x=data["panel_x"],
            panel_y=data["panel_y"],
            panel_z=data["panel_z"],
            coef_z=data["coef_z"],
            coef_x=data["coef_x"],
            coef_y=data["coef_y"],
            resid_z=data["resid_z"],
            resid_x=data["resid_x"],
            resid_y=data["resid_y"],
            residual_mean_z=data["residual_mean_z"],
            residual_mean_x=data["residual_mean_x"],
            residual_mean_y=data["residual_mean_y"],
            gamma_z=data["gamma_z"],
            gamma_x=data["gamma_x"],
            gamma_y=data["gamma_y"],
            gap_scale=float(data["gap_scale"]),
            x_clip=data["x_clip"],
            y_clip=data["y_clip"],
            z_clip=data["z_clip"],
            metadata=dict(data["metadata"][0]),
        )


def aggregate_city_to_target_t(
    csv_path: Path,
    city: str,
    source_t: int,
    target_t: int = 6,
) -> pd.DataFrame:
    if source_t % target_t != 0:
        raise ValueError(f"source_t={source_t} must be divisible by target_t={target_t}.")
    group_size = source_t // target_t
    df = pd.read_csv(csv_path)
    required = {"date", "time", "gmv", "sum_online_time", "cnt_call"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    if int(df["time"].max()) != int(source_t):
        raise ValueError(f"{csv_path} has max time {df['time'].max()}, expected {source_t}.")
    out = df.copy()
    out["t"] = ((out["time"].astype(int) - 1) // group_size + 1).astype(int)
    agg = (
        out.groupby(["date", "t"], as_index=False)[["gmv", "sum_online_time", "cnt_call"]]
        .sum()
        .sort_values(["date", "t"])
    )
    agg["city"] = city
    agg = agg.rename(
        columns={
            "cnt_call": "X_demand",
            "sum_online_time": "Y_supply",
            "gmv": "Z_gmv",
        }
    )
    return agg[["date", "city", "t", "X_demand", "Y_supply", "Z_gmv"]]


def build_aa_panel(
    data_dir: Path,
    target_t: int = 6,
    city_specs: Mapping[str, Mapping[str, Any]] = CITY_SPECS,
) -> pd.DataFrame:
    rows = []
    for city, spec in city_specs.items():
        rows.append(
            aggregate_city_to_target_t(
                data_dir / str(spec["file"]),
                city=city,
                source_t=int(spec["source_t"]),
                target_t=target_t,
            )
        )
    panel = pd.concat(rows, ignore_index=True)
    common_dates = None
    for city in city_specs:
        dates = set(panel.loc[panel["city"] == city, "date"].unique())
        common_dates = dates if common_dates is None else common_dates & dates
    if not common_dates:
        raise ValueError("No common dates across AA city panels.")
    panel = panel[panel["date"].isin(sorted(common_dates))].copy()
    expected = len(common_dates) * len(city_specs) * target_t
    if len(panel) != expected:
        raise ValueError(f"Aggregated panel has {len(panel)} rows, expected {expected}.")
    return panel.sort_values(["date", "city", "t"]).reset_index(drop=True)


def _ols_fit(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if y.ndim != 1:
        y = y.reshape(-1)
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional.")
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    resid = y - pred
    sse = float(np.sum(resid**2))
    centered = y - float(np.mean(y))
    sst = float(np.sum(centered**2))
    r2 = 1.0 - sse / sst if sst > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean(resid**2)))
    return beta.astype(float), resid.astype(float), {"r2": float(r2), "rmse": rmse}


def _predict_linear(coef: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return coef[..., 0] + coef[..., 1] * x + coef[..., 2] * y


def _response_transform(action: np.ndarray, scale: float) -> np.ndarray:
    scale = max(float(scale), 1e-8)
    return scale * np.log1p(np.maximum(action, 0.0) / scale)


def _gap_gate(x: np.ndarray, y: np.ndarray, gap_scale: float) -> np.ndarray:
    scale = max(float(gap_scale), 1e-8)
    positive_gap_std = np.maximum(x - y, 0.0) / scale
    return 0.10 + 1.90 / (1.0 + np.exp(-6.0 * (positive_gap_std - 0.50)))


def _gap_policy_actions(
    x: np.ndarray,
    y: np.ndarray,
    daily_budget: float,
) -> np.ndarray:
    score = np.maximum(x - y, 0.0)
    if float(score.sum()) <= 1e-12:
        score = np.ones_like(score)
    return float(daily_budget) * score / max(float(score.sum()), 1e-12)


def calibrate_aa_simulator(
    data_dir: Path,
    cfg: AADataSimulatorConfig,
) -> Tuple[AADataCalibration, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = build_aa_panel(data_dir=data_dir, target_t=cfg.target_t)
    city_names = sorted(panel["city"].unique().tolist())
    dates = np.asarray(sorted(panel["date"].unique()), dtype=int)
    n_days = len(dates)
    m = len(city_names)
    T = int(cfg.target_t)

    city_to_idx = {city: i for i, city in enumerate(city_names)}
    date_to_idx = {date: i for i, date in enumerate(dates)}
    panel_x = np.zeros((n_days, m, T), dtype=float)
    panel_y = np.zeros((n_days, m, T), dtype=float)
    panel_z = np.zeros((n_days, m, T), dtype=float)
    for row in panel.itertuples(index=False):
        d = date_to_idx[int(row.date)]
        i = city_to_idx[str(row.city)]
        t = int(row.t) - 1
        panel_x[d, i, t] = float(row.X_demand)
        panel_y[d, i, t] = float(row.Y_supply)
        panel_z[d, i, t] = float(row.Z_gmv)

    coef_z = np.zeros((m, T, 3), dtype=float)
    coef_x = np.zeros((m, T, 3), dtype=float)
    coef_y = np.zeros((m, T, 3), dtype=float)
    resid_z = np.zeros((n_days, m, T), dtype=float)
    resid_x = np.zeros((n_days, m, T), dtype=float)
    resid_y = np.zeros((n_days, m, T), dtype=float)
    diag_rows: List[Dict[str, Any]] = []

    for i, city in enumerate(city_names):
        for t in range(T):
            design = np.column_stack([np.ones(n_days), panel_x[:, i, t], panel_y[:, i, t]])
            beta_z, rz, metrics_z = _ols_fit(panel_z[:, i, t], design)
            coef_z[i, t] = beta_z
            resid_z[:, i, t] = rz
            diag_rows.append(
                {
                    "city": city,
                    "t": t + 1,
                    "equation": "gmv_z",
                    "n": n_days,
                    "intercept": beta_z[0],
                    "beta_x": beta_z[1],
                    "beta_y": beta_z[2],
                    "r2": metrics_z["r2"],
                    "rmse": metrics_z["rmse"],
                    "residual_mean": float(np.mean(rz)),
                    "residual_sd": float(np.std(rz, ddof=1)),
                }
            )
            if t < T - 1:
                beta_x, rx, metrics_x = _ols_fit(panel_x[:, i, t + 1], design)
                beta_y, ry, metrics_y = _ols_fit(panel_y[:, i, t + 1], design)
                coef_x[i, t] = beta_x
                coef_y[i, t] = beta_y
                resid_x[:, i, t] = rx
                resid_y[:, i, t] = ry
                diag_rows.extend(
                    [
                        {
                            "city": city,
                            "t": t + 1,
                            "equation": "next_x_demand",
                            "n": n_days,
                            "intercept": beta_x[0],
                            "beta_x": beta_x[1],
                            "beta_y": beta_x[2],
                            "r2": metrics_x["r2"],
                            "rmse": metrics_x["rmse"],
                            "residual_mean": float(np.mean(rx)),
                            "residual_sd": float(np.std(rx, ddof=1)),
                        },
                        {
                            "city": city,
                            "t": t + 1,
                            "equation": "next_y_supply",
                            "n": n_days,
                            "intercept": beta_y[0],
                            "beta_x": beta_y[1],
                            "beta_y": beta_y[2],
                            "r2": metrics_y["r2"],
                            "rmse": metrics_y["rmse"],
                            "residual_mean": float(np.mean(ry)),
                            "residual_sd": float(np.std(ry, ddof=1)),
                        },
                    ]
                )
            else:
                # Terminal transition coefficients are unused. Keep identity-like
                # placeholders so arrays remain rectangular and easy to save.
                coef_x[i, t] = np.asarray([0.0, 1.0, 0.0])
                coef_y[i, t] = np.asarray([0.0, 0.0, 1.0])

    positive_gap = np.maximum(panel_x - panel_y, 0.0)
    gap_positive_vals = positive_gap[positive_gap > 1e-12]
    gap_scale = float(np.quantile(gap_positive_vals, 0.75)) if gap_positive_vals.size else 1.0
    gap_scale = max(gap_scale, 1e-6)

    base_gamma_profile = np.linspace(0.75, 1.25, T, dtype=float)
    mean_x = panel_x.mean(axis=0)
    mean_y = panel_y.mean(axis=0)
    gap_action = _gap_policy_actions(mean_x, mean_y, cfg.daily_budget)
    gate = _gap_gate(mean_x, mean_y, gap_scale)
    raw_uplift = float(
        np.sum(base_gamma_profile.reshape(1, T) * _response_transform(gap_action, cfg.subsidy_response_scale) * gate)
    )
    target_gross = float(cfg.target_gap_gross_uplift_per_budget) * float(cfg.daily_budget)
    gamma_scale = target_gross / max(raw_uplift, 1e-8)
    gamma_z_t = np.clip(base_gamma_profile * gamma_scale, 0.05, 10.0)
    gamma_z = np.tile(gamma_z_t.reshape(1, T), (m, 1))
    gamma_x = float(cfg.gamma_x_ratio) * gamma_z
    gamma_y = float(cfg.gamma_y_ratio) * gamma_z

    residual_mean_z = resid_z.mean(axis=0)
    residual_mean_x = resid_x.mean(axis=0)
    residual_mean_y = resid_y.mean(axis=0)
    x_clip = np.quantile(panel_x, 0.995, axis=0) * float(cfg.state_ceiling_multiplier)
    y_clip = np.quantile(panel_y, 0.995, axis=0) * float(cfg.state_ceiling_multiplier)
    z_clip = np.quantile(panel_z, 0.995, axis=0) * float(cfg.state_ceiling_multiplier)
    x_clip = np.maximum(x_clip, panel_x.max(axis=0) + 1e-3)
    y_clip = np.maximum(y_clip, panel_y.max(axis=0) + 1e-3)
    z_clip = np.maximum(z_clip, panel_z.max(axis=0) + 1e-3)

    calibration = AADataCalibration(
        city_names=city_names,
        dates=dates,
        target_t=T,
        daily_budget=float(cfg.daily_budget),
        panel_x=panel_x,
        panel_y=panel_y,
        panel_z=panel_z,
        coef_z=coef_z,
        coef_x=coef_x,
        coef_y=coef_y,
        resid_z=resid_z,
        resid_x=resid_x,
        resid_y=resid_y,
        residual_mean_z=residual_mean_z,
        residual_mean_x=residual_mean_x,
        residual_mean_y=residual_mean_y,
        gamma_z=gamma_z,
        gamma_x=gamma_x,
        gamma_y=gamma_y,
        gap_scale=gap_scale,
        x_clip=x_clip,
        y_clip=y_clip,
        z_clip=z_clip,
        metadata={
            "aggregation_method": AA_AGGREGATION_METHOD,
            "aggregation_group_size_by_city": {
                city: int(CITY_SPECS[city]["source_t"]) // int(T)
                for city in city_names
                if city in CITY_SPECS
            },
            "daily_budget": float(cfg.daily_budget),
            "subsidy_response_scale": float(cfg.subsidy_response_scale),
            "overspend_waste": float(cfg.overspend_waste),
            "gamma_x_ratio": float(cfg.gamma_x_ratio),
            "gamma_y_ratio": float(cfg.gamma_y_ratio),
            "target_gap_gross_uplift_per_budget": float(cfg.target_gap_gross_uplift_per_budget),
            "gamma_calibration_raw_uplift": raw_uplift,
            "gamma_calibration_target_gross": target_gross,
            "gap_scale": gap_scale,
        },
    )

    residual_rows = []
    for d_idx, date in enumerate(dates):
        for i, city in enumerate(city_names):
            for t in range(T):
                residual_rows.append(
                    {
                        "date": int(date),
                        "city": city,
                        "t": t + 1,
                        "resid_z": resid_z[d_idx, i, t],
                        "resid_x_next": resid_x[d_idx, i, t],
                        "resid_y_next": resid_y[d_idx, i, t],
                    }
                )

    coef_rows = []
    for i, city in enumerate(city_names):
        for t in range(T):
            for equation, arr in [("gmv_z", coef_z), ("next_x_demand", coef_x), ("next_y_supply", coef_y)]:
                if t == T - 1 and equation != "gmv_z":
                    continue
                coef_rows.append(
                    {
                        "city": city,
                        "t": t + 1,
                        "equation": equation,
                        "intercept": arr[i, t, 0],
                        "beta_x": arr[i, t, 1],
                        "beta_y": arr[i, t, 2],
                    }
                )

    return (
        calibration,
        panel,
        pd.DataFrame(coef_rows),
        pd.DataFrame(residual_rows),
        pd.DataFrame(diag_rows),
    )


class AADataBasedSubsidyEnv:
    """
    AA data-calibrated true environment with synthetic subsidy effects.

    Each episode samples one observed AA day with replacement. The initial
    state and the full residual path both come from that same day, preserving
    the empirical within-day and cross-city residual structure.
    """

    def __init__(
        self,
        calibration: AADataCalibration,
        cfg: Optional[AADataSimulatorConfig] = None,
        seed: Optional[int] = None,
        project_action: bool = True,
    ):
        self.calibration = calibration
        self.cfg = cfg or AADataSimulatorConfig(daily_budget=calibration.daily_budget, target_t=calibration.T)
        self.M = calibration.n_cities
        self.T = calibration.T
        self.project_action = bool(project_action)
        self.device = torch.device(self.cfg.device)
        self.dtype = self.cfg.dtype
        self.rng = np.random.default_rng(self.cfg.seed if seed is None else int(seed))
        self._resid_z_np, self._resid_x_np, self._resid_y_np = calibration.residual_arrays(
            center=bool(self.cfg.center_residual)
        )

        self.t = 0
        self.H: torch.Tensor
        self.X: np.ndarray
        self.Y: np.ndarray
        self.C: np.ndarray
        self.O: np.ndarray
        self._bootstrap_day_idx = 0
        self._day_idx = 0
        self._resid_idx = 0
        self._has_reset = False

    @property
    def state_dim(self) -> int:
        return 4 * self.M + 1 + 3

    @property
    def action_dim(self) -> int:
        return self.M

    def _time_features(self) -> torch.Tensor:
        tt = self.t + 1
        angle = 2.0 * np.pi * float(tt) / float(self.T)
        remaining = float(self.T - self.t) / float(self.T)
        return torch.tensor([np.sin(angle), np.cos(angle), remaining], device=self.device, dtype=self.dtype)

    def _context(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.maximum(x - y, 0.0) / max(float(self.calibration.gap_scale), 1e-8)

    def _get_state(self) -> torch.Tensor:
        if not self._has_reset:
            raise RuntimeError("Call reset() before requesting state.")
        h = self.H.reshape(1)
        tensors = [
            torch.as_tensor(self.X, device=self.device, dtype=self.dtype),
            torch.as_tensor(self.Y, device=self.device, dtype=self.dtype),
            torch.as_tensor(self.C, device=self.device, dtype=self.dtype),
            torch.as_tensor(self.O, device=self.device, dtype=self.dtype),
            h,
            self._time_features(),
        ]
        return torch.cat(tensors, dim=0)

    def reset(self) -> torch.Tensor:
        self.t = 0
        self.H = torch.tensor(float(self.cfg.daily_budget), device=self.device, dtype=self.dtype)
        self._bootstrap_day_idx = int(self.rng.integers(0, self.calibration.n_days))
        self._day_idx = self._bootstrap_day_idx
        self._resid_idx = self._bootstrap_day_idx
        self.X = self.calibration.panel_x[self._bootstrap_day_idx, :, 0].astype(float).copy()
        self.Y = self.calibration.panel_y[self._bootstrap_day_idx, :, 0].astype(float).copy()
        self.C = self._context(self.X, self.Y)
        city_mean_gmv = self.calibration.panel_z.mean(axis=(0, 2))
        self.O = (city_mean_gmv - city_mean_gmv.mean()) / max(float(city_mean_gmv.std()), 1e-8)
        self._has_reset = True
        return self._get_state()

    def _project_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.to(device=self.device, dtype=self.dtype).reshape(-1)
        if tuple(action.shape) != (self.M,):
            raise ValueError(f"action must have shape {(self.M,)}, got {tuple(action.shape)}.")
        if not torch.isfinite(action).all():
            raise ValueError("action contains nonfinite entries.")
        action = torch.clamp(action, min=0.0)
        total = action.sum()
        if total > self.H:
            action = action * (self.H / total.clamp_min(1e-8))
        return action

    def step(self, action: torch.Tensor):
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")
        if self.t >= self.T:
            raise RuntimeError("Episode is already done. Call reset().")

        A = self._project_action(action) if self.project_action else action.to(device=self.device, dtype=self.dtype)
        a_np = A.detach().cpu().numpy().astype(float)
        t = self.t
        cal = self.calibration

        coef_z_t = cal.coef_z[:, t, :]
        z_base = _predict_linear(coef_z_t, self.X, self.Y) + self._resid_z_np[self._resid_idx, :, t]
        psi = _response_transform(a_np, self.cfg.subsidy_response_scale)
        gate = _gap_gate(self.X, self.Y, cal.gap_scale)
        subsidy_z = cal.gamma_z[:, t] * psi * gate
        waste = float(self.cfg.overspend_waste) * np.square(a_np)
        z_np = np.clip(z_base + subsidy_z - waste, float(self.cfg.state_floor), cal.z_clip[:, t])

        if t < self.T - 1:
            x_base = _predict_linear(cal.coef_x[:, t, :], self.X, self.Y) + self._resid_x_np[self._resid_idx, :, t]
            y_base = _predict_linear(cal.coef_y[:, t, :], self.X, self.Y) + self._resid_y_np[self._resid_idx, :, t]
            x_next = x_base + cal.gamma_x[:, t] * psi * gate
            y_next = y_base + cal.gamma_y[:, t] * psi * gate
            self.X = np.clip(x_next, float(self.cfg.state_floor), cal.x_clip[:, t + 1])
            self.Y = np.clip(y_next, float(self.cfg.state_floor), cal.y_clip[:, t + 1])
        self.C = self._context(self.X, self.Y)

        reward = torch.as_tensor(float(np.sum(z_np) - np.sum(a_np)), device=self.device, dtype=self.dtype)
        self.H = torch.clamp(self.H - A.sum(), min=0.0)
        self.t += 1
        done = self.t >= self.T
        next_state = None if done else self._get_state()

        info: Dict[str, Any] = {
            "A": A.detach().clone(),
            "Z": torch.as_tensor(z_np, device=self.device, dtype=self.dtype),
            "spend": A.sum().detach().clone(),
            "remaining_budget": self.H.detach().clone(),
            "t": self.t,
            "bootstrap_day_idx": self._bootstrap_day_idx,
            "day_idx": self._day_idx,
            "residual_idx": self._resid_idx,
            "baseline_z": torch.as_tensor(z_base, device=self.device, dtype=self.dtype),
            "subsidy_z": torch.as_tensor(subsidy_z, device=self.device, dtype=self.dtype),
            "gate": torch.as_tensor(gate, device=self.device, dtype=self.dtype),
        }
        return next_state, reward, done, info


def _policy_action(policy: str, state: torch.Tensor, env: AADataBasedSubsidyEnv) -> torch.Tensor:
    x = state[: env.M]
    y = state[env.M : 2 * env.M]
    h = state[4 * env.M]
    remaining = max(env.T - env.t, 1)
    spend = h / float(remaining)
    if policy == "no_subsidy":
        return torch.zeros(env.M, device=state.device, dtype=state.dtype)
    if policy == "uniform_pacing":
        return torch.full((env.M,), spend / float(env.M), device=state.device, dtype=state.dtype)
    if policy == "demand_supply_gap":
        score = torch.clamp(x - y, min=0.0)
        if float(score.sum().detach().cpu()) <= 1e-12:
            score = torch.ones_like(score)
        return spend * score / score.sum().clamp_min(1e-8)
    if policy == "random_pacing":
        score = torch.rand(env.M, device=state.device, dtype=state.dtype)
        return spend * score / score.sum().clamp_min(1e-8)
    raise ValueError(f"Unknown policy: {policy}")


def rollout_policy(
    calibration: AADataCalibration,
    cfg: AADataSimulatorConfig,
    policy: str,
    n_episodes: int,
    seed: int,
) -> Dict[str, float]:
    rewards = []
    response_scores = []
    spends = []
    final_budgets = []
    for ep in range(int(n_episodes)):
        env = AADataBasedSubsidyEnv(calibration, cfg, seed=seed + ep, project_action=True)
        state = env.reset()
        total_reward = 0.0
        total_response = 0.0
        total_spend = 0.0
        done = False
        while not done:
            action = _policy_action(policy, state, env)
            state, reward, done, info = env.step(action)
            total_reward += float(reward.detach().cpu())
            total_response += float(info["Z"].sum().detach().cpu())
            total_spend += float(info["spend"].detach().cpu())
        rewards.append(total_reward)
        response_scores.append(total_response)
        spends.append(total_spend)
        final_budgets.append(float(env.H.detach().cpu()))
    arr_reward = np.asarray(rewards, dtype=float)
    arr_response = np.asarray(response_scores, dtype=float)
    arr_spend = np.asarray(spends, dtype=float)
    arr_budget = np.asarray(final_budgets, dtype=float)
    return {
        "policy": policy,
        "episodes": int(n_episodes),
        "net_reward_mean": float(arr_reward.mean()),
        "net_reward_se": float(arr_reward.std(ddof=1) / np.sqrt(len(arr_reward))) if len(arr_reward) > 1 else 0.0,
        "response_score_mean": float(arr_response.mean()),
        "spend_mean": float(arr_spend.mean()),
        "final_budget_mean": float(arr_budget.mean()),
    }


def write_calibration_outputs(
    out_dir: Path,
    calibration: AADataCalibration,
    panel: pd.DataFrame,
    coef_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    diag_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_dir / f"aa_t{calibration.T}_panel.csv", index=False)
    panel.to_csv(out_dir / "aa_panel.csv", index=False)
    coef_df.to_csv(out_dir / "aa_ols_coefficients.csv", index=False)
    residual_df.to_csv(out_dir / "aa_residual_paths.csv", index=False)
    diag_df.to_csv(out_dir / "aa_calibration_summary.csv", index=False)
    mechanism_rows = []
    city_mean_gmv = calibration.panel_z.mean(axis=(0, 2))
    city_o = (city_mean_gmv - city_mean_gmv.mean()) / max(float(city_mean_gmv.std()), 1e-8)
    for i, city in enumerate(calibration.city_names):
        for t in range(calibration.T):
            mechanism_rows.append(
                {
                    "city": city,
                    "t": t + 1,
                    "daily_budget": calibration.daily_budget,
                    "subsidy_response_scale": calibration.metadata.get("subsidy_response_scale", float("nan")),
                    "overspend_waste": calibration.metadata.get("overspend_waste", float("nan")),
                    "gap_scale": calibration.gap_scale,
                    "gamma_z": calibration.gamma_z[i, t],
                    "gamma_x": calibration.gamma_x[i, t],
                    "gamma_y": calibration.gamma_y[i, t],
                    "city_feature_o": city_o[i],
                    "city_mean_gmv": city_mean_gmv[i],
                }
            )
    pd.DataFrame(mechanism_rows).to_csv(out_dir / "aa_subsidy_mechanism.csv", index=False)
    calibration.save_npz(out_dir / "aa_calibration_artifact.npz")


def make_sanity_plots(out_dir: Path, diag_df: pd.DataFrame, sanity_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    gmv = diag_df[diag_df["equation"] == "gmv_z"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    for city, sub in gmv.groupby("city"):
        ax.plot(sub["t"], sub["r2"], marker="o", label=city)
    ax.set_xlabel("Time interval")
    ax.set_ylabel("OLS R2")
    ax.set_title("AA GMV baseline OLS fit by city and time")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "aa_gmv_ols_r2_by_time.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = sanity_df.sort_values("net_reward_mean", ascending=False)
    ax.bar(order["policy"], order["net_reward_mean"], yerr=order["net_reward_se"], capsize=3)
    ax.set_ylabel("Mean net reward")
    ax.set_title("AA data-based true simulator sanity policies")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plot_dir / "aa_policy_sanity_net_reward.png", dpi=200)
    plt.close(fig)
