from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import torch

SUBSIDY_RESPONSE_SCALE = 0.10
CITY_OVERSPEND_WASTE = 0.03
FLOAT_DTYPES = {torch.float32, torch.float64}
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


# =========================================================
# 0. Config
# =========================================================

@dataclass
class SimConfig:
    """
    One-dimensional city-level log-scale DGP simulator.

    Indices:
      d = 1,...,D: i.i.d. days / episodes
      i = 1,...,M: cities
      t = 1,...,T: within-day time intervals

    Variables:
      O_di      : day-city-level exogenous market-scale covariate
      X_dit     : log-scale demand indicator
      Y_dit     : log-scale supply indicator
      Q_dit     : transformed matched-market proxy, min(X_dit, Y_dit)
      U_dit     : positive transformed demand-supply gap, (X_dit - Y_dit)_+
      C_dit     : context / congestion / market condition
      A_dit     : nonnegative subsidy decision
      Z_dit     : log-scale market-response score

    DGP convention:
      Demand, supply, and market response are represented directly as log-scale
      synthetic indicators. The simulator generates and evaluates X/Y/Z on this
      scale and does not apply inverse scale transformations inside the DGP. In
      particular, Z is a transformed response score used for controlled policy
      comparison, not original-currency GMV.
      Subsidy cost and budget accounting use the raw A_dit units. Marketplace
      responses use the concave transform a0*log(1 + A_dit/a0), with a0=0.10 in
      synthetic cost units. The transformed response is multiplied by a bounded
      positive-gap gate based on (X_dit - Y_dit)_+. The response-score equation
      also includes a light city-level overspending waste term. Together these
      choices create diminishing marginal subsidy returns while making targeted
      dynamic allocation materially different from uniform spending.
      The synthetic net-reward index subtracts raw subsidy spending directly:
      sum(Z) - sum(A). Thus A and B are synthetic cost units on the same
      experiment scale as log-scale Z.
    """

    M: int = 8
    Time_intervals: int = 12
    D: int = 30

    daily_budget: float = 10.0

    # Noise-scale multipliers. The effective X/Y/C shock standard deviations are
    # 0.10*sigma_x, 0.10*sigma_y, and 0.08*sigma_c. The effective AR(1) Z shock
    # variance is 0.01*sigma2_z.
    sigma_x: float = 1.0
    sigma_y: float = 1.0
    sigma_c: float = 1.0
    sigma2_z: float = 1.5
    rho_corr: float = 0.5

    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 2026


# =========================================================
# 1. Random and transform helpers
# =========================================================

def make_generator(device: str = "cpu", seed: Optional[int] = None) -> torch.Generator:
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))
    return gen


def rand_uniform(
    low: float,
    high: float,
    size,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(
        size, device=device, dtype=dtype, generator=generator
    )


def rand_normal(
    mean: float,
    std: float,
    size,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if std < 0:
        raise ValueError("std must be nonnegative.")
    return mean + std * torch.randn(
        size, device=device, dtype=dtype, generator=generator
    )


def assert_finite(name: str, x: torch.Tensor) -> None:
    """Fail fast when the synthetic DGP produces nonfinite values."""
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"{name} contains nonfinite values.")


def validate_sim_config(cfg: SimConfig) -> None:
    if cfg.dtype not in FLOAT_DTYPES:
        raise ValueError("dtype must be torch.float32 or torch.float64.")
    if cfg.M <= 0:
        raise ValueError("M must be positive.")
    if cfg.Time_intervals <= 0:
        raise ValueError("Time_intervals must be positive.")
    if cfg.D <= 0:
        raise ValueError("D must be positive.")
    numeric_fields = {
        "daily_budget": cfg.daily_budget,
        "sigma_x": cfg.sigma_x,
        "sigma_y": cfg.sigma_y,
        "sigma_c": cfg.sigma_c,
        "sigma2_z": cfg.sigma2_z,
        "rho_corr": cfg.rho_corr,
    }
    for name, value in numeric_fields.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if cfg.daily_budget < 0:
        raise ValueError("daily_budget must be nonnegative.")
    if cfg.sigma_x < 0 or cfg.sigma_y < 0 or cfg.sigma_c < 0:
        raise ValueError("sigma_x, sigma_y, and sigma_c must be nonnegative.")
    if cfg.sigma2_z <= 0:
        raise ValueError("sigma2_z must be positive.")
    if not (-1.0 < cfg.rho_corr < 1.0):
        raise ValueError("rho_corr must be in (-1, 1).")


def validate_dgp_coefs(coefs: Dict[str, torch.Tensor], cfg: SimConfig) -> None:
    """Validate externally supplied DGP coefficients before simulation."""
    validate_sim_config(cfg)
    expected_device = torch.device(cfg.device)
    expected_shape = (cfg.Time_intervals,)

    missing = [key for key in DGP_COEF_KEYS if key not in coefs]
    if missing:
        raise ValueError(f"DGP coefficients are missing keys: {missing}.")

    extra = [key for key in coefs if key not in DGP_COEF_KEYS]
    if extra:
        raise ValueError(f"DGP coefficients contain unexpected keys: {extra}.")

    for key in DGP_COEF_KEYS:
        value = coefs[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"DGP coefficient {key} must be a torch.Tensor.")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"DGP coefficient {key} must have shape {expected_shape}, "
                f"got {tuple(value.shape)}."
            )
        if value.dtype != cfg.dtype:
            raise ValueError(f"DGP coefficient {key} has dtype {value.dtype}, expected {cfg.dtype}.")
        if value.device.type != expected_device.type:
            raise ValueError(
                f"DGP coefficient {key} is on device {value.device}, expected {expected_device}."
            )
        if expected_device.index is not None and value.device.index != expected_device.index:
            raise ValueError(
                f"DGP coefficient {key} is on device {value.device}, expected {expected_device}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"DGP coefficient {key} contains nonfinite values.")


def effective_sigma_x(cfg: SimConfig) -> float:
    return 0.10 * float(cfg.sigma_x)


def effective_sigma_y(cfg: SimConfig) -> float:
    return 0.10 * float(cfg.sigma_y)


def effective_sigma_c(cfg: SimConfig) -> float:
    return 0.08 * float(cfg.sigma_c)


def effective_sigma2_z(cfg: SimConfig) -> float:
    if cfg.sigma2_z <= 0:
        raise ValueError("sigma2_z must be positive.")
    return 0.01 * float(cfg.sigma2_z)


def budget_atol(reference: torch.Tensor, dtype: torch.dtype) -> float:
    """Tolerance for float roundoff in remaining-budget checks."""
    scale = float(reference.detach().abs().max().cpu().item())
    if dtype == torch.float64:
        return 1e-10 * max(1.0, scale)
    return 1e-5 * max(1.0, scale)


def subsidy_response(A: torch.Tensor) -> torch.Tensor:
    """Concave subsidy-response transform used in the synthetic DGP."""
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
    """
    Bounded positive-gap subsidy-effect multipliers.

    Subsidies are more productive in cities with larger positive transformed
    demand-supply gaps. This simple gate keeps the DGP interpretable and makes
    targeted allocation distinct from uniform pacing.
    """
    _ = O  # City heterogeneity still enters elsewhere through O.
    positive_gap = torch.clamp(X - Y, min=0.0)
    gate = 0.10 + 1.90 * torch.sigmoid(6.0 * (positive_gap - 0.50))
    z_multiplier = gate
    x_multiplier = 0.75 * gate
    y_multiplier = 0.90 * gate
    return z_multiplier, x_multiplier, y_multiplier


def blockwise_uniform(
    centers: Tuple[float, float, float, float],
    half_width: float,
    T: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw a common time-varying coefficient sequence from four time blocks."""
    vals = []
    for t in range(T):
        block = min(3, int(4 * t / max(T, 1)))
        center = centers[block]
        vals.append(
            rand_uniform(
                center - half_width,
                center + half_width,
                (),
                device,
                dtype,
                generator,
            )
        )
    return torch.stack(vals)


def halfwise_uniform(
    first_half: Tuple[float, float],
    second_half: Tuple[float, float],
    T: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw a time-varying coefficient sequence from low/high intraday regimes."""
    split = max(1, T // 2)
    vals = []
    for t in range(T):
        low, high = first_half if t < split else second_half
        vals.append(rand_uniform(low, high, (), device, dtype, generator))
    return torch.stack(vals)


def timewise_uniform(
    low: float,
    high: float,
    T: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return rand_uniform(low, high, (T,), device, dtype, generator)


# =========================================================
# 2. Time-correlated noise for log-scale Z
# =========================================================

def build_ar1_cov(
    T: int,
    rho: float = 0.5,
    sigma2: float = 1.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be in (-1, 1) for a valid AR(1)-type covariance.")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")

    idx = torch.arange(T, device=device)
    dist = torch.abs(idx[:, None] - idx[None, :]).to(dtype)
    return (sigma2 * (rho ** dist)).to(dtype=dtype)


def sample_mvn_zero(
    cov: torch.Tensor,
    shape_prefix: Tuple[int, ...],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix.")

    T = cov.shape[0]
    device = cov.device
    dtype = cov.dtype
    jitter = 1e-6 * torch.eye(T, device=device, dtype=dtype)
    L = torch.linalg.cholesky(cov + jitter)
    eps = torch.randn((*shape_prefix, T), device=device, dtype=dtype, generator=generator)
    return eps @ L.T


def sample_ar1_zero(
    T: int,
    rho: float,
    sigma2: float,
    shape_prefix: Tuple[int, ...],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample a stationary zero-mean AR(1) Gaussian path with covariance
    Cov(e_t, e_s) = sigma2 * rho**|t-s|.

    This is distribution-equivalent to drawing from the Toeplitz covariance used
    by build_ar1_cov/sample_mvn_zero, but avoids a Cholesky factorization at
    every synthetic day or environment reset.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be in (-1, 1) for a valid AR(1)-type covariance.")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")

    out = torch.empty((*shape_prefix, T), device=device, dtype=dtype)
    std0 = math.sqrt(float(sigma2))
    innov_std = math.sqrt(float(sigma2) * (1.0 - float(rho) ** 2))
    out[..., 0] = std0 * torch.randn((*shape_prefix,), device=device, dtype=dtype, generator=generator)
    if T == 1:
        return out
    eps = innov_std * torch.randn((*shape_prefix, T - 1), device=device, dtype=dtype, generator=generator)
    rho_t = torch.tensor(float(rho), device=device, dtype=dtype)
    for t in range(1, T):
        out[..., t] = rho_t * out[..., t - 1] + eps[..., t - 1]
    return out


# =========================================================
# 3. Generate DGP coefficients
# =========================================================

def generate_dgp_coefs(
    cfg: SimConfig,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate fixed-across-day, common time-varying coefficients for the
    log-scale DGP. Coefficient shapes are (T,), and the same coefficient
    sequence is shared across cities; city heterogeneity comes from O and
    stochastic shocks.
    """
    validate_sim_config(cfg)
    T = cfg.Time_intervals
    device, dtype = cfg.device, cfg.dtype

    coefs: Dict[str, torch.Tensor] = {}

    # Time-varying intercepts and response coefficients, following the
    # blockwise style in the original R DGP.
    coefs["alpha_x"] = halfwise_uniform((0.35, 0.60), (0.70, 1.00), T, device, dtype, generator)
    coefs["alpha_y"] = halfwise_uniform((0.20, 0.45), (0.50, 0.78), T, device, dtype, generator)
    coefs["alpha_z"] = halfwise_uniform((1.20, 1.55), (1.45, 1.85), T, device, dtype, generator)
    coefs["alpha_c"] = timewise_uniform(-0.05, 0.08, T, device, dtype, generator)

    coefs["phi_x"] = blockwise_uniform((0.35, 0.62, 0.55, 0.60), 0.04, T, device, dtype, generator)
    coefs["phi_y"] = blockwise_uniform((0.40, 0.58, 0.50, 0.65), 0.04, T, device, dtype, generator)
    coefs["phi_u"] = blockwise_uniform((0.03, 0.08, 0.05, 0.10), 0.02, T, device, dtype, generator)
    coefs["rho_c"] = timewise_uniform(0.45, 0.65, T, device, dtype, generator)

    coefs["lambda_x"] = timewise_uniform(0.02, 0.06, T, device, dtype, generator)
    coefs["lambda_y"] = timewise_uniform(0.02, 0.06, T, device, dtype, generator)
    coefs["gamma_x"] = halfwise_uniform((-0.03, 0.25), (0.10, 0.40), T, device, dtype, generator)
    coefs["gamma_y"] = halfwise_uniform((-0.03, 0.28), (0.12, 0.42), T, device, dtype, generator)
    coefs["gamma_z"] = halfwise_uniform((1.35, 3.30), (3.30, 6.00), T, device, dtype, generator)
    coefs["chi_c"] = timewise_uniform(0.04, 0.10, T, device, dtype, generator)

    coefs["beta_O"] = timewise_uniform(0.08, 0.18, T, device, dtype, generator)
    coefs["beta_1"] = halfwise_uniform((0.25, 0.45), (0.45, 0.75), T, device, dtype, generator)
    coefs["beta_2"] = halfwise_uniform((0.05, 0.15), (0.12, 0.25), T, device, dtype, generator)
    coefs["beta_c"] = timewise_uniform(0.03, 0.08, T, device, dtype, generator)

    validate_dgp_coefs(coefs, cfg)
    return coefs


# =========================================================
# 4. Generate random subsidy decisions for batch simulation
# =========================================================

def generate_random_subsidies(
    cfg: SimConfig,
    generator: Optional[torch.Generator] = None,
    logging_budget_fraction: float = 1.0,
    spend_all_budget: bool = True,
) -> torch.Tensor:
    """
    Generate batch random subsidies for the standalone batch simulator helper.

    The maintained iterative experiment uses collect_logged_trajectories() in
    iterative_synthetic_sacsa.py for logging-policy data. This helper is kept
    only for direct batch DGP simulation and uses the full daily budget unless
    an explicit logging_budget_fraction is supplied.
    """
    validate_sim_config(cfg)
    D, M, T = cfg.D, cfg.M, cfg.Time_intervals
    device, dtype = cfg.device, cfg.dtype
    logging_budget_fraction = float(logging_budget_fraction)
    if not 0.0 <= logging_budget_fraction <= 1.0:
        raise ValueError("logging_budget_fraction must be between 0 and 1.")

    raw = torch.rand((D, M, T), device=device, dtype=dtype, generator=generator)
    raw_sum = raw.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    daily_cap = cfg.daily_budget * logging_budget_fraction

    if spend_all_budget:
        spend = torch.ones((D, 1, 1), device=device, dtype=dtype)
    else:
        spend = torch.rand((D, 1, 1), device=device, dtype=dtype, generator=generator)

    return spend * daily_cap * raw / raw_sum


def check_budget(A: torch.Tensor, cfg: SimConfig, atol: Optional[float] = None) -> None:
    expected_shape = (cfg.D, cfg.M, cfg.Time_intervals)
    if A.ndim != 3:
        raise ValueError(f"Subsidy tensor must be 3-dimensional, got ndim={A.ndim}.")
    if tuple(A.shape) != expected_shape:
        raise ValueError(f"Subsidy tensor must have shape {expected_shape}, got {tuple(A.shape)}.")
    if not torch.is_floating_point(A):
        raise ValueError("Subsidy tensor must use a floating-point dtype.")
    daily_cap = torch.tensor(float(cfg.daily_budget), device=A.device, dtype=A.dtype)
    if not torch.isfinite(A).all():
        raise ValueError("Subsidy tensor has nonfinite entries.")
    if torch.any(A < 0):
        min_spend = A.min().item()
        raise ValueError(f"Subsidy tensor has negative entries: min = {min_spend:.6f}.")

    daily_spend = A.sum(dim=(1, 2))
    if atol is None:
        atol = budget_atol(daily_cap, A.dtype)

    if torch.any(daily_spend > daily_cap + atol):
        max_spend = daily_spend.max().item()
        raise ValueError(
            f"Budget violation: max daily spend = {max_spend:.6f}, "
            f"daily cap = {float(daily_cap.item()):.6f}."
        )


# =========================================================
# 5. Shared log-scale DGP equations
# =========================================================

def initial_latent_state(
    O: torch.Tensor,
    coefs: Dict[str, torch.Tensor],
    cfg: SimConfig,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = cfg.device, cfg.dtype

    x0 = (
        coefs["alpha_x"][0]
        + 0.15 * O
        + rand_normal(0.0, effective_sigma_x(cfg), O.shape, device, dtype, generator)
    )
    y0 = (
        coefs["alpha_y"][0]
        + 0.10 * O
        + rand_normal(0.0, effective_sigma_y(cfg), O.shape, device, dtype, generator)
    )
    c0 = (
        coefs["alpha_c"][0]
        + 0.10 * O
        + rand_normal(0.0, effective_sigma_c(cfg), O.shape, device, dtype, generator)
    )
    assert_finite("X_initial", x0)
    assert_finite("Y_initial", y0)
    assert_finite("C_initial", c0)
    return x0, y0, c0


def one_step_log_scale(
    X: torch.Tensor,
    Y: torch.Tensor,
    C: torch.Tensor,
    O: torch.Tensor,
    A: torch.Tensor,
    t: int,
    coefs: Dict[str, torch.Tensor],
    Ez_t: torch.Tensor,
    Ex_t: torch.Tensor,
    Ey_t: torch.Tensor,
    Ec_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    matched = torch.minimum(X, Y)
    excess = torch.clamp(X - Y, min=0.0)
    A_response = subsidy_response(A)
    z_subsidy_multiplier, x_subsidy_multiplier, y_subsidy_multiplier = subsidy_effect_multipliers(X, Y, O)

    Z = (
        coefs["alpha_z"][t]
        + coefs["beta_O"][t] * O
        + coefs["beta_1"][t] * matched
        + coefs["beta_2"][t] * excess
        + coefs["beta_c"][t] * C
        + coefs["gamma_z"][t] * A_response * z_subsidy_multiplier
        - CITY_OVERSPEND_WASTE * A.pow(2)
        + Ez_t
    )

    x_next = (
        coefs["alpha_x"][t]
        + coefs["phi_x"][t] * X
        + coefs["lambda_x"][t] * C
        + coefs["gamma_x"][t] * A_response * x_subsidy_multiplier
        + Ex_t
    )
    y_next = (
        coefs["alpha_y"][t]
        + coefs["phi_y"][t] * Y
        + coefs["phi_u"][t] * excess
        + coefs["lambda_y"][t] * C
        + coefs["gamma_y"][t] * A_response * y_subsidy_multiplier
        + Ey_t
    )
    C_next = (
        coefs["alpha_c"][t]
        + coefs["rho_c"][t] * C
        + coefs["chi_c"][t] * O
        + Ec_t
    )

    assert_finite("Z", Z)
    assert_finite("X_next", x_next)
    assert_finite("Y_next", y_next)
    assert_finite("C_next", C_next)
    return Z, x_next, y_next, C_next, matched, excess


# =========================================================
# 6. Batch simulator
# =========================================================

def simulate_dgp(
    cfg: SimConfig,
    A: Optional[torch.Tensor] = None,
    coefs: Optional[Dict[str, torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    validate_sim_config(cfg)
    if generator is None:
        generator = make_generator(cfg.device, cfg.seed)

    D, M, T = cfg.D, cfg.M, cfg.Time_intervals
    device, dtype = cfg.device, cfg.dtype

    if coefs is None:
        coefs = generate_dgp_coefs(cfg, generator=generator)
    validate_dgp_coefs(coefs, cfg)

    if A is None:
        A = generate_random_subsidies(cfg, generator=generator)
    else:
        A = A.to(device=device, dtype=dtype)
    check_budget(A, cfg)

    O = rand_normal(0.0, 1.0, (D, M), device, dtype, generator)

    X = torch.zeros((D, M, T + 1), device=device, dtype=dtype)
    Y = torch.zeros((D, M, T + 1), device=device, dtype=dtype)
    C = torch.zeros((D, M, T + 1), device=device, dtype=dtype)
    Z = torch.zeros((D, M, T), device=device, dtype=dtype)
    Q = torch.zeros((D, M, T), device=device, dtype=dtype)
    U = torch.zeros((D, M, T), device=device, dtype=dtype)

    x0, y0, c0 = initial_latent_state(O, coefs, cfg, generator)
    C[:, :, 0] = c0
    X[:, :, 0] = x0
    Y[:, :, 0] = y0

    Ez = sample_ar1_zero(
        T=T,
        rho=cfg.rho_corr,
        sigma2=effective_sigma2_z(cfg),
        shape_prefix=(D, M),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    Ex = rand_normal(0.0, effective_sigma_x(cfg), (D, M, T), device, dtype, generator)
    Ey = rand_normal(0.0, effective_sigma_y(cfg), (D, M, T), device, dtype, generator)
    Ec = rand_normal(0.0, effective_sigma_c(cfg), (D, M, T), device, dtype, generator)

    for t in range(T):
        Q[:, :, t] = torch.minimum(X[:, :, t], Y[:, :, t])
        U[:, :, t] = torch.clamp(X[:, :, t] - Y[:, :, t], min=0.0)
        Z_t, x_next, y_next, C_next, _, _ = one_step_log_scale(
            X[:, :, t],
            Y[:, :, t],
            C[:, :, t],
            O,
            A[:, :, t],
            t,
            coefs,
            Ez[:, :, t],
            Ex[:, :, t],
            Ey[:, :, t],
            Ec[:, :, t],
        )
        Z[:, :, t] = Z_t
        C[:, :, t + 1] = C_next
        X[:, :, t + 1] = x_next
        Y[:, :, t + 1] = y_next

    return {
        "O": O,
        "Q": Q,
        "U": U,
        "A": A,
        "Z": Z,
        "X": X,
        "Y": Y,
        "C": C,
        "Ez": Ez,
        "Ex": Ex,
        "Ey": Ey,
        "Ec": Ec,
        "coefs": coefs,
    }


# =========================================================
# 7. Step-wise RL environment
# =========================================================

class SubsidyAllocationEnv:
    """
    Step-wise RL environment for dynamic subsidy allocation.

    Policy state:
      [X_t, Y_t, C_t, O, H_t,
       sin(2pi (t+1)/T), cos(2pi (t+1)/T), remaining_time_fraction]

    X, Y, and Z are log-scale synthetic market indicators.
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
        self.project_action = project_action

        self.generator = make_generator(
            cfg.device,
            cfg.seed if seed is None else seed,
        )

        if coefs is None:
            coefs = generate_dgp_coefs(cfg, generator=self.generator)
        validate_dgp_coefs(coefs, cfg)
        self.coefs = coefs

        self.t: int = 0
        self.H: torch.Tensor
        self.O: torch.Tensor
        self.X: torch.Tensor
        self.Y: torch.Tensor
        self.C: torch.Tensor
        self.Ez: torch.Tensor
        self.Ex: torch.Tensor
        self.Ey: torch.Tensor
        self.Ec: torch.Tensor
        self._has_reset = False

    @property
    def state_dim(self) -> int:
        return 4 * self.M + 1 + 3

    @property
    def action_dim(self) -> int:
        return self.M

    def reset(self) -> torch.Tensor:
        self.t = 0
        self.H = torch.tensor(float(self.cfg.daily_budget), device=self.device, dtype=self.dtype)
        self.O = rand_normal(0.0, 1.0, (self.M,), self.device, self.dtype, self.generator)

        x0, y0, c0 = initial_latent_state(self.O, self.coefs, self.cfg, self.generator)
        self.C = c0
        self.X = x0
        self.Y = y0

        self.Ez = sample_ar1_zero(
            T=self.T,
            rho=self.cfg.rho_corr,
            sigma2=effective_sigma2_z(self.cfg),
            shape_prefix=(self.M,),
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )
        self.Ex = rand_normal(0.0, effective_sigma_x(self.cfg), (self.M, self.T), self.device, self.dtype, self.generator)
        self.Ey = rand_normal(0.0, effective_sigma_y(self.cfg), (self.M, self.T), self.device, self.dtype, self.generator)
        self.Ec = rand_normal(0.0, effective_sigma_c(self.cfg), (self.M, self.T), self.device, self.dtype, self.generator)

        self._has_reset = True
        return self._get_state()

    def _time_features(self) -> torch.Tensor:
        tt = self.t + 1
        angle = 2.0 * torch.pi * torch.tensor(
            float(tt) / float(self.T),
            device=self.device,
            dtype=self.dtype,
        )
        remaining_fraction = torch.tensor(
            float(self.T - self.t) / float(self.T),
            device=self.device,
            dtype=self.dtype,
        )
        return torch.stack([torch.sin(angle), torch.cos(angle), remaining_fraction])

    def _get_state(self) -> torch.Tensor:
        if not self._has_reset:
            raise RuntimeError("Call reset() before requesting the environment state.")
        h = self.H.reshape(1)
        return torch.cat([self.X, self.Y, self.C, self.O, h, self._time_features()], dim=0)

    def _project_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.to(device=self.device, dtype=self.dtype)
        if action.shape != (self.M,):
            raise ValueError(f"action must have shape {(self.M,)}, got {tuple(action.shape)}.")
        if not torch.isfinite(action).all():
            raise ValueError("Action has nonfinite entries.")
        if not torch.isfinite(self.H):
            raise FloatingPointError("Remaining budget is nonfinite.")
        action = torch.clamp(action, min=0.0)
        total = action.sum()
        if not torch.isfinite(total):
            raise ValueError("Projected action has nonfinite total spend.")
        if total > self.H:
            action = action * (self.H / total.clamp_min(1e-8))
        if not torch.isfinite(action).all():
            raise FloatingPointError("Projected action contains nonfinite entries.")
        return action

    def step(self, action: torch.Tensor):
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")
        if self.t >= self.T:
            raise RuntimeError("Episode is already done. Call reset().")

        if self.project_action:
            A = self._project_action(action)
        else:
            A = action.to(device=self.device, dtype=self.dtype)
            if A.shape != (self.M,):
                raise ValueError(f"action must have shape {(self.M,)}, got {tuple(A.shape)}.")
            if not torch.isfinite(A).all():
                raise ValueError("Action has nonfinite entries.")
            if torch.any(A < 0):
                raise ValueError("Action has negative entries.")
            total_spend = A.sum()
            if not torch.isfinite(total_spend):
                raise ValueError("Action has nonfinite total spend.")
            action_atol = budget_atol(self.H, self.dtype)
            if total_spend > self.H + action_atol:
                raise ValueError("Action violates remaining budget.")
            if total_spend > self.H:
                A = A * (self.H / total_spend.clamp_min(1e-8))
                if not torch.isfinite(A).all():
                    raise FloatingPointError("Budget-adjusted action contains nonfinite entries.")

        Q = torch.minimum(self.X, self.Y)
        U = torch.clamp(self.X - self.Y, min=0.0)

        Z, x_next, y_next, C_next, matched, excess = one_step_log_scale(
            self.X,
            self.Y,
            self.C,
            self.O,
            A,
            self.t,
            self.coefs,
            self.Ez[:, self.t],
            self.Ex[:, self.t],
            self.Ey[:, self.t],
            self.Ec[:, self.t],
        )

        reward = Z.sum() - A.sum()
        self.H = torch.clamp(self.H - A.sum(), min=0.0)
        self.C = C_next
        self.X = x_next
        self.Y = y_next
        self.t += 1

        done = self.t >= self.T
        next_state = None if done else self._get_state()

        info: Dict[str, Any] = {
            "A": A.detach().clone(),
            "Z": Z.detach().clone(),
            "Q": Q.detach().clone(),
            "U": U.detach().clone(),
            "matched": matched.detach().clone(),
            "excess": excess.detach().clone(),
            "spend": A.sum().detach().clone(),
            "remaining_budget": self.H.detach().clone(),
            "t": self.t,
        }
        return next_state, reward, done, info


# =========================================================
# 8. Example / sanity check
# =========================================================

if __name__ == "__main__":
    cfg = SimConfig(
        M=10,
        Time_intervals=12,
        D=30,
        daily_budget=10.0,
        rho_corr=0.5,
        sigma2_z=1.5,
        device="cpu",
        dtype=torch.float32,
        seed=2026,
    )

    gen = make_generator(cfg.device, cfg.seed)
    data = simulate_dgp(cfg, generator=gen)

    print("Batch simulation:")
    for key in ["O", "Q", "U", "A", "Z", "X", "Y", "C"]:
        print(f"{key}:", data[key].shape)
    print("daily spend:", data["A"].sum(dim=(1, 2))[:5])
    print(
        "X/Y/Z ranges:",
        (data["X"].min().item(), data["X"].max().item()),
        (data["Y"].min().item(), data["Y"].max().item()),
        (data["Z"].min().item(), data["Z"].max().item()),
    )

    Q_check = torch.minimum(data["X"][:, :, :-1], data["Y"][:, :, :-1])
    U_check = torch.clamp(data["X"][:, :, :-1] - data["Y"][:, :, :-1], min=0.0)
    print("Q check max abs diff:", (data["Q"] - Q_check).abs().max().item())
    print("U check max abs diff:", (data["U"] - U_check).abs().max().item())

    env = SubsidyAllocationEnv(cfg, seed=cfg.seed, project_action=False)
    state = env.reset()
    print("\nRL env:")
    print("state shape:", state.shape)
    print("state_dim:", env.state_dim)
    print("action_dim:", env.action_dim)
    total_reward = 0.0
    done = False
    while not done:
        a = torch.ones(cfg.M) * (env.H / max(env.T - env.t, 1) / cfg.M)
        state, r, done, info = env.step(a)
        total_reward += r.item()
    print("total reward:", total_reward)
    print("final remaining budget:", env.H.item())
